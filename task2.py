import gzip
import datasets
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
from torch.utils.data import (Dataset, DataLoader, random_split, SequentialSampler)
import math
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau

import conlleval
import itertools

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NERDataset(Dataset):
    def __init__(self, dataset, vocab, tagset, no_targets=False):
        self.dataset = dataset
        self.no_targets = no_targets
        self.vocab = vocab
        self.tag_set = tagset

        self.words, self.targets = self.get_words()
        self.vocab_map = {word: i for (i, word) in enumerate(self.vocab)}

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        words = self.words[idx]
        case_bool = torch.Tensor([
            0 if i.lower() == i else 1 for i in words
        ])

        word = torch.LongTensor([self.vocab_map.get(i.lower(), 0)
                                 for i in words])
        n_words = len(word)

        if not self.no_targets:
            targets = self.targets[idx]
            targets = torch.Tensor([self.tag_set[i] for i in targets])
            return (word, case_bool, n_words), targets.type(torch.LongTensor)
        else:
            return (word, case_bool, n_words), None

    def get_words(self):
        data_set = []
        targets = []

        tag2idx = {'<PAD>': -1, 'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6,
                   'B-MISC': 7, 'I-MISC': 8}
        idx_to_tag = {i: tag for tag, i in tag2idx.items()}

        for key in range(self.dataset.num_rows):
            words = self.dataset[key]['tokens']
            ner_tags = self.dataset[key]['ner_tags']

            data_set.append(words)

            if not self.no_targets:
                targets.append([idx_to_tag[i] for i in ner_tags])

        return data_set, targets


def collate_fn(data):
    max_len = max([l for (_, _, l), _ in data])
    batch_len = len(data)
    sentences_batched = torch.zeros((batch_len, max_len), dtype=torch.long)
    case_batched = torch.zeros((batch_len, max_len), dtype=torch.bool)
    lengths_batched = []
    targets_batched = torch.zeros((batch_len, max_len), dtype=torch.long)

    for i, ((sentence, case_bool, length), target) in enumerate(data):
        pad_length = max_len - length
        padding = torch.nn.ConstantPad1d((0, pad_length), 0)
        tag_padding = torch.nn.ConstantPad1d((0, pad_length), -1)
        sentence = padding(sentence)
        sentences_batched[i, :] = sentence

        case_bool = padding(case_bool)
        case_batched[i, :] = case_bool

        if target is not None:
            target = tag_padding(target)
            targets_batched[i, :] = target

        lengths_batched.append(length)

    sentences_batched = torch.Tensor(sentences_batched)
    case_batched = torch.Tensor(case_batched)
    lengths_batched = torch.Tensor(lengths_batched)

    targets_batched = torch.Tensor(targets_batched)

    return (sentences_batched, case_batched, lengths_batched), targets_batched


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=100, word_embeddings=None, **kwargs):
        super(BiLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.tagset_size = tagset_size

        if word_embeddings is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding.from_pretrained(word_embeddings, freeze=True, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embedding_dim + 1,
            hidden_size=kwargs.get('hidden_size', 256),
            num_layers=kwargs.get('num_layers', 1),
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=kwargs.get('dropout', 0.33))
        self.linear = nn.Linear(
            in_features=2 * kwargs.get('hidden_size', 256),
            out_features=128
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=self.tagset_size),
        )
        self.elu = nn.ELU()

    def forward(self, sentences, case_bool, lengths):
        x = self.embedding(sentences)

        case_bool = torch.unsqueeze(case_bool, dim=2)

        x = torch.cat([x, case_bool], dim=2)
        x = pack_padded_sequence(x, lengths, batch_first=True,
                                 enforce_sorted=False)

        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.dropout(x)

        x = self.linear(x)
        x = self.elu(x)
        x = self.classifier(x)

        return x


def train_model(model, optimizer, criterion,
                train_loader,
                test_loader,
                num_epochs=30,
                lr_scheduler=None):
    device_model = model.to(device)
    for epoch in range(num_epochs):
        metrics = {
            'train_acc': 0,
            'train_loss': 0.0,
            'valid_acc': 0,
            'valid_loss': 0.0
        }

        for i, ((X, case, lengths), y) in enumerate(train_loader):
            device_model.train()
            optimizer.zero_grad()

            # Move to device
            X = X.to(device)
            y = y.to(device)
            case = case.to(device)

            # Forward pass
            outputs = model(X, case, lengths)
            outputs = outputs.permute(0, 2, 1)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            metrics['train_acc'] += (torch.argmax(outputs, axis=1) == y).float().sum() / sum(lengths)
            metrics['train_loss'] += loss

        metrics['train_acc'] /= len(train_loader)
        metrics['train_loss'] /= len(train_loader)

        for i, ((X, case, lengths), y) in enumerate(test_loader):
            device_model.eval()

            # Move to GPU
            X = X.to(device)
            y = y.to(device)
            case = case.to(device)

            # Forward pass
            outputs = model(X, case, lengths)
            outputs = outputs.permute(0, 2, 1)

            # Calculate the accuracy
            metrics['valid_acc'] += (torch.argmax(outputs, axis=1) == y).float().sum() / sum(lengths)
            # Calculate the loss
            metrics['valid_loss'] += loss

        if lr_scheduler is not None:
            if isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(metrics['valid_loss'])
            else:
                lr_scheduler.step()

        metrics['valid_acc'] /= len(test_loader)
        metrics['valid_loss'] /= len(test_loader)

        print(f"Epoch: {epoch + 1}/{num_epochs}")
        print("Mode\tLoss\tAcc")
        print(f"Train\t{metrics['train_loss']:.2f}\t{metrics['train_acc']:.2f}")
        print(f"Valid\t{metrics['valid_loss']:.2f}\t{metrics['valid_acc']:.2f}")
    return model


def read_data(dataset):
    tag2idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    idx_to_tag = {i: tag for tag, i in tag2idx.items()}

    counts = {}
    tags = []
    for num in range(dataset.num_rows):
        words = dataset[num]['tokens']
        ner_tags = dataset[num]['ner_tags']
        for word in words:
            word = word.lower()
            tags.extend([idx_to_tag[i] for i in ner_tags])
            if word in counts:
                counts[word] += 1
            else:
                counts[word] = 1

    counts = {k: v for k, v in sorted(counts.items(),
                                      key=lambda item: item[1],
                                      reverse=True)}
    vocab = ['<unk>']

    unknown_count = 0

    min_freq_thresh = 1

    for i, (word, count) in enumerate(counts.items(), start=1):
        if count >= min_freq_thresh:
            vocab.append(word)
        else:
            unknown_count += count

    tag_weight = compute_class_weight(
        class_weight='balanced',
        classes=np.array(list(tag2idx.keys())),
        y=tags)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag_weight = torch.Tensor(tag_weight).to(device)
    num_tags = len(tag2idx)
    return vocab, tag2idx, tag_weight, num_tags, idx_to_tag


def load_glove_vec():
    glove_vec = {'<unk>': np.zeros(100)}
    with gzip.open('glove.6B.100d.gz', 'rb') as file:
        lines = file.readlines()

    for line in lines:
        line = line.decode('utf-8').split(' ')
        glove_vec.__setitem__(
            line[0],
            np.asarray(line[1:], dtype='float32')
        )

    return glove_vec


train_data, test_data, val_data = datasets.load_dataset("conll2003", split=('train', 'test', 'validation'))

to_remove = ['id', 'pos_tags', 'chunk_tags']

train_data = train_data.remove_columns(to_remove)
train_vocab, train_tag2idx, train_tag_weight, train_num_tags, idx2tag = read_data(train_data)

glove_vec = load_glove_vec()
word_embeddings = torch.Tensor(np.vstack(list(glove_vec.values())))
num_tags = len(train_tag2idx)

tag_weight = torch.Tensor(train_tag_weight).to(device)

seed = 42
batch_size_2 = 64

train_dataset = NERDataset(train_data, list(glove_vec.keys()), train_tag2idx)
valid_dataset = NERDataset(val_data, list(glove_vec.keys()), train_tag2idx)
test_dataset = NERDataset(test_data, list(glove_vec.keys()), train_tag2idx)

train_len = math.floor(0.8 * len(train_dataset))
validation_len = len(train_dataset) - train_len
train_dataset, val_dataset = random_split(
    train_dataset,
    [train_len, validation_len],
    torch.Generator().manual_seed(seed))

val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size_2,
    shuffle=False,
    collate_fn=collate_fn,
    generator=torch.Generator().manual_seed(seed)
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size_2,
    shuffle=True,
    collate_fn=collate_fn,
    generator=torch.Generator().manual_seed(seed)
)

model = BiLSTM(
    vocab_size=len(list(glove_vec.keys())),
    tagset_size=num_tags,
    embedding_dim=100,
    word_embeddings=word_embeddings
)

criterion = nn.CrossEntropyLoss(weight=tag_weight, ignore_index=-1)
optim = torch.optim.SGD(model.parameters(), lr=0.8, momentum=0.5)

model = train_model(
    model=model,
    optimizer=optim,
    criterion=criterion,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    num_epochs=30
)


def model_test(model, test_dataset, idx2tag):
    sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn,
                                 sampler=sampler)
    model = model.to(device)

    tag_true = []
    tag_preds = []
    for i, ((X, case_bool, lengths), y) in enumerate(tqdm(test_dataloader)):
        model.eval()

        # Move to GPU
        X = X.to(device)
        y = y.to(device)
        case_bool = case_bool.to(device)

        output = model(X, case_bool, lengths)
        output = torch.argmax(output, axis=2)

        for j in range(len(output)):
            tags_a = []
            tags_b = []
            for k in range(int(lengths[j])):
                tags_a.append(idx2tag[int(y[j][k])])
                tags_b.append(idx2tag[int(output[j][k])])
            tag_true.append(tags_a)
            tag_preds.append(tags_b)

    precision, recall, f1 = conlleval.evaluate(itertools.chain(*tag_true), itertools.chain(*tag_preds))

    print(f"Precision\tRecall\tF1_score")
    print(f"{precision:.2f}\t{recall:.2f}\t{f1:.2f}")


print("test data :")
model_test(model, test_dataset, idx2tag)

print("validation data :")
model_test(model, valid_dataset, idx2tag)
