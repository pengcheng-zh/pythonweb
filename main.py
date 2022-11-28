import os.path

from flask import Flask, url_for
from flask import render_template
from flask import request
from flask import jsonify
from flask import session
from flask import redirect
import hashlib
import re
from Database import *

app = Flask(__name__, template_folder='./template', static_folder='./assets')


@app.route('/sign_in', methods=['GET'])
def sign_in():
    return render_template('sign_in.html')


@app.route('/sign_up', methods=['GET'])
def sign_up():
    return render_template('sign_up.html')


@app.route('/api_sign_in', methods=['POST'])
def api_sign_in():
    email = request.form.get('email')
    password = request.form.get('password')
    if email == '' or email is None:
        return jsonify({
            "code": 0,
            "message": "email is need"
        })
    if password == '' or password is None:
        return jsonify({
            "code": 0,
            "message": "password is need"
        })

    user = get_user_by_email(email)
    if not user:
        return jsonify({
            "code": 0,
            "message": "the email hasn't sign up"
        })
    if user['password'] != password:
        return jsonify({
            "code": 0,
            "message": "password incorrect"
        })
    session['user_id'] = user['id']
    return jsonify({
        "code": 1,
        "message": "success"
    })


@app.route('/api_sign_up', methods=['POST'])
def api_sign_up():
    email = request.form.get('email')
    password = request.form.get('password')
    name = request.form.get('name')
    telephone = request.form.get('telephone')
    if email == '' or email is None:
        return jsonify({
            "code": 0,
            "message": "email is need"
        })
    if password == '' or password is None:
        return jsonify({
            "code": 0,
            "message": "password is need"
        })
    if name == '' or name is None:
        return jsonify({
            "code": 0,
            "message": "name is need"
        })

    if telephone == '' or telephone is None:
        return jsonify({
            "code": 0,
            "message": "telephone is need"
        })
    check_result = check_password(password)
    if check_result != 'success':
        return jsonify({
            "code": 0,
            "message": check_result
        })

    user = get_user_by_email(email)
    if user:
        return jsonify({
            "code": 0,
            "message": "the email has sign up"
        })
    insert_sql = "insert into user(`email`, `password`, `name`, `telephone`, `role`) value('%s', '%s', '%s', '%s', " \
                 "'%s')" % (email, password, name, telephone, 'guest')
    db.prepare(insert_sql)
    db.commit()

    user = get_user_by_email(email)
    session['user_id'] = user['id']
    return jsonify({
        "code": 1,
        "message": "success"
    })


@app.route('/forget_password', methods=['GET'])
def forget_password():
    return render_template('forget_password.html')


@app.route('/api_forget_password', methods=['POST'])
def api_forget_password():
    email = request.form.get('email')
    telephone = request.form.get('telephone')
    if email == '' or email is None:
        return jsonify({
            "code": 0,
            "message": "email is need"
        })
    if telephone == '' or telephone is None:
        return jsonify({
            "code": 0,
            "message": "password is need"
        })

    sql = "select * from user where email = '%s' and telephone = '%s' limit 1" % (email, telephone)
    db.prepare(sql)
    result = db.cursor.fetchone()
    if not result:
        return jsonify({
            "code": 0,
            "message": "the email hasn't sign up"
        })
    user = {}
    index = 0
    for key in db.cursor.description:
        user[key[0]] = result[index]
        index += 1
    return jsonify({
        "code": 1,
        "message": "your password is: %s" % user['password']
    })


def check_password(password):
    reg_number = ".*\\d+.*"
    reg_upper_case = ".*[A-Z]+.*"
    reg_lower_case = ".*[a-z]+.*"
    reg_symbol = ".*[~!@#$%^&*()_+|<>,.?/:;'\\[\\]{}\"]+.*"
    if len(password) < 6:
        return "password length should at least 6"
    if not re.match(reg_number, password):
        return "password should contain at least one number"
    if not re.match(reg_upper_case, password):
        return "password should contain at least on upper letter"
    if not re.match(reg_lower_case, password):
        return "password should contain at least on lower letter"
    if not re.match(reg_symbol, password):
        return "password should contain at least on special letter"

    return "success"


def get_user_by_email(email):
    sql = "select * from user where email = '%s' limit 1" % email
    db.prepare(sql)
    result = db.cursor.fetchone()
    if not result:
        return None
    user = {}
    index = 0
    for key in db.cursor.description:
        user[key[0]] = result[index]
        index += 1
    return user


def get_user_by_id(user_id):
    sql = "select * from user where id = %d limit 1" % user_id
    db.prepare(sql)
    result = db.cursor.fetchone()
    if not result:
        return None
    user = {}
    index = 0
    for key in db.cursor.description:
        user[key[0]] = result[index]
        index += 1
    return user


@app.route('/evaluation', methods=['GET'])
def evaluation_page():
    user_id = session.get('user_id')
    if user_id is None:
        return redirect(url_for('sign_in'))

    user = get_user_by_id(user_id)
    if not user:
        return redirect(url_for('sign_in'))

    admin = user['role'] == 'admin'
    return render_template('evaluation_page.html', is_admin=admin)


@app.route('/evaluation', methods=['POST'])
def api_evaluation():
    user_id = session.get('user_id')
    if user_id is None:
        return redirect(url_for('sign_in'))

    user = get_user_by_id(user_id)
    if not user:
        return redirect(url_for('sign_in'))

    file_path = ''
    f = request.files['file']
    if f:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
        f.save(file_path)

    comment = request.form.get('evaluation')
    if comment == '':
        return redirect(url_for('evaluation_page'))

    contact = request.form.get('contact')
    sql = "insert into evaluation(`user_id`, `comment`, `contact`, `file_path`) value(%d, '%s', '%s', '%s')" % (user_id, comment, contact, file_path)
    db.prepare(sql)
    db.commit()
    if user['role'] == 'admin':
        return redirect(url_for('evaluation_list_page'))
    else:
        return redirect(url_for('evaluation_page'))


@app.route('/evaluation_list', methods=['GET'])
def evaluation_list_page():
    user_id = session.get('user_id')
    if user_id is None:
        return redirect(url_for('sign_in'))

    user = get_user_by_id(user_id)
    if not user:
        return redirect(url_for('sign_in'))

    if user['role'] != 'admin':
        return redirect(url_for('evaluation'))

    sql = 'select evaluation.id, evaluation.user_id, evaluation.comment, evaluation.file_path, evaluation.contact, ' \
          'user.name from evaluation join user on user.id = evaluation.user_id'
    db.prepare(sql)
    result = db.cursor.fetchall()
    if not result:
        return render_template('evaluation_list_page.html', comment_list=[])

    comment_list = []
    for item in result:
        index = 0
        comment = {}
        for key in db.cursor.description:
            comment[key[0]] = item[index]
            index += 1
        comment_list.append(comment)
    return render_template('evaluation_list_page.html', comment_list=comment_list)


if __name__ == '__main__':
    db = Database()
    app.secret_key = 'Lovejoy'
    app.config['UPLOAD_FOLDER'] = './files'
    app.run(port=8000)