import pymysql


class Database(object):
    def __init__(self):
        self.db = pymysql.connect(host="localhost", user="root", password="123456", port=3306, database="demo")
        self.cursor = self.db.cursor()

    def prepare(self, sql):
        return self.cursor.execute(sql)

    def commit(self):
        self.db.commit()

    def close(self):
        self.db.close()