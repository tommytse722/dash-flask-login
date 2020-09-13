from sqlalchemy import Table
from sqlalchemy.sql import select
from flask_sqlalchemy import SQLAlchemy
from config import engine

db = SQLAlchemy()


class Strategy(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), unique=True)
    x_name = db.Column(db.String(20))
    y_name = db.Column(db.String(20))
    x_default = db.Column(db.Integer)
    y_default = db.Column(db.Integer)
    x_start = db.Column(db.Integer)
    x_end = db.Column(db.Integer)
    x_step = db.Column(db.Integer)
    y_start = db.Column(db.Integer)
    y_end = db.Column(db.Integer)
    y_step = db.Column(db.Integer)

Strategy_tbl = Table('strategy', Strategy.metadata)


def create_strategy_table():
    Strategy.metadata.create_all(engine)
    
    
def drop_strategy_table():
    Strategy_tbl.drop(engine)


def add_strategy(name, x_name, y_name, x_default, y_default, x_start, x_end, x_step, y_start, y_end, y_step):

    insert_cmd = Strategy_tbl.insert().values(name=name, x_name=x_name, y_name=y_name, x_default=x_default, y_default=y_default, x_start=x_start, x_end=x_end, x_step=x_step, y_start=y_start, y_end=y_end, y_step=y_step)

    conn = engine.connect()
    conn.execute(insert_cmd)
    conn.close()


def del_strategy(name):
    delete_cmd = Strategy_tbl.delete().where(Strategy_tbl.c.name == name)

    conn = engine.connect()
    conn.execute(delete_cmd)
    conn.close()


def show_strategy():
    select_cmd = select([Strategy_tbl.c.id, Strategy_tbl.c.name, Strategy_tbl.c.x_name, Strategy_tbl.c.y_name])

    conn = engine.connect()
    rs = conn.execute(select_cmd)

    for row in rs:
        print(row)

    conn.close()
