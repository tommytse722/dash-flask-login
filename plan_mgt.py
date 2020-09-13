from sqlalchemy import Table
from sqlalchemy.sql import select
from flask_sqlalchemy import SQLAlchemy
from config import engine
import pandas as pd

db = SQLAlchemy()


class Plan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer)
    strategy_id = db.Column(db.Integer)
    stock_code = db.Column(db.String(20))
    capital = db.Column(db.Integer)


Plan_tbl = Table('plan', Plan.metadata)


def create_plan_table():
    Plan.metadata.create_all(engine)
    
    
def drop_plan_table():
    Plan_tbl.drop(engine)


def add_plan(user_id, strategy_id, stock_code, capital):
    insert_cmd = Plan_tbl.insert().values(user_id=user_id, strategy_id=strategy_id, stock_code=stock_code, capital=capital)
    conn = engine.connect()
    conn.execute(insert_cmd)
    conn.close()
    

def del_plan(user_id):
    delete_cmd = Plan_tbl.delete().where(Plan_tbl.c.user_id == user_id)

    conn = engine.connect()
    conn.execute(delete_cmd)
    conn.close()

def show_plan():
    select_cmd = select([Plan_tbl.c.id, Plan_tbl.c.user_id, Plan_tbl.c.strategy_id, Plan_tbl.c.stock_code, Plan_tbl.c.capital])

    conn = engine.connect()
    rs = conn.execute(select_cmd)

    for row in rs:
        print(row)

    conn.close()
