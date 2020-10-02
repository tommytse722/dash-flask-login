from sqlalchemy import Table
from sqlalchemy.sql import select
from flask_sqlalchemy import SQLAlchemy
from config import engine
import pandas as pd
import datetime

db = SQLAlchemy()


class Optimal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    strategy_name = db.Column(db.String(20))
    stock_code = db.Column(db.String(20))
    capital = db.Column(db.Integer)
    x = db.Column(db.Integer)
    y = db.Column(db.Integer)
    last_updated = db.Column(db.DateTime(), default=datetime.datetime.now)

Optimal_tbl = Table('optimal', Optimal.metadata)


def create_optimal_table():
    Optimal.metadata.create_all(engine)
    
    
def drop_optimal_table():
    Optimal_tbl.drop(engine)


def add_optimal(strategy_name, stock_code, capital, x, y):
    insert_cmd = Optimal_tbl.insert().values(strategy_name=strategy_name, stock_code=stock_code, capital=capital, x=x, y=y)
    conn = engine.connect()
    conn.execute(insert_cmd)
    conn.close()