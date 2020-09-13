from sqlalchemy import Table
from sqlalchemy.sql import select
from flask_sqlalchemy import SQLAlchemy
from config import engine
import pandas as pd

db = SQLAlchemy()


class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(20), unique=True)
    name = db.Column(db.String(50))
    board_lot = db.Column(db.Integer)


Stock_tbl = Table('stock', Stock.metadata)


def create_stock_table():
    Stock.metadata.create_all(engine)
    
    
def drop_stock_table():
    Stock_tbl.drop(engine)

def get_stock_df():
    hkex_excel = pd.read_excel('https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx', skiprows = 2)
    stock_df = hkex_excel[['Stock Code', 'Name of Securities', 'Board Lot']].copy()
    stock_df = stock_df.rename(columns={"Stock Code": "code", "Name of Securities": "name", "Board Lot": "board_lot"})
    stock_df = stock_df[stock_df['code'].apply(lambda x: len(str(x)) <= 4)]
    stock_df['code'] = stock_df['code'].apply(format_stock)
    stock_df['board_lot'] = stock_df['board_lot'].apply(format_board_lot)
    return stock_df

def format_stock(code):
    return str(code).zfill(4) + ".HK"

def format_board_lot(board_lot):
    return int(board_lot.replace(",", ""))

def add_stock():
    df = get_stock_df()
    batch_df = df.iloc[0:2000]
    batch_df.to_sql('stock', engine, if_exists='append', index=False)
    batch_df = df.iloc[2000:]
    batch_df.to_sql('stock', engine, if_exists='append', index=False)

def del_stock(name):
    delete_cmd = Stock_tbl.delete().where(Stock_tbl.c.code == code)

    conn = engine.connect()
    conn.execute(delete_cmd)
    conn.close()

def show_stock():
    select_cmd = select([Stock_tbl.c.id, Stock_tbl.c.code, Stock_tbl.c.name, Stock_tbl.c.board_lot])

    conn = engine.connect()
    rs = conn.execute(select_cmd)

    for row in rs:
        print(row)

    conn.close()
