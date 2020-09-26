from sqlalchemy import Table
from sqlalchemy.sql import select
from flask_sqlalchemy import SQLAlchemy
from config import engine
import pandas as pd
from datetime import date, datetime, timedelta
import yfinance as yf
import stock_mgt

def download_data_from_yf(period, interval, stock):
    data = yf.download(tickers = stock, period = period, interval = interval, group_by = 'ticker', prepost = True)
    return data

def get_stock_historial_data(stock, period='2y'):
    yf_data = download_data_from_yf(period, '1d', stock).copy()
    yf_data.rename(columns = {'Open':'open',
                              'High':'high',
                              'Low':'low',
                              'Close':'close',
                              'Adj Close':'adj_close',
                              'Volume':'volume',
                             }, inplace=True)
    yf_data.index.names = ["date"]
    yf_data.reset_index(level=0, inplace=True)
    data = yf_data.round(3).copy()
    data['code'] = stock
    return data

db = SQLAlchemy()

class ticker(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(20))
    date = db.Column(db.Date())
    open = db.Column(db.Float())
    high = db.Column(db.Float())
    low = db.Column(db.Float())
    close = db.Column(db.Float())
    adj_close = db.Column(db.Float())
    volume = db.Column(db.Integer)
    
ticker_tbl = Table('ticker', ticker.metadata)

def download_ticker():
    for stock in stock_mgt.get_indice_stock_list():
        print(stock)
        df = get_stock_historial_data(stock)
        df.to_sql('ticker', engine, if_exists='append', index=False)
    
def create_ticker_table():
    ticker.metadata.create_all(engine)
    
def drop_ticker_table():
    ticker_tbl.drop(engine)

