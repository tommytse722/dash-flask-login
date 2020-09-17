import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
import yfinance as yf

def download_data_from_yf(period, interval, stock_list):
    data = yf.download(tickers = stock_list, period = period, interval = interval, group_by = 'ticker', prepost = True)
    return data

def get_stock_historial_data(stock_list, period='2y'):
    yf_data = download_data_from_yf(period, '1d', stock_list).copy()
    yf_data.rename(columns = {'Open':'open',
                              'High':'high',
                              'Low':'low',
                              'Close':'close',
                              'Adj Close':'adj_close',
                              'Volume':'volume',
                             }, inplace=True)
    yf_data.index.names = ["date"]
    data = yf_data.round(3).copy()
    return data

def get_historial_data(stock):
    signals = get_stock_historial_data(stock).copy()
    if stock == '0017.HK':
        split_date = datetime(2020,6,9)
        for item, row in signals.iterrows():
            if item<=split_date:
                signals.loc[item, 'close'] = row.close * 4
    return signals

def SMA(strategy, stock):
    short_window = "20"
    long_window = "50"
    signals = get_historial_data(stock)
    signals[strategy+short_window] = signals.close.rolling(window=int(short_window)).mean().astype(float)
    signals[strategy+long_window] = signals.close.rolling(window=int(long_window)).mean().astype(float)
    signals['strength'] = np.where(signals[strategy+short_window] > signals[strategy+long_window], 1, 0).astype(int)
    signals['strategy'] = strategy
    signals['stock'] = stock
    signals['action'] = np.diff(signals['strength'], prepend=0).astype(int)
    return signals

def RSI(strategy, stock):
    window = '14'
    amplitude = '20'
    resistance = 50 + int(amplitude)
    support = 50 - int(amplitude)
    signals = get_historial_data(stock)
    signals['change'] = np.diff(signals.close, prepend=0)
    signals['up_strength'] = np.where(signals['change'] > 0, signals['change'], 0)
    signals['dn_strength'] = np.where(signals['change'] < 0, abs(signals['change']), 0)
    signals['up_win'] = signals['up_strength'].rolling(window=int(window)).mean().astype(float)
    signals['dn_win'] = signals['dn_strength'].rolling(window=int(window)).mean().astype(float)
    signals['up_win'].fillna(1, inplace=True)
    signals['dn_win'].fillna(1, inplace=True)
    signals[strategy+window] = (100 * signals['up_win'] / (signals['up_win'] + signals['dn_win'] ))
    signals[strategy+window+'='+str(resistance)] = resistance
    signals[strategy+window+'='+str(support)] = support
    signals['strength'] = int(0)
    signals['strength'] = np.where(signals[strategy+window] > signals[strategy+window+'='+str(resistance)], resistance, signals['strength'])
    signals['strength'] = np.where(signals[strategy+window] < signals[strategy+window+'='+str(support)], -1 * support, signals['strength']) 
    signals['edge'] = np.diff(signals['strength'], prepend=0).astype(int)
    signals['strategy'] = strategy
    signals['stock'] = stock
    signals['action'] = int(0)
    signals['action'] = np.where(signals['edge']==support, 1, signals['action']) 
    signals['action'] = np.where(signals['edge']==(-1 * resistance), -1, signals['action'])
    return signals