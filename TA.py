import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

def get_tx_cost(transaction_amount):
    total = 0
    if float(transaction_amount)>0:
        Commission_fee = float(transaction_amount) * 0.25 /100
        Transaction_levy = float(transaction_amount) * 0.003 /100
        Trading_fee = float(transaction_amount) * 0.005 /100
        Stamp_duty = np.ceil(float(transaction_amount) * 0.1 /100)
        CCASS_fee = max(5, float(transaction_amount) * 0.002 /100)
        total = Commission_fee + Transaction_levy + Trading_fee + Stamp_duty + CCASS_fee
    return float(total)

def get_tx_shares(item, position, position_row, last_cash, last_shares):
    tx_shares = 0
    # Use all the cash allocated to a particular stock to buy as much shares as possible            
    if int(position_row.action) == 1:
        if int(last_shares)==0:
            tx_shares = np.floor( (last_cash - get_tx_cost(last_cash)) / (position_row.close * position_row.board_lot) ) * position_row.board_lot
    # Sell all the shares in hand
    if int(position_row.action) == -1:
        tx_shares = last_shares 
    tx_shares = tx_shares * position_row.action
    return int(tx_shares)

def backtesting(stock, strategy, capital, df):
    df = df.set_index('date')
    position = eval(strategy)(strategy, stock, df)
    position['tx_shares'] = int(0)
    position['tx_cost'] = 0.0
    position['shares'] = int(0)
    position['cash'] = 0.0
    position['value'] = 0.0

    last_cash = int(capital)
    last_shares = 0  
    for item, position_row in position.iterrows():
        position.loc[item, 'tx_shares'] = int(get_tx_shares(item, position, position_row, last_cash, last_shares))
        position.loc[item, 'tx_cost'] = get_tx_cost(position_row.close * abs(position.loc[item, 'tx_shares']))
        position.loc[item, 'shares'] = int(last_shares + position.loc[item, 'tx_shares'])
        position.loc[item, 'cash'] = last_cash - position_row.close * position.loc[item, 'tx_shares'] - position.loc[item, 'tx_cost']
        position.loc[item, 'value'] = position.loc[item, 'cash'] + position_row.close * position.loc[item, 'shares']

        last_cash = position.loc[item, 'cash']
        last_shares = position.loc[item, 'shares']
    return position

def SMA(strategy, stock, df, short_window = "20", long_window = "50"):
    signals = df.copy()
    signals[strategy+short_window] = signals.close.rolling(window=int(short_window)).mean().astype(float)
    signals[strategy+long_window] = signals.close.rolling(window=int(long_window)).mean().astype(float)
    signals['strength'] = np.where(signals[strategy+short_window] > signals[strategy+long_window], 1, 0).astype(int)
    signals['strategy'] = strategy
    signals['stock'] = stock
    signals['action'] = np.diff(signals['strength'], prepend=0).astype(int)
    return signals

def RSI(strategy, stock, df, window = '14', amplitude = '20'):
    resistance = 50 + int(amplitude)
    support = 50 - int(amplitude)
    signals = df.copy()
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