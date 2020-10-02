import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta
import sqlite3
import optimal_mgt as om
import Email

def get_tickers(stocks):
    conn = sqlite3.connect('database.db')
    stock_list = str(', '.join("'{0}'".format(s) for s in stocks))
    df = pd.read_sql_query("SELECT stock.code, date, open, high, low, close, adj_close, volume, board_lot FROM stock, ticker where stock.code in (" + stock_list +") and stock.code = ticker.code", conn)
    conn.close()
    return df

def select_strategy(name):
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query("SELECT * FROM strategy where name = '{}'".format(name), conn)
    conn.close()
    x = 0
    y = 0
    if len(df)>0:
        x = str(df.head(1)['x_default'][0])
        y = str(df.head(1)['y_default'][0])
    return x, y

def get_strategy_parameters(strategy, stock, capital):
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query("SELECT * FROM optimal where strategy_name='{}' and stock_code='{}' and capital={} order by last_updated DESC".format(strategy, stock, capital), conn)
    conn.close()
    x = 100
    y = 100
    if len(df)>0:
        x = str(df.head(1)['x'][0])
        y = str(df.head(1)['y'][0])
    else:
        x, y = select_strategy(strategy)
        om.add_optimal(strategy, stock, capital, x, y)
        get_strategy_parameters(strategy, stock, capital)
    return x, y
 
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

def get_tx_df(position_df):
    column_list = ['strategy', 'stock', 'close', 'tx_shares', 'tx_cost', 'shares', 'cash', 'value']
    df1 = position_df.head(1)[column_list].copy()
    df2 = position_df[position_df.tx_shares!=0][column_list].copy()
    frames = [df1, df2]
    tx_df = pd.concat(frames)
    tx_df.tx_cost = tx_df.tx_cost.round(2)
    tx_df.cash = tx_df.cash.round(2)
    tx_df.value = tx_df.value.round(2)
    return tx_df

def get_current_shares_value(position_df):
    current_shares_value = 0
    if position_df.shape[0]>0:
        current_close = float(position_df.tail(1)['close'][0])
        current_shares = int(position_df.tail(1)['shares'][0])
        current_shares_value = current_close*current_shares - get_tx_cost(current_close*current_shares)
    return current_shares_value

def get_trade_df(tx_df, current_shares_value):
    df = tx_df[tx_df.tx_shares!=0][['strategy', 'stock', 'close', 'tx_shares', 'tx_cost']].copy()
    if df.shape[0]>0:
        count = 1
        df['trade_no'] = count
        for item, row in df.iterrows():
            df.loc[item, 'trade_no'] = count
            if row.tx_shares < 0:
                count=count+1

        df['total_tx_shares'] = -1 * df['tx_shares']
        df['net_profit'] = -1 * (df['close'] * df['tx_shares'] + df['tx_cost'])
        trade_df = df.groupby('trade_no').agg({'strategy': 'max', 'stock': 'max', 'tx_cost': 'sum', 'net_profit': 'sum'})
        trade_df.columns = ['strategy', 'stock', 'tx_cost', 'net_profit']
        trade_df = trade_df.reset_index()
        trade_df.iloc[-1, trade_df.columns.get_loc('net_profit')] = trade_df.iloc[-1, trade_df.columns.get_loc('net_profit')] + current_shares_value
        trade_df.net_profit = trade_df.net_profit.round(2)  
    else:
        trade_df = pd.DataFrame(columns = ['trade_no', 'strategy', 'stock', 'tx_cost', 'net_profit'])
        new_row = {'trade_no':0, 'strategy':tx_df.strategy[0], 'stock':tx_df.stock[0], 'tx_cost':0.0, 'net_profit':0.0}
        trade_df = trade_df.append(new_row, ignore_index=True)
    return trade_df

def get_performance_df(tx_df, trade_df):
    strategy = tx_df.strategy[0]
    stock = tx_df.stock[0]
    no_of_win = trade_df[trade_df['net_profit']>0]['net_profit'].count()
    no_of_loss = trade_df[trade_df['net_profit']<=0]['net_profit'].count()
    no_of_trade = int(no_of_win + no_of_loss)
    total_win = trade_df[trade_df['net_profit']>0]['net_profit'].sum().round(0)
    total_loss = trade_df[trade_df['net_profit']<=0]['net_profit'].sum().round(0)
    total_profit = total_win + total_loss
    total_cost = trade_df['tx_cost'].sum().round(0)
    initial_value = float(tx_df.groupby('stock').first().agg({'value': 'sum'})[0])
    final_value = initial_value + total_profit
    ROI = ((final_value-initial_value)/initial_value)
    return [strategy, stock, no_of_win, no_of_loss, no_of_trade, total_win, total_loss, total_profit, total_cost, ROI]

def backtesting(stock, strategy, capital, df, x="0", y="0", user_id="0"):
    df = df.set_index('date')
    position = eval(strategy)(strategy, stock, capital, df, x, y)
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
        
    tx_df = get_tx_df(position)
    trade_df = get_trade_df(tx_df, get_current_shares_value(position))
    performance = get_performance_df(tx_df, trade_df)
    
    #Email.send_order_signal(current_user.email, tx_df, performance, date.today())
        
    return position, tx_df, trade_df, performance

def SMA(strategy, stock, capital, df, short_window, long_window):
    if short_window == "0" or long_window == "0":
        short_window, long_window = get_strategy_parameters(strategy, stock, capital)
    signals = df.copy()
    signals[strategy+short_window] = signals.close.rolling(window=int(short_window)).mean().astype(float)
    signals[strategy+long_window] = signals.close.rolling(window=int(long_window)).mean().astype(float)
    signals['strength'] = np.where(signals[strategy+short_window] > signals[strategy+long_window], 1, 0).astype(int)
    signals['strategy'] = strategy
    signals['stock'] = stock
    signals['action'] = np.diff(signals['strength'], prepend=0).astype(int)
    return signals

def RSI(strategy, stock, capital, df, window, amplitude):
    if window == "0" or amplitude == "0":
        window, amplitude = get_strategy_parameters(strategy, stock, capital)
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