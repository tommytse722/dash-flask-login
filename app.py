# index page
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from server import app, server
from flask_login import logout_user, current_user
from views import success, login, login_fd, logout
from strategy_mgt import db, Strategy

import plan_mgt
import strategy_mgt
import stock_mgt

# My code

import numpy as np
import pandas as pd
import csv
from datetime import date, datetime, timedelta
import sqlite3

import yfinance as yf

#Data Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

def download_data_from_yf(period, interval, stock_list):
    data = yf.download(tickers = stock_list, period = period, interval = interval, group_by = 'ticker', prepost = True)
    return data

def get_stock_historial_data(stock_list, period='1y'):
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

def select_strategy():
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query("SELECT * FROM strategy", conn)
    conn.close()
    return df

def get_strategy_details(id):
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query("SELECT * FROM strategy where id = " + str(id), conn)
    conn.close()
    return df

def select_stock():
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query("SELECT * FROM stock", conn)
    conn.close()
    return df

def select_stock_board_lot(stock_code):
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query("SELECT * FROM stock where code = '" + str(stock_code) +"'", conn)
    conn.close()
    return df

def get_user_strategy(user_id):
    value = ''
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query("SELECT strategy_id FROM plan where plan.user_id=" + str(user_id), conn)
    if len(df)>0:
        value = int(df.head(1)['strategy_id'][0])
    conn.close()
    return value

def get_user_stock(user_id):
    value = ''
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query("SELECT stock_code FROM plan where plan.user_id=" + str(user_id), conn)
    if len(df)>0:
        value = df['stock_code'].tolist()
    conn.close()
    return value

def get_user_capital(user_id):
    value = 100000
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query("SELECT capital FROM plan where plan.user_id=" + str(user_id), conn)
    if len(df)>0:
        value = str(df.head(1)['capital'][0])
    conn.close()
    return value

def get_all_plans():
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query("SELECT * FROM plan", conn)
    conn.close()
    return df

def get_plan(id):
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query("SELECT * FROM plan where plan.user_id=" + str(id), conn)
    conn.close()
    return df

def download_data_from_yf(period, interval, stock_list):
    import yfinance as yf
    data = yf.download(tickers = stock_list, period = period, interval = interval, group_by = 'ticker', auto_adjust = False, prepost = True, threads = True, proxy = None)
    return data

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
            #print(position[0:position.index.get_loc(item)+1]['close'])
            #Trend Prediction
            #stock_data = stock_historial_data2[user_stock_row.stock].copy()
            #print(get_future_trend(stock_data[0:stock_data.index.get_loc(item)+1]))
            board_lot = select_stock_board_lot(position_row.stock)['board_lot'][0]
            tx_shares = np.floor( (last_cash - get_tx_cost(last_cash)) / (position_row.close * board_lot) ) * board_lot
    # Sell all the shares in hand
    if int(position_row.action) == -1:
        tx_shares = last_shares 
    tx_shares = tx_shares * position_row.action
    return int(tx_shares)

def plot_signals(df):  
    
    strategy = str(df['strategy'][0])
    stock = str(df['stock'][0])
    
    # Create figure
    fig = go.Figure()
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    for column in df.columns:
        if strategy in str(column):
            fig.add_trace(go.Scatter(x=list(df.index), y=list(df[str(column)]), name=str(column), visible = "legendonly"), secondary_y=True)

    #fig.add_trace(go.Scatter(x=list(df.index), y=list(df.open), name="open", visible = "legendonly"), secondary_y=False)
    #fig.add_trace(go.Scatter(x=list(df.index), y=list(df.high), name="high", visible = "legendonly"), secondary_y=False)
    #ig.add_trace(go.Scatter(x=list(df.index), y=list(df.low), name="low", visible = "legendonly"), secondary_y=False)
    fig.add_trace(go.Scatter(x=list(df.index), y=list(df.close), name="close"), secondary_y=False)
    
    df2 = df[df.tx_shares>0][['close', 'tx_shares']]
    fig.add_trace(go.Scatter(x=list(df2.index), y=list(df2.close), name="Buy", 
                             mode='markers', marker_size=12, marker_color="green", marker_symbol=5,
                             text=df2['tx_shares'],
                             hovertemplate="%{x}<br>%{text} shares @ $%{y}"
                            ), secondary_y=False)
    
    df3 = df[df.tx_shares<0][['close', 'tx_shares']]
    fig.add_trace(go.Scatter(x=list(df3.index), y=list(df3.close), name="Sell", 
                            mode='markers', marker_size=12, marker_color="red", marker_symbol=6,
                             text=df3['tx_shares'],
                             hovertemplate="%{x}<br>%{text} shares @ $%{y}"
                            ), secondary_y=False)
    
    # Set title
    fig.update_layout(
        title_text='Trading Signals of ' + strategy + ' on ' + stock
    )

    # Add range slider
    fig.update_layout(
        width=960,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=3,
                         label="3m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        )
    )

    # Set y-axes titles
    fig.update_yaxes(title_text="Close Price", secondary_y=False)
    fig.update_yaxes(title_text=strategy, secondary_y=True)

    return fig
    
header = html.Div(
    className='header',
    children=html.Div(
        className='container-width',
        style={'height': '100%'},
        children=[
            html.Img(
                src='assets/dash-logo-stripe.svg',
                className='logo'
            ),
            html.Div(className='links', children=[
                html.Div(id='user-name', className='link'),
                html.Div(id='logout', className='link')
            ])
        ]
    )
)


app.layout = html.Div(
    [
        header,
        html.Div([
            html.Div(
                html.Div(id='page-content', className='content'),
                className='content-container'
            ),
        ], className='container-width'),
        dcc.Location(id='url', refresh=False),
    ]
)


success_layout = html.Div(children=[
    dcc.Location(id='url_login_success', refresh=True),
    dcc.Dropdown(
        id='strategy-dropdown',
        options=[{'label': row['name'], 'value': row['id']} for index, row in select_strategy().iterrows()],
        value=''
    ),
    dcc.Dropdown(
        id='stock-dropdown',
        options=[{'label': row['code'] + ' ' + row['name'], 'value': row['code']} for index, row in select_stock().iterrows()],
        value='',
        multi=True
        ),
    dcc.Input(id='capital-text', type='number', value='', min=10000, step=10000, style={'width': '110px', 'textAlign': 'right'}),

     html.P(children=[
        html.Button('Create', type='submit', id='create-button', n_clicks=0),
        ]),
    html.Div(id='display-selected-values'),
    dash_table.DataTable(id='table', columns = [{"name": i, "id": i} for i in get_all_plans().columns]),
    html.Div(id='container'),
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return login.layout
    elif pathname == '/login':
        return login.layout
    elif pathname == '/success':
        if current_user.is_authenticated:
            return success_layout
        else:
            return login_fd.layout
    elif pathname == '/logout':
        if current_user.is_authenticated:
            logout_user()
            return logout.layout
        else:
            return logout.layout
    else:
        return '404'


@app.callback(
    Output('user-name', 'children'),
    [Input('page-content', 'children')])
def cur_user(input1):
    if current_user.is_authenticated:
        return html.Div(current_user.email)
        # 'User authenticated' return username in get_id()
    else:
        return ''
    
    
@app.callback(
    Output('strategy-dropdown', 'value'),
    [Input('page-content', 'children')])
def cur_user(input1):
    if current_user.is_authenticated:
        return get_user_strategy(current_user.id)
    else:
        return ''
    
    
@app.callback(
    Output('stock-dropdown', 'value'),
    [Input('page-content', 'children')])
def cur_user(input1):
    if current_user.is_authenticated:
        return get_user_stock(current_user.id)
    else:
        return ''
    
    
@app.callback(
    Output('capital-text', 'value'),
    [Input('page-content', 'children')])
def cur_user(input1):
    if current_user.is_authenticated:
        return get_user_capital(current_user.id)
    else:
        return 100000
    

@app.callback(
    Output('logout', 'children'),
    [Input('page-content', 'children')])
def user_logout(input1):
    if current_user.is_authenticated:
        return html.A('Logout', href='/logout')
    else:
        return ''
    
    
@app.callback(
    [Output('container', 'children'),
    Output('table', 'data')],
    [Input('create-button', 'n_clicks')],
    [State('strategy-dropdown', 'value'),
    State('stock-dropdown', 'value'),
    State('capital-text', 'value')])
def create_plan(n_clicks, strategy, stocks, capital):
    if n_clicks>0:
        plan_mgt.del_plan(current_user.id)
    for stock in stocks:
        plan_mgt.add_plan(current_user.id, strategy, stock, capital)
    graphs = []
    for stock in stocks:
        position = RSI(get_strategy_details(strategy)['name'][0], stock)
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

        graphs.append(
            dcc.Graph(
            id='graph-{}'.format(stock),
            figure= plot_signals(position)

            )
        )
        
    return html.Div(graphs), get_plan(current_user.id).to_dict('records')

    
if __name__ == '__main__':
    app.run_server(debug=True)
