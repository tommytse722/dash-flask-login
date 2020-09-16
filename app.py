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

#Email
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

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
    df = pd.read_sql_query("SELECT strategy_name FROM plan where plan.user_id=" + str(user_id), conn)
    if len(df)>0:
        value = df.head(1)['strategy_name'][0]
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
    df = pd.read_sql_query("SELECT strategy_name as 'Strategy', stock_code as 'Stock Code', capital as 'Initial Capital' FROM plan", conn)
    conn.close()
    return df

def get_plan(id):
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query("SELECT strategy_name as 'Strategy', stock_code as 'Stock Code', capital as 'Initial Capital' FROM plan where plan.user_id=" + str(id), conn)
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
#    fig.update_layout(
#        title_text='Trading Signals of ' + strategy + ' on ' + stock
#    )
    
    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.1,
    xanchor="right",
    x=1
    ))

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
    
def plot_tx(tx_df):
    strategy = ', '.join(tx_df.strategy.unique())
    stock = ', '.join(tx_df.stock.unique())
    tx_df['Date'] = pd.DatetimeIndex(tx_df.index).strftime("%Y-%m-%d")
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Date', 'Strategy', 'Stock', 'Close Price', 'Shares (Tx)', 'Cost (Tx)', 'Shares in hand', 'Cash in hand', 'Value'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[tx_df.Date, tx_df.strategy, tx_df.stock, tx_df.close, tx_df.tx_shares, tx_df.tx_cost, tx_df.shares, tx_df.cash, tx_df.value],
                   fill_color='lavender',
                   align='right'))
    ])
    
    title = 'Transaction records of ' + strategy + ' on ' + stock
    fig.update_layout(
        width=960,
        title_text=title)
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

def get_trade_df(tx_df, current_shares_value):
    df = tx_df[tx_df.tx_shares!=0][['strategy', 'stock', 'close', 'tx_shares', 'tx_cost']].copy()
    
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
    return trade_df

def plot_trade(trade_df):
    strategy = ', '.join(trade_df.strategy.unique())
    stock = ', '.join(trade_df.stock.unique())
    fig = go.Figure(data=[go.Table(
        header=dict(values=['Strategy', 'Stock', 'Trade No.', 'Net Profit'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[trade_df.strategy, trade_df.stock, trade_df.trade_no, trade_df.net_profit],
                   fill_color='lavender',
                   align='right'))
    ])
    
    title = 'Trade records of ' + strategy + ' on ' + stock
    fig.update_layout(
        width=960,
        title_text=title)
    return fig

def get_current_shares_value(position_df):
    current_close = float(position_df.tail(1)['close'][0])
    current_shares = int(position_df.tail(1)['shares'][0])
    current_shares_value = current_close*current_shares - get_tx_cost(current_close*current_shares)
    return current_shares_value

def get_performance_df(tx_df, trade_df):
    strategy = ', '.join(tx_df.strategy.unique())
    stock = ', '.join(tx_df.stock.unique())
    
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

    #no_of_trade = no_of_win+no_of_loss
    #win_factor = no_of_win/no_of_trade
    #max_drawdown = trade_record['net_profit'].min()
    #total_profit = total_win+total_loss
    #average_profit = total_profit/no_of_trade
    #profit_factor = total_win/(-1*total_loss)
    return [strategy, stock, no_of_win, no_of_loss, no_of_trade, total_win, total_loss, total_profit, total_cost, ROI]

def plot_performance(performance):

    strategy = str(performance[0])
    stock = str(performance[1])
    
    no_of_win = int(performance[2])
    no_of_loss = int(performance[3])
    no_of_trade = int(performance[4])
    total_win = int(performance[5])
    total_loss = int(performance[6])
    total_profit = int(performance[7])
    total_cost = int(performance[8])
    ROI = float(performance[9])

    colors = ['green','red']
    
    count_labels = ['Win','Loss']
    count_values = [no_of_win, no_of_loss]
    count_data = {'labels':count_labels, 'values':count_values }
    df_count = pd.DataFrame(count_data)
    
    sum_labels = ['Win','Loss']
    sum_values = [total_win, -1*total_loss]
    sum_data = {'labels':sum_labels, 'values':sum_values }
    df_sum = pd.DataFrame(sum_data)

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.3,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=['Trade(s): '+str(no_of_trade), 'Profit($): '+str(total_profit)]
    )

    fig.add_trace(go.Pie(labels=count_labels, name='', values=count_values, hole=0.6, rotation=180, marker=dict(colors=colors), textinfo='value+percent'),
                  row=1, col=1)

    fig.add_trace(go.Pie(labels=sum_labels, name='', values=sum_values, hole=0.6, rotation=180, marker=dict(colors=colors), textinfo='value'),
                  row=1, col=2)
    

#    title = 'Performance of ' + strategy + ' ( ROI: ' + '{:.2%}'.format(ROI) + ' )' + ' on ' + stock
#    fig.update_layout(
#    title_text=title,
#    annotations=[dict(text='No. of Trade: '+str(no_of_trade), x=0.13, y=0.5, font_size=18, showarrow=False),
#                 dict(text='Net Profit: '+str(total_profit), x=0.9, y=0.5, font_size=18, showarrow=False)])

    fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.1,
    xanchor="right",
    x=1
    ))
    
    return fig

def send_html_email(receiver, subject, html_body):
    sender = "mdascalgotrading@gmail.com"
    password = "trend2020"
    message = MIMEMultipart("alternative")
    message["From"] = sender
    message["To"] = receiver
    message["Subject"] = subject
    message.attach(MIMEText(html_body, "html"))
    # Create secure connection with server and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender, password)
        server.sendmail(
            sender, receiver, message.as_string()
        )
    return True

def send_order_signal(receiver, tx_df, performance, execution_date):
    
    delta = 0
    last_tx_date = datetime.strftime(tx_df.index[-1] - timedelta(delta), '%Y-%m-%d')
    #last_tx_date = '2020-09-02'
    #execution_date = '2020-09-02'

    if str(last_tx_date)==str(execution_date):

        price = float(tx_df.close[-1])
        tx_shares = int(tx_df.tx_shares[-1])
        if tx_shares>0:
            action = "Buy"
        else:
            action = "Sell"
        tx_shares = abs(tx_shares)

        strategy = str(performance[0])
        stock = str(performance[1])

        no_of_win = int(performance[2])
        no_of_loss = int(performance[3])
        no_of_trade = int(performance[4])
        win_rate = no_of_win/no_of_trade
        total_win = float(performance[5])
        total_loss = float(performance[6])
        total_profit = float(performance[7])
        total_cost = float(performance[8])
        ROI = float(performance[9])

        subject = "{action} {tx_shares} shares of {stock} at ${price}"

        subject = subject.replace('{stock}', stock)
        subject = subject.replace('{action}', action)
        subject = subject.replace('{price}', str(price))
        subject = subject.replace('{tx_shares}', str(tx_shares))

        html_body = """\
          <html>
            <body>
                Win (%): {no_of_win}/{no_of_trade} ({win_rate}) <br>
                Profit (%):  {total_profit} ({ROI}) <br>
                by {strategy} on {execution_date}
            </body>
          </html>
          """
        html_body = html_body.replace('{no_of_win}', str(no_of_win))
        html_body = html_body.replace('{no_of_trade}', str(no_of_trade))
        html_body = html_body.replace('{win_rate}', '{:.1%}'.format(win_rate))
        html_body = html_body.replace('{total_profit}', str(total_profit))
        html_body = html_body.replace('{ROI}', '{:.1%}'.format(ROI))
        html_body = html_body.replace('{strategy}', strategy)
        html_body = html_body.replace('{execution_date}', str(execution_date))

        send_html_email(receiver, subject, html_body)
        
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
        id='stock-dropdown',
        options=[{'label': row['code'] + ' ' + row['name'], 'value': row['code']} for index, row in select_stock().iterrows()],
        value='',
        multi=True
        ),
    dcc.Dropdown(
         id='strategy-dropdown',
         options=[{'label': row['name'], 'value': row['name']} for index, row in select_strategy().iterrows()],
         value=''
    ),
    dcc.Input(id='capital-text', type='number', value='', min=10000, step=10000, style={'width': '110px', 'textAlign': 'right'}),
    html.Button('Create', type='submit', id='create-button', n_clicks=0),
#   dash_table.DataTable(id='table', columns = [{"name": i, "id": i} for i in get_all_plans().columns]),
    html.Div(id='container',style = {'width': '100%'}),
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
    Output('container', 'children'),
    [Input('create-button', 'n_clicks')],
    [State('strategy-dropdown', 'value'),
    State('stock-dropdown', 'value'),
    State('capital-text', 'value')])
def create_plan(n_clicks, strategy, stocks, capital):
    if n_clicks>0:
        plan_mgt.del_plan(current_user.id)
    for stock in stocks:
        plan_mgt.add_plan(current_user.id, strategy, stock, capital)
    tabs = []
    all_tx_df = pd.DataFrame(columns = ['strategy', 'stock', 'close', 'tx_shares', 'tx_cost', 'shares', 'cash', 'value'])
    all_trade_df = pd.DataFrame(columns = ['trade_no', 'strategy', 'stock', 'tx_cost', 'net_profit'])
    for stock in stocks:
        graphs = []
        position = eval(strategy)(strategy, stock)
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
        
#       graphs.append(
#            dcc.Graph(
#            id='tx-{}'.format(stock),
#            figure = plot_tx(tx_df)
#            )
#        )    
        
        trade_df = get_trade_df(tx_df, get_current_shares_value(position))
        
#        graphs.append(
#            dcc.Graph(
#            id='trade-{}'.format(stock),
#            figure = plot_trade(trade_df)
#            )
#        )  
        
        performance = get_performance_df(tx_df, trade_df)
        
        graphs.append(
            dcc.Graph(
            id='performance-{}'.format(stock),
            figure = plot_performance(performance)
            )
        )  
        
        graphs.append(
            dcc.Graph(
            id='graph-{}'.format(stock),
            figure = plot_signals(position)
            )
        )
        
        tabs.append(
            dcc.Tab(label='{} ({:.1%})'.format(stock, performance[9]), children=graphs)
        )
        
        send_order_signal(current_user.email, tx_df, performance, date.today())
        tx_frames = [all_tx_df, tx_df]
        all_tx_df = pd.concat(tx_frames)
        trade_frames = [all_trade_df, trade_df]
        all_trade_df = pd.concat(trade_frames)
       
    all_performance = get_performance_df(all_tx_df, all_trade_df)
    
    portfolio_graphs = []
    portfolio_graphs.append(
        dcc.Graph(
        id='performance-{}'.format('Portfolio'),
        figure = plot_performance(all_performance)
        )
    )  
    tabs.insert(0, 
                dcc.Tab(label='{} ({:.1%})'.format('Portfolio', all_performance[9]), children=portfolio_graphs)
               )
 
    return html.Div(dcc.Tabs(tabs))
    #get_plan(current_user.id).to_dict('records')

    
if __name__ == '__main__':
    app.run_server(debug=True)
