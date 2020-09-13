# index page
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

def select_strategy():
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query("SELECT * FROM strategy", conn)
    conn.close()
    return df

def select_stock():
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query("SELECT * FROM stock", conn)
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

def get_plan(id):
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query("SELECT * FROM plan where plan.user_id=" + str(id), conn)
    conn.close()
    return df.head(1)

def download_data_from_yf(period, interval, stock_list):
    import yfinance as yf
    data = yf.download(tickers = stock_list, period = period, interval = interval, group_by = 'ticker', auto_adjust = False, prepost = True, threads = True, proxy = None)
    return data


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
    html.Div(id='display-selected-values')
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
    Output('display-selected-values', 'children'),
    [Input('create-button', 'n_clicks')],
    [State('strategy-dropdown', 'value'),
    State('stock-dropdown', 'value'),
    State('capital-text', 'value')])
def create_plan(n_clicks, strategy, stocks, capital):
    if n_clicks>0:
        plan_mgt.del_plan(current_user.id)
    for stock in stocks:
        plan_mgt.add_plan(current_user.id, strategy, stock, capital)
    return ''

    
if __name__ == '__main__':
    app.run_server(debug=True)
