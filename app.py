# index page
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from server import app, server
from flask_login import logout_user, current_user
from views import success, login, login_fd, logout

import pandas as pd

def get_stock_df():
  hkex_excel = pd.read_excel('https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx', skiprows = 2)
  stock_df = hkex_excel[['Stock Code', 'Name of Securities', 'Board Lot']].copy()
  stock_df = stock_df.rename(columns={"Stock Code": "stock", "Name of Securities": "name", "Board Lot": "board_lot"})
  stock_df['stock'] = stock_df['stock'].apply(format_stock)
  stock_df['board_lot'] = stock_df['board_lot'].apply(format_board_lot)
  return stock_df

def format_stock(stock):
  return str(stock).zfill(4) + ".HK"

def format_board_lot(board_lot):
  return int(board_lot.replace(",", ""))

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


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return login.layout
    elif pathname == '/login':
        return login.layout
    elif pathname == '/success':
        if current_user.is_authenticated:
            layout = html.Div(children=[
                dcc.Location(id='url_login_success', refresh=True),
                dcc.Dropdown(
                    options=[
                        {'label': 'SMA', 'value': 'SMA'},
                        {'label': 'RSI', 'value': 'RSI'},
                        {'label': 'MACD', 'value': 'MACD'}
                    ],
                    value='SMA'
                ),
                dcc.Dropdown(
                    id='my-dropdown',
                    options=[{'label': row['stock'] + ' ' + row['name'], 'value': row['stock']} for index, row in get_stock_df().iterrows()],
                    value='',
                    multi=True
                    ),
            
                dcc.Input(type='number', value=100000, min=10000, step=10000, style={'width': '110px', 'textAlign': 'right'}),
            
                 html.P(children=[
                    html.Button('Create', id='submit-val', n_clicks=0),
                    ]),
            ])
            return layout
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
        return html.Div('Current user: ' + current_user.username)
        # 'User authenticated' return username in get_id()
    else:
        return ''


@app.callback(
    Output('logout', 'children'),
    [Input('page-content', 'children')])
def user_logout(input1):
    if current_user.is_authenticated:
        return html.A('Logout', href='/logout')
    else:
        return ''


if __name__ == '__main__':
    app.run_server(debug=True)
