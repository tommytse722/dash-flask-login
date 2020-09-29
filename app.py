import numpy as np
import pandas as pd
import csv
from datetime import date, datetime, timedelta
import sqlite3

import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

from server import app, server
from flask_login import logout_user, current_user
from views import success, login, login_fd, logout

import plan_mgt
import strategy_mgt
import stock_mgt as sm
import ticker_mgt as tm
import TA
import Email

from strategy_mgt import db, Strategy


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

def get_content(user_id):
    strategy = ''
    stocks = ''
    capital = 100000
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query("SELECT strategy_name, stock_code, capital FROM plan WHERE plan.user_id=" + str(user_id), conn)
    conn.close()
    if len(df)>0:
        strategy = df.head(1)['strategy_name'][0]
        stocks = df['stock_code'].tolist()
        capital = str(df.head(1)['capital'][0])
    
    return strategy, stocks, capital

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
    fig.add_trace(go.Scatter(x=list(df.index), y=list(df.close), name="close", marker_color="black"), secondary_y=False)
    
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
    
    fig.update_layout(
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.2, xanchor="right", x=1),
        margin=dict(l=10, r=10, b=20, t=10)
    )

    # Add range slider
    fig.update_layout(
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
        title_text=title)
    return fig

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
        rows=1, cols=2,
        specs=[[{"type": "domain"}, {"type": "domain"}]],
        subplot_titles=['Trade(s): '+str(no_of_trade), 'Profit($): '+str(total_profit)]
    )

    fig.add_trace(go.Pie(labels=count_labels, name='', values=count_values, hole=0.5, rotation=180, marker=dict(colors=colors), textinfo='value+percent'),
                  row=1, col=1)

    fig.add_trace(go.Pie(labels=sum_labels, name='', values=sum_values, hole=0.5, rotation=180, marker=dict(colors=colors), textinfo='value'),
                  row=1, col=2)

    fig.update_layout(
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.2, xanchor="right", x=1),
        margin=dict(l=10, r=10, b=20, t=10)
    )
    
    return fig


tabs_styles = {
    
}

def plot_value(stocks, all_df):  
    # Create figure
    fig = go.Figure()
    
    for stock in stocks:
        df = all_df[all_df['stock']==stock]
        fig.add_trace(go.Scatter(x=list(df.index), y=list(df.value), name=stock, visible = "legendonly"))

    fig.add_trace(go.Scatter(x=list(df.index), y=list(all_df.reset_index().groupby(['index']).mean().value), name='Average', marker_color="black", visible = "legendonly"))
    
    fig.add_trace(go.Scatter(x=list(df.index), y=list(all_df.reset_index().groupby(['index']).sum().value), name='Total', marker_color="black"))
    
    fig.update_layout(
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.2, xanchor="right", x=1),
        margin=dict(l=10, r=10, b=20, t=10)
    )
    
    # Add range slider
    fig.update_layout(
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
    fig.update_yaxes(title_text="Value")

    return fig

def show_plan(stocks, strategy, capital):
    tabs = []
    ticker_df = TA.get_tickers(stocks)
    all_position_df = pd.DataFrame(columns = ['strategy', 'stock', 'close', 'tx_shares', 'tx_cost', 'shares', 'cash', 'value'])
    all_tx_df = pd.DataFrame(columns = ['strategy', 'stock', 'close', 'tx_shares', 'tx_cost', 'shares', 'cash', 'value'])
    all_trade_df = pd.DataFrame(columns = ['trade_no', 'strategy', 'stock', 'tx_cost', 'net_profit'])
    for stock in stocks:
        graphs = []
        position, tx_df, trade_df, performance = TA.backtesting(stock, strategy, capital, ticker_df[ticker_df.code==stock])
        
        graphs.append(
            dcc.Graph(
                id='performance-{}'.format(stock),
                figure = plot_performance(performance),
                config={'displayModeBar': False}
            )
        )  
        
        graphs.append(
            dcc.Graph(
                id='graph-{}'.format(stock),
                figure = plot_signals(position),
                config={'displayModeBar': False}
            )
        )
        
        tabs.append(dcc.Tab(label='{} ({:.1%})'.format(stock, performance[9]), children=graphs))
        
        #Email.send_order_signal(current_user.email, tx_df, performance, date.today())
        
        position_frames = [all_position_df, position]
        all_position_df = pd.concat(position_frames)
        
        tx_frames = [all_tx_df, tx_df]
        all_tx_df = pd.concat(tx_frames)
        
        trade_frames = [all_trade_df, trade_df]
        all_trade_df = pd.concat(trade_frames)
       
    if len(stocks)>0:        
        all_performance = TA.get_performance_df(all_tx_df, all_trade_df)

        portfolio_graphs = []
        portfolio_graphs.append(
            dcc.Graph(
                id='performance-{}'.format('Portfolio'),
                figure = plot_performance(all_performance),
                config={
                    'displayModeBar': False
                }
            )
        )  
        
        portfolio_graphs.append(
            dcc.Graph(
                id='value-{}'.format('Portfolio'),
                figure = plot_value(stocks, all_position_df),
                config={
                    'displayModeBar': False
                }
            )
        )  

        tabs.insert(0, dcc.Tab(label='{} ({:.1%})'.format('Portfolio', all_performance[9]), children=portfolio_graphs))
    
    return html.Div(dcc.Tabs(tabs, style=tabs_styles))


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
    html.Div(
            children=[
                html.Label("HK$: ", style={'display': 'inline-block'}),
                dcc.Input(id='capital-text', type='number', value='', min=10000, step=10000, style={'width': '110px', 'textAlign': 'right','margin-left': 5, 'display': 'inline-block'}),
                html.Button('Load', type='submit', id='load-button', n_clicks=0, style={'margin-left': 20,'display': 'inline-block'}),
                html.Button('Save', type='submit', id='create-button', n_clicks=0, style={'margin-left': 5,'display': 'inline-block'}),
            ]
    ),
    html.Div(
            children=[
                html.Button('Update', type='submit', id='update-button', n_clicks=0, style={'display': 'inline-block'}),
                html.Label(str(datetime.now()), id='lastupdated', style={'display': 'inline-block'})
                ]
    ),
    html.Div(id='plan_container',style = {'width': '100%'}),
    html.P(id='placeholder'),
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
    Output('logout', 'children'),
    [Input('page-content', 'children')])
def user_logout(input1):
    if current_user.is_authenticated:
        return html.A('Logout', href='/logout')
    else:
        return ''
    
    
@app.callback(
    [Output('strategy-dropdown', 'value'),
     Output('stock-dropdown', 'value'),
     Output('capital-text', 'value')
    ],
    [Input('page-content', 'children'), Input('load-button', 'n_clicks')])
def cur_user(input1, load_clicks):
    if current_user.is_authenticated:
        return get_content(current_user.id)
    else:
        return '', '', 100000
    
    
@app.callback(
    Output('placeholder', 'children'),
    [Input('create-button', 'n_clicks')],
    [State('strategy-dropdown', 'value'),
    State('stock-dropdown', 'value'),
    State('capital-text', 'value')])
def create_plan(create_clicks, strategy, stocks, capital):
    if create_clicks>0:
        plan_mgt.del_plan(current_user.id)
        for stock in stocks:
            plan_mgt.add_plan(current_user.id, strategy, stock, capital)
    return ''


@app.callback(
    Output('plan_container', 'children'),
    [Input('stock-dropdown', 'value'),
     Input('strategy-dropdown', 'value'),
    Input('capital-text', 'value')])
def create_plan(stocks, strategy, capital):
    return show_plan(stocks, strategy, capital)


@app.callback(
    Output('lastupdated', 'children'),
    [Input('update-button', 'n_clicks')])
def update_data(update_clicks):
    if update_clicks>0:
        sm.drop_stock_table()
        sm.create_stock_table()
        sm.download_stock()
        tm.drop_ticker_table()
        tm.create_ticker_table()
        tm.download_ticker()
    return str(datetime.now())


if __name__ == '__main__':
    app.run_server(debug=True)
