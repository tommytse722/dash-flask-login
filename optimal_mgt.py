from sqlalchemy import Table
from sqlalchemy.sql import select
from flask_sqlalchemy import SQLAlchemy
from config import engine
import pandas as pd
import datetime

import numpy as np
import sqlite3
import TA
import numpy as np

#Data Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

db = SQLAlchemy()


class Optimal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    strategy_name = db.Column(db.String(20))
    stock_code = db.Column(db.String(20))
    capital = db.Column(db.Integer)
    x = db.Column(db.Integer)
    y = db.Column(db.Integer)
    last_updated = db.Column(db.DateTime(), default=datetime.datetime.now)

Optimal_tbl = Table('optimal', Optimal.metadata)


def create_optimal_table():
    Optimal.metadata.create_all(engine)
    
    
def drop_optimal_table():
    Optimal_tbl.drop(engine)


def add_optimal(strategy_name, stock_code, capital, x, y):
    insert_cmd = Optimal_tbl.insert().values(strategy_name=strategy_name, stock_code=stock_code, capital=capital, x=x, y=y)
    conn = engine.connect()
    conn.execute(insert_cmd)
    conn.close()
    
def select_plan():
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query("SELECT user_id, strategy_name, stock_code, capital, strategy.* FROM plan, strategy where strategy_name = strategy.name", conn)
    conn.close()
    return df

def get_ticker(stock):
    conn = sqlite3.connect('database.db')
    stock_code = "'{0}'".format(stock)
    df = pd.read_sql_query("SELECT stock.code, date, open, high, low, close, adj_close, volume, board_lot FROM stock, ticker where stock.code = " + stock_code + " and stock.code = ticker.code", conn)
    conn.close()
    return df

def get_ROI_distribution(r):
    rows = []
    for y in range(r.y_start, r.y_end+1, r.y_step):
        row = []
        for x in range(r.x_start, r.x_end+1, r.x_step):
            position, tx_df, trade_df, performance = TA.backtesting(r.stock_code, r.strategy_name, r.capital, get_ticker(r.stock_code), str(x), str(y))
            ROI = performance[9]
            #print('x:{}; y:{}; z:{}'.format(x, y, ROI))
            row.append(ROI)
        rows.append(row)
    return rows
    
def convolve2D(image, kernel, padding=0, strides=1):
    # Cross Correlation
    kernel = np.flipud(np.fliplr(kernel))

    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = image.shape[0]
    yImgShape = image.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / strides) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / strides) + 1)
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = image
    else:
        imagePadded = image

    # Iterate through image
    rows = []
    for x in range(image.shape[0]):
        # Exit Convolution
        if x > image.shape[0] - xKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if x % strides == 0:
            row = []
            for y in range(image.shape[1]):
            # Go to next row once kernel is out of bounds
                if y > image.shape[1] - yKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if y % strides == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).mean()
                        row.append(output[x, y])
                except:
                    break
        rows.append(row)
    return rows

def forward_convolution(ROI_distribution):
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    layers = []
    length = len(ROI_distribution)
    output = ROI_distribution
    layers.append(output)
    while length > 3:
        output = convolve2D(np.array(output), kernel)
        layers.append(output)
        length = length - 2
    return layers

def backward_convolution(r, layers):
    layers.reverse()
    # For the 1st layer, choose the best one
    x_start = 0
    y_start = 0
    x_end = len(layers[0])
    y_end = len(layers[0])
    n = 0

    for layer in layers:
        #print(layer)
        max_x = 0
        max_y = 0
        max_z = -10000
        #print('x_start:{}; x_end:{}; y_start:{}; y_end:{}'.format(x_start, x_end, y_start, y_end))
        n = n + 1
        if n == len(layers):
            # For the nth layer, always choose the middle one
            max_x = x_start + 1
            max_y = y_start + 1
            max_z = layer[max_y][max_x]
        else:
            # For the 2nd to (n-1)th layer, always choose the middle one
            for y in range(y_start, y_end):
                #print(layer[y]) 
                for x in range(x_start, x_end): 
                    #print(layer[y][x]) 
                    if layer[y][x]>max_z:
                        max_x = x
                        max_y = y
                        max_z = layer[y][x]
            x_start = max_x - 1 + 1
            y_start = max_y - 1 + 1
            x_end =   max_x + 1 + 1 + 1
            y_end =   max_y + 1 + 1 + 1
    return int((max_x+r.x_start/r.x_step)*r.x_step), int((max_y+r.y_start/r.y_step)*r.y_step), max_z

def plot_ROI_distribution(r, z):
    x, y = np.linspace(r.x_start, r.x_end, num=int((r.x_end-r.x_start)/r.x_step)+1), np.linspace(r.y_start, r.y_end, num=int((r.y_end-r.y_start)/r.y_step)+1)
    fig = go.Figure(data=[go.Surface(x=x, y=y, z=z)])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True))
    fig.update_layout(title='ROI Distribution', autosize=False, width=700, height=700, scene_camera_eye=dict(x=-2, y=-2, z=0.2))  
    fig.show()
    
def plan_optimization():
    for _, r in select_plan().iterrows():
        ROI_distribution = get_ROI_distribution(r)
        x, y, z = backward_convolution(r, forward_convolution(ROI_distribution))
        print('The optimal value of (x, y, z) is ({}, {}, {})'.format(x, y, z))
        plot_ROI_distribution(r, np.array(ROI_distribution))
        add_optimal(r.strategy_name, r.stock_code, r.capital, x, y)