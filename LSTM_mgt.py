#Import the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from datetime import date, datetime, timedelta

import sqlite3

def prediction():
    ticker_df = get_stock_ticker()
    for stock in list(ticker_df.code.unique()):
        get_future_trend(stock, ticker_df[ticker_df.code==stock], 60, 5)
    
def get_stock_ticker():
    conn = sqlite3.connect('database.db')
    df = pd.read_sql_query("SELECT * FROM ticker", conn)
    conn.close()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    return df

def get_next_trading_day(next_trading_day):
    while next_trading_day.weekday() >= 5:
        next_trading_day = next_trading_day + timedelta(days=1)     
    return next_trading_day

def split_data(scaler, signals, days_before):
    #Create a new dataframe with only the 'Close' column
    data = signals.filter(['close'])
    #Converting the dataframe to a numpy array
    dataset = data.values
    #Scale the all of the data to be values between 0 and 1 
    scaled_data = scaler.fit_transform(dataset)
    #Get /Compute the number of rows to train the model on
    training_ratio = 0.8
    training_data_len = math.ceil( len(scaled_data) * training_ratio) 
    #Create the scaled training data set 
    train_data = scaled_data[0:training_data_len,:]
    #Test data set
    test_data = scaled_data[training_data_len - days_before: , : ]
    return train_data, test_data

def get_model_data(days_before, data):
    #Create the x and y data sets
    x = []
    y = []
    for i in range(days_before, len(data)):
        x.append(data[i-days_before:i,0])
        y.append(data[i,0])
    #Convert x and y to a numpy array 
    x, y = np.array(x), np.array(y)
    #Reshape the data into the shape accepted by the LSTM
    x = np.reshape(x, (x.shape[0],x.shape[1],1)) 
    return x,y

def train_model(data, days_before):
    X_train, y_train = get_model_data(days_before, data)
    #Build the LSTM network model
    model = Sequential()
    #Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units = 1))

    # Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    # Fitting the RNN to the Training set
    model.fit(X_train, y_train, epochs = 100, batch_size = 16)
    return model

def test_model(model, scaler, data, days_before):
    x_test, y_test = get_model_data(days_before, data)
    #Getting the models predicted price values
    prediction = model.predict(x_test) 
    prediction = scaler.inverse_transform(prediction)#Undo scaling
    #Calculate/Get the value of RMSE
    rmse=np.sqrt(np.mean(((prediction - y_test)**2)))
    print('rmse:'+str(rmse))
    return prediction

def one_step_ahead(scaler, model, df, days_before):
    #Get the closing price of past days 
    past_days = df[-days_before:].values
    #Scale the data to be values between 0 and 1
    past_days_scaled = scaler.transform(past_days)
    #Create an empty list
    x_test = []
    #Append the past days
    x_test.append(past_days_scaled)
    #Convert the x_test data set to a numpy array
    x_test = np.array(x_test)
    #Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    #Get the predicted scaled price
    predicted_price = model.predict(x_test)
    #undo the scaling 
    predicted_price = scaler.inverse_transform(predicted_price)
    new_row = pd.DataFrame([float(predicted_price[0][0])], columns=['close'], index=[get_next_trading_day(df.index[-1] + timedelta(days=1))])
    df = pd.concat([df, pd.DataFrame(new_row)], ignore_index=False)
    return df

def forecast(scaler, data, model, days_before, days_after):
    df = data.filter(['close'])
    for _ in range(days_after):
        df = one_step_ahead(scaler, model, df, days_before)
    return df

def plot_data(stock, data, test, future, days_after):
    #Plot/Create the data for the graph
    #Visualize the data
    plt.figure(figsize=(16,8))
    plt.title('LSTM on ' + stock)
    plt.xlabel('date', fontsize=18)
    plt.ylabel('close', fontsize=18)
    plt.plot(data['close'])
    plt.plot(test[['close', 'Prediction']])
    plt.plot(future.tail(days_after+1)['close'])
    plt.legend(['Train', 'Test', 'Predict', 'Forecast'], loc='lower left')
    plt.show()
    
def get_trend(data, order=1):
    trend = 0
    coeffs = np.polyfit(np.arange(len(data)), list(data), order)
    slope = coeffs[-2]
    if slope>0:
        trend=1
    if slope<0:
        trend=-1  
    return int(trend)

def get_future_trend(stock, data, days_before, days_after):
    scaler = MinMaxScaler(feature_range=(0, 1)) 
    train_data, test_data = split_data(scaler, data, days_before)
    model = train_model(train_data, days_before)
    prediction = test_model(model, scaler, test_data, days_before)
    test = data[-len(prediction):]
    test.insert(1, "Prediction", prediction) 
    future = forecast(scaler, data, model, days_before, days_after)
    plot_data(stock, data, test, future, days_after)
    trend = get_trend(future.tail(days_after+1)['close'])
    return trend