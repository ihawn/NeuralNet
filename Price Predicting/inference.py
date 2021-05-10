import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import datetime, timedelta
import yfinance as yf
import os



#Parameters
read = 'BTC-USD'
future_days = 1
start = '2015-01-10'
adj_start = datetime.strptime(start, '%Y-%m-%d') - timedelta(future_days)
today = datetime.now().strftime('%Y-%m-%d')
bat_size = 64
chunk_size = 60
epochs = 75
infer_ = [
    'ETH-USD', 'BNB-USD', 'DOGE-USD', 'XRP-USD', 'ADA-USD', 'BCH-USD', 'LTC-USD', 'LINK-USD',
    'XLM-USD', 'THETA-USD', 'TRX-USD', 'EOS-USD', 'XMR-USD', 'NEO-USD'
]

difs = []



for k in range(len(infer_)):

    infer = infer_[k]
    print("Training ", infer_[k])

    try:
        os.remove('in.csv')
        os.remove('out.csv')
    except:
        print()


    x = yf.download(read, start=adj_start, end=datetime.now(), progress=False).dropna()
    y = yf.download(infer, start=start, end=today, progress=False).dropna()
    x.to_csv(r'in.csv')
    y.to_csv(r'out.csv')


    df_x = pd.read_csv('in.csv').dropna()
    df_y = pd.read_csv('out.csv').dropna()
    df_x = df_x[df_x['Date'] >= str(datetime.strptime(df_y['Date'].iloc[0], '%Y-%m-%d') - timedelta(-2))].copy()
    df_full = df_x


    #Parse data
    df_x = df_x.drop(['Date', 'Adj Close'], axis=1)
    df_y = df_y.drop(['Date', 'Adj Close', 'Open', 'High', 'Low', 'Volume'], axis=1)

    scaler = MinMaxScaler()
    data_training_x = scaler.fit_transform(df_x)
    data_training_y = scaler.fit_transform(df_y)


    #print(df_x.iloc[0].to_numpy())

    train_x = []
    train_y = []
    test_x = []

    for i in range(chunk_size, data_training_x.shape[0] - future_days):
        train_x.append(data_training_x[i-chunk_size:i])
        train_y.append(data_training_y[i, 0])
    for i in range(data_training_x.shape[0] - future_days, data_training_x.shape[0]):
        test_x.append(data_training_x[i-chunk_size:i])


    train_x, train_y = np.array(train_x), np.array(train_y)
    test_x = np.array(test_x)

    # Model
    model = Sequential()

    model.add(LSTM(units=60, activation='tanh', return_sequences=True, input_shape=(train_x.shape[1], 5)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=60, activation='tanh', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=80, activation='tanh', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=120, activation='tanh'))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(train_x, train_y, epochs=epochs, batch_size=bat_size)

    pred_y = model.predict(test_x)

    scale = 1 / scaler.scale_[0]
    pred_y *= scale

    difs.append(-(df_y['Close'].iloc[-1] - pred_y) / df_y['Close'].iloc[-1])






arr = np.array(difs).reshape(len(difs))
m = np.mean(arr)
arr = np.add(arr, -m)

data = {'Date': str(datetime.now().date()), 'Crypto': infer_, 'Adj Diff': arr}
df = pd.DataFrame(data)
df.to_csv('price_changes.csv', mode='a', header=False)

for i in range(len(infer_)):
    print(infer_[i], ": ", arr[i])
