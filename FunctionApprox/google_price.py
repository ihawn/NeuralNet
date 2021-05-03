#https://kgptalkie.com/google-stock-price-prediction-using-rnn-lstm/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as ts
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

###########
#Parse data
data = pd.read_csv('GOOG.csv', date_parser=True) #read file
data_training = data[data['Date'] < '2020-01-01'].copy() #take all entries before 2020-01-01 as train data
data_test = data[data['Date'] >= '2020-01-01'].copy() #take all entries from 2020 on as test data
data_training = data_training.drop(['Date', 'Adj Close'], axis=1) #drop Date and Adj Close columns from training

scaler = MinMaxScaler()
data_training = scaler.fit_transform(data_training) #scale data to fit between 0 and 1 (improves accuracy)

x_train = []
y_train = []

#divide the data into chunks of 60 rows
for i in range(60, data_training.shape[0]):
    x_train.append(data_training[i-60:i])
    y_train.append(data_training[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train) # convert x_train and y_train into numpy arrays
###########



###########
#Model
model = Sequential()

model.add(LSTM(units=60, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 5)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=80, activation = 'relu', return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units=120, activation = 'relu'))
model.add(Dropout(0.2))

model.add(Dense(units=1))
###########



###########
#Train model
model.compile(optimizer='adam')
###########