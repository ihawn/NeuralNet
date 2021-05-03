#https://kgptalkie.com/google-stock-price-prediction-using-rnn-lstm/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as ts
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.dates as mdates

test_date_pivot = '2020-01-01'
prediction_window = 60

###########
#Parse data
data = pd.read_csv('GOOG.csv', date_parser=True) #read file
data_training = data[data['Date'] < test_date_pivot].copy() #take all entries before 2020-01-01 as train data
data_test = data[data['Date'] >= test_date_pivot].copy() #take all entries from 2020 on as test data
data_training = data_training.drop(['Date', 'Adj Close'], axis=1) #drop Date and Adj Close columns from training

scaler = MinMaxScaler()
data_training = scaler.fit_transform(data_training) #scale data to fit between 0 and 1 (improves accuracy)

x_train = []
y_train = []

#divide the data into chunks of 60 rows
for i in range(prediction_window, data_training.shape[0]):
    x_train.append(data_training[i-prediction_window:i])
    y_train.append(data_training[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train) # convert x_train and y_train into numpy arrays
###########



###########
#Model
model = Sequential()

model.add(LSTM(units=60, activation='tanh', return_sequences=True, input_shape=(x_train.shape[1], 5)))
model.add(Dropout(0.2))

model.add(LSTM(units=60, activation='tanh', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=80, activation='tanh', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=120, activation='tanh'))
model.add(Dropout(0.2))

model.add(Dense(units=1))
###########



###########
#Train model
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=20, batch_size=1024)
###########



###########
#Predict
past_days = data_test.tail(prediction_window) #store past days. We need this to predict the next day

df = past_days.append(data_test, ignore_index=True) #append the test data
df = df.drop(['Date', 'Adj Close'], axis=1) #drop date and adj close columns as before
inputs = scaler.transform(df) #scale values to be between 0 and 1


x_test = []
y_test = []

for i in range(prediction_window, inputs.shape[0]):
    x_test.append(inputs[i-prediction_window:i])
    y_test.append(inputs[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_pred = model.predict(x_test) #actual predict

scale = 1/scaler.scale_[0] #scale test data back to original scale
y_pred *= scale
y_test *= scale
###########

test_dates = data[data['Date'] >= test_date_pivot]['Date'].copy() #store dates for plot labeling




fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test_dates, y_test, color='red', label='Real Google Stock Price')
ax.plot(y_pred, color='blue', label='Predicted Google Stock Price')
fmt_half_year = mdates.DayLocator(interval=6)
ax.xaxis.set_major_locator(fmt_half_year)
fmt_month = mdates.MonthLocator()
ax.xaxis.set_minor_locator(fmt_month)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
#ax.format_ydata = lambda x: f'${x:.2f}'  # Format the price.
ax.grid(True)
fig.autofmt_xdate()
plt.show()