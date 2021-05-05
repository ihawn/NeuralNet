#https://kgptalkie.com/google-stock-price-prediction-using-rnn-lstm/

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as ts
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.dates as mdates
import yfinance as yf
from datetime import datetime, timedelta

prediction_window = 3
test_date_pivot = (datetime.now() - timedelta(prediction_window)*3).strftime('%Y-%m-%d')
stock = 'DOGE.csv'

startdate = '2014-09-15'
today = datetime.now().strftime('%Y-%m-%d')
yesterday = (datetime.now() - timedelta(1)).strftime('%Y-%m-%d')


#Download data from Yahoo finance and write to file
data = yf.download('DOGE-USD',
                      start=startdate,
                      end=yesterday,
                      progress=False)

data.to_csv(r'DOGE.csv')


#Calculate average of last prediction_window rows and append to the end of the file
step = prediction_window
m_df = pd.read_csv(stock)
open_ave = m_df['Open'].tail(step).mean()
high_ave = m_df['High'].tail(step).mean()
low_ave = m_df['Low'].tail(step).mean()
close_ave = m_df['Close'].tail(step).mean()
volume_ave = m_df['Volume'].tail(step).mean()

o = []
o += step*[open_ave]
h = []
h += step*[high_ave]
l = []
l += step*[low_ave]
c = []
c += step*[close_ave]
v = []
v += step*[volume_ave]
dates = pd.date_range(datetime.today(), periods=step).tolist()
future_dates = [datetime.strftime(d, '%Y-%m-%d') for d in dates]

ave_df = pd.DataFrame({"Date": future_dates, "Open": o, "High": h, "Low": l, "Close": c, "Volume": v})



###########
#Parse data

data = pd.read_csv(stock, date_parser=True) #read file
data = data.append(ave_df, ignore_index=True)
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
model.fit(x_train, y_train, epochs=5000, batch_size=1024)
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

print(inputs.shape[0] - prediction_window)

x_test, y_test = np.array(x_test), np.array(y_test)

y_pred = model.predict(x_test) #actual predict


scale = 1/scaler.scale_[0] #scale test data back to original scale
y_pred *= scale
y_test *= scale
###########

test_dates = data[data['Date'] >= test_date_pivot]['Date'].copy() #store dates for plot labeling

# Visualising the results
y_test = y_test[:len(y_test) - prediction_window]
plt.figure(figsize=(14,5))
plt.plot(y_test, color='red', label='Real Stock Price')
plt.plot(test_dates, y_pred, color='blue', label='Predicted Stock Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()




