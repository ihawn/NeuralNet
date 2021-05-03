import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

#read data
data = pd.read_csv('FunctionData.csv')
testData = pd.read_csv('FunctionTestData.csv')

#model
model = keras.Sequential()
model.add(keras.layers.Dense(40, input_dim=1, activation='relu'))
model.add(keras.layers.Dense(20, activation='relu'))
model.add(keras.layers.Dense(10, activation='relu'))
model.add(keras.layers.Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(data['X'], data['Y'], epochs=25, batch_size=256)

#test the model
results = model.evaluate(testData['X'])
print(results)

#define function and plot data alongside function
def f(x):
    return np.sin(x)

predictions = model.predict(testData['X'])

x1 = np.arange(-20.0, 20.0, 0.1)
plt.plot(x1, f(x1))
plt.plot(testData['X'], predictions, "o", markersize=3)
plt.show()




