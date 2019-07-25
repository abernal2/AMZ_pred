#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 14:32:14 2019

@author: abernal2
"""

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler as sc
import matplotlib.pyplot as plt
from keras.models import model_from_json

# Training
training_data = pd.read_csv('AMZNtrain.csv')
num_samples = training_data.shape[0]

trainX =training_data.iloc[:,1:-2]

# Scaling each dimension/feature
list_sc = [sc(feature_range=(0,1)) for i in range(4)]

for i in range(4):
    trainX.iloc[:,i] = list_sc[i].fit_transform(trainX.iloc[:,i].values.reshape(num_samples,1))
    
trainX_scaled = trainX.values

lookback = 60
X_train = []
y_train = []
for i in range(lookback, num_samples):
    # Grab all features, i.e open, high, low, and close
    X_train.append(trainX_scaled[i-lookback:i])
    # I want to predict the 'close' price, i.e. the last column
    y_train.append(trainX_scaled[i, -1])
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

def build_model(X, units_=[50]*4, dropout=0.2, optimizer = 'adam', loss = 'mean_squared_error'):
    
    regressor = Sequential()
    
    regressor.add(LSTM(units = units_[0], return_sequences = True, input_shape = (X.shape[1], 1)))
    regressor.add(Dropout(dropout))
    
    for i in range(len(units_-2)):
        regressor.add(LSTM(units = units_[1+i], return_sequences = True))
        regressor.add(Dropout(dropout))

    regressor.add(LSTM(units = units_[-1]))
    regressor.add(Dropout(dropout))
    
    regressor.add(Dense(units = 1))
    
    regressor.compile(optimizer = optimizer, loss = loss)
    
    return regressor

regressor = build_model(X_train)

print("Printing model summary.............")
regressor.summary()

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


# Testing
test_data = pd.read_csv('AMZNtest.csv')
num_test_samples = test_data.shape[0]

testX =test_data.iloc[:,1:-2]
closing_price = testX[:,-1]

for i in range(4):
    testX.iloc[:,i] = list_sc[i].transform(testX.iloc[:,i].values.reshape(num_test_samples,1))
    
testX = pd.concat( [pd.DataFrame(data=trainX_scaled[-60:,:],columns=testX.columns),testX], axis=0, ignore_index = True)
testX = testX.values

X_test = []
for i in range(lookback, lookback+num_test_samples):
    X_test.append(testX[i-60:i])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_train.shape[2]))

# Predict
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = list_sc[-1].inverse_transform(predicted_stock_price)

# Plotting results
plt.plot(closing_price, color = 'black', label = 'AMZ Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted AMZ Stock Price')
plt.title('AMZ Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('AMZ Stock Price')
plt.legend()
plt.show()

# serialize model to JSON
model_json = regressor.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
regressor.save_weights("model.h5")
print("Saved model to disk")


#keras.optimizers.Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
#keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
#keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

#import datetime
#customdate = datetime.datetime(2016, 1, 1)
#x_axis = [customdate + datetime.timedelta(days=i) for i in range(8)]
#plt.plot(x,y)
#plt.gcf().autofmt_xdate() # rotates x-labels 
#plt.show()

#np.newaxis

## load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
## load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")