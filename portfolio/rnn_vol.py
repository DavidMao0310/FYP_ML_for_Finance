import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import math
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter('ignore')
plt.style.use('ggplot')

use_data = pd.read_csv('dataset/rolling.csv')
use_data['Date'] = pd.to_datetime(use_data['Date'])
use_data.set_index('Date', inplace=True)
print(use_data)

#Hyperparameters
look_back = 3
epochs = 10


target_series = use_data['returnVol'].values
target_series = target_series.reshape(len(target_series), 1)
target_series = target_series.astype('float32')

# Normalize the values to a range from zero to 1
scaler = MinMaxScaler(feature_range=(0, 1))
target_series = scaler.fit_transform(target_series)

# Split the data into a training and test set
# We are using 80 percent of the data as training set and 20% as the test set.
train_size = int(len(target_series) * 0.80)
test_size = len(target_series) - train_size

train, test = target_series[0:train_size], target_series[train_size:len(target_series)]


# Create Data Set
# dataX is the is the rolling window of past oberservations
# dataY becomes the the value that is one day ahead of the rolling window.
# This is the label/prediction for the past values

def create_dataset(time_series, look_back):
    dataX, dataY = [], []

    for i in range(1, len(time_series) - look_back - 1):
        x = time_series[i:i + look_back]
        dataX.append(x)

        y = time_series[i + look_back + 1]
        dataY.append(y)

    return np.array(dataX), np.array(dataY)


# Create the dataset with rolling window for the training set and test set
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
print(len(trainX),len(trainY))
# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Create the model
# LSTM Neural Network with 128 Nodes is 3 Layers, followed by dense layer that outputs the prediction

model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(1, look_back)))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(1))
optimizer = Adam(lr=0.0001)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# TODO: Try diffrent (smaller) model architectures, add regularization

# Train the model, epochs is the number of iterations

history = model.fit(trainX, trainY, epochs=epochs, batch_size=1, verbose=1)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.tight_layout()
plt.show()

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# Inverse the normalization procedure of the data
trainY = np.reshape(trainY, (trainY.shape[0],))
testY = np.reshape(testY, (testY.shape[0],))

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# Calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.6f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.6f RMSE' % (testScore))

#plt.plot(crix_data['vol'].values, color = 'grey')

# Shift train predictions for plotting
trainPredictPlot = np.empty_like(target_series)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# Shift test predictions for plotting
testPredictPlot = np.empty_like(target_series)
testPredictPlot[:, :]= np.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 2:len(target_series)-2, :] = testPredict



new = pd.DataFrame(scaler.inverse_transform(target_series)/100,columns=['volatility'],index=use_data.index)
new['prediction_train_vol']=trainPredictPlot/100
new['prediction_test_vol']=testPredictPlot/100
print(new)
# Plot baseline and predictions
fig = plt.figure(figsize=(12,6))
new['volatility'].plot(alpha=0.8,color='skyblue')
new['prediction_train_vol'].plot(alpha=0.8,color='red')
new['prediction_test_vol'].plot(alpha=0.8,color='orange')
plt.legend()
plt.xlabel('Trading Days')
plt.ylabel('Volatility')
plt.title('LSTM Prediction')
plt.tight_layout()
plt.show()

print(new)
#new.to_csv('dataset/new_rnn_vol.csv')



