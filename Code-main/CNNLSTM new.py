import numpy as np
import pandas as pd
import csv
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Conv2D, MaxPool2D, Flatten, Reshape, Dropout
from attention import Attention
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.preprocessing import StandardScaler

WindowSize = 5
TrainPercentage = 0.8

# load data
data_original = pd.read_csv(r'C:\Users\iris0\OneDrive\桌面\ml\yahoo_stock.csv')
yahoo_stock_complete = data_original.to_numpy()
yahoo_stock = yahoo_stock_complete[:, 1:6]


# normalize the dataset
scaler_features = StandardScaler()
scaler_features = scaler_features.fit(yahoo_stock)
yahoo_stock_normalized = scaler_features.transform(yahoo_stock)

OneDimFeatures = yahoo_stock_normalized

# construct training and testing data
ThreeDimFeatures = np.zeros([len(OneDimFeatures) - WindowSize, WindowSize, yahoo_stock.shape[1]])
ThreeDimTargets = np.zeros([len(OneDimFeatures) - WindowSize])

for FIdx in range(ThreeDimFeatures.shape[0]):
    for SIdx in range(WindowSize):
        for TIdx in range(ThreeDimFeatures.shape[1]):
            ThreeDimFeatures[FIdx][SIdx][TIdx] = OneDimFeatures[FIdx + SIdx][TIdx]

        ThreeDimTargets[FIdx] = yahoo_stock[FIdx + WindowSize][3]

x_train = np.zeros([int(ThreeDimFeatures.shape[0] * TrainPercentage), WindowSize, ThreeDimFeatures.shape[1]])
y_train = np.zeros([int(ThreeDimFeatures.shape[0] * TrainPercentage)])
x_test = np.zeros([(ThreeDimFeatures.shape[0] - x_train.shape[0]), WindowSize, ThreeDimFeatures.shape[1]])
y_test = np.zeros([ThreeDimFeatures.shape[0] - x_train.shape[0]])

for FIdx in range(x_train.shape[0]):
    for SIdx in range(WindowSize):
        for TIdx in range(x_train.shape[1]):
            x_train[FIdx][SIdx][TIdx] = ThreeDimFeatures[FIdx][SIdx][TIdx]
    
    y_train[FIdx] = ThreeDimTargets[FIdx]

for FIdx in range(x_test.shape[0]):
    for SIdx in range(WindowSize):
        for TIdx in range(x_test.shape[1]):
            x_test[FIdx][SIdx][TIdx] = ThreeDimFeatures[FIdx + x_train.shape[0]][SIdx][TIdx]
    
    y_test[FIdx] = ThreeDimTargets[FIdx + x_train.shape[0]]

# model construction
model = Sequential()
model.add(Conv2D(32,(2,2), input_shape=(x_train.shape[1], x_train.shape[2], 1)))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())  
model.add(Flatten()) 
model.add(Reshape((-1, 1)))
model.add(LSTM(16, activation='tanh', return_sequences=True))
model.add(Attention(units=16))
model.add(Dense(8, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='relu')) 
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse', 'mean_absolute_percentage_error'])
model.fit(x_train, y_train, batch_size=1, epochs=2000, verbose=2)
results = model.predict(x_test)
eva = model.evaluate(x_test, y_test)

OutputFile = open('results.txt', 'w')
OutputFile.write('mean absolute error: ' + str(eva[1]) + '\n')
OutputFile.write('mean squared error: ' + str(eva[2]) + '\n')
OutputFile.write(' : ' + str(eva[3]) + '\n\n')
OutputFile.write('real value, predicted value\n')
for Idx in range(len(results)):
    OutputFile.write(str(y_test[Idx]) + ',' + str(results[Idx][0]) + '\n')
OutputFile.close()

