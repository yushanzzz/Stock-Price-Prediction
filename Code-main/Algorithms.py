import numpy as np
import pandas as pd
import csv
from tensorflow import keras
from keras.models import Sequential
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

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
TwoDimFeatures = np.zeros([len(OneDimFeatures) - WindowSize, WindowSize * yahoo_stock.shape[1]])
TwoDimTargets = np.zeros([len(OneDimFeatures) - WindowSize])

for FIdx in range(TwoDimFeatures.shape[0]):
    for SIdx in range(WindowSize):
        for TIdx in range(OneDimFeatures.shape[1]):
            TwoDimFeatures[FIdx][SIdx * OneDimFeatures.shape[1] + TIdx] = OneDimFeatures[FIdx + SIdx][TIdx]

        TwoDimTargets[FIdx] = yahoo_stock[FIdx + WindowSize][3]

x_train = np.zeros([int(TwoDimFeatures.shape[0] * TrainPercentage), TwoDimFeatures.shape[1]])
y_train = np.zeros([int(TwoDimFeatures.shape[0] * TrainPercentage)])
x_test = np.zeros([(TwoDimFeatures.shape[0] - x_train.shape[0]), TwoDimFeatures.shape[1]])
y_test = np.zeros([TwoDimFeatures.shape[0] - x_train.shape[0]])

for FIdx in range(x_train.shape[0]):
    for SIdx in range(x_train.shape[1]):
        x_train[FIdx][SIdx] = TwoDimFeatures[FIdx][SIdx]
    
    y_train[FIdx] = TwoDimTargets[FIdx]

for FIdx in range(x_test.shape[0]):
    for SIdx in range(x_test.shape[1]):
        x_test[FIdx][SIdx] = TwoDimFeatures[FIdx + x_train.shape[0]][SIdx]
    
    y_test[FIdx] = TwoDimTargets[FIdx + x_train.shape[0]]

#SVR: beginning
print('start SVR:\n')

OutputFile = open('Algorithms.txt', 'w')
OutputFile.write('SVR:\n')

for CValueExp in range(-5, 21, 2):
    for GammaExp in range(-15, 8, 2):
        CValue = 2**CValueExp
        Gamma = 2**GammaExp
        svm = SVR(kernel='rbf', gamma=Gamma, C=CValue)
        svm.fit(x_train, y_train)
        y_predict = svm.predict(x_test)
        
        test_score_mse = mean_squared_error(y_test, y_predict)
        test_score_mae = mean_absolute_error(y_test, y_predict)
        test_score_mape = mean_absolute_percentage_error(y_test, y_predict)
        OutputFile.write("CValue:" + str(CValueExp) + ", Gamma:" + str(GammaExp) + ", MSE:" + str(test_score_mse) + ", MAE:" + str(test_score_mae) + ", MAPE:" + str(test_score_mape))
        OutputFile.write('\n')

OutputFile.write('\n\n')
OutputFile.close()
#SVR: end

#CART decision tree: beginning
print('start CART:\n')

OutputFile = open('Algorithms.txt', 'a')
OutputFile.write('CART:\n')

CARTtree = tree.DecisionTreeRegressor()
CARTtree.fit(x_train, y_train)
y_predict = CARTtree.predict(x_test)

test_score_mse = mean_squared_error(y_test, y_predict)
test_score_mae = mean_absolute_error(y_test, y_predict)
test_score_mape = mean_absolute_percentage_error(y_test, y_predict)
OutputFile.write("MSE:" + str(test_score_mse) + ", MAE:" + str(test_score_mae) + ", MAPE:" + str(test_score_mape))

OutputFile.write('\n\n')
OutputFile.close()
#CART decision tree: end

#KNN: beginning
print('start KNN:\n')

OutputFile = open('Algorithms.txt', 'a')
OutputFile.write('KNN:\n')

for K in range(1, 16, 2):
    neigh = KNeighborsRegressor(n_neighbors=K)
    neigh.fit(x_train, y_train)
    y_predict = neigh.predict(x_test)

    test_score_mse = mean_squared_error(y_test, y_predict)
    test_score_mae = mean_absolute_error(y_test, y_predict)
    test_score_mape = mean_absolute_percentage_error(y_test, y_predict)
    OutputFile.write("K: " + str(K) + ", MSE:" + str(test_score_mse) + ", MAE:" + str(test_score_mae) + ", MAPE:" + str(test_score_mape))
    OutputFile.write('\n')

OutputFile.write('\n\n')
OutputFile.close()
#KNN: end

#Linear Regression: beginning
print('start Linear Regression:\n')

OutputFile = open('Algorithms.txt', 'a')
OutputFile.write('Linear Regression:\n')

LR = LinearRegression()
LR.fit(x_train, y_train)
y_predict = LR.predict(x_test)

test_score_mse = mean_squared_error(y_test, y_predict)
test_score_mae = mean_absolute_error(y_test, y_predict)
test_score_mape = mean_absolute_percentage_error(y_test, y_predict)
OutputFile.write("MSE:" + str(test_score_mse) + ", MAE:" + str(test_score_mae) + ", MAPE:" + str(test_score_mape))
OutputFile.write('\n\n')

OutputFile.close()
#KNN: end