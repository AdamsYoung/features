# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 14:19:47 2016

@author: Young
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


#==============================================================================
# 利用前step个数据进行预测
#==============================================================================
def creat_dataset(dataset, step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-step-1):
        a = dataset[i:(i+step)]
        dataX.append(a)
        dataY.append(dataset[i+step])
    return np.array(dataX), np.array(dataY)

#==============================================================================
# 录入数据，若要更换品种，只需把以下'RB'的位置替换成相应的品种的简称
# 把指定列改为时序数据，并进行排序且设定为index
# 添加一列close_rate，记录收盘价变化率
# 添加一列logreturn，记录对数收益率
#==============================================================================
data = pd.read_csv('C:\Users\MSI\Desktop\deep learning\code\data\RB\RB dominant contract.csv')
data['Date'] = [pd.datetime.strptime(i, '%Y/%m/%d') for i in data['Date']]
data = data.sort_values('Date')
data = data.set_index('Date')
percent = np.array(data['close_diff'][1:]) / np.array(data['Close'][:-1])
data['close_rate'] = np.append(0, percent)
logreturn = np.array(np.log(data['Close'][1:])) - np.array(np.log(data['Close'][:-1]))
data['logreturn'] = np.append(0, logreturn)


#==============================================================================
# 计算14期的RSI
#==============================================================================
up = [i if i>0 else 0 for i in data['close_diff']]
down = [-i if i<0 else 0 for i in data['close_diff']]
RSI = []
for i in range(len(up)-14):
    increase = sum(up[i:i+14])
    decrease = sum(down[i:i+14]) if sum(down[i:i+14])>0 else 0.0001
    RS = float(increase)/decrease
    RS = 100 * RS / (1+RS)
    RSI.append(RS)
data['RSI'] = np.append([0]*14, RSI)


#==============================================================================
# 计算MACD及相关指标，包括DEA,DIF等
#==============================================================================
EMA12 = [data['Close'][0]]
EMA26 = [data['Close'][0]]
DIF = [0]
DEA9 = [0]
for i in range(len(data['Close'])-1):
    EMA_short = EMA12[-1] * 11.0/13 + data['Close'].values[i+1] * 2.0/13
    EMA_long = EMA26[-1] * 25.0/27 + data['Close'].values[i+1] * 2.0/27
    dif = EMA_short - EMA_long
    DEA = DEA9[-1] * 8.0/10 + dif * 2.0/10
    EMA12.append(EMA_short)
    EMA26.append(EMA_long)
    DIF.append(dif)
    DEA9.append(DEA)
data['EMA12'] = EMA12
data['EMA26'] = EMA26
data['DIF'] = DIF
data['DEA9'] = DEA9

MACD = (data['DIF'] - data['DEA9']) * 2
data['MACD'] = MACD


#==============================================================================
# 数据预处理，使得数据落在0,1之间
# 区分训练集和测试集，2016以前为训练集，以后为测试集
#==============================================================================
scaler = MinMaxScaler(feature_range=(0,1))
data.iloc[:, 5:] = scaler.fit_transform(data.iloc[:, 5:])

data_train = data[:'2015']
data_test = data['2016']


#==============================================================================
# 利用前一期数据进行预测
# LSTM的输入格式为(n_samples, n_steps, n_features)，因此需要修改输入的维数
#==============================================================================
step = 1
trainX, trainY = creat_dataset(data_train['close_diff'], step)
testX, testY = creat_dataset(data_test['close_diff'], step)

trainX = np.reshape(trainX, (trainX.shape[0], step, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], step, testX.shape[1]))


#==============================================================================
# 建立模型并进行拟合，每期1个输入4个LSTM单元1个输出
# 优化函数选择adam.?
#==============================================================================
model = Sequential()
model.add(LSTM(4, input_dim=step))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=100, batch_size=1)


#==============================================================================
# 输出模型在训练集和测试集上的标准误差
#==============================================================================
trainScore = model.evaluate(trainX, trainY)
print trainScore
trainScore = np.sqrt(trainScore)
print trainScore
trainScore = scaler.inverse_transform(np.array([[trainScore]]))
print 'Train Score: %.2f RMSE' % trainScore

testScore = model.evaluate(testX, testY)
print testScore
testScore = np.sqrt(testScore)
print testScore
testScore = scaler.inverse_transform(np.array([[testScore]]))
print 'Test Score: %.2f RMSE' % testScore


