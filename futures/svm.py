# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 16:09:18 2016

@author: Young
"""

import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import preprocessing
import matplotlib.pyplot as plt

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
# 区分训练集和测试集，2016以前为训练集，以后为测试集
#==============================================================================
data_train = data[:'2015']
data_test = data['2016']


#==============================================================================
# 读取对应数据，对齐时间，计算特征，并对特征正规化，
# 采取的特征为：两期收盘价差，两期对数收益率，RSI, MACD, DIF, DEA
# 计算标签，观察样本是否均衡
#==============================================================================
daterange = data_train.index
close_diff = data_train['close_diff'][13:-1]
logreturn = data_train['logreturn'][13:-1]
high = data_train['High'][14:-1]
low = data_train['Low'][14:-1]
volume = data_train['Volume'][14:-1]
RSI = data_train['RSI'][15:]
MACD = data_train['MACD'][14:-1]
EMA12 = data_train['EMA12'][14:-1]
DIF = data_train['DIF'][14:-1]
DEA = data_train['DEA9'][14:-1]
EMA12 = data_train['EMA12'][14:-1]
EMA26 = data_train['EMA26'][14:-1]
HL_diff = np.array(np.log(high)) - np.array(np.log(low))
      
X = np.column_stack([close_diff[1:], close_diff[:-1], logreturn[1:], logreturn[:-1], RSI, MACD, DIF, DEA])
normalizer = preprocessing.Normalizer().fit(X)
X = normalizer.transform(X)

label_X = [1 if i>0 else -1 for i in data_train['close_diff'][15:]]
print sum([i==1 for i in label_X])/float(len(label_X))


#==============================================================================
# 拟合模型，输出训练集内正确率
#==============================================================================
clf = svm.SVC(C=1000, gamma=10e-1000000)
clf.fit(X, label_X)
predict_X = clf.predict(X)
print sum(predict_X==label_X)/float(len(X))


#==============================================================================
# 计算测试集的特征并正规化，滚动预测，为避免用到未来的数据，首先统一截断最后一天的数据
# 同时由于某些特征需要前2期的数据，因此在2016年的数据前加入2015年的最后2期数据  
# 同时预测测试集的标签，作为买入卖出的信号，并输出测试集的正确率
#==============================================================================
daterange = data_test.index
close_diff = np.append(data_train['close_diff'][-2:], data_test['close_diff'][:-1])
logreturn = np.append(data_train['logreturn'][-2:], data_test['logreturn'][:-1])
high = np.append(data_train['High'][-1:], data_test['High'][:-1])
low = np.append(data_train['Low'][-1:], data_test['Low'][:-1])
volume = np.append(data_train['Volume'][-1:], data_test['Volume'][:-1])
RSI = data_test['RSI']
MACD = np.append(data_train['MACD'][-1:], data_test['MACD'][:-1])
EMA12 = np.append(data_train['EMA12'][-1:], data_test['EMA12'][:-1])
EMA26 = np.append(data_train['EMA26'][-1:], data_test['EMA26'][:-1])
DIF = np.append(data_train['DIF'][-1:], data_test['DIF'][:-1])
DEA = np.append(data_train['DEA9'][-1:], data_test['DEA9'][:-1])
HL_diff = np.array(np.log(high)) - np.array(np.log(low))

Y = np.column_stack([close_diff[1:], close_diff[:-1], logreturn[1:], logreturn[:-1], RSI, MACD, DIF, DEA])
Y = normalizer.transform(Y)

label_Y = [1 if i>0 else -1 for i in data_test['close_diff']]

buy_sell = clf.predict(Y)
print sum(buy_sell==label_Y)/float(len(Y))


#==============================================================================
# 作出资金变化曲线、收益率变化曲线以及对数收益率变化曲线
# 同时作出一条辅助曲线，即全部买入各变量的变化曲线
#==============================================================================
capital = data_test['close_diff'].multiply(buy_sell, axis=0)
capital = capital.cumsum()

plt.figure(figsize=(17,8))
plt.plot(daterange, capital, '-', label='Capital Curve')
plt.legend()
plt.grid(1)

rate = data_test['close_rate'].multiply(buy_sell, axis=0)
rate = rate.cumsum()

plt.figure(figsize=(17,8))
plt.plot(daterange, rate, '-', label='Rate')
plt.plot(daterange, data_test['close_rate'].cumsum(), '-', label='Close Rate')
plt.legend()
plt.grid(1)

logreturn_2016 = data_test['logreturn']
return_rate = logreturn_2016.multiply(buy_sell)
return_rate = return_rate.cumsum()
yield_rate = logreturn_2016.cumsum()

plt.figure(figsize=(17,8))
plt.plot(daterange, return_rate, '-', label='Return Rate')
plt.plot(daterange, yield_rate, '-', label='Yield Rate')
plt.legend()
plt.grid(1)