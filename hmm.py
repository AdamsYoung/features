# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 15:11:56 2016

@author: Young
"""

from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


#==============================================================================
# 录入数据，若要更换品种，只需把以下'RB'的位置替换成相应的品种的简称
# 把指定列改为时序数据，并进行排序且设定为index
# 添加一列close_rate，记录收盘价变化率
# 添加一列logreturn，记录对数收益率
#==============================================================================
data = pd.read_csv('C:\Users\MSI\Desktop\deep learning\code\data\RB\RB dominant contract.csv')
data['Date'] = [pd.datetime.strptime(date, '%Y/%m/%d') for date in data['Date']]
data = data.sort_values('Date')
data = data.set_index('Date')
percent = np.array(data['close_diff'][1:])/np.array(data['Close'][:-1])
data['close_rate'] = np.append(0, percent)
logreturn = np.array(np.log(data['Close'][1:])) - np.array(np.log(data['Close'][:-1]))
data['logreturn'] = np.append(0, logreturn)

data_train = data[:'2015']
data_test = data['2016']


#==============================================================================
# 读取对应数据，计算特征
# 特征为：昨天和前天的收盘价变化，前天的对数收益率，前天的高低价差
#==============================================================================
daterange = data_train.index
close_diff = data_train['close_diff'][1:]
logreturn = data_train['logreturn'][1:]
high = data_train['High'][1:]
low = data_train['Low'][1:]
HL_diff = np.array(np.log(high)) - np.array(np.log(low))

X = np.column_stack([close_diff[:-1], close_diff[1:], logreturn[:-1], HL_diff[:-1]])


#==============================================================================
# 拟合模型并输出状态转移矩阵和各状态的输出函数参数
#==============================================================================
model = GaussianHMM(n_components=4, covariance_type='diag', n_iter=5000).fit(X)
hidden_states = model.predict(X)

print "Transition matrix"
print model.transmat_
print 

print("Means and vars of each hidden state")
for i in range(model.n_components):
    print "{0}th hidden state".format(i)
    print "mean = ", model.means_[i]
    print "var = ", np.diag(np.sqrt(model.covars_[i]))
    print 


#==============================================================================
# 区分每个状态所对应的含义，并对每个状态收益率作图
# 由于每次跑程序各状态的编号都是随机的，因此用label变量来保存对状态的买卖判断
#==============================================================================
plt.figure(figsize=(17, 15))
for i in range(model.n_components):
    state = (hidden_states == i)
    plt.subplot(model.n_components, 1, i+1)
    plt.plot(daterange[state], logreturn[state], '.', label='hidden state %d' %i)
    plt.legend(loc='best')
    plt.grid(1)
    
data = pd.DataFrame({'daterange':daterange[1:-1], 'logreturn':logreturn[:-1],
                     'state':hidden_states})

label = []  #label用于保存对每个状态是买还是卖的判断
plt.figure(figsize=(17, 8))
for i in range(model.n_components):
    state = (hidden_states == i)
    state = np.append(0, state[:-1])
    data['state %d_return' %i] = data.logreturn.multiply(state, axis=0)
    rate = np.exp(data['state %d_return' %i].cumsum())
    plt.plot(data.daterange, rate, label='hidden state %d' %i)
    if rate.values[-1] > 1.2:
        label.append(1)
    elif rate.values[-1] < 0.8:
        label.append(-1)
    else:
        label.append(0)
    plt.legend()
    plt.grid(1)
    
    
#==============================================================================
# 进行回测，对历史数据的收益率进行计算并作图
#==============================================================================
buy_sell = [0] * len(hidden_states)
for i in range(model.n_components):
    if label[i] > 0:
        buy_sell = buy_sell + (hidden_states == i)
    elif label[i] < 0:
        buy_sell = buy_sell - (hidden_states == i)
buy_sell = np.append(0, buy_sell[:-1])

backtest_return = data.logreturn.multiply(buy_sell, axis=0)

plt.figure(figsize=(17, 8))
plt.plot_date(data.daterange, np.exp(backtest_return.cumsum()), '-', label='backtest')
plt.legend()
plt.grid(1)  


#==============================================================================
# 处理测试集的数据，滚动预测，为避免用到未来的数据，首先统一截断最后一天的数据
# 同时由于特征需要前2期的数据，因此在2016年的数据前加入2015年的最后2期数据    
#==============================================================================
daterange = data_test.index
close_diff = np.append(data_train['close_diff'][-2:], data_test['close_diff'][:-1])
logreturn = np.append(data_train['logreturn'][-2:], data_test['logreturn'][:-1])
high = np.append(data_train['High'][-2:], data_test['High'][:-1])
low = np.append(data_train['Low'][-2:], data_test['Low'][:-1])
HL_diff = np.array(np.log(high)) - np.array(np.log(low))

Y = np.column_stack([close_diff[:-1], close_diff[1:], logreturn[:-1], HL_diff[:-1]])

predict_states = model.predict(Y)

buy_sell = [0] * len(data_test)
for i in range(model.n_components):
    if label[i] > 0:
        buy_sell = buy_sell + (predict_states == i)
    elif label[i] < 0:
        buy_sell = buy_sell - (predict_states == i)
        
        
#==============================================================================
# 作出资金曲线和收益率曲线
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
return_rate = logreturn_2016 * buy_sell
return_rate = return_rate.cumsum()
yield_rate = logreturn_2016.cumsum()

plt.figure(figsize=(17,8))
plt.plot(daterange, return_rate, '-', label='Return Rate')
plt.plot(daterange, yield_rate, '-', label='Yield Rate')
plt.legend()
plt.grid(1)

