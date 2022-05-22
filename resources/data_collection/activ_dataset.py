# -*- coding: utf-8 -*-

#Authors Gissel Velarde 14.05.2022
#Suggested citation: Velarde, G., Brañez, P., Bueno, A., Heredia, R., & Lopez, M. (Accepted). An Open-Source and Reproducible Implementation of GRU and LSTM Networks for Time Series Forecasting. Submitted to ITISE CONFERENCE: 8th International Conference on Time Series and Forecasting, June 27th-30th, 2022.
#creates a synthetic dataset considering 5 days of high activity and 2 days of low activity
#Saves a X matrix  (Un-normalized)
#It shows that recovering the matrix from CSV returns the same matrix

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt

s = np.zeros(7)
s[0:5] = 1
for d in range(9):
  s=np.concatenate((s, s), axis=None)
ni=np.random.randint(5, size=(1, len(s)))*0.01
s = s + ni

s.shape[1]

s.shape

x = np.arange(1,s.shape[1]+1)

y = x*0.0001

plt.plot(x,y)

for r in range(4):
  s = np.concatenate((s, s), axis=0)
  
s.shape
  
for r in range(15):
  s[r,:] = np.roll(s[0,:], r*2)

ni=np.random.randint(9, size=(1, s.shape[1]))*0.01
s[1,:] = s[1,:] + ni

ni=np.random.randint(15, size=(1, s.shape[1]))*0.01
s[2,:] = s[2,:] + ni

ni=np.random.randint(50, size=(1, s.shape[1]))*0.01
s[3,:] = s[3,:] + ni

ni=np.random.randint(20, size=(1, s.shape[1]))*0.01
s[4,:] = s[4,:] + ni

ni=np.random.randint(5, size=(1, s.shape[1]))*0.01
s[6,:] = s[6,:] + ni

ni=np.random.randint(40, size=(1, s.shape[1]))*0.01
s[8,:] = s[8,:] + ni

s.shape

y.shape

y = np.reshape(y,(1,s.shape[1]))

for r in range(1,10):
  s[r,:] = s[r,:]+y
  s[r,:] = s[r,:]*r
  
X = s[0:10,:]
  
X.shape

np.savetxt('activities.csv', X, delimiter=',')

plt.plot(s[0,0:50], label = 's')

X.shape

for r in range (10):
  plt.plot(X[r,3400:3500], label = 's')

plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')