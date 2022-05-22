# -*- coding: utf-8 -*-

#Authors Gissel Velarde, Pedro Brañez, Alejandro Bueno, Rodrigo Heredia, & Mateo Lopez
#04.02.2022
#Suggested citation: Velarde, G., Brañez, P., Bueno, A., Heredia, R., & Lopez, M. (Accepted). An Open-Source and Reproducible Implementation of GRU and LSTM Networks for Time Series Forecasting. Submitted to ITISE CONFERENCE: 8th International Conference on Time Series and Forecasting, June 27th-30th, 2022.
#Downloads the BANKEX dataset
#Saves a 10 by 3032 matrix with the closing prices of time series (Un-normalized)
#Recovers the matrix from CSV
import math
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr

pip install yfinance --upgrade --no-cache-dir

"""Collecting BANKEX dataset"""

import yfinance as yf
yf.pdr_override()

X=[]
Y=[]

bank = ["AXISBANK.BO", "BANKBARODA.BO", "FEDERALBNK.BO", "HDFCBANK.BO", "ICICIBANK.BO", "INDUSINDBK.BO", "KOTAKBANK.BO", "PNB.BO", "SBIN.BO", "YESBANK.BO"]

for i in range(len(bank)):
  df1 = pdr.get_data_yahoo(bank[i], start='2005-07-12', end='2017-11-03')
  data = df1.filter(["Close"])
  dataset = data.values
  scaled_data = dataset
  if scaled_data.shape[0] > 3033:
      X.append(scaled_data[3:3035, 0])  
  else:
      X.append(scaled_data[0:3032, 0])
  
X = np.array(X)

np.savetxt('BANKEX.csv', X, delimiter=',')

from numpy import genfromtxt
X_n = genfromtxt('BANKEX.csv', delimiter=',')

X_n.shape