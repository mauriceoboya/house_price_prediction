#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:25:58 2023

@author: fibonacci
"""

import tensorflow as tf
import keras
from tensorflow.keras import layers,models
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import numpy as np
import datacleaner
from fasteda import fast_eda
from datacleaner import autoclean
data=pd.read_csv('boston.csv')
data=autoclean(data)

fast_eda(data)
y=data['MEDV']
x=data.drop(columns=['MEDV'],axis=1)

from sklearn.preprocessing import StandardScaler

str_x=StandardScaler()
x=str_x.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.28)

### fit a traditional linear regression
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model=model.fit(x_train,y_train)
ypred=model.predict(x_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test, ypred)
print(r2)

#fitting a neural network

model=keras.Sequential([
    keras.layers.Dense(64,input_dim=x_train.shape[1],activation='relu'),
    keras.layers.Dense(32,activation='relu'),
    keras.layers.Dense(6,activation='relu'), 
    keras.layers.Dense(1,activation='relu')
    ])

model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,epochs=1000,batch_size=32,verbose=1)

y_preds=model.predict(x_test)

r2=r2_score(y_test, y_preds)
print(r2)
