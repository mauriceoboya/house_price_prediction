#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:25:58 2023

@author: fibonacci
"""

import tensorflow as tf
from tensorflow.keras import layers,models
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data=pd.read_csv('boston.csv')

y=data['MEDV']
x=data.drop(columns=['MEDV'],axis=1)

from sklearn.preprocessing import StandardScaler
str_x=StandardScaler()
x=str_x.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.28)

### fit a traditional linear regression
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model=model.fit(x_train,y_train)
ypred=model.predict(x_test)
from sklearn.metrics import r2_score
r2=r2_score(y_test, ypred)
print(r2)

#fitting a neural network