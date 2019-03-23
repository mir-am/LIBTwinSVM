#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This module is created for testing different components of the library.
# It may be removed in the near future.

from preprocess import read_data
from estimators import LSTSVM
from model_selection import eval_metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
    
X, y, filename = read_data('../dataset/australian.csv')
    
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

lstsvm_model = LSTSVM('linear', 0.25, 0.5)

lstsvm_model.fit(x_train, y_train)
pred = lstsvm_model.predict(x_test)

test = eval_metrics(y_test, pred)
