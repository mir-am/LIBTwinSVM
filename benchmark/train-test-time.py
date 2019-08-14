#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# LIBTwinSVM: A Library for Twin Support Vector Machines
# Developers: Mir, A. and Mahdi Rahbar
# License: GNU General Public License v3.0

"""
A benchmark script that both the training and prediction time.
"""

from libtsvm.preprocess import DataReader
from libtsvm.estimators import TSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from os.path import join
import time

# Assign the path to folder that contains your own dataset
data_path = '/home/mir/mir-projects/NDC'

# Specify the dataset's filename
dataset = DataReader(join(data_path, 'NDC-train-1l.csv'), ',', False)
dataset.load_data(False, False)
X, y, _ = dataset.get_data()

print("Loaded the dataset...")

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)


print(X_train.shape)
print("Split train/test sets...")

# A TSVM-based estimator
tsvm_model = TSVM()

train_t = time.time()
tsvm_model.fit(X_train, y_train)
print("Train time: %.5f" % (time.time() - train_t))

test_t = time.time()
pred = tsvm_model.predict(X_test)
print("Test time: %.5f" % (time.time() - test_t))

acc = accuracy_score(y_test, pred)
print("Acc: %.2f" % (acc *  100))


