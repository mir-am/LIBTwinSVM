#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# LIBTwinSVM: A Library for Twin Support Vector Machines
# Developers: Mir, A. and Mahdi Rahbar
# License: GNU General Public License v3.0

"""
A benchmark script that both the training and prediction time.
"""

from libtsvm.preprocess import DataReader
from libtsvm.estimators import LSTSVM, TSVM
from sklearn.model_selection import train_test_split
from os.path import join
import time

# Assign the path to folder that contains your own dataset
data_path = '/home/mir/mir-projects/datasets/NDC'

# Specify the dataset's filename
dataset = DataReader(join(data_path, 'NDC-train-10K.csv'), ',', False)
dataset.load_data(False, False)
X, y, _ = dataset.get_data()

print("Loaded the dataset...")

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

print("Split train/test sets...")

# A TSVM-based estimator
tsvm_model = TSVM()

train_t = time.time()
tsvm_model.fit(X_train, y_train)
print("Train time: %.5f" % (time.time() - train_t))

test_t = time.time()
tsvm_model.predict(X_test)
print("Test time: %.5f" % (time.time() - test_t))

