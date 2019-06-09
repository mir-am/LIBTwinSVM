# -*- coding: utf-8 -*-

# LIBTwinSVM: A Library for Twin Support Vector Machines
# Developers: Mir, A. and Mahdi Rahbar
# License: GNU General Public License v3.0

"""
A benchmark script for demonstrating the effectiveness of LIBTwinSVM in terms of
prediction accuracy. 
"""

from libtsvm.preprocess import DataReader
from libtsvm.estimators import TSVM
from libtsvm.model_selection import Validator, grid_search

data_path = '../dataset/australian.csv'

dataset = DataReader(data_path, ',', True)
dataset.load_data(True, False)
X, y, _ = dataset.get_data()

tsvm_clf = TSVM(kernel='linear')

val = Validator(X, y, ('CV', 5), tsvm_clf)
eval_method = val.choose_validator()

params = {'C1': (-5, 5), 'C2': (-5, 5), 'gamma': None}

best_acc, best_acc_std, opt_params, _ = grid_search(eval_method, params)

print("Best accuracy: %.2f+-%.2f | Optimal parameters: %s" % (best_acc, best_acc_std,
                                          str(opt_params)))
