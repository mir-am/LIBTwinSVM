#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Developers: Mir, A. and Mahdi Rahbar
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0

"""
This test module tests the functionalities of estimators.py module
"""

# A temprory workaround to import LIBTwinSVM for running tests
import sys
sys.path.append('./')

from libtsvm.estimators import TSVM, LSTSVM
from libtsvm.preprocess import read_data
from sklearn.utils.testing import assert_greater
import unittest
import numpy as np

# Load the dataset for testing
X, y, file_name = read_data('./dataset/hepatitis.csv')

class TestLSTSVM(unittest.TestCase):
    
    """
    It tests the functionalities of the LSTSVM estimator
    """
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
    def test_lstsvm_set_get_params_linear(self):
        
        """
        It checks that set_params and get_params works correctly for linear 
        LSTSVM
        """
        
        expected_output = {'gamma': 1, 'C1': 0.1, 'rect_kernel': 1, 'C2': 0.25,
                           'kernel': 'linear'}
        
        lstsvm_cls = LSTSVM('linear')
        lstsvm_cls.set_params(**{'C1': 0.1, 'C2':0.25})
        
        self.assertEqual(lstsvm_cls.get_params(), expected_output,
                         'set_params and get_params output don\'t match')
        
    def test_lstsvm_set_get_params_rbf(self):
        
        """
        It checks that set_params and get_params works correctly for non-linear
        LSTSVM
        """
        
        expected_output = {'C2': 0.625, 'C1': 0.05, 'rect_kernel': 1,
                           'gamma': 0.5, 'kernel': 'RBF'}
        
        lstsvm_cls = LSTSVM('RBF')
        lstsvm_cls.set_params(**{'C1': 0.05, 'C2': 0.625, 'gamma': 0.5})
        
        self.assertEqual(lstsvm_cls.get_params(), expected_output,
                         'set_params and get_params output don\'t match')
        
    def test_linear_lstsvm_hepatitis(self):
        
        """
        It tests linear LSTSVM on hepatits dataset
        """
        
        clf = LSTSVM('linear', 1, 0.5, 0.5)
        clf.fit(X, y)
        pred = clf.predict(X)
        
        assert_greater(np.mean(y == pred), 0.85)
        
    def test_rbf_lstsvm_hepatitis(self):
        
        """
        It tests non-linear LSTSVM on hepatitis dataset
        """
        
        clf = LSTSVM('RBF', 1, 0.5, 0.5, 0.1)
        clf.fit(X, y)
        pred = clf.predict(X)
        
        assert_greater(np.mean(y == pred), 0.95)
        
    def test_rectangular_lstsvm_hepatitis(self):
        
        """
        It tests LSTSVM with rectangular on hepatitis dataset
        """
        
        clf = LSTSVM('RBF', 0.75, 0.5, 2, 0.1)
        clf.fit(X, y)
        pred = clf.predict(X)
        
        assert_greater(np.mean(y == pred), 0.75)
        

class TestTSVM(unittest.TestCase):
    
    """
    It tests the functionalities of the TSVM estimator
    """
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
    def test_linear_tsvm_set_get_params(self):
        
        """
        It checks that set_params and get_params works correctly for linear 
        TSVM estimator
        """
        
        expected_output = {'gamma': 1, 'C1': 0.5, 'rect_kernel': 1, 'C2': 1,
                           'kernel': 'linear'}
        
        tsvm_clf = TSVM('linear')
        tsvm_clf.set_params(**{'C1': 0.5, 'C2':1})
        
        self.assertEqual(tsvm_clf.get_params(), expected_output,
                         'set_params and get_params output don\'t match')


if __name__ == '__main__':

    unittest.main()
