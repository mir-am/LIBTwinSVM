#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Developers: Mir, A. and Mahdi Rahbar
# Version: 0.1 - 2019-03-20
# License:

"""
This test module tests the functionalities of estimators.py module
"""

# A temprory workaround to import LightTwinSVM for running tests
import sys
sys.path.append('./')

from libtsvm.estimators import LSTSVM
import unittest

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
        

if __name__ == '__main__':

    unittest.main()
