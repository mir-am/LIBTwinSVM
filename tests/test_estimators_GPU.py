#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Developers: Mir, A.
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0

"""
This test module tests the TSVM-based estimators on the GPU.
"""

# A temprory workaround to import LIBTwinSVM for running tests
import sys
sys.path.append('./')

from libtsvm.estimators import TSVM, LSTSVM
from libtsvm.preprocess import DataReader
from libtsvm import set_dev_GPU_tests, set_device_CPU
from sklearn.utils.testing import assert_greater
from sklearn.metrics import accuracy_score
import unittest
import numpy as np


class TestEstimatorsGPU(unittest.TestCase):
    """
    It tests both TSVM and LSTSVM classifiers on the GPU.
    """
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
    
    @classmethod
    def setUpClass(self):
        
        set_dev_GPU_tests()
        
        data_1 = DataReader('./dataset/duke.csv')
        data_1.load_data()
        
        self.X_1, self.y_1, f_n_1 = data_1.get_data()
        
        
        data_2 = DataReader('./dataset/checkerboard.csv')
        data_2.load_data()
        
        self.X_2, self.y_2, f_n_2 = data_2.get_data()
        
    @classmethod
    def tearDownClass(self):
        
        set_device_CPU()
        
    def test_linear_lstsvm_gpu(self):
        """
        It tests linear LSTSVM classifier using GPU.
        """
        
        clf = LSTSVM('linear', 1, 0.5, 0.5)
        clf.fit(self.X_1, self.y_1)
        pred = clf.predict(self.X_1)
        
        assert_greater(np.mean(self.y_1 == pred), 0.55)
        
    def test_rbf_lstsvm_gpu(self):
        """
        It tests LSTSVM classifier with RBF kernel using GPU.
        """
        
        clf = LSTSVM('RBF', 1, 1, 1, 2**-1)
        clf.fit(self.X_2, self.y_2)
        pred = clf.predict(self.X_2)
        
        assert_greater(np.mean(self.y_2 == pred), 0.75)
        
    def test_rect_lstsvm_gpu(self):
        """
        It tests LSTSVM classifier with Rectangular kernel using GPU.
        """
        
        clf = LSTSVM('RBF', 0.5, 1, 1, 2**-1)
        clf.fit(self.X_2, self.y_2)
        pred = clf.predict(self.X_2)
        
        assert_greater(np.mean(self.y_2 == pred), 0.75)
        
    def test_linear_tsvm_gpu(self):
        """
        It tests linear TSVM classifier using GPU.
        """
        
        clf = TSVM('linear', 1, 0.5, 0.5)
        clf.fit(self.X_1, self.y_1)
        pred = clf.predict(self.X_1)
        
        assert_greater(np.mean(self.y_1 == pred), 0.95)
        
    def test_rbf_tsvm_gpu(self):
        """
        It tests TSVM classifier with RBF kernel using GPU.
        """
        
        clf = TSVM('RBF', 1, 1, 1, 2**-1)
        clf.fit(self.X_2, self.y_2)
        pred = clf.predict(self.X_2)
        
        assert_greater(np.mean(self.y_2 == pred), 0.85)
        
    def test_rect_tsvm_gpu(self):
        """
        It tests TSVM classifier with Rectangular kernel using GPU.
        """
        
        clf = TSVM('RBF', 0.5, 1, 1, 2**-1)
        clf.fit(self.X_2, self.y_2)
        pred = clf.predict(self.X_2)
        
        assert_greater(np.mean(self.y_2 == pred), 0.83)
        