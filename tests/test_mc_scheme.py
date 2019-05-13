# -*- coding: utf-8 -*-

# LIBTwinSVM: A Library for Twin Support Vector Machines
# Developers: Mir, A.
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0

"""
This test module tests the functionalities of mc_scheme.py module
"""

from libtsvm.estimators import LSTSVM
from libtsvm.mc_scheme import OneVsAllClassifier, OneVsOneClassifier
from libtsvm.preprocess import DataReader
from sklearn.utils.testing import assert_greater
import unittest
import numpy as np

# Load the dataset for testing
data = DataReader('./dataset/iris.csv', ',', True)
data.load_data(False, False)
X, y, file_name = data.get_data()

class TestOVA(unittest.TestCase):
    """
    It tests the functionalities of One-vs-All classifier.
    """
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
    def test_linear_OVA(self):
        """
        It trains and tests a linear OVA classifier.
        """
        
        bin_clf = LSTSVM('linear')
        ova_clf = OneVsAllClassifier(bin_clf)
        
        ova_clf.fit(X, y)
        pred = ova_clf.predict(X)
        
        assert_greater(np.mean(y == pred), 0.82)
        
    def test_rbf_OVA(self):
        """
        It trains and tests a non-linear OVA classifier with RBF kernel.
        """
        
        bin_clf = LSTSVM('RBF', 1, 1, 1, 2**-5)
        ova_clf = OneVsAllClassifier(bin_clf)
        
        ova_clf.fit(X, y)
        pred = ova_clf.predict(X)
        
        assert_greater(np.mean(y == pred), 0.95)
        
    def test_rect_OVA(self):
        """
        It trains and tests a non-linear OVA classifier with rectangular kernel.
        """
        
        bin_clf = LSTSVM('RBF', 0.5, 1, 1, 2**-3)
        ova_clf = OneVsAllClassifier(bin_clf)
        
        ova_clf.fit(X, y)
        pred = ova_clf.predict(X)
        
        assert_greater(np.mean(y == pred), 0.96)
        

class TestOVO(unittest.TestCase):
    """
    It tests the functionalities of One-vs-One classifier.
    """
        
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
    
    def test_linear_OVO(self):
        """
        It trains and tests a linear OVO classifier.
        """
        
        bin_clf = LSTSVM('linear')
        ova_clf = OneVsOneClassifier(bin_clf)
        
        ova_clf.fit(X, y)
        pred = ova_clf.predict(X)
        
        #print("Acc: %.2f" % (np.mean(y == pred) * 100))
        
        assert_greater(np.mean(y == pred), 0.96)
        
    def test_RBF_OVO(self):
        """
        It trains and tests a non-linear OVO classifier with RBF kernel.
        """
        
        bin_clf = LSTSVM('RBF', 1, 1, 1, 2**-4)
        ova_clf = OneVsOneClassifier(bin_clf)
        
        ova_clf.fit(X, y)
        pred = ova_clf.predict(X)
        
        assert_greater(np.mean(y == pred), 0.97)
        
    def test_rect_OVO(self):
        """
        It trains and tests a non-linear OVO classifier with rectangular kernel.
        """
        
        bin_clf = LSTSVM('RBF', 0.75, 1, 1, 2**-2)
        ova_clf = OneVsOneClassifier(bin_clf)
        
        ova_clf.fit(X, y)
        pred = ova_clf.predict(X)
        
        assert_greater(np.mean(y == pred), 0.96)
