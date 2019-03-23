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
        


if __name__ == '__main__':

    unittest.main()
