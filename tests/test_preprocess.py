# -*- coding: utf-8 -*-

# LIBTwinSVM: A Library for Twin Support Vector Machines
# Developers: Mir, A. and Mahdi Rahbar
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0

"""
It tests the functionalites of the preprocess module
"""

from libtsvm.preprocess import DataReader
import unittest
import numpy as np


class TestPreprocess(unittest.TestCase):
    """
    It tests the methods of the DataReader class.
    """
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.data = DataReader('./dataset/test.csv', ',', True)
        self.data_mat = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
        self.labels = np.array([1, 1, 1, -1])
        
        
    def test_load_data_shuffle(self):
        """
        It tests loading a dataset with shuffling.
        """
        
        # Test shuffle
        self.data.load_data(True, False)

        self.assertNotEqual(np.array_equal(self.data_mat, self.data.X_train),
                            True)
        
    def test_get_data(self):
        """
        It tests whether the returned data is correct.
        """
        
        expected_output = [True, True, 'test']
        
        self.data.load_data(False, False)
        X, y, f_name = self.data.get_data()
        
        self.assertEqual([np.array_equal(X, self.data_mat),
                          np.array_equal(y, self.labels), f_name], expected_output)
    
    def test_get_data_info(self):
        """
        It tests whether the returned data characteristics is correct.
        """
        
        expected_output = [4, 2, 2, [-1, 1], ['x1', 'x2']]
        
        self.data.load_data(False, False)
        data_info = self.data.get_data_info()
        
        self.assertEqual([data_info.no_samples, data_info.no_features,
                          data_info.no_class, list(data_info.class_labels),
                          data_info.header_names], expected_output)
    