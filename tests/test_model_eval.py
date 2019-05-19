# -*- coding: utf-8 -*-

# LIBTwinSVM: A Library for Twin Support Vector Machines
# Developers: Mir, A.
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0

"""
This test module tests the functionalities of model_eval.py module
"""

from libtsvm.estimators import TSVM
from libtsvm.model_selection import Validator
from libtsvm.preprocess import DataReader
from libtsvm.model_eval import save_model, load_model
from os.path import join, isfile
import unittest


class TestModel(unittest.TestCase):
    """
    It tests load and save abilities for fitted TSVM-based models.
    """
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
        self.save_model_path = join('tests', 'test_tsvm_model.joblib')
        
        self.data = DataReader('./dataset/2d-synthetic.csv', ',', True)
        self.data.load_data(False, False)
        self.X, self.y, _ = self.data.get_data()
        
    def test_save_model(self):
        """
        It saves a TSVM-based model on the disk for test purpose.
        """
        
        if not isfile(self.save_model_path):
        
            tsvm_model = TSVM()
            eval_m = Validator(self.X, self.y, ('CV', 5), tsvm_model)
            save_model(eval_m, {'C1': 1, 'C2': 1}, self.save_model_path)
        
    def test_load_model(self):
        """
        It loads a pre-trained TSVM-based model for test purpose. 
        """
        
        expected_output = {'model_name': 'TSVM',
                           'kernel': 'linear', 'rect_kernel': 1,
                           'no_params': '6', 'h_params': {'C1': 1, 'C2': 1,
                           'gamma': 1, 'kernel': 'linear', 'rect_kernel': 1},
                           'clf_type': 'Binary'}
        
        if not isfile(self.save_model_path):
            
            self.test_save_model()
            
        model, output = load_model(self.save_model_path)
        
        self.assertEqual(output, expected_output)
        