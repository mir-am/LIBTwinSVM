# -*- coding: utf-8 -*-

# Developers: Mir, A.
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0

"""
This test module tests the functionalities of mc_scheme.py module
"""

# A temprory workaround to import LIBTwinSVM for running tests
import sys
sys.path.append('./')

from libtsvm.estimators import LSTSVM, TSVM
from libtsvm.mc_scheme import OneVsAllClassifier, OneVsOneClassifier
from libtsvm.model import UserInput
import unittest


class TestUserInput(unittest.TestCase):
    """
    It tests the functionalities of UserInput class.
    """
    
    def __init__(self, *args, **kwargs):
        
        super().__init__(*args, **kwargs)
        
    def test_get_kernel_selection(self):
        """
        It tests the name of the kernel function that is returned.
        """
    
        expected_output = ['Linear', 'Gaussian (RBF)',
                           'Rectangular (50.0% of samples)']
        output = []
        
        user_in = UserInput()
        user_in.kernel_type = 'linear'
        output.append(user_in._get_kernel_selection())
        
        user_in.kernel_type = 'RBF'
        output.append(user_in._get_kernel_selection())
        
        user_in.rect_kernel = 0.5
        output.append(user_in._get_kernel_selection())
        
        self.assertEqual(output, expected_output)
        
    def test_get_eval_method(self):
        """
        It tests the name of the evaluation method that is returned.
        """
        
        expected_output = ['5-Fold cross-validation',
                           'Train/Test split (70%/30%)']
        output = []
    
        user_in = UserInput()
        user_in.test_method_tuple = ('CV', 5)
        output.append(user_in._get_eval_method())
        
        user_in.test_method_tuple = ('t_t_split', 70)
        output.append(user_in._get_eval_method())
        
        self.assertEqual(output, expected_output)
        
    def test_get_mc_scheme(self):
        """
        It tests the name of multi-class classifcation scheme that is returned.
        """
        
        expected_output = ["Binary", "One-vs-All", "One-vs-One"]
        output = []
        
        user_in = UserInput()
        user_in.class_type = 'binary'
        output.append(user_in._get_mc_scheme())
        
        user_in.class_type = 'multiclass'
        user_in.mc_scheme = 'ova'
        output.append(user_in._get_mc_scheme())
        
        user_in.mc_scheme = 'ovo'
        output.append(user_in._get_mc_scheme())
        
        self.assertEqual(output, expected_output)
        
    def test_get_clf_name(self):
        """
        It tests the name of selected classifier that is returned.
        """
        
        expected_output = ['TSVM', 'LSTSVM', 'OVA-LSTSVM', 'OVO-LSTSVM',
                           'OVO-TSVM', 'OVA-TSVM']
        output = []
        
        user_in = UserInput()
        user_in.class_type = 'binary'
        user_in.clf_type = 'tsvm'
        output.append(user_in._get_clf_name())
        
        user_in.clf_type = 'lstsvm'
        output.append(user_in._get_clf_name())
        
        user_in.class_type = 'multiclass'
        user_in.mc_scheme = 'ova'
        output.append(user_in._get_clf_name())
        
        user_in.mc_scheme = 'ovo'
        output.append(user_in._get_clf_name())
        
        user_in.clf_type = 'tsvm'
        output.append(user_in._get_clf_name())
        
        user_in.mc_scheme = 'ova'
        output.append(user_in._get_clf_name())
        
        self.assertEqual(output, expected_output)
        
    def test_get_current_selection(self):
        """
        It tests user's current selection that is returned.
        """
        
        expected_output = ("Dataset: Iris\nClassifier: Standard TwinSVM\nKernel: Gaussian (RBF)\n"
            "Multi-class scheme: One-vs-One\nEvaluation method: 5-Fold cross-validation\n"
            "Range of parameters for grid search: (step:1.00)\nC1: 2^-5 to 2^5 |"
            "C2: 2^-5 to 2^5 | u: 2^-10 to 2^2\n"
            "Results' path:/home/results/\nLog File: No")
        
        user_in = UserInput()
        user_in.data_filename = 'Iris'
        user_in.clf_type = 'tsvm'
        user_in.kernel_type = 'RBF'
        user_in.class_type = 'multiclass'
        user_in.mc_scheme = 'ovo'
        user_in.test_method_tuple = ('CV', 5)
        user_in.C1_range = (-5, 5)
        user_in.C2_range = (-5, 5)
        user_in.u_range = (-10, 2)
        user_in.result_path = '/home/results/'
        user_in.log_file = False
        user_in.input_complete = True
        
        output = user_in.get_current_selection()
        
        self.assertEqual(output, expected_output)
        
    def test_get_selected_clf(self):
        """
        It tests the classifier object that is returned.
        """
        
        # TODO: This test can be improved by re-thinking the classifiers
        # equality test 
        output = []
        
        user_in = UserInput()
        user_in.kernel_type = 'linear'
        user_in.rect_kernel = 1.0
        user_in.clf_type = 'tsvm'
        output.append(user_in.get_selected_clf())
        
        user_in.clf_type = 'lstsvm'
        output.append(user_in.get_selected_clf())
        
        user_in.class_type = 'multiclass'
        user_in.mc_scheme = 'ova'
        output.append(user_in.get_selected_clf())
        
        user_in.mc_scheme = 'ovo'
        output.append(user_in.get_selected_clf())
        
        eql_clfs = all([isinstance(output[0], TSVM), isinstance(output[1],
                        LSTSVM), isinstance(output[2], OneVsAllClassifier),
                       isinstance(output[3], OneVsOneClassifier)])
            
        self.assertEqual(eql_clfs, True)
        
    def test_get_clf_params(self):
        """
        It tests hyper-parameters of a classifier that is returned.
        """
        
        expected_output = [{'C1': 0.5, 'C2': 1}, {'C1': 1, 'C2': 1,
                           'gamma': 0.125}]
        output = []
        
        user_in = UserInput()
        user_in.kernel_type = 'linear'
        user_in.C1 = 0.5
        user_in.C2 = 1
        output.append(user_in.get_clf_params())
        
        user_in.kernel_type = 'RBF'
        user_in.C1 = 1
        user_in.C2 = 1
        user_in.u = 0.125
        output.append(user_in.get_clf_params())
        
        self.assertEqual(output, expected_output)
        