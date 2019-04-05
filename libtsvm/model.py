# -*- coding: utf-8 -*-

# Developers: Mir, A. and Mahdi Rahbar
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0

"""
This modules models data, user input in classes and functions
"""

class DataInfo:
    """
    It stores dataset characteristics such as no. samples, no. features and etc.
    
    Parameters
    ----------
    no_samples : int
        Number of samples in dataset.
        
    no_features : init
        Number of features in dataset.
        
    no_class : int
        Number of classes in dataset.
        
    class_labels: array-like
        Unique class labels.
    
    header_names: list
        Name of every feature in dataset.
    """
    
    def __init__(self, no_samples, no_features, no_class, class_labels,
                 header_names):
        
        self.no_samples = no_samples
        self.no_features = no_features
        self.no_class = no_class
        self.class_labels = class_labels
        self.header_names = header_names
        
        
class UserInput:
    """
    It encapsulates a user's input.
    
    Attributes
    ----------
    X_train : array-like, shape (n_samples, n_features)
              Training feature vectors, where n_samples is the number of samples
              and n_features is the number of features.
              
    y_train : array-like, shape(n_samples,)
              Target values or class labels.
              
    clf_type : str, {'tsvm', 'lstsvm'}
        Type of the classifier.
              
    class_type : str, {'binary', 'multiclass'}
        Type of classification problem.
        
    mc_scheme : str, {'ova', 'ovo'}
        The multi-class strategy
        
    result_path : str
        Path for saving classification results.
        
    kernel_type : str, {'linear', 'RBF'} 
        Type of the kernel function
        
    rect_kernel : float (default=1.0)
        Percentage of training samples for Rectangular kernel.
        
    test_method_tuple : tuple
        A two-element tuple which contains type of evaluation method and its
        parameter.
        
    C1_range : range
        Lower and upper bound for C1 penalty parameter.
        
    C2_range : range
        Lower and upper bound for C2 penalty parameter.
        
    u_range : range
        Lower and upper bound for gamma parameter.
    """
    
    def __init__(self):

        self.X_train, self.y_train = None, None
        self.clf_type = None
        self.class_type = None
        self.mc_scheme = None
        #self.filename = None
        self.result_path = './result'
        self.kernel_type = None
        self.rect_kernel = 1.0
        self.test_method_tuple = None
        # Lower and upper bounds of hyper-parameters
        self.C1_range = None
        self.C2_range = None
        self.u_range = None
        # Whether all the input varabiles are inserted or not.
        self.input_complete = False 
        
    def get_current_selection(self):
        """
        It returns a user's current selection for confirmation
        """
        
        curr_selection = ''
        
        if self.input_complete:
            
            curr_selection = """
            data
            """
            
        
    
    