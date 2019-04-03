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
    
    pass
    