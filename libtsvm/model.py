# -*- coding: utf-8 -*-

# Developers: Mir, A. and Mahdi Rahbar
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0

"""
This modules models data, user input in classes and functions
"""

from datetime import datetime
from libtsvm.estimators import TSVM, LSTSVM
from libtsvm.mc_scheme import OneVsAllClassifier, OneVsOneClassifier


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
              
    data_filename : str
        The filename of a user's dataset.
              
    clf_type : str, {'tsvm', 'lstsvm'}
        Type of the classifier.
              
    class_type : str, {'binary', 'multiclass'}
        Type of classification problem.
        
    mc_scheme : str, {'ova', 'ovo'}
        The multi-class strategy
        
    result_path : str
        Path for saving classification results.
        
    save_clf_results : boolean (default=True)
        Whether to save the classification results or not.
        
    save_best_model : boolean (default=False)
        Whether to save the best fitted model or not.    
    
    log_file : boolean
        Whether to create a log file or not.
        
    kernel_type : str, {'linear', 'RBF'} 
        Type of the kernel function
        
    rect_kernel : float (default=1.0)
        Percentage of training samples for Rectangular kernel.
        
    test_method_tuple : tuple
        A two-element tuple which contains type of evaluation method and its
        parameter.
        
    step_size : float
        Step size for generating search elements.
        
    C1_range : tuple
        Lower and upper bound for C1 penalty parameter.
        example: (-4, 5), first element is lower bound and second element is
        upper bound
        
    C2_range : tuple
        Lower and upper bound for C2 penalty parameter.
        
    u_range : tuple
        Lower and upper bound for gamma parameter.
        
    C1 : float
        The penalty parameter.
        
    C2 : float
        The penalty parameter.
        
    u : float
        The parameter of the RBF kernel function.
        
    input_complete : boolean
        Whether all the required inputs are set.
        
    linear_db : boolean
        Whether to plot decision boundary or not.
        
    fig_save : boolean
        Whether to save the figure or not.
        
    fig_dpi : int
        DPI of the figure. It determines the quality of the output image.
        
    fig_save_path : str
        The path at which a figure will be saved.
        
    pre_trained_model : object
        A pre-trained TSVM-based classifer.
        
    save_pred : boolean
        Whether to save predicted labels of test samples in a file or not.
        
    save_pred_path : str
        The path at which the file of predicted labels will be saved.
    """
    
    def __init__(self):
        
        # Data
        self.X_train, self.y_train = None, None
        self.data_filename = ''
        
        # Classify
        self.clf_type = None
        self.class_type = None
        self.mc_scheme = None
        #self.filename = None
        self.result_path = ''
        self.save_clf_results = True
        self.save_best_model = False
        self.log_file = False
        self.kernel_type = None
        self.rect_kernel = 1.0
        self.test_method_tuple = None
        self.step_size = 1.0
        # Lower and upper bounds of hyper-parameters
        self.C1_range = None
        self.C2_range = None
        self.u_range = None
        # Whether all the input varabiles are inserted or not.
        self.input_complete = False
        
        # Visualization
        self.C1 = 1.0
        self.C2 = 1.0
        self.u = 1.0
        self.linear_db = False # Linear decision boundary
        self.fig_save = False
        self.fig_dpi = None
        self.fig_save_path = None
        
        # Model
        self.pre_trained_model = None
        self.save_pred = False
        self.save_pred_path = ''
        
    def _get_kernel_selection(self):
        """
        It returns the name of the user's selected kernel function.
        
        Returns
        -------
        str
            Name of kernel function
        """
        
        if self.kernel_type == 'linear':
            
            return 'Linear'
        
        elif self.rect_kernel == 1.0:
            
            return 'Gaussian (RBF)'
        
        else:
                
            return 'Rectangular (%s%% of samples)' % (self.rect_kernel * 100)
        
    def _get_eval_method(self):
        """
        It returns the name of the user's selected evaluation method.
        
         Returns
        -------
        str
            Name of evaluation method.
        """
        
        if self.test_method_tuple[0] == 'CV':
            
            return "%d-Fold cross-validation" % self.test_method_tuple[1]
        
        elif self.test_method_tuple[0] == 't_t_split':
            
            return "Train/Test split (%d%%/%d%%)" % (100-(self.test_method_tuple[1]*100),
                                     self.test_method_tuple[1]*100)
            
    def _get_mc_scheme(self):
        """
        It returns type of multi-class classifcation
        
        Returns
        -------
        str
            Name of mult-class strategy.
        """
        
        if self.class_type == 'binary':
            
            return "Binary"
        
        elif self.class_type == 'multiclass':
          
            if self.mc_scheme == 'ova':
                
                return "One-vs-All"
            
            elif self.mc_scheme == 'ovo':
                
                return "One-vs-One"
            
    def _get_clf_name(self):
        """
        It returns the name of the user's selected classifier.
        """
        
        if self.clf_type == 'tsvm':
            
            clf = 'TSVM'
            
        elif self.clf_type == 'lstsvm':
            
            clf = 'LSTSVM'
            
        if self.class_type == 'binary':
            
            return clf
        
        elif self.class_type == 'multiclass':
          
            if self.mc_scheme == 'ova':
                
                return "OVA-" + clf
            
            elif self.mc_scheme == 'ovo':
                
                return "OVO-" + clf
            
    def get_current_selection(self):
        """
        It returns a user's current selection for confirmation
        """
        
        if self.input_complete:
            
            u_param = " | u: 2^%d to 2^%d" % (self.u_range[0], self.u_range[-1]) \
            if self.kernel_type == 'RBF' else ''
            clf = "Standard TwinSVM" if self.clf_type == 'tsvm' else "LeastSquares TwinSVM"
            
            return ("Dataset: %s\nClassifier: %s\nKernel: %s\n"
            "Multi-class scheme: %s\nEvaluation method: %s\n"
            "Range of parameters for grid search: (step:%.2f)\nC1: 2^%d to 2^%d |"
            "C2: 2^%d to 2^%d%s\n"
            "Results' path:%s\nLog File: %s"
            ) % (self.data_filename, clf, self._get_kernel_selection(),
                self._get_mc_scheme(), self._get_eval_method(), self.step_size,
                self.C1_range[0], self.C1_range[1], self.C2_range[0],
                self.C2_range[1], u_param, self.result_path, 'Yes' \
                if self.log_file else 'No')
            
        else:
            
            raise RuntimeError("input_complete has not been set yet! "
                               "Check out UserInput Class Docs.")
            
    def get_selected_clf(self):
        """
        It returns the classifier that is selected by user.
        
        Returns
        -------
        clf_obj : object
            An estimator object.
        
        .. warning::
            
        """
        
        clf_obj = None
    
        if self.clf_type == 'tsvm':
            
            clf_obj = TSVM(self.kernel_type, self.rect_kernel)
            
        elif self.clf_type == 'lstsvm':
            
            clf_obj = LSTSVM(self.kernel_type, self.rect_kernel)
            
        if self.class_type == 'multiclass':
            
            if self.mc_scheme == 'ova':
                
                clf_obj = OneVsAllClassifier(clf_obj)
                
            elif self.mc_scheme == 'ovo':
                
                clf_obj = OneVsOneClassifier(clf_obj)
                
        return clf_obj
    
    def get_clf_params(self):
        """
        It returns hyper-parameters of the classifier in a dictionary.
        
        Returns
        -------
        dict
            Hyper-parameters of the classifier.   
        """
        
        if self.kernel_type == 'linear':
        
            return {'C1': self.C1, 'C2': self.C2}
        
        elif self.kernel_type == 'RBF':
            
            return {'C1': self.C1, 'C2': self.C2, 'gamma': self.u}
        
    def get_fig_name(self):
        """
        Returns the figure's name based on the user's selection for saving a file.
        """
        
        return "Plot_%s_%s_%s_%s" % (self._get_clf_name(), self.kernel_type,
                                        self.data_filename,
                                        datetime.now().strftime('%Y-%m-%d %H-%M'))
        
        
        