# -*- coding: utf-8 -*-

# Developers: Mir, A.
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0

"""
This module contains code for saving, loading, and evaluating pre-trained
models.
"""

from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal
from sklearn.metrics import accuracy_score
from joblib import dump, load
from libtsvm.estimators import BaseTSVM
from libtsvm.mc_scheme import OneVsAllClassifier, OneVsOneClassifier, mc_clf_no_params
from libtsvm.misc import time_fmt
from datetime import datetime
from os.path import join
from numpy import savetxt


def save_model(validator, params, output_file):
    """
    It saves an estimator with specified hyper-parameters and 
    a evaluation method.
    
    Parameters
    ----------
    validator : object
        An evaluation method.
        
    params : dict
        Hyper-parameters of the estimator.
        
    output_file : str
        The full path and filename of the saved model.
    """
    
    # The evaluation method
    eval_func = validator.choose_validator()
    eval_func(params)
    
    dump(validator.estimator, output_file)
    
    
def load_model(model_path):
    """
    It loads a pre-trained TSVM-based estimator.
    
    Parameters
    ----------
    model_path : str
        The path at which the model is stored.
    
    Returns
    -------
    object
        A pre-trained estimator.
        
    dict
        Model information.
    """
    
    pre_trained_clf = load(model_path)
    
    if isinstance(pre_trained_clf, BaseTSVM):
        
        kernel_name = pre_trained_clf.kernel
        rect_kernel = pre_trained_clf.rect_kernel
        model_no_params = str(pre_trained_clf.w1.shape[0] + \
                              pre_trained_clf.w2.shape[0] + 2)
        model_h_param = pre_trained_clf.get_params()
    
    elif isinstance(pre_trained_clf, OneVsAllClassifier) or \
         isinstance(pre_trained_clf, OneVsOneClassifier):
    
        kernel_name = pre_trained_clf.estimator.kernel
        rect_kernel = pre_trained_clf.estimator.rect_kernel
        model_no_params = str(mc_clf_no_params(pre_trained_clf.bin_clf_))
        model_h_param = pre_trained_clf.estimator.get_params()
        
    else:
    
        raise ValueError("An unsupported estimator is loaded!")             
             
    return pre_trained_clf, {'model_name': pre_trained_clf.clf_name,
                             'kernel': kernel_name, 'rect_kernel': rect_kernel,
                             'no_params': model_no_params, 'h_params': model_h_param}
    
    
class ModelThread(QObject):
    """
    Evaluates a pre-trained model in a thread.
    
    Parameters
    ----------
    usr_input : object
        An instance of :class:`UserInput` class which holds the user input.
    """
    
    sig_update_model_eval = pyqtSignal(str, str)
    
    def __init__(self, usr_in):
    
        super(ModelThread, self).__init__()
        
        self.usr_in = usr_in
        
    @pyqtSlot()
    def eval_model(self):
        """
        It evaluates a pre-trained model on test samples.
        """
        
        start_t = datetime.now()
        
        pred = self.usr_in.pre_trained_model.predict(self.usr_in.X_train)
        
        test_acc = accuracy_score(self.usr_in.y_train, pred) * 100
        
        elapsed_t = datetime.now() - start_t 
        
        self.sig_update_model_eval.emit("%.2f%%" % test_acc,
                                        time_fmt(elapsed_t.seconds))
        
        if self.usr_in.save_pred:
            
            f_name = 'test_labels_model_%s_%s_%s_%s.txt' % (
                    self.usr_in.pre_trained_model.clf_name,
                    self.usr_in.kernel_type, self.usr_in.data_filename,
                    datetime.now().strftime('%Y-%m-%d %H-%M'))
            
            savetxt(join(self.usr_in.save_pred_path, f_name), pred, fmt='%d')
    