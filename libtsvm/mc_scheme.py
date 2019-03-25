# -*- coding: utf-8 -*-

# Developers: Mir, A. and Mahdi Rahbar
# Version: 0.1 - 2019-03-20
# License:
    
"""
In this module, multi-class schemes such as One-vs-One and One-vs-All are
implemented.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils import column_or_1d
import numpy as np


class OneVsOneClassifier(BaseEstimator, ClassifierMixin):

    """
    Multi-class classification using One-vs-One scheme
    
    The :class:`OneVsOneClassifier` is scikit-learn compatible, which means
    scikit-learn tools such as `cross_val_score <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html>`_ 
    and `GridSearchCV <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`_
    can be used for an instance of :class:`OneVsOneClassifier`
    
    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit` and `predict`.
               
    Attributes
    ----------
    cls_name : str
        Name of the classifier.
    
    bin_TSVM_models_ : list
        Stores intances of each binary :class:`TSVM` classifier.
    """    
    
    def __init__(self, estimator):
               
        self.estimator = estimator
        self.cls_name = 'TSVM_OVO'
        
    def _validate_targets(self, y):
        
        """
        Validates labels for training and testing classifier
        """
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        self.classes_, y = np.unique(y_, return_inverse=True)
        
        return np.asarray(y, dtype=np.int)
    
    def _validate_for_predict(self, X):
        
        """
        Checks that the classifier is already trained and also test samples are
        valid
        """
        
        check_is_fitted(self, ['bin_TSVM_models_'])
        X = check_array(X, dtype=np.float64)
        
        n_samples, n_features = X.shape
        
        if n_features != self.shape_fit_[1]:
            
            raise ValueError("X.shape[1] = %d should be equal to %d," 
                             "the number of features of training samples" % 
                             (n_features, self.shape_fit_[1]))
        
        return X
    
    def fit(self, X, y):
        
        """
        It fits the OVO-classfier model according to the given training data.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features) 
            Training feature vectors, where n_samples is the number of samples
            and n_features is the number of features.
           
        y : array-like, shape(n_samples,)
            Target values or class labels.
            
        Returns
        -------
        self : object
        """
        
        y = self._validate_targets(y)
        X, y = check_X_y(X, y, dtype=np.float64)
         
        # Allocate n(n-1)/2 binary TSVM classifiers
        self.bin_TSVM_models_ = ((self.classes_.size * (self.classes_.size - 1))
                               // 2 ) * [None]
        
        p = 0
        
        for i in range(self.classes_.size):
            
            for j in range(i + 1, self.classes_.size):
                
                #print("%d, %d" % (i, j))
                
                # Break multi-class problem into a binary problem
                sub_prob_X_i_j = X[(y == i) | (y == j)]
                sub_prob_y_i_j = y[(y == i) | (y == j)]
                
                #print(sub_prob_y_i_j)
                
                # For binary classification, labels must be {-1, +1}
                # i-th class -> +1 and j-th class -> -1
                sub_prob_y_i_j[sub_prob_y_i_j == j] = -1
                sub_prob_y_i_j[sub_prob_y_i_j == i] = 1
                
                self.bin_TSVM_models_[p] = TSVM(self.kernel, 1, self.C1, self.C2, \
                               self.gamma)
                
                self.bin_TSVM_models_[p].fit(sub_prob_X_i_j, sub_prob_y_i_j)
                
                p = p + 1
                
        self.shape_fit_ = X.shape
                
        return self
         
    def predict(self, X):
        
        """
        Performs classification on samples in X using the OVO-classifier model.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature vectors of test data.
        
        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted class lables of test data.
        """
        
        X = self._validate_for_predict(X)
        
        # Initialze votes
        votes = np.zeros((X.shape[0], self.classes_.size), dtype=np.int)
        
        # iterate over test samples
        for k in range(X.shape[0]):
            
            p = 0
        
            for i in range(self.classes_.size):
                
                for j in range(i + 1, self.classes_.size):
                    
                    y_pred = self.bin_TSVM_models_[p].predict(X[k, :].reshape(1, X.shape[1]))
                    
                    if y_pred == 1:
                        
                        votes[k, i] = votes[k, i] + 1
                        
                    else:
                        
                        votes[k, j] = votes[k, j] + 1
                        
                    p = p + 1
                        
        
         # Labels of test samples based max-win strategy
        max_votes = np.argmax(votes, axis=1)
            
        return self.classes_.take(np.asarray(max_votes, dtype=np.int))
