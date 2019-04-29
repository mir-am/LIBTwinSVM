# -*- coding: utf-8 -*-

# Developers: Mir, A. and Mahdi Rahbar
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0

"""
In this module, multi-class schemes such as One-vs-One and One-vs-All are
implemented.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from sklearn.utils import column_or_1d
from sklearn.base import clone
from libtsvm import get_current_device, __GPU_enabled
import numpy as np

if __GPU_enabled:
    
    from libtsvm import cp


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
    clf_name : str
        Name of the classifier.

    bin_clf_ : list
        Stores intances of each binary :class:`TSVM` classifier.
    """

    def __init__(self, estimator):

        self.estimator = estimator
        self.clf_name = 'OVO-' + estimator.clf_name
        
        if get_current_device() == 'CPU':
            
            self.fit_method = self.__fit_CPU
            self.pred_method = self.__predict_CPU
            
        elif get_current_device() == 'GPU':
            
            print("Selected the GPU.")
            
            self.fit_method = self.__fit_GPU
            self.pred_method = self.__predict_GPU

    def _validate_targets(self, y):
        """
        Validates labels for training and testing classifier
        """
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        self.classes_, y = np.unique(y_, return_inverse=True)
        
        if len(self.classes_) < 2:
            
            raise ValueError(
                "The number of classes has to be greater than one; got %d"
                " class" % len(self.classes_))

        return np.asarray(y, dtype=np.int)

    def _validate_for_predict(self, X):
        """
        Checks that the classifier is already trained and also test samples are
        valid
        """

        check_is_fitted(self, ['bin_clf_'])
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
        
        X, y = check_X_y(X, y, dtype=np.float64)
        y = self._validate_targets(y)
        
        return self.fit_method(X, y)
        
    def __fit_CPU(self, X, y):
        """
        It fits the OVO-classfier model on the CPU.

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
        
        print("%s - Fits on the CPU." % self.clf_name) 

        # Allocate n(n-1)/2 binary classifiers
        self.bin_clf_ = [clone(self.estimator) for i in range(((self.classes_.size * \
                        (self.classes_.size - 1)) // 2))]

        p = 0

        for i in range(self.classes_.size):

            for j in range(i + 1, self.classes_.size):

                #print("%d, %d" % (i, j))

                # Break multi-class problem into a binary problem
                sub_prob_X_i_j = X[(y == i) | (y == j)]
                sub_prob_y_i_j = y[(y == i) | (y == j)]

                # print(sub_prob_y_i_j)

                # For binary classification, labels must be {-1, +1}
                # i-th class -> +1 and j-th class -> -1
                sub_prob_y_i_j[sub_prob_y_i_j == j] = -1
                sub_prob_y_i_j[sub_prob_y_i_j == i] = 1

                self.bin_clf_[p].fit(sub_prob_X_i_j, sub_prob_y_i_j)

                p = p + 1

        self.shape_fit_ = X.shape

        return self
    
    def __fit_GPU(self, X, y):
        """
        It fits the OVO-classfier model on the GPU.

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
        
        print("%s - Fits on the GPU." % self.clf_name) 

        # Move dataset to the GPU device
        X = cp.asarray(X)
        y = cp.asarray(y)
        
        # Allocate n(n-1)/2 binary classifiers
        self.bin_clf_ = [clone(self.estimator) for i in range(((self.classes_.size * \
                        (self.classes_.size - 1)) // 2))]

        p = 0

        for i in range(self.classes_.size):

            for j in range(i + 1, self.classes_.size):

                #print("%d, %d" % (i, j))

                # Break multi-class problem into a binary problem
                sub_prob_X_i_j = X[(y == i) | (y == j)]
                sub_prob_y_i_j = y[(y == i) | (y == j)]

                # print(sub_prob_y_i_j)

                # For binary classification, labels must be {-1, +1}
                # i-th class -> +1 and j-th class -> -1
                sub_prob_y_i_j[sub_prob_y_i_j == j] = -1
                sub_prob_y_i_j[sub_prob_y_i_j == i] = 1

                self.bin_clf_[p].fit(sub_prob_X_i_j, sub_prob_y_i_j)

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
        
        return self.pred_method(X)

    def __predict_CPU(self, X):
        """
        Performs classification on samples in X using the CPU

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature vectors of test data.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted class lables of test data.
        """
        
        print("%s - Predict on the CPU." % self.clf_name)

        # Initialze votes
        votes = np.zeros((X.shape[0], self.classes_.size), dtype=np.int)

        # iterate over test samples
        for k in range(X.shape[0]):

            p = 0

            for i in range(self.classes_.size):

                for j in range(i + 1, self.classes_.size):

                    y_pred = self.bin_clf_[p].predict(X[k, :].reshape(1, X.shape[1]))

                    if y_pred == 1:

                        votes[k, i] = votes[k, i] + 1

                    else:

                        votes[k, j] = votes[k, j] + 1

                    p = p + 1

         # Labels of test samples based max-win strategy
        max_votes = np.argmax(votes, axis=1)

        return self.classes_.take(np.asarray(max_votes, dtype=np.int))
    
    def __predict_GPU(self, X):
        """
        Performs classification on samples in X using the GPU

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature vectors of test data.

        Returns
        -------
        y_pred : array, shape (n_samples,)
            Predicted class lables of test data.
        """
        
        print("%s - Predict on the GPU." % self.clf_name)
        
        X = cp.asarray(X)
        
        votes = cp.zeros((X.shape[0], self.classes_.size), dtype=cp.int16)
        
        for k in range(X.shape[0]):

            p = 0

            for i in range(self.classes_.size):

                for j in range(i + 1, self.classes_.size):

                    y_pred = self.bin_clf_[p].predict(X[k, :].reshape(1, X.shape[1]))

                    if y_pred == 1:

                        votes[k, i] = votes[k, i] + 1

                    else:

                        votes[k, j] = votes[k, j] + 1

                    p = p + 1

         # Labels of test samples based max-win strategy
        max_votes = cp.argmax(votes, axis=1)
        
        return self.classes_.take(np.asarray(cp.asnumpy(max_votes),
                                             dtype=np.int))


class OneVsAllClassifier(BaseEstimator, ClassifierMixin):
    
    """
    Multi-class classification using One-vs-One scheme
    
    Parameters
    ----------
    estimator : estimator object
        An estimator object implementing `fit` and `predict`.
        
    Attributes
    ----------
    clf_name : str
        Name of the classifier.
        
    bin_clf_ : list
        Stores intances of each binary :class:`TSVM` classifier.
    """
    
    def __init__(self, estimator):
        
        self.estimator = estimator
        self.clf_name = 'OVA-' + estimator.clf_name
        
    def _validate_targets(self, y):
        """
        Validates labels for training and testing classifier
        """
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        self.classes_, y = np.unique(y_, return_inverse=True)
        
        if len(self.classes_) < 2:
            
            raise ValueError(
                "The number of classes has to be greater than one; got %d"
                " class" % len(self.classes_))

        return np.asarray(y, dtype=np.int)
    
    def _validate_for_predict(self, X):
        """
        Checks that the classifier is already trained and also test samples are
        valid
        """

        check_is_fitted(self, ['bin_clf_'])
        X = check_array(X, dtype=np.float64)

        n_samples, n_features = X.shape

        if n_features != self.shape_fit_[1]:

            raise ValueError("X.shape[1] = %d should be equal to %d,"
                             "the number of features of training samples" %
                             (n_features, self.shape_fit_[1]))

        return X
        
    def fit(self, X, y):
        """
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
        
        X, y = check_X_y(X, y, dtype=np.float64)
        y = self._validate_targets(y)
        
        # Allocate n binary classifiers
        # Note that an estimator should be cloned for training a multi-class method
        self.bin_clf_ =  [clone(self.estimator) for i in range(self.classes_.size)]
        
        for i in range(self.classes_.size):
            
            # labels of samples of i-th class and other classes
            mat_y_i = y[(y == i) | (y != i)]
                    
            # For binary classification, labels must be {-1, +1}
            # i-th class -> +1 and other class -> -1
            mat_y_i[y == i] = 1
            mat_y_i[y != i] = -1
            
            self.bin_clf_[i].fit(X, mat_y_i)
            
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
        test_labels : array, shape (n_samples,)
            Predicted class lables of test data.
        """
        
        X = self._validate_for_predict(X)
        
        pred = np.zeros((X.shape[0], self.classes_.size), dtype=np.float64)
        
        for i in range(X.shape[0]):
            
            for j in range(self.classes_.size):
                
                pred[i, j] = self.bin_clf_[j].decision_function(X[i, :].reshape(1,
                    X.shape[1]))[0, 1]
                #pred[i, j] = self.bin_clf_[j].predict(X[i, :].reshape(1, X.shape[1]))
        
        test_lables = np.argmin(pred, axis=1)  
        return self.classes_.take(np.asarray(test_lables, dtype=np.int))
