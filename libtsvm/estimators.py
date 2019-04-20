# -*- coding: utf-8 -*-

# Developers: Mir, A. and Mahdi Rahbar
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0

"""
In this module, Standard TwinSVM and Least Squares TwinSVM estimators are defined.
"""

from sklearn.base import BaseEstimator
from libtsvm import get_current_device
from libtsvm.optimizer import clipdcd
from cupy import prof
import cupy as cp
import numpy as np


class BaseTSVM(BaseEstimator):
    """
    Base class for TSVM-based estimators
    
    Parameters
    ----------
    kernel : str 
        Type of the kernel function which is either 'linear' or 'RBF'.

    rect_kernel : float
        Percentage of training samples for Rectangular kernel.

    C1 : float
        Penalty parameter of first optimization problem.

    C2 : float
        Penalty parameter of second optimization problem.

    gamma : float
        Parameter of the RBF kernel function.

    Attributes
    ----------
    mat_C_t : array-like, shape = [n_samples, n_samples]
        A matrix that contains kernel values.

    cls_name : str
        Name of the classifier.

    w1 : array-like, shape=[n_features]
        Weight vector of class +1's hyperplane.

    b1 : float
        Bias of class +1's hyperplane.

    w2 : array-like, shape=[n_features]
        Weight vector of class -1's hyperplane.

    b2 : float
        Bias of class -1's hyperplane.
    """
    
    def __init__(self, kernel, rect_kernel, C1, C2, gamma):
        
        self.C1 = C1
        self.C2 = C2
        self.gamma = gamma
        self.kernel = kernel
        self.rect_kernel = rect_kernel
        self.mat_C_t = None
        self.clf_name = None
        
        # Two hyperplanes attributes
        self.w1, self.b1, self.w2, self.b2 = None, None, None, None
        
        if get_current_device() == 'CPU':
            
            self.pred_method = self.__predict__CPU
            self.df_method = self.__decision_function_CPU
            
        elif get_current_device() == 'GPU':
            
            self.pred_method = self.__predict_GPU
            self.df_method = self.__decision_function_GPU
        
    def get_params_names(self):
        """
        For retrieving the names of hyper-parameters of the TSVM-based estimator.

        Returns
        -------
        parameters : list of str, {['C1', 'C2'], ['C1', 'C2', 'gamma']}
            Returns the names of the hyperparameters which are same as
            the class' attributes.
        """

        return ['C1', 'C2'] if self.kernel == 'linear' else ['C1', 'C2',
               'gamma']
        
    def fit(self, X, y):
        """
        It fits a TSVM-based estimator. 
        THIS METHOD SHOULD BE IMPLEMENTED IN CHILD CLASS.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape(n_samples,)
            Target values or class labels.
        """
        
        pass # Impelement fit method in child class
        
    def predict(self, X):
        """
        Performs classification on samples in X using the TSVM-based model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature vectors of test data.

        Returns
        -------
        array, shape (n_samples,)
            Predicted class lables of test data.
        """
        
        return self.pred_method(X)
    
    def decision_function(self, X):
        """
        Computes distance of test samples from both non-parallel hyperplanes
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        
        Returns
        -------
        : array, shape(n_samples, 2)
            distance from both hyperplanes.
        """
        
        return self.df_method(X)
    
    def __predict__CPU(self, X):
        """
        Performs prediction on the CPU.
        """
        
        print("Performs prediction on the CPU.")
        
        # Assign data points to class +1 or -1 based on distance from hyperplanes
        return 2 * np.argmin(self.__decision_function_CPU(X), axis=1) - 1
    
    def __predict_GPU(self, X):
        """
        Performs prediction on the GPU.
        """
        
        print("Performs prediction on the GPU.")
        
        return cp.asnumpy(2 * cp.argmin(self.__decision_function_GPU(X),
                                        axis=1) - 1)
    
    def __decision_function_CPU(self, X):
        """
        Computes distances on the CPU.
        """
        
        print("Computes distances on the CPU.")
        
        #dist = np.zeros((X.shape[0], 2), dtype=np.float64)
        
        kernel_f = {'linear': lambda : X,
                    'RBF': lambda : rbf_kernel(X, self.mat_C_t, self.gamma)}
        
#        # TODO: prediction can be sped up using NumPy's np.apply?!. It removes
#        # below for loop.
#        for i in range(X.shape[0]):
#
#            # Prependicular distance of data pint i from hyperplanes
#            dist[i, 1] = np.abs(np.dot(kernel_f[self.kernel](i), self.w1) \
#                + self.b1)
#
#            dist[i, 0] = np.abs(np.dot(kernel_f[self.kernel](i), self.w2) \
#                + self.b2)
            
        return np.column_stack((np.abs(np.dot(kernel_f[self.kernel](), \
                                self.w2) + self.b2), np.abs(np.dot(kernel_f[self.kernel](), \
                                                           self.w1) + self.b1)))
    
    def __decision_function_GPU(self, X):
        """
        Computes distances on the GPU.
        """
        
        print("Computes distances on the GPU.")
        
        X = cp.asarray(X)
        
        kernel_f = {'linear': lambda : X,
                    'RBF': lambda : GPU_rbf_kernel(X, self.mat_C_t, self.gamma)}
    
        return cp.column_stack((cp.abs(cp.dot(kernel_f[self.kernel](), \
                                self.w2) + self.b2), cp.abs(cp.dot(kernel_f[self.kernel](), \
                                                           self.w1) + self.b1)))
        
class TSVM(BaseTSVM):

    """
    Standard Twin Support Vector Machine for binary classification.
    It inherits attributes of :class:`BaseTSVM`.

    Parameters
    ----------
    kernel : str, optional (default='linear')
        Type of the kernel function which is either 'linear' or 'RBF'.

    rect_kernel : float, optional (default=1.0)
        Percentage of training samples for Rectangular kernel.

    C1 : float, optional (default=1.0)
        Penalty parameter of first optimization problem.

    C2 : float, optional (default=1.0)
        Penalty parameter of second optimization problem.

    gamma : float, optional (default=1.0)
        Parameter of the RBF kernel function.
    """

    def __init__(self, kernel='linear', rect_kernel=1, C1=2**0, C2=2**0,
                 gamma=2**0):

        super(TSVM, self).__init__(kernel, rect_kernel, C1, C2, gamma)

        self.clf_name = 'TSVM'
        
        # Regulariztion term used for ill-possible condition
        self.__reg_term = 2 ** float(-7)
        
        if get_current_device() == 'CPU':
            
            self.fit_method = self.__fit_CPU
            
        elif get_current_device() == 'GPU':
            
            print("Selected the GPU.")
            
            self.fit_method = self.__fit_GPU
            
    def fit(self, X, y):
        """
        It fits the binary TwinSVM model according to the given training data.
        
        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
           Training feature vectors, where n_samples is the number of samples
           and n_features is the number of features.

        y_train : array-like, shape(n_samples,)
            Target values or class labels.
        """
        
        return self.fit_method(X, y)
        

    def __fit_CPU(self, X_train, y_train):
        """
        It fits the binary TwinSVM model using CPU
        
        Parameters
        ----------
        X_train : array-like, shape (n_samples, n_features)
           Training feature vectors, where n_samples is the number of samples
           and n_features is the number of features.

        y_train : array-like, shape(n_samples,)
            Target values or class labels.
        """
        
        print("Fits on the CPU.")

        # Matrix A or class 1 samples
        mat_A = X_train[y_train == 1]

        # Matrix B  or class -1 data
        mat_B = X_train[y_train == -1]

        # Vectors of ones
        mat_e1 = np.ones((mat_A.shape[0], 1))
        mat_e2 = np.ones((mat_B.shape[0], 1))

        if self.kernel == 'linear':  # Linear kernel

            mat_H = np.column_stack((mat_A, mat_e1))
            mat_G = np.column_stack((mat_B, mat_e2))

        elif self.kernel == 'RBF':  # Non-linear

            # class 1 & class -1
            mat_C = np.row_stack((mat_A, mat_B))

            self.mat_C_t = np.transpose(mat_C)[:, :int(mat_C.shape[0] * self.rect_kernel)]

            mat_H = np.column_stack((rbf_kernel(mat_A, self.mat_C_t, self.gamma), mat_e1))
            mat_G = np.column_stack((rbf_kernel(mat_B, self.mat_C_t, self.gamma), mat_e2))

        mat_H_t = np.transpose(mat_H)
        mat_G_t = np.transpose(mat_G)

        # Compute inverses:
        # Regulariztion term used for ill-possible condition
        reg_term = 2 ** float(-7)

        mat_H_H = np.linalg.inv(np.dot(mat_H_t, mat_H) + (reg_term * np.identity(mat_H.shape[1])))
        mat_G_G = np.linalg.inv(np.dot(mat_G_t, mat_G) + (reg_term * np.identity(mat_G.shape[1])))
        
        # Wolfe dual problem of class 1
        mat_dual1 = np.dot(np.dot(mat_G, mat_H_H), mat_G_t)
        # Wolfe dual problem of class -1
        mat_dual2 = np.dot(np.dot(mat_H, mat_G_G), mat_H_t)

        # Obtaining Lagrange multipliers using ClipDCD optimizer
        alpha_d1 = np.array(clipdcd.optimize(mat_dual1, self.C1)).reshape(mat_dual1.shape[0], 1)
        alpha_d2 = np.array(clipdcd.optimize(mat_dual2, self.C2)).reshape(mat_dual2.shape[0], 1)

        # Obtain hyperplanes
        hyper_p_1 = -1 * np.dot(np.dot(mat_H_H, mat_G_t), alpha_d1)

        # Class 1
        self.w1 = hyper_p_1[:hyper_p_1.shape[0] - 1, :]
        self.b1 = hyper_p_1[-1, :]

        hyper_p_2 = np.dot(np.dot(mat_G_G, mat_H_t), alpha_d2)

        # Class -1
        self.w2 = hyper_p_2[:hyper_p_2.shape[0] - 1, :]
        self.b2 = hyper_p_2[-1, :]
        
    def __fit_GPU(self, X, y):
        """
        It fits the binary TwinSVM model using GPU

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape(n_samples,)
            Target values or class labels.
        """

        print("Fits on the GPU.") 

        X = cp.asarray(X)
        y = cp.asarray(y)
        
        # Matrix A or class 1 data
        mat_A = X[y == 1]
        
        # Matrix B or class -1 data
        mat_B = X[y == -1]
        
        mat_e1 = cp.ones((mat_A.shape[0], 1))
        mat_e2 = cp.ones((mat_B.shape[0], 1))
        
        if self.kernel == 'linear':  # Linear kernel

            mat_H = cp.column_stack((mat_A, mat_e1))
            mat_G = cp.column_stack((mat_B, mat_e2))

        elif self.kernel == 'RBF':  # Non-linear

            # class 1 & class -1
            mat_C = cp.vstack((mat_A, mat_B))

            self.mat_C_t = cp.transpose(mat_C)[:, :int(mat_C.shape[0] * self.rect_kernel)]

            mat_H = cp.column_stack((GPU_rbf_kernel(mat_A, self.mat_C_t, self.gamma), mat_e1))
            mat_G = cp.column_stack((GPU_rbf_kernel(mat_B, self.mat_C_t, self.gamma), mat_e2))
            
        mat_H_t = cp.transpose(mat_H)
        mat_G_t = cp.transpose(mat_G)

        # Compute inverses: 
        mat_H_H = cp.linalg.inv(cp.dot(mat_H_t, mat_H) + (self.__reg_term * cp.identity(mat_H.shape[1])))
        mat_G_G = cp.linalg.inv(cp.dot(mat_G_t, mat_G) + (self.__reg_term * cp.identity(mat_G.shape[1])))

        # Wolfe dual problem of class 1
        mat_dual1 = cp.dot(cp.dot(mat_G, mat_H_H), mat_G_t)

        # Wolfe dual problem of class -1
        mat_dual2 = cp.dot(cp.dot(mat_H, mat_G_G), mat_H_t)

        # Obtaining Lagrange multipliers using ClipDCD optimizer
        alpha_d1 = cp.array(clipdcd.optimize(cp.asnumpy(mat_dual1), \
                                             self.C1)).reshape(mat_dual1.shape[0], 1)
        alpha_d2 = cp.array(clipdcd.optimize(cp.asnumpy(mat_dual2), \
                                             self.C2)).reshape(mat_dual2.shape[0], 1)

        # Obtain hyperplanes
        hyper_p_1 = -1 * cp.dot(cp.dot(mat_H_H, mat_G_t), alpha_d1)

        # Class 1
        self.w1 = hyper_p_1[:hyper_p_1.shape[0] - 1, :]
        self.b1 = hyper_p_1[-1, :]

        hyper_p_2 = cp.dot(cp.dot(mat_G_G, mat_H_t), alpha_d2)

        # Class -1
        self.w2 = hyper_p_2[:hyper_p_2.shape[0] - 1, :]
        self.b2 = hyper_p_2[-1, :]
        
class LSTSVM(BaseTSVM):
    """
    Least Squares Twin Support Vector Machine (LSTSVM) for binary classification
    It inherits attributes of :class:`BaseTSVM`.
    
    Parameters
    ----------
    kernel : str, optional (default='linear')
    Type of the kernel function which is either 'linear' or 'RBF'.

    rect_kernel : float, optional (default=1.0)
        Percentage of training samples for Rectangular kernel.

    C1 : float, optional (default=1.0)
        Penalty parameter of first optimization problem.

    C2 : float, optional (default=1.0)
        Penalty parameter of second optimization problem.

    gamma : float, optional (default=1.0)
        Parameter of the RBF kernel function.
    """

    def __init__(self, kernel='linear', rect_kernel=1, C1=2**0, C2=2**0,
                 gamma=2**0):

        super(LSTSVM, self).__init__(kernel, rect_kernel, C1, C2, gamma)

        self.clf_name = 'LSTSVM'
        # Regulariztion term used for ill-possible condition
        self.__reg_term = 2 ** float(-7)
        
        if get_current_device() == 'CPU':
            
            self.fit_method = self.__fit_CPU
            
        elif get_current_device() == 'GPU':
            
            print("Selected the GPU.")
            
            self.fit_method = self.__fit_GPU
            
    def fit(self, X, y):
        """
        It fits the binary Least Squares TwinSVM model according to the given
        training data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape(n_samples,)
            Target values or class labels.
        """
        
        return self.fit_method(X, y)

    def __fit_CPU(self, X, y):
        """
        It fits the binary Least Squares TwinSVM model using CPU.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape(n_samples,)
            Target values or class labels.
        """
        
        print("Fits on the CPU.")

        # Matrix A or class 1 data
        mat_A = X[y == 1]

        # Matrix B or class -1 data
        mat_B = X[y == -1]

        # Vectors of ones
        mat_e1 = np.ones((mat_A.shape[0], 1))
        mat_e2 = np.ones((mat_B.shape[0], 1))

        if self.kernel == 'linear':

            mat_H = np.column_stack((mat_A, mat_e1))
            mat_G = np.column_stack((mat_B, mat_e2))

            mat_H_t = np.transpose(mat_H)
            mat_G_t = np.transpose(mat_G)

            # Determine parameters of two non-parallel hyperplanes
            hyper_p_1 = -1 * np.dot(np.linalg.inv(np.dot(mat_G_t, mat_G) +
                                                  (1 / self.C1) * np.dot(mat_H_t, mat_H)), np.dot(mat_G_t,
                                                                                                  mat_e2))

            self.w1 = hyper_p_1[:hyper_p_1.shape[0] - 1, :]
            self.b1 = hyper_p_1[-1, :]

            hyper_p_2 = np.dot(np.linalg.inv(np.dot(mat_H_t, mat_H) + (1 / self.C2)
                                             * np.dot(mat_G_t, mat_G)), np.dot(mat_H_t, mat_e1))

            self.w2 = hyper_p_2[:hyper_p_2.shape[0] - 1, :]
            self.b2 = hyper_p_2[-1, :]

        elif self.kernel == 'RBF':

            # class 1 & class -1
            mat_C = np.row_stack((mat_A, mat_B))

            self.mat_C_t = np.transpose(mat_C)[:, :int(mat_C.shape[0] * self.rect_kernel)]

            mat_H = np.column_stack((rbf_kernel(mat_A, self.mat_C_t, self.gamma),
                                     mat_e1))

            mat_G = np.column_stack((rbf_kernel(mat_B, self.mat_C_t, self.gamma),
                                     mat_e2))

            mat_H_t = np.transpose(mat_H)
            mat_G_t = np.transpose(mat_G)
            
            mat_I_H = np.identity(mat_H.shape[0]) # (m_1 x m_1)
            mat_I_G = np.identity(mat_G.shape[0]) # (m_2 x m_2)
            
            mat_I = np.identity(mat_G.shape[1]) # (n x n)

            if mat_A.shape[0] < mat_B.shape[0]:
                
                y = (1 / self.__reg_term) * (mat_I - np.dot(np.dot(mat_G_t, \
                    np.linalg.inv((self.__reg_term * mat_I_G) + np.dot(mat_G, mat_G_t))), mat_G))
                
                mat_H_y = np.dot(mat_H, y)
                mat_y_Ht = np.dot(y, mat_H_t)
                mat_H_y_Ht = np.dot(mat_H_y, mat_H_t)
                
                h_surf1_inv = np.linalg.inv(self.C1 * mat_I_H + mat_H_y_Ht)
                h_surf2_inv = np.linalg.inv((mat_I_H / self.C2) + mat_H_y_Ht)

                hyper_surf1 = np.dot(-1 * (y - np.dot(np.dot(mat_y_Ht, h_surf1_inv), mat_H_y)), \
                                     np.dot(mat_G_t, mat_e2))

                hyper_surf2 = np.dot(self.C2 * (y - np.dot(np.dot(mat_y_Ht, h_surf2_inv), mat_H_y)), \
                                     np.dot(mat_H_t, mat_e1))

                # Parameters of hypersurfaces
                self.w1 = hyper_surf1[:hyper_surf1.shape[0] - 1, :]
                self.b1 = hyper_surf1[-1, :]

                self.w2 = hyper_surf2[:hyper_surf2.shape[0] - 1, :]
                self.b2 = hyper_surf2[-1, :]

            else:

                z = (1 / self.__reg_term) * (mat_I - np.dot(np.dot(mat_H_t, \
                    np.linalg.inv(self.__reg_term * mat_I_H + np.dot(mat_H, mat_H_t))), mat_H))
                    
                mat_G_z = np.dot(mat_G, z)
                mat_z_Gt = np.dot(z, mat_G_t)
                mat_G_y_Gt = np.dot(mat_G_z, mat_G_t)
                
                g_surf1_inv = np.linalg.inv((mat_I_G / self.C1) + mat_G_y_Gt)
                g_surf2_inv = np.linalg.inv(self.C2 * mat_I_G + mat_G_y_Gt)

                hyper_surf1 = np.dot(-self.C1 * (z - np.dot(np.dot(mat_z_Gt, g_surf1_inv), mat_G_z)), \
                                     np.dot(mat_G_t, mat_e2))

                hyper_surf2 = np.dot((z - np.dot(np.dot(mat_z_Gt, g_surf2_inv), mat_G_z)), \
                                     np.dot(mat_H_t, mat_e1))

                self.w1 = hyper_surf1[:hyper_surf1.shape[0] - 1, :]
                self.b1 = hyper_surf1[-1, :]

                self.w2 = hyper_surf2[:hyper_surf2.shape[0] - 1, :]
                self.b2 = hyper_surf2[-1, :]


    def __fit_GPU(self, X, y):
        """
        It fits the binary Least Squares TwinSVM model using GPU

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training feature vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape(n_samples,)
            Target values or class labels.
        """

        print("Fits on the GPU.")        
        
        X = cp.asarray(X)
        y = cp.asarray(y)
        
        # Matrix A or class 1 data
        mat_A = X[y == 1]
        
        # Matrix B or class -1 data
        mat_B = X[y == -1]
        
        # Vectors of ones
        mat_e1 = cp.ones((mat_A.shape[0], 1))
        mat_e2 = cp.ones((mat_B.shape[0], 1))
        
        if self.kernel == 'linear':
        
            mat_H = cp.column_stack((mat_A, mat_e1))
            mat_G = cp.column_stack((mat_B, mat_e2))
    
            mat_H_t = cp.transpose(mat_H)
            mat_G_t = cp.transpose(mat_G)
            
            # Determine parameters of two non-parallel hyperplanes
            hyper_p_1 = -1 * cp.dot(cp.linalg.inv(cp.dot(mat_G_t, mat_G) + \
                                   (1 / self.C1) * cp.dot(mat_H_t, mat_H)), \
                                    cp.dot(mat_G_t, mat_e2))
            
            self.w1 = hyper_p_1[:hyper_p_1.shape[0] - 1, :]
            self.b1 = hyper_p_1[-1, :]
            
            hyper_p_2 = cp.dot(cp.linalg.inv(cp.dot(mat_H_t, mat_H) + (1 / self.C2)
                                                 * cp.dot(mat_G_t, mat_G)), cp.dot(mat_H_t, mat_e1))
    
            self.w2 = hyper_p_2[:hyper_p_2.shape[0] - 1, :]
            self.b2 = hyper_p_2[-1, :]
            
        elif self.kernel == 'RBF':
            
            mat_C = cp.vstack((mat_A, mat_B))
            
            self.mat_C_t = cp.transpose(mat_C)[:, :int(mat_C.shape[0] * self.rect_kernel)]
            
            mat_H = cp.column_stack((GPU_rbf_kernel(mat_A, self.mat_C_t,
                                                    self.gamma), mat_e1))
            
            mat_G = cp.column_stack((GPU_rbf_kernel(mat_B, self.mat_C_t,
                                                    self.gamma), mat_e2))
            
            mat_H_t = cp.transpose(mat_H)
            mat_G_t = cp.transpose(mat_G)
            
            mat_I_H = cp.identity(mat_H.shape[0])
            mat_I_G = cp.identity(mat_G.shape[0])
            
            mat_I = cp.identity(mat_G.shape[1])
            
            if mat_A.shape[0] < mat_B.shape[0]:
                
                y = (1 / self.__reg_term) * (mat_I - cp.dot(cp.dot(mat_G_t, \
                    cp.linalg.inv((self.__reg_term * mat_I_G) + cp.dot(mat_G, mat_G_t))), mat_G))
                
                mat_H_y = cp.dot(mat_H, y)
                mat_y_Ht = cp.dot(y, mat_H_t)
                mat_H_y_Ht = cp.dot(mat_H_y, mat_H_t)
                
                h_surf1_inv = cp.linalg.inv(self.C1 * mat_I_H + mat_H_y_Ht)
                h_surf2_inv = cp.linalg.inv((mat_I_H / self.C2) + mat_H_y_Ht)
                
                hyper_surf1 = cp.dot(-1 * (y - cp.dot(cp.dot(mat_y_Ht, h_surf1_inv), mat_H_y)), \
                                     cp.dot(mat_G_t, mat_e2))

                hyper_surf2 = cp.dot(self.C2 * (y - cp.dot(cp.dot(mat_y_Ht, h_surf2_inv), mat_H_y)), \
                                     cp.dot(mat_H_t, mat_e1))
                
            else:
                
                z = (1 / self.__reg_term) * (mat_I - cp.dot(cp.dot(mat_H_t, \
                    cp.linalg.inv(self.__reg_term * mat_I_H + cp.dot(mat_H, mat_H_t))), mat_H))
                
                mat_G_z = cp.dot(mat_G, z)
                mat_z_Gt = cp.dot(z, mat_G_t)
                mat_G_y_Gt = cp.dot(mat_G_z, mat_G_t)
                
                g_surf1_inv = cp.linalg.inv((mat_I_G / self.C1) + mat_G_y_Gt)
                g_surf2_inv = cp.linalg.inv(self.C2 * mat_I_G + mat_G_y_Gt)
                
                hyper_surf1 = cp.dot(-self.C1 * (z - cp.dot(cp.dot(mat_z_Gt, g_surf1_inv), mat_G_z)), \
                                     cp.dot(mat_G_t, mat_e2))

                hyper_surf2 = cp.dot((z - cp.dot(cp.dot(mat_z_Gt, g_surf2_inv), mat_G_z)), \
                                     cp.dot(mat_H_t, mat_e1))
                
            self.w1 = hyper_surf1[:hyper_surf1.shape[0] - 1, :]
            self.b1 = hyper_surf1[-1, :]

            self.w2 = hyper_surf2[:hyper_surf2.shape[0] - 1, :]
            self.b2 = hyper_surf2[-1, :]
            

def rbf_kernel(x, y, u):
    """
    It transforms samples into higher dimension using Gaussian (RBF) kernel.

    Parameters
    ----------
    x, y : array-like, shape (n_features,)
        A feature vector or sample.

    u : float
        Parameter of the RBF kernel function.

    Returns
    -------
    float
        Value of kernel matrix for feature vector x and y.
    """

    return np.exp(-2 * u) * np.exp(2 * u * np.dot(x, y))

# GPU implementation of estimators ############################################


    
    
def GPU_rbf_kernel(x, y, u):
    """
    It transforms samples into higher dimension using Gaussian (RBF) kernel.

    Parameters
    ----------
    x, y : array-like, shape (n_features,)
        A feature vector or sample.

    u : float
        Parameter of the RBF kernel function.

    Returns
    -------
    float
        Value of kernel matrix for feature vector x and y.
    """

    return cp.exp(-2 * u) * cp.exp(2 * u * cp.dot(x, y))
        
#    class GPU_LSTSVM():
#        """
#        GPU implementation of least squares twin support vector machine
#        
#        Parameters
#        ----------
#        kernel : str 
#            Type of the kernel function which is either 'linear' or 'RBF'.
#    
#        rect_kernel : float
#            Percentage of training samples for Rectangular kernel.
#    
#        C1 : float
#            Penalty parameter of first optimization problem.
#    
#        C2 : float
#            Penalty parameter of second optimization problem.
#    
#        gamma : float
#            Parameter of the RBF kernel function.
#    
#        Attributes
#        ----------
#        mat_C_t : array-like, shape = [n_samples, n_samples]
#            A matrix that contains kernel values.
#    
#        cls_name : str
#            Name of the classifier.
#    
#        w1 : array-like, shape=[n_features]
#            Weight vector of class +1's hyperplane.
#    
#        b1 : float
#            Bias of class +1's hyperplane.
#    
#        w2 : array-like, shape=[n_features]
#            Weight vector of class -1's hyperplane.
#    
#        b2 : float
#            Bias of class -1's hyperplane.
#        """
#        
#        def __init__(self, kernel='linear', rect_kernel=1, C1=2**0, C2=2**0,
#                     gamma=2**0):
#            
#            self.C1 = C1
#            self.C2 = C2
#            self.gamma = gamma
#            self.kernel = kernel
#            self.rect_kernel = rect_kernel
#            self.mat_C_t = None
#            self.clf_name = None
#            
#            # Two hyperplanes attributes
#            self.w1, self.b1, self.w2, self.b2 = None, None, None, None
#        
#        #@prof.TimeRangeDecorator()
#        def fit(self, X, y):
#            """
#            It fits the binary Least Squares TwinSVM model according to the given
#            training data.
#    
#            Parameters
#            ----------
#            X : array-like, shape (n_samples, n_features)
#                Training feature vectors, where n_samples is the number of samples
#                and n_features is the number of features.
#    
#            y : array-like, shape(n_samples,)
#                Target values or class labels.
#            """
#            
#            X = cp.asarray(X)
#            y = cp.asarray(y)
#            
#            # Matrix A or class 1 data
#            mat_A = X[y == 1]
#            
#            # Matrix B or class -1 data
#            mat_B = X[y == -1]
#            
#            # Vectors of ones
#            mat_e1 = cp.ones((mat_A.shape[0], 1))
#            mat_e2 = cp.ones((mat_B.shape[0], 1))
#            
#            mat_H = cp.column_stack((mat_A, mat_e1))
#            mat_G = cp.column_stack((mat_B, mat_e2))
#    
#            mat_H_t = cp.transpose(mat_H)
#            mat_G_t = cp.transpose(mat_G)
#            
#            # Determine parameters of two non-parallel hyperplanes
#            hyper_p_1 = -1 * cp.dot(cp.linalg.inv(cp.dot(mat_G_t, mat_G) + \
#                                   (1 / self.C1) * cp.dot(mat_H_t, mat_H)), \
#                                    cp.dot(mat_G_t, mat_e2))
#            
#            self.w1 = hyper_p_1[:hyper_p_1.shape[0] - 1, :]
#            self.b1 = hyper_p_1[-1, :]
#            
#            hyper_p_2 = cp.dot(cp.linalg.inv(cp.dot(mat_H_t, mat_H) + (1 / self.C2)
#                                                 * cp.dot(mat_G_t, mat_G)), cp.dot(mat_H_t, mat_e1))
#    
#            self.w2 = hyper_p_2[:hyper_p_2.shape[0] - 1, :]
#            self.b2 = hyper_p_2[-1, :]
#        
#        #@prof.TimeRangeDecorator()
#        def predict(self, X):
#            """
#            Performs classification on samples in X using the Least Squares 
#            TwinSVM model.
#            
#            Parameters
#            ----------
#            X_test : array-like, shape (n_samples, n_features)
#                Feature vectors of test data.
#                    
#            Returns
#            -------
#            output : array, shape (n_samples,)
#                Predicted class lables of test data.
#                
#            """
#            
#            X = cp.asarray(X)
#    
#            # Calculate prependicular distances for new data points 
#    #        prepen_distance = cp.zeros((X.shape[0], 2))
#    #
#    #        kernel_f = {'linear': lambda i: X[i, :] ,
#    #                    'RBF': lambda i: GPU_rbf_kernel(X[i, :], self.mat_C_t, self.gamma)}
#    #
#    #        for i in range(X.shape[0]):
#    #
#    #            # Prependicular distance of data pint i from hyperplanes
#    #            prepen_distance[i, 1] = cp.abs(cp.dot(kernel_f[self.kernel](i), self.w1) + self.b1)[0]
#    #        
#    #            prepen_distance[i, 0] = cp.abs(cp.dot(kernel_f[self.kernel](i), self.w2) + self.b2)[0]
#            
#            dist = cp.column_stack((cp.abs(cp.dot(X, self.w2) + self.b2), 
#                                    cp.abs(cp.dot(X, self.w1) + self.b1)))
#            
#        
#            
#            # Assign data points to class +1 or -1 based on distance from hyperplanes
#            output = 2 * cp.argmin(dist, axis=1) - 1
#    
#            return cp.asnumpy(output)

#except ImportError:
#            
#    print("Cannot run GPU implementation. Install CuPy package.")
###############################################################################


if __name__ == '__main__':

    from preprocess import read_data
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y, filename = read_data('../dataset/australian.csv')

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    tsvm_model = TSVM('linear', 0.25, 0.5)
    
    tsvm_model.fit(x_train, y_train)
    pred = tsvm_model.predict(x_test)
    
    print("Accuracy: %.2f" % (accuracy_score(y_test, pred) * 100))
