# -*- coding: utf-8 -*-

# Developers: Mir, A. and Mahdi Rahbar
# Version: 0.1 - 2019-03-20
# License:

"""
In this module, Standard TwinSVM and Least Squares TwinSVM estimators are defined.
"""

import numpy as np

# TwinSVM code




# End of TwinSVM code

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

# Least Squares TwinSVM
class LSTSVM:

	"""
	Least Squares Twin Support Vector Machine (LSTSVM) for binary classification
	
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

	def __init__(self, kernel='linear', rect_kernel=1, C1=2**0, C2=2**0,
                 gamma=2**0):
        
         self.kernel = kernel
         self.rect_kernel = rect_kernel
         self.C1 = C1
         self.C2 = C2
         self.gamma = gamma
         self.mat_C_t = None
         self.cls_name = 'LSTSVM'
         
         # Two hyperplanes attributes
         self.w1, self.b1, self.w2, self.b2 = None, None, None, None
        
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
            
            # Regulariztion term used for ill-possible condition
            reg_term = 2 ** float(-7)
            
            # TODO: There are redundant computation below, which needs to be fixed
            # Determine parameters of hypersurfaces # Using SMW formula
            if mat_A.shape[0] < mat_B.shape[0]:
                
                y = (1 / reg_term) * (np.identity(mat_G.shape[1]) -
                     np.dot(np.dot(mat_G_t, np.linalg.inv((reg_term * \
                     np.identity(mat_G.shape[0])) + np.dot(mat_G, mat_G_t))),
                     mat_G))
                
                hyper_surf1 = np.dot(-1 * (y - np.dot(np.dot(np.dot(y, mat_H_t),
                              np.linalg.inv(self.C1 * np.identity(mat_H.shape[0]) \
                              + np.dot(np.dot(mat_H, y), mat_H_t))), np.dot(mat_H,
                              y))), np.dot(mat_G_t, np.ones((mat_G.shape[0], 1))))
                
                hyper_surf2 = np.dot(self.C2 * (y - np.dot(np.dot(np.dot(y, mat_H_t),
                              np.linalg.inv((np.identity(mat_H.shape[0]) / self.C2) \
                              + np.dot(np.dot(mat_H, y), mat_H_t))), np.dot(mat_H,
                              y))), np.dot(mat_H_t, np.ones((mat_H.shape[0], 1))))
                
                # Parameters of hypersurfaces
                self.w1 = hyper_surf1[:hyper_surf1.shape[0] - 1, :]
                self.b1 = hyper_surf1[-1, :]
                
                self.w2 = hyper_surf2[:hyper_surf2.shape[0] - 1, :]
                self.b2 = hyper_surf2[-1, :]
                
            else:
                
                z = (1 / reg_term) * (np.identity(mat_H.shape[1]) - 
                     np.dot(np.dot(mat_H_t, np.linalg.inv(reg_term * \
                     np.identity(mat_H.shape[0]) + np.dot(mat_H, mat_H_t))),
                     mat_G))
                
                hyper_surf1 = np.dot(self.C1 * (z - np.dot(np.dot(np.dot(z, mat_G_t),
                              np.linalg.inv((np.identity(mat_G.shape[0]) / self.C1) \
                              + np.dot(np.dot(mat_G, z), mat_G_t))), np.dot(mat_G,
                              z))), np.dot(mat_G_t, np.ones((mat_G.shape[0], 1))))
                
                hyper_surf2 = np.dot((z - np.dot(np.dot(np.dot(z, mat_G_t),
                              np.linalg.inv(self.C2 * np.identity(mat_G.shape[0]) \
                              + np.dot(np.dot(mat_G, z), mat_G_t))), np.dot(mat_G,
                              z))), np.dot(mat_H_t, np.ones((mat_H.shape[0], 1))))
                
                self.w1 = hyper_surf1[:hyper_surf1.shape[0] - 1, :]
                self.b1 = hyper_surf1[-1, :]
                
                self.w2 = hyper_surf2[:hyper_surf2.shape[0] - 1, :]
                self.b2 = hyper_surf2[-1, :]
		
	def predict(self, X):
	
         """
         Performs classification on samples in X using the TwinSVM model.
        
         Parameters
         ----------
         X : array-like, shape (n_samples, n_features)
            Feature vectors of test data.
                
         Returns
         -------
         output : array, shape (n_samples,)
             Predicted class lables of test data.
		 """
        
         # Calculate prependicular distances for new data points 
         prepen_distance = np.zeros((X.shape[0], 2))

         kernel_f = {'linear': lambda i: X[i, :], 'RBF': lambda i: 
                     rbf_kernel(X[i, :], self.mat_C_t, self.gamma)}

         for i in range(X.shape[0]):

            # Prependicular distance of data pint i from hyperplanes
            prepen_distance[i, 1] = np.abs(np.dot(kernel_f[self.kernel](i),
                                    self.w1) + self.b1)

            prepen_distance[i, 0] = np.abs(np.dot(kernel_f[self.kernel](i),
                                    self.w2) + self.b2)

        # Assign data points to class +1 or -1 based on distance from hyperplanes
         output = 2 * np.argmin(prepen_distance, axis=1) - 1

         return output


if __name__ == '__main__':
    
    
    from preprocess import read_data
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    X, y, filename = read_data('../dataset/australian.csv')
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    lstsvm_model = LSTSVM('linear', 0.25, 0.5)
    
    lstsvm_model.fit(x_train, y_train)
    pred = lstsvm_model.predict(x_test)
    
    print("Accuracy: %.2f" % (accuracy_score(y_test, pred) * 100))

