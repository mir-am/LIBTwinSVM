# -*- coding: utf-8 -*-

# Developers: Mir, A. and Mahdi Rahbar
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0

"""
In this module, functions for reading and processing datasets are defined.
"""


from os.path import splitext, split
from sklearn.datasets import load_svmlight_file
from libtsvm.model import DataInfo
import numpy as np
import pandas as pd
import csv


def load_data(file_path, sep, header, shuffle, normalize):
    """
    It reads a CSV file into pandas DataFrame.
        
    Parameters
    ----------
    file_path : str
        Path to the dataset file.
        
    sep : str
        Separator character
        
    header : boolean
        whether the dataset has header names or not.
        
    shuffle : boolean
        whether to shuffle the dataset or not.
        
    normalize : boolean
        whether to normalize the dataset or not.
    
    Returns
    -------
    data_train : array-like, shape (n_samples, n_features) 
        Training samples in NumPy array.
        
    data_labels : array-like, shape(n_samples,) 
        Class labels of training samples.
        
    file_name : str
        Dataset's filename.
    """
    
    df = pd.read_csv(file_path, sep=sep)

    # First extract class labels
    y_true = df.iloc[:, 0].values
    df.drop(df.columns[0], axis=1, inplace=True)
        
    if normalize:
    
        df = (df - df.mean()) / df.std()
    
        #print(df)
        
    if shuffle:
        
        df = df.sample(frac=1).reset_index(drop=True)
        
        #print(df)
    
    X_train = df.iloc[:, 1:].values # Feature values

    return X_train, y_true, list(df.columns.values) if header else []


def get_data_info(X, y, header):
    """
    It gets data characteristics from dataset.
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training feature vectors, where n_samples is the number of samples
        and n_features is the number of features.
        
    y : array-like, shape(n_samples,)
        Target values or class labels.
        
    header : list
        Header names
        
    Return
    ------
    object
        data characteristics
    """
    
    no_samples = X.shape[0]
    no_features = X.shape[1]
    class_labels = np.unique(y)
    
    return DataInfo(no_samples, no_features, class_labels.size, class_labels,
                    header)
    

def conv_str_fl(data):
    """
    It converts string data to float for computation.
    
    Parameters
    ----------
    data : array-like, shape (n_samples, n_features)
        Training samples, where n_samples is the number of samples
        and n_features is the number of features.
        
    Returns
    -------
    array-like
        A numerical dataset which is suitable for futher computation.
    """
    
    temp_data = np.zeros(data.shape)
    
    # Read rows
    for i in range(data.shape[0]):
        
        # Read coloums
        for j in range(data.shape[1]):
            
            temp_data[i][j] = float(data[i][j])
            
    return temp_data

def read_data(filename, header=True):
    
    """
    It converts a CSV dataset to NumPy arrays for further operations
    like training the TwinSVM classifier.
        
    Parameters
    ----------
    filename : str
        Path to the dataset file.
        
    header : boolean, optional (default=True)
        Ignores first row of dataset which contains header names.
            
    Returns
    -------
    data_train : array-like, shape (n_samples, n_features) 
        Training samples in NumPy array.
        
    data_labels : array-like, shape(n_samples,) 
        Class labels of training samples.
        
    file_name : str
        Dataset's filename.
    """
    
    data = open(filename, 'r')
    
    data_csv = csv.reader(data, delimiter=',')
    
    # Ignore header names
    if not header:
        
        data_array = np.array(list(data_csv))
        
    else:
        
        data_array = np.array(list(data_csv)[1:]) # [1:] for removing headers
    
    data.close()
    
    # Shuffle data
    #np.random.shuffle(data_array)                        
    
    # Convers string data to float
    data_train = conv_str_fl(data_array[:, 1:])                     
                         
    data_labels = np.array([int(i) for i in data_array[:, 0]])
    
    file_name = splitext(split(filename)[-1])[0]
    
    return data_train, data_labels, file_name 


def read_libsvm(filename):
    """
    It reads `LIBSVM <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/>`_
    data files for doing classification using the TwinSVM model.

    Parameters
    ---------- 
    filename : str 
    Path to the LIBSVM data file.
	
    Returns
    -------
    array-like 
    Training samples.
   
    array-like
    Class labels of training samples.
   
    str
    Dataset's filename
    """

    libsvm_data = load_svmlight_file(filename)
    file_name = splitext(split(filename)[-1])[0]
	
	# Converting sparse CSR matrix to NumPy array
    return libsvm_data[0].toarray(), libsvm_data[1].astype(np.int), file_name

