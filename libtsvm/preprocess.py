# -*- coding: utf-8 -*-

# LIBTwinSVM: A Library for Twin Support Vector Machines
# Developers: Mir, A. and Mahdi Rahbar
# License: GNU General Public License v3.0

"""
In this module, functions for reading and processing datasets are defined.
"""


from os.path import splitext, split
from sklearn.datasets import load_svmlight_file
from libtsvm.model import DataInfo
import numpy as np
import pandas as pd


class DataReader():
    """
    It handels data-related tasks like reading, etc.

    Parameters
    ----------
    file_path : str
        Path to the dataset file.

    sep : str
        Separator character

    header : boolean
        whether the dataset has header names or not.

    Attributes
    ----------
    X_train : array-like, shape (n_samples, n_features)
        Training samples in NumPy array.

    y_train :  array-like, shape(n_samples,)
        Class labels of training samples.

    hdr_names : list
        Header names of datasets.

    filename : str
        dataset's filename
    """

    def __init__(self, file_path, sep, header):

        self.file_path = file_path
        self.sep = sep
        self.header = header

    def load_data(self, shuffle, normalize):
        """
        It reads a CSV file into pandas DataFrame.

        Parameters
        ----------
        shuffle : boolean
            Whether to shuffle the dataset or not.

        normalize : boolean
            Whether to normalize the dataset or not.
        """

        f_name, f_ext = splitext(self.file_path)

        if f_ext == '.csv':

            df = pd.read_csv(self.file_path, sep=self.sep)
            self.hdr_names = list(df.columns.values)[1:] if self.header else []

        elif f_ext == '.libsvm':

            X, y, _ = read_libsvm(self.file_path)

            df = pd.DataFrame(np.hstack((y.reshape(X.shape[0], 1), X)))
            self.hdr_names = []
            
            # Check that the lables of binary problems are +1 and -1.
            class_label = df.iloc[:, 0].unique()
            
            if class_label.size == 2:
                
                if not(1 in class_label and -1 in class_label):
                    
                    df.iloc[:, 0][df.iloc[:, 0] == class_label[0]] = 1 
                    df.iloc[:, 0][df.iloc[:, 0] == class_label[1]] = -1 

        else:

            raise ValueError("Dataset format is not supported: %s" % f_ext)

        if shuffle:

            df = df.sample(frac=1).reset_index(drop=True)

            # print(df)

        # extract class labels
        self.y_train = df.iloc[:, 0].values
        df.drop(df.columns[0], axis=1, inplace=True)

        if normalize:

            df = (df - df.mean()) / df.std()

        self.X_train = df.values  # Feature values
        self.filename = splitext(split(f_name)[-1])[0]
        # print(self.filename)

    def get_data(self):
        """
        It returns processed dataset.

        Returns
        -------
        array-like
            Training samples in NumPy array.

        array-like
            Class labels of training samples.

        str
            The dataset's filename
        """

        if all([hasattr(self, attr) for attr in ['X_train', 'y_train',
                'filename']]):

            return self.X_train, self.y_train, self.filename

        else:

            raise AttributeError("The dataset has not been loaded yet!"
                                 "Run load_data() method.")

    def get_data_info(self):
        """
        It returns data characteristics from dataset.

        Returns
        ------
        object
            data characteristics
        """

        unq_cls_lables = np.unique(self.y_train)

        return DataInfo(self.X_train.shape[0], self.X_train.shape[1],
                        unq_cls_lables.size, unq_cls_lables, self.hdr_names)


# def conv_str_fl(data):
#    """
#    It converts string data to float for computation.
#
#    Parameters
#    ----------
#    data : array-like, shape (n_samples, n_features)
#        Training samples, where n_samples is the number of samples
#        and n_features is the number of features.
#
#    Returns
#    -------
#    array-like
#        A numerical dataset which is suitable for futher computation.
#    """
#
#    temp_data = np.zeros(data.shape)
#
#    # Read rows
#    for i in range(data.shape[0]):
#
#        # Read coloums
#        for j in range(data.shape[1]):
#
#            temp_data[i][j] = float(data[i][j])
#
#    return temp_data

# def read_data(filename, header=True):
#
#    """
#    It converts a CSV dataset to NumPy arrays for further operations
#    like training the TwinSVM classifier.
#
#    Parameters
#    ----------
#    filename : str
#        Path to the dataset file.
#
#    header : boolean, optional (default=True)
#        Ignores first row of dataset which contains header names.
#
#    Returns
#    -------
#    data_train : array-like, shape (n_samples, n_features) 
#        Training samples in NumPy array.
#
#    data_labels : array-like, shape(n_samples,) 
#        Class labels of training samples.
#
#    file_name : str
#        Dataset's filename.
#    """
#
#    data = open(filename, 'r')
#
#    data_csv = csv.reader(data, delimiter=',')
#
#    # Ignore header names
#    if not header:
#
#        data_array = np.array(list(data_csv))
#       
#    else:
#  
#        data_array = np.array(list(data_csv)[1:]) # [1:] for removing headers
#   
#    data.close()
#   
#    # Shuffle data
#    #np.random.shuffle(data_array)                        
#   
#    # Convers string data to float
#    data_train = conv_str_fl(data_array[:, 1:])                     
#                         
#    data_labels = np.array([int(i) for i in data_array[:, 0]])
#    
#    file_name = splitext(split(filename)[-1])[0]
#    
#    return data_train, data_labels, file_name 


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
