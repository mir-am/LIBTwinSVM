# -*- coding: utf-8 -*-

# Developers: Mir, A. and Mahdi Rahbar
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0

"""
In this module, functions for reading and processing datasets are defined.
"""


from os.path import splitext, split
from sklearn.datasets import load_svmlight_file
import numpy as np
import csv


def conv_str_fl(data):
    """
        It converts string data to float for computation.

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
        It converts CSV file to NumPy arrays for further operations like training

        Input:
            file_name: Path to the dataset file
            ignore_header: Ignoring first row of dataset because of header names

        output:
            data_samples: Training samples in NumPy array
            data_labels: labels of samples in NumPy array
            file_name: Name of dataset

    """

    data = open(filename, 'r')

    data_csv = csv.reader(data, delimiter=',')

    # Ignore header names
    if not header:

        data_array = np.array(list(data_csv))

    else:

        data_array = np.array(list(data_csv)[1:])  # [1:] for removing headers

    data.close()

    # Shuffle data
    # np.random.shuffle(data_array)

    # Convers string data to float
    data_train = conv_str_fl(data_array[:, 1:])

    data_labels = np.array([int(i) for i in data_array[:, 0]])

    file_name = splitext(split(filename)[-1])[0]

    return data_train, data_labels, file_name


def read_libsvm(filename):
    """
    It reads LIBSVM data files for doing classification with TwinSVM
    Input:
            file_name: Path to the dataset file

    output:
    data_samples: Training samples in NumPy array
    data_labels: labels of samples in NumPy array
    file_name: Name of dataset
    """

    libsvm_data = load_svmlight_file(filename)
    file_name = splitext(split(filename)[-1])[0]

    # Converting sparse CSR matrix to NumPy array
    return libsvm_data[0].toarray(), libsvm_data[1].astype(np.int), file_name
