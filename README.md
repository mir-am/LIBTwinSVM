# LIBTwinSVM
<a href="https://opensource.org/licenses/GPL-3.0"><img src="https://img.shields.io/badge/License-GPL%20v3-blue.svg" alt="License"></a>
<a href="https://travis-ci.org/mir-am/LIBTwinSVM"><img src="https://api.travis-ci.org/mir-am/LIBTwinSVM.svg?branch=master" alt="Travis-CI"></a>
[![Documentation Status](https://readthedocs.org/projects/libtwinsvm/badge/?version=latest)](https://libtwinsvm.readthedocs.io/en/latest/?badge=latest)


# Contents
1. [Core Features](#core-features)
2. [Installation Guide](#installation-guide)
3. [Quick Start](#quick-start)
4. [Documentation](#documentation) 
5. [License](#license)

## Core Features
LIBTwinSVM is an easy-to-use implementation of Twin Support Vector Machine.  It is licensed under the terms of GNU GPL v3. This application comes with a user interface which makes using the algorithm so handy for Data Scientists, Machine Learning Researchers and whoever else that is interested in Machine Learning.
<br>

- A **simple** and **user-friendly Graphical User Interface** (GUI).
- Supports both **standard TwinSVM** and **Least Squares TwinSVM** classifiers.
- Easy to import data in **CSV** & **LIBSVM** format.
- A dataset can be loaded with **shuffling** and **normalization**.
- A **fast optimizer** (clipDCD) is improved and implemented in C++ to solve optimization problems of TwinSVMs.
- Supports **Linear**, **RBF** and **Rectangular** kernel.
- Supports **Binary** and **Multi-class** classification (One-vs-All & One-vs-One).
- The OVA and OVO estimators are **compatible with scikit-learn** tools such as GridSearchCV, cross_val_score, etc.
- A classifier can be evaluated using either **K-fold cross-validation** or **Training/Test split**.
- Supports **grid search** over estimators' hyper-parameters.
- The detailed classification results can be saved in an **Excel-format spreadsheet** file.
- The classification results can be logged during the grid search process to not lose results in case of power failure.
- A **feature-rich visualization tool** to show decision boundaries and geometrical interpretation of TwinSVMs.
- The best-fitted classifier can be saved on the disk after the grid search process.
- The pre-trained models can be loaded and evaluated on the test samples.


## Installation Guide
### Dependencies

LIBTwinSVM depends on the following packages.

|   Package                                      |                      Description                               |       License         |
| ---------------------------------------------- | -------------------------------------------------------------- | --------------------- |
| [Cython](https://cython.org/)                  |  To use C++ code in Python.                                    | Apache License 2.0    |
| [NumPy](https://www.numpy.org/)                |  Fast linear algebra operations.                               | BSD 3-Clause          | 
| [Matplotlib](https://matplotlib.org/)          |  Visualization and geometrical representation of classifiers.  | [Matplotlib License](https://matplotlib.org/users/license.html)                    |
| [PyQt5](https://www.riverbankcomputing.com/software/pyqt/intro)  | To create a GUI for using the LIBTwinSVM's features.| GPL |
| [Scikit-learn](https://scikit-learn.org/)      | For TwinSVM-based models evaluation and selection.             | BSD 3-Clause |
| [Pandas](https://pandas.pydata.org/)           | For reading and processing datasets.                           | BSD 3-Clause |
| [XlsxWriter](https://xlsxwriter.readthedocs.io/) | For saving classification results in an Excel file.          | BSD 3-Clause |
| [Joblib](https://joblib.readthedocs.io)   | For saving and loading TwinSVM-based models.                   | BSD 3-Clause |
| [numpydoc](https://numpydoc.readthedocs.io/en/latest/) | API code documentation.                                | BSD License  |


### Quick Installation

For Installing LIBTwinSVM you can choose one of the following commands based on your Operating System and your Python version. Please note that for installing the latest bugfixes and features, we recommend you to install the library from the source. 

* Linux & Mac OS:
```python
pip3 install libtwinsvm
```
* Windows:
```python
pip install libtwinsvm
```

### Installation from the source

**1.  Downloading LIBTWinSVM**

First, make sure that [Git](https://git-scm.com/) is installed as it is required for getting the source code. Then open Git in any arbitrary path and enter the following command:

```
git clone --recursive https://github.com/mir-am/LIBTwinSVM
```

**2. Downloading Requirements**

Before installing any packages, we recommend you to upgrade pip:

* Linux & Mac OS:
```python
pip3 install -U pip
```
* Windows:
```python
pip install -U pip
```

For LIBTwinSVM installation, *Numpy* and *Cython* must be installed. You can install them by entering the following commands in your terminal. If you already have *Cython* and *Numpy* installed, you can skip to the next section.

* Linux & Mac OS: For installing Numpy & Cython on your computer, you should enter the following command in the terminal.
```
pip3 install numpy cython
```

* Windows: for installing the requirements on windows, you should enter the following command in the command prompt. 
```
pip install numpy cython
```

**3. Installing LIBTwinSVM**

Go to LIBTwinSVM folder where you have downloaded the source code in step 2. Then enter the following command in the terminal:

* Linux & Mac OS:
```python
pip3 install .
```
* Windows:
```python
pip install .
```

## Quick start

After LIBTwinSVM was installed, for running the GUI application, open the terminal and enter the following code:
* Linux & Mac OS:
```python
python3 -m libtsvm
```
* Windows:
```python
python -m libtsvm


## Uninstall LIBTwinSVM

For uninstalling LIBTwinSVM enter the following command in the terminal:

* Linux & Mac OS:
```python
pip3 uninstall libtsvm
```
* Windows:
```python
pip uninstall libtsvm
```

## Documentation
Usage examples and API reference can be found on the project's [Read the Docs page](https://libtwinsvm.readthedocs.io/en/latest/). 


## License
LIBTwinSVM library is licensed under the terms of GNU General Public License v3. This library can be used for both academic and commercial purposes. For more information, check out the [LICENSE](https://github.com/mir-am/LIBTwinSVM/blob/master/LICENSE.txt) file.
