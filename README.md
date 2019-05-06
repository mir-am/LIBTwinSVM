# LIBTwinSVM

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



## Installation Guide
### Dependencies

LIBTwinSVM depends on the following packages.

| Package  | Description | License |
| :-------------: | :-------------: | :-------------: |
| Cython  |  | Apache License 2.0 |
| Numpy |  | BSD 3-Clause |
| matplotlib  |  | _ |
| PyQt5  |  | GPL |
| Scikit-learn  |  | BSD 3-Clause |
| Pandas  |  | BSD 3-Clause |
| xlsxwriter |  | BSD 3-Clause |
| joblib |  | BSD 3-Clause |
| NumPydoc |  | _ |