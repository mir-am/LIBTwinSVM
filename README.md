# LIBTwinSVM
***
***
## Core Features
LIBTwinSVM is an easy-to-use implementation of Twin Support Vector Machine.  It is licensed under the terms of GNU GPL v3. This application comes with a user interface which makes using the algorithm so handy for Data Scientists, Machine Learning Researchers and whoever else that is interested in Machine Learning.
<br>

You can read some of its core features below:
- A **simple** and **user-friendly Graphical User Interface** (GUI) for running TwinSVM Classifier.
- Easy to import data in **CSV** & **LIBSVM** format.
- Loading data with **shuffling** and **normalizing** ability.
- **Fast optimization algorithm**: The clipDCD algorithm was improved and is implemented in **C++** for solving optimization problems of TwinSVM.
- Supporting **Linear**, **RBF kernel** and **Rectangular**.
- Supporting **Binary** and **Multi-class** classification (One-vs-All & One-vs-One).
- The OVO estimator is **compatible with sci-kit-learn** tools such as GridSearchCV, cross_val_score, etc.
- The classifier can be evaluated using either **K-fold cross-validation** or **Training/Test split**.
- It supports **grid search** over C and gamma parameters.
- **Saving** the detailed classification result in an **Excel-format spreadsheet** file in the custom location.
- **Keeping log** of the entire training session to keep track of best training results.