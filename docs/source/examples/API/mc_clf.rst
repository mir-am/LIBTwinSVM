An example of multi-class classification using OVO-LSTSVM
=========================================================

This example shows how you can use Least Squares TwinSVM classifier with One-vs-One
strategy to solve a multi-class classification problem.

.. code-block:: python

	from libtsvm.preprocess import DataReader
	from libtsvm.estimators import LSTSVM
	from libtsvm.mc_scheme import OneVsOneClassifier
	from libtsvm.model_selection import Validator

	# Step 1: Load your dataset
	data_path = '../../dataset/iris.csv'
	sep_char = ',' # separtor character of the CSV file
	header = True # Whether the dataset has header names.

	dataset = DataReader(data_path, sep_char, header)

	shuffle_data = True
	normalize_data = False

	dataset.load_data(shuffle_data, normalize_data)
	X, y, file_name = dataset.get_data()

	# Step 2: Choose a TSVM-based estimator
	kernel = 'RBF'
	lstsvm_clf = LSTSVM(kernel=kernel)

	# Step 3: Select a multi-class approach
	ovo_lstsvm = OneVsOneClassifier(lstsvm_clf)

	# Step 4: Evaluate the multi-class estimator using train/test split
	eval_method = 't_t_split' # Train/Test split
	test_set_size = 20 # 20% of samples

	val = Validator(X, y, (eval_method, test_set_size), ovo_lstsvm)
	eval_func = val.choose_validator()

	# Hyper-parameters of the classifier
	h_params =  {'C1': 2**-2, 'C2': 2**-2, 'gamma': 2**-7}

	acc, std, full_report = eval_func(h_params)

	print("Accuracy: %.2f" % acc)
	print(full_report)