An example of model evaluation with cross-validation
=====================================================

This user guide is provided to help you evaluate the model with cross validation.

.. code-block:: python

	from libtsvm.preprocess import DataReader
	from libtsvm.estimators import TSVM
	from libtsvm.model_selection import Validator

	# Step 1: Load your dataset
	data_path = '../../dataset/hepatits.csv'
	sep_char = ',' # separtor character of the CSV file
	header = True # Whether the dataset has header names.

	dataset = DataReader(data_path, sep_char, header)

	shuffle_data = True
	normalize_data = False

	dataset.load_data(shuffle_data, normalize_data)
	X, y, file_name = dataset.get_data()

	# Step 2: Choose a TSVM-based estimator
	kernel = 'linear'
	tsvm_clf = TSVM(kernel=kernel)

	# Step 3: Evaluate the estimator using cross validation
	eval_method = 'CV' # Cross validation
	folds = 5 

	val = Validator(X, y, (eval_method, folds), tsvm_clf)
	eval_func = val.choose_validator()

	# Hyper-parameters of the classifier
	h_params =  {'C1': 2**-2, 'C2': 2**1}

	acc, std, full_report = eval_func(h_params)

	print("Accuracy: %.2f" % acc)
	print(full_report)
