An example of binary classification using TSVM-based classifiers
================================================================

Here, we provided an example to help you classify using binary TSVM-based classifiers that are available
in the library's API. Comments are also provided in the code example to make API usage clear. 

.. code-block:: python

	from libtsvm.preprocess import DataReader
	from libtsvm.estimators import TSVM
	from libtsvm.model_selection import Validator

	# Step 1: Load your dataset
	data_path = '../../dataset/australian.csv'
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

	# Step 3: Evaluate the estimator using train/test split
	eval_method = 't_t_split' # Train/Test split
	test_set_size = 30 # 30% of samples

	val = Validator(X, y, (eval_method, test_set_size), tsvm_clf)
	eval_func = val.choose_validator()

	# Hyper-parameters of the classifier
	h_params =  {'C1': 2**-3, 'C2': 2**-5}

	acc, std, full_report = eval_func(h_params)

	print("Accuracy: %.2f" % acc)
	print(full_report)
