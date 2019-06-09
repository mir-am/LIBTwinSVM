An example of model selection with grid search and cross-validation
===================================================================

In this example, a code samples is provided to help you find the best model for your classification task. The best model will be found using grid search and cross validation that are available in the library's API.
At the end, the classification results is also saved in a speadsheet file for further analysis.

.. code-block:: python

	from libtsvm.preprocess import DataReader
	from libtsvm.estimators import TSVM
	from libtsvm.model_selection import Validator, grid_search, save_result

	# Step 1: Load your dataset
	data_path = './dataset/australian.csv'
	sep_char = ',' # separtor character of the CSV file
	header = True # Whether the dataset has header names.

	dataset = DataReader(data_path, sep_char, header)

	shuffle_data = True
	normalize_data = False

	dataset.load_data(True, False)
	X, y, _ = dataset.get_data()

	# Step 2: Choose a TSVM-based estimator
	tsvm_clf = TSVM(kernel='RBF')

	# Step 3: Choose an evaluation method.
	val = Validator(X, y, ('CV', 5), tsvm_clf) # 5-fold cross-validation
	eval_method = val.choose_validator()

	# Step 4: Specify range of each hyper-parameter for a TSVM-based estimator.
	params = {'C1': (-2, 2), 'C2': (-2, 2), 'gamma': (-8, 2)}

	best_acc, best_acc_std, opt_params, clf_results = grid_search(eval_method, params)

	print("Best accuracy: %.2f+-%.2f | Optimal parameters: %s" % (best_acc, best_acc_std,
											  str(opt_params)))

	# Step 5: Save the classification results
	clf_type = 'binary' # Type of classification problem
	save_result(val, clf_type, clf_results, 'TSVM-RBF-Australian')
	