# clone Armadillo which is a C++ Linear Algebra library
# Armadillo is licensed under the Apache License, Version 2.0
if [ -d "armadillo-code" ]
then
	echo "Found Armadillo repository. No need to clone again."
	
else
	# clones Armadillo which is a C++ Linear Algebra library
	# Armadillo is licensed under the Apache License, Version 2.0
	git clone https://github.com/mir-am/armadillo-code.git
fi


# Generate C++ extension module (Optimizer) using Cython
python3 setup.py build_ext --inplace
