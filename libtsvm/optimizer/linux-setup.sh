# clone Armadillo which is a C++ Linear Algebra library
# Armadillo is licensed under the Apache License, Version 2.0
git clone https://github.com/mir-am/armadillo-code.git

# Generate C++ extension module (Optimizer) using Cython
python setup.py build_ext --inplace
