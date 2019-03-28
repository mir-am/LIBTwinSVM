# -*- coding: utf-8 -*-

"""
LightTwinSVM Program - Simple and Fast
Version: 0.2.0-alpha - 2018-05-30
Developer: Mir, A. (mir-am@hotmail.com)
License: GNU General Public License v3.0

Module: setup.py
Building C++ extension module for Windows OS using Cython

Externel dependencies:
- Armadillo C++ Linear Agebra Library (http://arma.sourceforge.net)
- LAPACK and BLAS libaray (http://www.netlib.org/lapack)
- Cython (http://cython.org/)
"""

from distutils.core import setup, Extension
from Cython.Build import cythonize
from os.path import join
import numpy as np


setup(name='clipdcd',
      version='0.2.0',
      author='Mir, A.',
      author_email='mir-am@hotmail.com',
      url='https://github.com/mir-am/LightTwinSVM',
      description='clipDCD opimtizer implemented in C++ and improved by Mir, A.',
      ext_modules=cythonize(Extension(
        "clipdcd",
        sources=[join("src", "clipdcd.pyx"), join("src", "clippdcd_opt.cpp")],
        language="c++",
        libraries=['lapack_win64_MT', 'blas_win64_MT'],
        library_dirs=['.\\armadillo-code\\lib_win64'],
        )),
      include_dirs=[np.get_include(), './armadillo-code/include'])
      
