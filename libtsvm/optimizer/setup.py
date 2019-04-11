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
from sys import platform, exit
from ctypes.util import find_library
import numpy as np


def check_libs_exist():
    """
    It checks whether required libraries exists on user's system for compiling
    the C++ extension module.
    """
    
    if not (find_library('lapack') and find_library('blas')):
        
        print("Could not find the required libraries. Please use your "
              "distribution package manager to install LAPACK and BLAS "
              "libraries. Otherwise, C++ extension cannot be generated.")
        
        exit()

# Choose appropriate libraries depeneding on the OS
if platform == 'win32':
    
    libs = ['lapack_win64_MT', 'blas_win64_MT']

elif platform == 'linux':
    
    check_libs_exist()
    
    libs = ['blas', 'lapack']
    
elif platform == 'darwin':
    # TODO: Add libs for OSX.
    pass


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
        libraries=libs,
        library_dirs=['.\\armadillo-code\\lib_win64'],
        )),
      include_dirs=[np.get_include(), './armadillo-code/include'])
      