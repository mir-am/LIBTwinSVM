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

#from distutils.core import setup, Extension
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

libs = []
link_args = []
compile_args = ['-std=c++11', '-DARMA_DONT_USE_WRAPPER']

# Choose appropriate libraries depeneding on the OS
if platform == 'win32':
    
    libs = ['lapack_win64_MT', 'blas_win64_MT']
    link_args = link_args + ['/MT']

elif platform == 'linux':
    
    check_libs_exist()
    
    libs = ['blas', 'lapack']
    
elif platform == 'darwin':
    # TODO: Add libs for OSX.
    
    #link_args = ['-framework Accelerate']
    compile_args = compile_args + ['-stdlib=libc++', '-mmacosx-version-min=10.9']


#ext_clipdcd = Extension("clipdcd",
#        sources=[join("src", "clipdcd.pyx"), join("src", "clippdcd_opt.cpp")],
#        language="c++",
#        libraries=libs,
#        library_dirs=['.\\armadillo-code\\lib_win64'],
#        extra_compile_args=compile_args,
#        extra_link_args=link_args,
#        include_dirs=[np.get_include(), './armadillo-code/include'])

def configuration(parent_package='', top_path=None):
    """A setup config for building clipdcd extension module """
    
    from numpy.distutils.misc_util import Configuration
    from Cython.Build import cythonize
    
    config = Configuration('optimizer', parent_package, top_path)
    
    # TODO: Check whether compile and link flags for both library and extension
    # is needed.
    config.add_library('clipdcd-lib',
                       language='c++',
                       sources=[join("src", "clippdcd_opt.cpp"),
                                join("src", "clippdcd_opt.h")],
                        include_dirs=[np.get_include(), join('armadillo-code',
                                       'include')],
                        extra_compiler_args=compile_args,
                     )
            
    
    sources = [join("src", "clipdcd.pyx")]
    depends = [join("src", "clippdcd_opt.cpp"), join("src", "clippdcd_opt.h")]
    
    config.add_extension('clipdcd',
                         sources=sources,
                         depends=depends,
                         language="c++",
                         libraries=['clipdcd-lib']+libs,
                         library_dirs=[join('armadillo-code', 'lib_win64')],
                         extra_compile_args=compile_args,
                         extra_link_args=link_args,
                         include_dirs=[np.get_include(), join('armadillo-code',
                                       'include')])
    
    config.ext_modules = cythonize(config.ext_modules)
    
    return config

if __name__ == '__main__':
    
    from numpy.distutils.core import setup
    
    setup(**configuration(top_path='').todict())
    
    
#setup(name='clipdcd',
#      version='0.2.0',
#      author='Mir, A.',
#      author_email='mir-am@hotmail.com',
#      url='https://github.com/mir-am/LightTwinSVM',
#      description='clipDCD opimtizer implemented in C++ and improved by Mir, A.',
#      ext_modules=cythonize(Extension(
#        "clipdcd",
#        sources=[join("src", "clipdcd.pyx"), join("src", "clippdcd_opt.cpp")],
#        language="c++",
#        libraries=libs,
#        library_dirs=['.\\armadillo-code\\lib_win64'],
#        extra_compile_args=compile_args,
#        extra_link_args=link_args,
#        include_dirs=[np.get_include(), './armadillo-code/include']
#        )))
      