# cython: language_level=3
# LightTwinSVM Program - Simple and Fast
# Version: 0.2.0-alpha - 2018-05-30
# Developer: Mir, A. (mir-am@hotmail.com)
# License: GNU General Public License v3.0

# A Cython module for wrapping C++ code (ClippDCD optimizer)
# It generates Python extension module for Windows OS.

from libcpp.vector cimport vector
import numpy as np
cimport numpy as np


cdef extern from "clippdcd_opt.h":
    
	vector[double] optimizer(vector[vector[double]] &dual, const double c)
	

def optimize(np.ndarray[double, ndim=2] mat_dual, const double c):
	
	"""
	It solves a dual optimization problem using clipDCD algorithm
    
	Parameters
    ----------
    mat_dual : array-like, shape (n_samples, n_samples)
        The Q matrix of dual QPP problems.
        
	c : float
        The penalty parameter
        
	Returns
    -------
	list
        Lagrange multipliers
	"""

	return optimizer(mat_dual, c)
	