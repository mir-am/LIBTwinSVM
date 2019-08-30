# cython: language_level=3
# LightTwinSVM Program - Simple and Fast
# Version: 0.2.0-alpha - 2018-05-30
# Developer: Mir, A. (mir-am@hotmail.com)
# License: GNU General Public License v3.0

# A Cython module for wrapping C++ code (ClippDCD optimizer)
# It generates Python extension module for Windows OS.

cimport cython
cimport numpy
import numpy

numpy.import_array()

from libcpp.vector cimport vector
from libcpp cimport bool

############### Armadillo & NumPy conversion #################################
"""
This section converts numpy arrays to Armadillo data structures without copying
the memory.

The credit for this module goes the mlpack's developers.

mlpack is free software; you may redistribute it and/or modify it under the
terms of the 3-clause BSD license.  You should have received a copy of the
3-clause BSD license along with mlpack.  If not, see
http://www.opensource.org/licenses/BSD-3-Clause for more information.
"""

# We have to include from mlpack/core.hpp so that everything is included in the
# right order and the Armadillo extensions from mlpack are set up right.
cdef extern from "<armadillo>" namespace "arma" nogil:
  # Import the index type.
  ctypedef int uword
  # Import the half-size index type.
  ctypedef int uhword

  cdef cppclass Mat[T]:
    # Special constructor that uses auxiliary memory.
    Mat(T* aux_mem,
        uword n_rows,
        uword n_cols,
        bool copy_aux_mem,
        bool strict) nogil

    # Constructor that initializes memory.
    Mat(uword n_rows, uword n_cols) nogil

    # Default constructor.
    Mat() nogil

    # Number of rows.
    const uword n_rows
    # Number of columns.
    const uword n_cols
    # Total number of elements.
    const uword n_elem

    # Memory state: preallocated, changeable, etc.
    const uhword mem_state

    # Access the memory pointer directly.
    T* memptr() nogil

  cdef cppclass Col[T]:
    # Special constructor that uses auxiliary memory.
    Col(T* aux_mem,
        uword n_elem,
        bool copy_aux_mem,
        bool strict) nogil

    # Constructor that initializes memory.
    Col(uword n_elem) nogil

    # Default constructor.
    Col() nogil

    # Number of rows (equal to number of elements).
    const uword n_rows
    # Number of columns (always 1).
    const uword n_cols
    # Number of elements (equal to number of rows).
    const uword n_elem

    # Memory state: preallocated, changeable, etc.
    const uword mem_state

    # Access the memory pointer directly.
    T* memptr() nogil

  cdef cppclass Row[T]:
    # Special constructor that uses auxiliary memory.
    Row(T* aux_mem,
        uword n_elem,
        bool copy_aux_mem,
        bool strict) nogil

    # Constructor that initializes memory.
    Row(uword n_elem) nogil

    # Default constructor.
    Row() nogil

    # Number of rows (always 1).
    const uword n_rows
    # Number of columns (equal to number of elements).
    const uword n_cols
    # Number of elements (equal to number of columns).
    const uword n_elem

    # Memory state: preallocated, changeable, etc.
    const uword mem_state

    # Access the memory pointer directly.
    T* memptr() nogil
    
cdef extern from "arma_util.hpp":
  void SetMemState[T](T& m, int state)
  size_t GetMemState[T](T& m)
  double* GetMemory(Mat[double]& m)
  double* GetMemory(Col[double]& m)
  double* GetMemory(Row[double]& m)
  size_t* GetMemory(Mat[size_t]& m)
  size_t* GetMemory(Col[size_t]& m)
  size_t* GetMemory(Row[size_t]& m)


cdef extern from "numpy/arrayobject.h":
  void PyArray_ENABLEFLAGS(numpy.ndarray arr, int flags)
  void PyArray_CLEARFLAGS(numpy.ndarray arr, int flags)
  

cdef Mat[double]* numpy_to_mat_d(numpy.ndarray[numpy.double_t, ndim=2] X, \
                                      bool takeOwnership) except +:
  """
  Convert a numpy ndarray to a matrix.  The memory will still be owned by numpy.
  """
  if not (X.flags.c_contiguous or X.flags.owndata):
    # If needed, make a copy where we own the memory.
    X = X.copy(order="C")
    takeOwnership = True

  cdef Mat[double]* m = new Mat[double](<double*> X.data, X.shape[1],\
      X.shape[0], False, False)

  # Take ownership of the memory, if we need to.
  if takeOwnership:
    PyArray_CLEARFLAGS(X, numpy.NPY_OWNDATA)
    SetMemState[Mat[double]](m[0], 0)

  return m

cdef numpy.ndarray[numpy.double_t, ndim=1] row_to_numpy_d(Row[double]& X) \
    except +:
  """
  Convert an Armadillo row vector to a one-dimensional numpy ndarray.
  """
  # Extract dimensions.
  cdef numpy.npy_intp dim = <numpy.npy_intp> X.n_elem
  cdef numpy.ndarray[numpy.double_t, ndim=1] output = \
      numpy.PyArray_SimpleNewFromData(1, &dim, numpy.NPY_DOUBLE, GetMemory(X))
  
  # No need to transer memory! It causes heap corruption on Windows OS.  
  # Transfer memory ownership, if needed.
#  if GetMemState[Row[double]](X) == 0:
#    SetMemState[Row[double]](X, 1)
#    PyArray_ENABLEFLAGS(output, numpy.NPY_OWNDATA)

  return output

##############################################################################

############################## ClipDCD optimizer #############################
cdef extern from "clippdcd_opt.h":
    
    Row[double] optimizer(Mat[double]* dual, const double c)
    
def optimize(numpy.ndarray[double, ndim=2] mat_dual, const double c):
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
    
    cdef Mat[double]* mat_dual_arma = numpy_to_mat_d(mat_dual, 0)
    cdef Row[double] lag_mults = optimizer(mat_dual_arma, c)
    
    return row_to_numpy_d(lag_mults)
	
##############################################################################