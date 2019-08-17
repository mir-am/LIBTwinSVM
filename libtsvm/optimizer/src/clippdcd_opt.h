/*
LIBTwinSVM: A Library for Twin Support Vector Machines
Developers: Mir, A. and Mahdi Rahbar
License: GNU General Public License v3.0

The clipDCD algorithm is implemented for solving dual optimization problems of TwinSVM.
Its speed is improved significantly by Mir, A.

This algorithm was proposed by:
Peng, X., Chen, D., & Kong, L. (2014). A clipping dual coordinate descent algorithm for solving support vector machines. Knowledge-Based Systems, 71, 266-278.

This C++ extension depends on the follwing libraries:
- Armadillo C++ Linear Agebra Library (http://arma.sourceforge.net)
- Cython for building C++ extension module (http://cython.org/)

Change log:
Mar 21, 2018: A bug related to the WLTSVM classifier was fixed. the bug caused poor accuracy.
Bug was in section which filters out indices.

Mar 23, 2018: execution time improved significantly by computing dot product in temp var.

May 4, 2018: A trick for improving dot product computation. It imporves speed by 4-5x times.

Aug 14, 2019: Moving NumPy array to Armadillo matrix without copying the memory.
This improves the speed of the optimizer by 10x times.

*/

#include <vector>
#include <armadillo>

using namespace arma;

Row<double> optimizer(Mat<double>* dualMatrix, const double c);
