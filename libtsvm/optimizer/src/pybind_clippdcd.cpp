/*
LightTwinSVM Program - Simple and Fast
Version: 0.2.0-alpha - 2018-05-30
Developer: Mir, A. (mir-am@hotmail.com)
License: GNU General Public License v3.0

The ClippDCD algorithm is implmeneted for solving dual optimization problems of TwinSVM.
Its speed is improved significantly by Mir, A. 

This algorithm was proposed by:
Peng, X., Chen, D., & Kong, L. (2014). A clipping dual coordinate descent algorithm for solving support vector machines. Knowledge-Based Systems, 71, 266-278.

This C++ extension depends on the follwing libraries:
- Armadillo C++ Linear Agebra Library (http://arma.sourceforge.net)
- pybind11 for creating python bindings on Linux (https://pybind11.readthedocs.io)

Change log:
Mar 21, 2018: A bug related to the WLTSVM classifier was fixed. the bug caused poor accuracy. 
Bug was in section which filters out indices.

Mar 23, 2018: execution time improved significantly by computing dot product in temp var.

May 4, 2018: A trick for improving dot product computation. It imporves speed by 4-5x times.

*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "clippdcd_opt.h"
#include "clippdcd_opt.cpp"

#define MAX_ITER 15000

PYBIND11_MODULE(clipdcd, m) {
    m.doc() = "ClippDCD opimizer implemented in C++ and improved by Mir, A.";

    m.def("clippDCD_optimizer", &clippDCD_optimizer, "ClippDCD algorithm - solves dual optimization problem");
}

