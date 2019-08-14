/*
LightTwinSVM Program - Simple and Fast
Version: 0.2.0-alpha - 2018-05-30
Developer: Mir, A. (mir-am@hotmail.com)
License: GNU General Public License v3.0

The clipDCD algorithm is implemented for solving dual optimization problems of TwinSVM.
Its speed is improved significantly by Mir, A.

This algorithm was proposed by:
Peng, X., Chen, D., & Kong, L. (2014). A clipping dual coordinate descent algorithm for
solving support vector machines. Knowledge-Based Systems, 71, 266-278.

This C++ extension depends on the following libraries:
- Armadillo C++ Linear Algebra Library (http://arma.sourceforge.net)
- pybind11 for creating python bindings on Linux (https://pybind11.readthedocs.io)
- Cython for building C++ extension module on Windows (http://cython.org/)

Change log:
Mar 21, 2018: A bug related to the WLTSVM classifier was fixed. the bug caused poor accuracy.
Bug was in section which filters out indices.

Mar 23, 2018: execution time improved significantly by computing dot product in temp var.

May 4, 2018: A trick for improving dot product computation. It improves speed by 4-5x times.

*/

#include "clippdcd_opt.h"
#include <iostream>
#include <numeric>

#define MAX_ITER 15000

void printAllElem(Mat<double>* x)
{
    cout << "no.rows: " << x->n_rows << " no_cols: " << x->n_cols << endl;
    cout << "First element: " << x->at(0, 0) << endl;
}


std::vector<double> optimizer(Mat<double>* dualMatrix, const double c)
{
//    // Type conversion - STD vector -> arma mat
//    mat dualMatrix = zeros<mat>(dual.size(), dual.size());
//
//    for(unsigned int i = 0; i < dualMatrix.n_rows; i++)
//    {
//        dualMatrix.row(i) = conv_to<rowvec>::from(dual[i]);
//
//    }

    // Step 1: Initial Lagrange multiplies
    vec alpha = zeros<vec>(dualMatrix->n_rows);

    // Number of iterations
    unsigned int iter = 0;

    // Tolerance value
    const double tolValue = pow(10, -5);

    // Max allowed iterations
    const unsigned int maxIter = MAX_ITER;

    // Index set
    std::vector<unsigned int> indexList(dualMatrix->n_rows);

    // Initialize index set
    std::iota(std::begin(indexList), std::end(indexList), 0);

    // Store dot product values
    std::vector<double> dotList(dualMatrix->n_rows);

    // For storing objective function value
    vec objList = zeros<vec>(dualMatrix->n_rows);

    // Create index set and computing dot products for all columns of dual mat
    for(unsigned int i = 0; i < indexList.size(); ++i)
    {

        // Computing dot product here, improves speed significantly
        double temp = dot(alpha, dualMatrix->col(indexList[i]));

        //double obj = (e(*it) - dot(alpha, dualMatrix.col(*it))) / dualMatrix(*it, *it);
        double obj = (1.0 - temp) / dualMatrix->at(indexList[i], indexList[i]);

        dotList[indexList[i]] = temp;

        // Remove index when it makes condition false - Filtering out indexes
        if( !((alpha(indexList[i]) < c) & (obj > 0)) )
        {
            //indexList.erase(it);
            indexList.erase(indexList.begin() + i);

        }
        else
        {
            objList(indexList[i]) = pow(1.0 - temp, 2) / dualMatrix->at(indexList[i], indexList[i]);
        }
    }

    // Step 2: Optimally condition
    while(iter <= maxIter)
    {

        // Find L-index
        unsigned int L_index = index_max(objList);

        // Compute lambda
        double lambda = (1.0 - dot(alpha, dualMatrix->col(L_index))) / dualMatrix->at(L_index, L_index);

        // Previous alpha value
        double preAlpha = alpha(L_index);

        // Step 2.2: Update multipliers
        alpha(L_index) = alpha(L_index) + std::max(0.0, std::min(lambda, c));

        double objValue = pow(1.0 - dot(alpha, dualMatrix->col(L_index)), 2) / dualMatrix->at(L_index, L_index);

        ++iter;

        // Check the convergence
        if(objValue < tolValue)
        {
            //cout << "Found!"  << "Iter: " << iter << endl;
            break;
        }

        // Zeroing!
        objList = zeros<vec>(dualMatrix->n_rows);

        // Computing index list
        for(unsigned int i = 0; i < indexList.size(); ++i)
        {

            // A trick for computing dot so much faster!
            dotList[indexList[i]] = (dotList[indexList[i]] - preAlpha * dualMatrix->at(indexList[i], L_index)) + (alpha(L_index) * dualMatrix->at(indexList[i], L_index));

            double obj = (1.0 - dotList[indexList[i]]) / dualMatrix->at(indexList[i], indexList[i]);

            // Remove index when it makes condition false - Filtering out indexes
            if( !((alpha(indexList[i]) < c) & (obj > 0)) )
            {
                indexList.erase(indexList.begin() + i);

            }
            else
            {
                objList(indexList[i]) = pow(1.0 - dotList[indexList[i]], 2) / dualMatrix->at(indexList[i], indexList[i]);

            }
        }

    }

    // Type conversion
    std::vector<double> alphaVec = conv_to<std::vector<double> >::from(alpha);

    return alphaVec;

}
