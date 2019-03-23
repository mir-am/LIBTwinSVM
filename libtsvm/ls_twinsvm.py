# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 20:36:55 2017

@author: Mir
"""

# Implementation of Least square Twin SVM

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from plotcl import make_mesh
from ls_wtsvm import accuracy
from dataproc import read_data
from weight import det_margin
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# Train LS-TwinSVM - Linear case
def train_ls_tsvm(X_train, X_train_label, c1, c2):
    
     # Matrix A or class 1 data
    mat_A = X_train[X_train_label == 1]
    
    # Matrix B  or class -1 data 
    mat_B = X_train[X_train_label == -1]
    
    # LSTSVM algorithm - Linear
    # Step 1: Define E=[A e] and F=[B e]
    mat_E = np.column_stack((mat_A, np.ones((mat_A.shape[0], 1), dtype=np.float64)))
    
    mat_F = np.column_stack((mat_B, np.ones((mat_B.shape[0], 1), dtype=np.float64)))
    
    # Transpose of E & F
    mat_E_t = np.transpose(mat_E)
    mat_F_t = np.transpose(mat_F)
    
    # e matrix
    mat_F_e = mat_F[:, -1].reshape(mat_F_t.shape[1], 1)
    mat_E_e = mat_E[:, -1].reshape(mat_E_t.shape[1], 1)
    
    # Step 2: Select penalty parameters C1 and C2
    c_1 = c1
    c_2 = c2
    
    # Step 3: Determine parameters of two non-parallel hyperplanes
    hyper_p_1 = -1 * np.dot(np.linalg.inv(np.dot(mat_F_t, mat_F) + (1 / c_1) * np.dot(mat_E_t, mat_E)) \
                                  ,np.dot(mat_F_t, mat_F_e))
    
    w_1 = hyper_p_1[:hyper_p_1.shape[0] - 1, :]
    b_1 = hyper_p_1[-1, :]
    
    hyper_p_2 = np.dot(np.linalg.inv(np.dot(mat_E_t, mat_E) + (1 / c_2) * np.dot(mat_F_t, mat_F)) \
                                  ,np.dot(mat_E_t, mat_E_e))
    
    w_2 = hyper_p_2[:hyper_p_2.shape[0] - 1, :]
    b_2 = hyper_p_2[-1, :]
    
    return w_1, b_1, w_2, b_2

# Predict LS-TwinSVM - Linear case
def predict_ls_tsvm(X_test, w_1, b_1, w_2, b_2):
    
    # Step 4: Calculate prependicular distances for new data points
    new_data = X_test # Test data
    
    prepen_distance = np.zeros((new_data.shape[0], 2))
    
    for i in range(new_data.shape[0]):
        
        # Prependicular distance of data pint i from hyperplane 2(class 1)
        prepen_distance[i, 1] = np.abs(np.dot(new_data[i, :], w_1) + b_1)
        
        # Prependicular distance of data pint i from hyperplane 1 (class -1)
        prepen_distance[i, 0] = np.abs(np.dot(new_data[i, :], w_2) + b_2)
        
    # Step 5: Assign data points to class +1 or -1 based on distance from hyperplanes
    output = 2 * np.argmin(prepen_distance, axis=1) - 1
    
    return output

# Linear LS-TSVM - Cross validation
def ls_tsvm(data_train, data_labels, k_fold, c1, c2):
    
    # K-Fold Cross validation, divide data into K subsets
    k_fold = KFold(k_fold)    
        
    # Store result after each run
    mean_accuracy = []
    # Postive class
    mean_recall_p, mean_precision_p, mean_f1_p = [], [], []
    # Negative class
    mean_recall_n, mean_precision_n, mean_f1_n = [], [], []
    
    # Count elements of confusion matrix
    tp, tn, fp, fn = 0, 0, 0, 0
    
    # Train and test LSTSVM K times
    for train_index, test_index in k_fold.split(data_train):
        
        # Extract data based on index created by k_fold
        X_train = np.take(data_train, train_index, axis=0) 
        X_test = np.take(data_train, test_index, axis=0)
        
        X_train_label = np.take(data_labels, train_index, axis=0)
        X_test_label = np.take(data_labels, test_index, axis=0)
                                     
        # fit - create two non-parallel hyperplanes
        hyper_para = train_ls_tsvm(X_train, X_train_label, c1, c2)
        
        # Predict
        output = predict_ls_tsvm(X_test, hyper_para[0], hyper_para[1], hyper_para[2], hyper_para[3])
        
        accuracy_test = accuracy(X_test_label, output)
                              
        mean_accuracy.append(accuracy_test[4])
        # Positive cass
        mean_recall_p.append(accuracy_test[5])
        mean_precision_p.append(accuracy_test[6])
        mean_f1_p.append(accuracy_test[7])
        # Negative class    
        mean_recall_n.append(accuracy_test[8])
        mean_precision_n.append(accuracy_test[9])
        mean_f1_n.append(accuracy_test[10])
        
        # Count
        tp = tp + accuracy_test[0]
        tn = tn + accuracy_test[1]
        fp = fp + accuracy_test[2]
        fn = fn + accuracy_test[3]
        
    # m_a=0, m_r_p=1, m_p_p=2, m_f1_p=3, k=4, c1=5, c2=6, w_1=7, b_1=8, w_2=9, b_2=10
    # m_r_n=11, m_p_n=12, m_f1_n=13, tp=14, tn=15, fp=16, fn=17    
    return mean_accuracy, mean_recall_p, mean_precision_p, mean_f1_p, k_fold.get_n_splits(), c1, c2, \
           hyper_para[0], hyper_para[1], hyper_para[2], hyper_para[3], mean_recall_n, mean_precision_n,\
           mean_f1_n, tp, tn, fp, fn 

# RBF Kernel
def rbf_kernel(x, y, u):
    
    return np.exp(-2 * u) * np.exp(2 * u * np.dot(x, y))

# Train LS-TwinSVM - non-Linear case
def train_nl_ls_tsvm(X_train, X_train_label, c1, c2, rbf_u, reg, rec_k_size=1):
    
    # Matrix A or class 1 data
    mat_A = X_train[X_train_label == 1]
    
    # Matrix B  or class -1 data 
    mat_B = X_train[X_train_label == -1]
    
    # class 1 & class -1
    mat_C = np.row_stack((mat_A, mat_B))
    
    # Tranpose of C
    mat_C_t = np.transpose(mat_C)
    
    # Use rectangular kernel
    #mat_C_t = mat_C_t[:, :mat_C_t.shape[0] * int(rec_k_size)]
    
    # Step2: Prepare G and H
    mat_G = np.column_stack((rbf_kernel(mat_A, np.transpose(mat_C), rbf_u), \
                             np.ones((mat_A.shape[0], 1), dtype=np.float64)))
    
    mat_H = np.column_stack((rbf_kernel(mat_B, np.transpose(mat_C), rbf_u), \
                             np.ones((mat_B.shape[0], 1), dtype=np.float64)))
    
    # Transpose of G & H
    mat_G_t = np.transpose(mat_G)
    mat_H_t = np.transpose(mat_H)
    
    # Select penalty parameters C1 & C2
    p_c1 = c1
    p_c2 = c2
    
    # regularization term
    reg_term = reg
    
    hyper_surf1 = None
    hyper_surf2 = None
    
    # Storing parameters of hypersurfaces
    u1, y1, u2, y2 = None, None, None, None
    
    # Determine parameters of hypersurfaces # Using SMW formula
    if mat_A.shape[0] < mat_B.shape[0]:
        
        y = (1 / reg_term) * (np.identity(mat_H.shape[1]) -  np.dot(np.dot(mat_H_t, np.linalg.inv((reg_term * \
             np.identity(mat_H.shape[0])) + np.dot(mat_H, mat_H_t))), mat_H))
        
        hyper_surf1 = np.dot(-1 * (y - np.dot(np.dot(np.dot(y, mat_G_t), np.linalg.inv(p_c1 * np.identity(mat_G.shape[0]) \
                                              + np.dot(np.dot(mat_G, y), \
                                                       mat_G_t))), np.dot(mat_G, y))), np.dot(mat_H_t, np.ones((mat_H.shape[0], 1))))
        
        hyper_surf2 = np.dot(p_c2 * (y - np.dot(np.dot(np.dot(y, mat_G_t), np.linalg.inv((np.identity(mat_G.shape[0]) / p_c2) \
                                              + np.dot(np.dot(mat_G, y), \
                                                       mat_G_t))), np.dot(mat_G, y))), np.dot(mat_G_t, np.ones((mat_G.shape[0], 1))))
        
        # Parameters of hypersurfaces
        u1 = hyper_surf1[:hyper_surf1.shape[0] - 1, :]
        y1 = hyper_surf1[-1, :]
        
        u2 = hyper_surf2[:hyper_surf2.shape[0] - 1, :]
        y2 = hyper_surf2[-1, :]
        
    else:
        
        z = (1 / reg_term) * (np.identity(mat_G.shape[1]) -  np.dot(np.dot(mat_G_t, np.linalg.inv(reg * \
             np.identity(mat_G.shape[0]) + np.dot(mat_G, mat_G_t))), mat_G))
        
        hyper_surf1 = np.dot(p_c1 * (z - np.dot(np.dot(np.dot(z, mat_H_t), np.linalg.inv((np.identity(mat_H.shape[0]) / p_c1) \
                                              + np.dot(np.dot(mat_H, z), \
                                                       mat_H_t))), np.dot(mat_H, z))), np.dot(mat_H_t, np.ones((mat_H.shape[0], 1))))
        
        hyper_surf2 = np.dot((z - np.dot(np.dot(np.dot(z, mat_H_t), np.linalg.inv(p_c2 * np.identity(mat_H.shape[0]) \
                                              + np.dot(np.dot(mat_H, z), \
                                                       mat_H_t))), np.dot(mat_H, z))), np.dot(mat_G_t, np.ones((mat_G.shape[0], 1))))
        
        # Parameters of hypersurfaces
        u1 = hyper_surf1[:hyper_surf1.shape[0] - 1, :]
        y1 = hyper_surf1[-1, :]
        
        u2 = hyper_surf2[:hyper_surf2.shape[0] - 1, :]
        y2 = hyper_surf2[-1, :]
        
    return u1, y1, u2, y2, mat_C_t

# Predict LS-TwinSVM - non-Linear case
def predict_nl_ls_tsvm(X_test, u1, y1, u2, y2, rbf_u, mat_C_t):
    
    # Step 4: Calculate prependicular distances for new data points
    new_data = X_test # Test data
    
    prepen_distance = np.zeros((new_data.shape[0], 2))
    
    for i in range(new_data.shape[0]):
        
        # Prependicular distance of data pint i from hypersurface2(class 1)
        prepen_distance[i, 1] = np.abs(np.dot(rbf_kernel(new_data[i, :], mat_C_t, rbf_u), u1) + y1)
        
        # Prependicular distance of data pint i from hypersurface1 (class -1)
        prepen_distance[i, 0] = np.abs(np.dot(rbf_kernel(new_data[i, :], mat_C_t, rbf_u), u2) + y2)
        
    # Step 5: Assign data points to class +1 or -1 based on distance from hyperplanes
    output = 2 * np.argmin(prepen_distance, axis=1) - 1

    return output

# non-Linear LS-TSVM - Cross validation
def nl_ls_tsvm(data_train, data_labels, k, c1, c2, rbf_u, reg):
    
    # K-Fold Cross validation, divide data into K subsets
    k_fold = KFold(k)    
        
    # Store result after each run
    mean_accuracy = []
    # Postive class
    mean_recall_p, mean_precision_p, mean_f1_p = [], [], []
    # Negative class
    mean_recall_n, mean_precision_n, mean_f1_n = [], [], []
    
    # Count elements of confusion matrix
    tp, tn, fp, fn = 0, 0, 0, 0
         
    # Train and test LSTSVM K times
    for train_index, test_index in k_fold.split(data_train):
        
        # Extract data based on index created by k_fold
        X_train = np.take(data_train, train_index, axis=0) 
        X_test = np.take(data_train, test_index, axis=0)
        
        X_train_label = np.take(data_labels, train_index, axis=0)
        X_test_label = np.take(data_labels, test_index, axis=0)
        
        # Obtain parameters of hypersurfaces                     
        hyper_s = train_nl_ls_tsvm(X_train, X_train_label, c1, c2, rbf_u, reg)
            
        # Predict
        output = predict_nl_ls_tsvm(X_test, hyper_s[0], hyper_s[1], hyper_s[2], hyper_s[3], \
                           rbf_u, hyper_s[4])
        
        accuracy_test = accuracy(X_test_label, output)
                              
        mean_accuracy.append(accuracy_test[4])
        # Positive cass
        mean_recall_p.append(accuracy_test[5])
        mean_precision_p.append(accuracy_test[6])
        mean_f1_p.append(accuracy_test[7])
        # Negative class    
        mean_recall_n.append(accuracy_test[8])
        mean_precision_n.append(accuracy_test[9])
        mean_f1_n.append(accuracy_test[10])
        
        # Count
        tp = tp + accuracy_test[0]
        tn = tn + accuracy_test[1]
        fp = fp + accuracy_test[2]
        fn = fn + accuracy_test[3]
            
    # m_a=0, m_r_p=1, m_p_p=2, m_f1_p=3, k=4, p_c1=5, p_c2=6, rbf=7, reg=8, u1=9, y1=10
    # u2= 11, y2=12, mat_c_t=13, m_r_n=14, m_p_n=15, m_f1_n=16, tp=17, tn=18, fp=19, fn=20    
    return mean_accuracy, mean_recall_p, mean_precision_p, mean_f1_p, k, c1, c2, \
           rbf_u, reg, hyper_s[0], hyper_s[1], hyper_s[2], hyper_s[3], hyper_s[4], mean_recall_n, mean_precision_n, mean_f1_n, \
           tp, tn, fp, fn

# Grid search for finding best parameters
def gs_ls_tsvm(train_data, labels, k, l_bound, u_bound, step, file_name):
       
    # Store 
    result_list = []
    
    # Create an excel file for 
    excel_write = pd.ExcelWriter(file_name, engine='xlsxwriter')
    
    # Search space
    search_array = np.arange(l_bound, u_bound, step)
    
    # Total number of search elements
    search_total = search_array.size * search_array.size
    
    # Count
    run = 1
           
    # Grid search for finding C1 & C2     
    for C1 in search_array:
        
        for C2 in search_array:
            
            # Parameters
            c_1 = 2 ** float(C1)
            c_2 = 2 ** float(C2)
                        
            # Save result after each run
            result = ls_tsvm(train_data, labels, k, c_1, c_2)

           # Format: accuracy, acc_std, recall_p, r_p_std, precision_p, p_p_std, f1_p, f1_p_std, recall_n, r_n_std,
           # precision_n, p_n_std, f1_n, f1_n_std, tp, tn, fp, fn c1, c2, k, run            
            result_list.append([np.mean(result[0]), np.std(result[0]), np.mean(result[1]), np.std(result[1]), np.mean(result[2]), \
                                 np.std(result[2]), np.mean(result[3]), np.std(result[3]), np.mean(result[11]), np.std(result[11]), \
                                 np.mean(result[12]), np.std(result[12]), np.mean(result[13]), np.std(result[13]), result[14], result[15], \
                                 result[16], result[17], result[5], result[6], result[4], run])
                    
            
            print("Run: %d | %d | C1: %.2f, C2: %.2f, Acc: %.2f" % (run, search_total, c_1, c_2, np.mean(result[0])))  
            run = run + 1

    
    # Create a panda data frame
    result_frame = pd.DataFrame(result_list, columns=['accuracy', 'acc_std', 'recall_p', 'r_p_std', 'precision_p', 'p_p_std', \
                                                      'f1_p', 'f1_p_std', 'recall_n', 'r_n_std', 'precision_n', 'p_n_std', 'f1_n',\
                                                      'f1_n_std', 'tp', 'tn', 'fp', 'fn', 'c1', 'c2', 'k_fold', 'run' ]) 
    # Write result to excel
    result_frame.to_excel(excel_write, sheet_name='Sheet1')
    
    excel_write.save()
    
    return result_frame

# Search for finding best parameters - non-linear        
def gs_nl_ls_tsvm(train_data, labels, k, l_bound, u_bound, step, rbf_lbound, rbf_ubound, step_u, file_name):
    
    # Store 
    result_list = []
    
    # Create an excel file for 
    excel_write = pd.ExcelWriter(file_name, engine='xlsxwriter')
    
    c_search = np.arange(l_bound, u_bound, step)
    rbf_search = np.arange(rbf_lbound, rbf_ubound, step_u)
    
    # Count
    run = 1
    
    # Search space
    search_space = c_search.size * c_search.size * rbf_search.size
               
    # Grid search for finding C1 & C2 & u    
    for C1 in c_search:
        
        for C2 in c_search:
            
            for u in rbf_search:
                
                # Parameters
                c_1 = 2 ** float(C1)
                c_2 = 2 ** float(C2)
                rbf_u = 2 ** float(u)
                 
                # Save result after each run
                result = nl_ls_tsvm(train_data, labels, k, c_1, c_2, rbf_u, 2**float(-7))
                
                result_list.append([np.mean(result[0]), np.std(result[0]), np.mean(result[1]), np.std(result[1]), np.mean(result[2]), \
                                     np.std(result[2]), np.mean(result[3]), np.std(result[3]), np.mean(result[14]), np.std(result[14]), \
                                     np.mean(result[15]), np.std(result[15]), np.mean(result[16]), np.std(result[16]), result[17], result[18], \
                                     result[19], result[20], result[5], result[6], result[7], result[8], result[4], run])
                
                print("Run: %d | %d | C1: %.2f, C2: %.2f, u: %.2f, Acc: %.2f" % (run, search_space, c_1, c_2,\
                                                                                 rbf_u, np.mean(result[0])))
                run = run + 1

                    
    # Create a panda data frame
    result_frame = pd.DataFrame(result_list, columns=['accuracy', 'acc_std', 'recall_p', 'r_p_std', 'precision_p', 'p_p_std', \
                                                      'f1_p', 'f1_p_std', 'recall_n', 'r_n_std', 'precision_n', 'p_n_std', 'f1_n',\
                                                      'f1_n_std', 'tp', 'tn', 'fp', 'fn', 'c1', 'c2', 'rbf_u', 'reg', 'k_fold', 'run' ]) 
    # Write result to excel
    result_frame.to_excel(excel_write, sheet_name='Sheet1')
    
    excel_write.save()                    


# Plotting decision boundary - LS-TSVM
def ls_tsvm_db(data, c1, c2, knn, gamma, file_name, cl_type='non-linear'):
    
    # color
    col2 = 'dimgray'
    
    # Split train data into separate class
    X_t_c1 = data[0][data[1] == 1]
    X_t_c2 = data[0][data[1] == -1]
    
    # Create a meshgrid
    xx, yy = make_mesh(data[0][:, 0], data[0][:, 1])
    
    # Datapoints in inputspace
    data_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Get index of margin points
    mp_idx = det_margin(data[0], data[1], knn) == 1
    
    if cl_type == 'linear':
        
        pass
    
    else:
        
        # Train
        model = train_nl_ls_tsvm(data[0], data[1], c1, c2, gamma, 2**float(-7))
        
        # Training error
        train_error = predict_nl_ls_tsvm(data[0], model[0], model[1], model[2], model[3], \
                                 gamma, model[4])
        
        print("Train-Error: %.2f " % (accuracy_score(data[1], train_error) *100))
        
        # Predict class of datapoints
        z = predict_nl_ls_tsvm(data_points, model[0], model[1], model[2], model[3], \
                                 gamma, model[4])
        
    z = z.reshape(xx.shape)
    
    fig = plt.figure(1)
    
    axes = plt.gca()
        
    # plot decision boundary
    lev = [-1, 0]
    plt.contourf(xx, yy, z, levels=lev, colors=col2, alpha=0.8)
    
    # Plot entire datapoints
    #plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1], s=30, cmap=plt.cm.Paired)
    
    # Plot Training data
    plt.scatter(X_t_c1[:, 0], X_t_c1[:, 1], marker='o', s=(35,), cmap=plt.cm.Paired)
    plt.scatter(X_t_c2[:, 0], X_t_c2[:, 1], marker='^', s=(35,), cmap=plt.cm.Paired)
    
    
    # Plot margin points
    plt.scatter(data[0][mp_idx, 0], data[0][mp_idx, 1], c='k', marker='x', s=(40,), \
			   edgecolors='none', label='Margin points')
    
    # Limit axis values
    axes.set_xlim(xx.min(), xx.max())
    axes.set_ylim(yy.min(), yy.max())
    
    # Show!
    plt.legend()
    plt.show()
    
    # Figure saved
    fig.savefig(file_name, format='png', dpi=500)
    
    # Clear plot
    #plt.clf()
    
    

if __name__ == '__main__':

    # Test module
    # Store intial time
    start_time = time.time()
    #
    ## Read data
    data = read_data('Dataset\\Synthetic\\check.csv')
    
    # Parameter
    c1 = 2 ** -9
    c2 = c1
    gamma = 2 ** -1
    k = 10
    
    ls_tsvm_db(data, c1, c2, k,  gamma, 'lstsvm-check-113.png')
    
    #
    ## Split data
    ##X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.33, random_state=42)
    #
    ## Train
    ##model = train_nl_ls_tsvm(X_train, y_train, 0.4 , 0.7, 2**-4, 2**-6)
    ## Test
    ##test = predict_nl_ls_tsvm(X_test, model[0], model[1], model[2], model[3], 2**-4, model[4])
    #acc = nl_ls_tsvm(data[0], data[1], 10, 0.4, 0.7, 2**-4, 2**-6)
    #
    ## Model accuracy
    #print("Accuracy: %.2f" % np.mean(acc[0]))
    #        
    #
    ## Finish time
    print("Finished %.2f Seconds" % (time.time() - start_time))