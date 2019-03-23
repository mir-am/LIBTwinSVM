#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Developers: Mir, A. and Mahdi Rahbar
# Version: 0.1 - 2019-03-20
# License:




def eval_metrics(y_true, y_pred):

    """
    It computes common evaluation metrics such as Accuracy, Recall, Precision,
    F1-measure, and elements of the confusion matrix.
    
    Parameters
    ----------
    y_true : array-like 
        Target values of samples.
        
    y_pred : array-like 
        Predicted class lables.
        
    Returns
    -------
    tp : int
        True positive.
        
    tn : int
        True negative.
        
    fp : int
        False positive.
        
    fn : int
        False negative.
        
    accuracy : float
        Overall accuracy of the model.
        
    recall_p : float
        Recall of positive class.
        
    precision_p : float
        Precision of positive class.
        
    f1_p : float
        F1-measure of positive class.
        
    recall_n : float
        Recall of negative class.
        
    precision_n : float
        Precision of negative class.
        
    f1_n : float
        F1-measure of negative class.
    """        
    
    # Elements of confusion matrix
    tp, tn, fp, fn = 0, 0, 0, 0
    
    for i in range(y_true.shape[0]):
        
        # True positive 
        if y_true[i] == 1 and y_pred[i] == 1:
            
            tp = tp + 1
        
        # True negative 
        elif y_true[i] == -1 and y_pred[i] == -1:
            
            tn = tn + 1
        
        # False positive
        elif y_true[i] == -1 and y_pred[i] == 1:
            
            fp = fp + 1
        
        # False negative
        elif y_true[i] == 1 and y_pred[i] == -1:
            
            fn = fn + 1
            
    # Compute total positives and negatives
    positives = tp + fp
    negatives = tn + fn

    # Initialize
    accuracy = 0
    # Positive class
    recall_p = 0
    precision_p = 0
    f1_p = 0
    # Negative class
    recall_n = 0
    precision_n = 0
    f1_n = 0
    
    try:
        
        accuracy = (tp + tn) / (positives + negatives)
        # Positive class
        recall_p = tp / (tp + fn)
        precision_p = tp / (tp + fp)
        f1_p = (2 * recall_p * precision_p) / (precision_p + recall_p)
        
        # Negative class
        recall_n = tn / (tn + fp)
        precision_n = tn / (tn + fn)
        f1_n = (2 * recall_n * precision_n) / (precision_n + recall_n)
        
    except ZeroDivisionError:
        
        pass # Continue if division by zero occured


    return tp, tn, fp, fn, accuracy * 100 , recall_p * 100, precision_p * 100, f1_p * 100, \
           recall_n * 100, precision_n * 100, f1_n * 100