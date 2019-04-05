#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This module is created for testing different components of the library.
# It may be removed in the near future.

#from libtsvm.preprocess import read_data
#from libtsvm.estimators import LSTSVM, TSVM
#from libtsvm.mc_scheme import OneVsOneClassifier
#from libtsvm.model_selection import eval_metrics
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import accuracy_score
from libtsvm.app import main

if __name__ == '__main__':
    
#    X, y, filename = read_data('./dataset/iris.csv')
#      
#    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#    #
#    tsvm_model = TSVM('linear', 1, 1, 0.5)
#    #
#    #lstsvm_model.fit(x_train, y_train)
#    #pred = lstsvm_model.predict(x_test)
#    #
#    #test = eval_metrics(y_test, pred)
#    
#    
#    ## Test OVO classifier
#    
#    ovo_model = OneVsOneClassifier(tsvm_model)
#    ovo_model.fit(x_train, y_train)
#    pred = ovo_model.predict(x_test)
#    
#    print("Accuracy: %.2f" % (accuracy_score(y_test, pred)))

    main()
    