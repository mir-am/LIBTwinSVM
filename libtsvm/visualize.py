# -*- coding: utf-8 -*-

# Developers: Mir, A.
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0

from os.path import join
from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


def hyperplane_eq(w, b):
    """
    It returns the slope and intercept of a hyperplane in 2-dimensional space.
    
    Parameters
    ----------
    w : array-like, (2,)
        2-dimensional weight vector.
        
    b : float
        Bias term.
        
    Returns
    -------
    slope : float
        Slope of the line.
    
    intercept : float
        Intercept of the line.
    """
    
    slope = -1 * (w[0] / w[1])
    intercept = -1 * (b / w[1]) 

    return slope, intercept


def make_mesh(x, y, h=0.02):
    """
    Creates a mesh grid of points.
    
    Parameters
    ----------
    x : array-like
        First dimension of training samples.
        
    y : array-like
        Second dimension of training samples.
    
    h : float
        Step size.
    
    Returns
    -------
    xx : array-like
        x-coordinates of data points.
        
    yy : array-like
        y-cooridinates of data points.
    """
    
    step = 0.5
    
    x_min, x_max = x.min() - step, x.max() + step
    y_min, y_max = y.min() - step, y.max() + step
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    return xx, yy


def bin_plot(estimator, X, y):
    """
    It plots hyperplanes of a binary TSVM-based estimator.
    
    Parameters
    ----------
    estimator : object
        A TSVM-based estimator.
     
    X : array-like, shape (n_samples, 2)
        Training feature vectors, where n_samples is the number of samples
        and number of features must be equal to 2.
        
    y : array-like, shape(n_samples,)
        Target values or class labels.
    """
 
    estimator.fit(X, y)

    # Line Equation hyperplane 1
    slope1, intercept1 = hyperplane_eq(estimator.w1, estimator.b1)  

    # Line Equation hyperplane 2
    slope2, intercept2 = hyperplane_eq(estimator.w2, estimator.b2) 
    
    fig = plt.gcf()
    axes = plt.gca()
    
    # Plot Training data
    plt.scatter(X[:, 0], X[:, 1], marker='^', color='red',
                label='Samples of class +1') #cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], marker='s', color='blue',
                label='Samples of class -1') #cmap=plt.cm.Paired)

    x_range = axes.get_xlim() # X-axis range
   
    # Min and Max of feature X1 and creating X values for creating line
    xx_1 = np.linspace(x_range[0], x_range[1])

    yy_1 = slope1 * xx_1 + intercept1
    yy_2 = slope2 * xx_1 + intercept2  

    # Plot two hyperplanes
    plt.plot(xx_1, yy_1, 'k-', label='Hyperplane') # Hyperplane of class 1
    plt.plot(xx_1, yy_2, 'k-') # Hyperplane of class 2        
   
    plt.ylim(-0.7, 8)
    plt.xlim(x_range[0], x_range[1])
   
    plt.legend()
    plt.show()



class VisualThread(QObject):
    """
    It runs the visualization in a separate thread.
    
    Parameters
    ----------
    usr_input : object
        An instance of :class:`UserInput` class which holds the user input.
    """
    
    sig_update_plt = pyqtSignal(object)
    
    def __init__(self, usr_input):
        
        super(VisualThread, self).__init__()
        self.usr_input = usr_input
        
        # Matplotlib initialization
        self.fig = Figure()
        self.canvas = FigureCanvas(self.fig)
        
    @pyqtSlot()
    def plot(self):
        """
        It plots the figure based on the user's input.
        """
        
        print("Plotting....")
        
        self.clf_obj = self.usr_input.get_selected_clf()
        
        if self.usr_input.class_type == 'binary':
            
            self.clf_obj.set_params(**self.usr_input.get_clf_params())
            
            #bin_plot(clf_obj, self.usr_input.X_train, self.usr_input.y_train)
            
            
        elif self.usr_input.class_type == 'multiclass':
            
            pass
        
        if self.usr_input.kernel_type == 'linear':
            
            self.fig_linear()
            
        elif self.usr_input.kernel_type == 'RBF':
            
            self.fig_non_linear()
        
        self.sig_update_plt.emit(self.canvas)
        
    @pyqtSlot()    
    def fig_linear(self):
        """
        Linear decision boundary for TSVM-based classifiers.
        """
        
        X_c1 = self.usr_input.X_train[self.usr_input.y_train == 1]
        X_c2 = self.usr_input.X_train[self.usr_input.y_train == -1]
        
        self.clf_obj.fit(self.usr_input.X_train, self.usr_input.y_train)

        # Line Equation hyperplane 1
        slope1, intercept1 = hyperplane_eq(self.clf_obj.w1, self.clf_obj.b1)  
    
        # Line Equation hyperplane 2
        slope2, intercept2 = hyperplane_eq(self.clf_obj.w2, self.clf_obj.b2) 
        
        ax = self.fig.add_subplot(111)
        ax.clear()
        
        # Plot Training data
        ax.scatter(X_c1[:, 0], X_c1[:, 1], marker='^', color='red',
                    label='Samples of class +1') #cmap=plt.cm.Paired)
        ax.scatter(X_c2[:, 0], X_c2[:, 1], marker='s', color='blue',
                    label='Samples of class -1') #cmap=plt.cm.Paired)
    
        x_range = ax.get_xlim() # X-axis range
        y_range = ax.get_ylim()
       
        # Min and Max of feature X1 and creating X values for creating line
        xx_1 = np.linspace(x_range[0], x_range[1])
    
        yy_1 = slope1 * xx_1 + intercept1
        yy_2 = slope2 * xx_1 + intercept2  
        
        # Plot two hyperplanes
        ax.plot(xx_1, yy_1, 'k-', label='Hyperplane') # Hyperplane of class 1
        ax.plot(xx_1, yy_2, 'k-') # Hyperplane of class 2        
       
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_xlim(x_range[0], x_range[1])
        
        ax.legend()
        
        if self.usr_input.fig_save:
            
            self.fig.savefig(join(self.usr_input.fig_save_path, 
                             self.usr_input.get_fig_name() + '.png'),
                             format='png', dpi=self.usr_input.fig_dpi)
        
        self.canvas.draw()
    
    @pyqtSlot()
    def fig_non_linear(self):
        """
        Non-linear decision boundary for TSVM-based classifiers.
        """
        
        xx, yy = make_mesh(self.usr_input.X_train[:, 0], self.usr_input.X_train[:, 1])
        
        # Datapoints in inputspace
        data_points = np.c_[xx.ravel(), yy.ravel()]
        
        self.clf_obj.fit(self.usr_input.X_train, self.usr_input.y_train)
        
        z = self.clf_obj.predict(data_points)
        z = z.reshape(xx.shape)
        
        # Plot
        ax = self.fig.add_subplot(111)
        
        ax.contourf(xx, yy, z, levels=[-1, 0], colors='dimgray', alpha=0.8)
        
        # Split training points into separate classes
        X_c1 = self.usr_input.X_train[self.usr_input.y_train == 1]
        X_c2 = self.usr_input.X_train[self.usr_input.y_train == -1]
        
        # plot training samples of both classes
        ax.scatter(X_c1[:, 0], X_c1[:, 1], marker='^', s=(50,), c='b', cmap=plt.cm.coolwarm)
        ax.scatter(X_c2[:, 0], X_c2[:, 1], marker='o', s=(50,), c='r', cmap=plt.cm.coolwarm)
        
        # Limit axis values
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        
        if self.usr_input.fig_save:
            
            self.fig.savefig(join(self.usr_input.fig_save_path,
                                  self.usr_input.get_fig_name() + '.png'),
                                  format='png', dpi=self.usr_input.fig_dpi)
        
        self.canvas.draw()
        
        