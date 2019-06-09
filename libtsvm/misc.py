# -*- coding: utf-8 -*-

# LIBTwinSVM: A Library for Twin Support Vector Machines
# Developers: Mir, A. and Mahdi Rahbar
# License: GNU General Public License v3.0

"""
In this module, several miscellaneous functions are defined for using
in other module, such as date time formatting.
"""

from os.path import isdir


def time_fmt(t_delta):
    """
    It converts datetime objects to formatted string.
    
    Parameters
    ----------
    t_delta : object
        The difference between two dates or time.
        
    Returns
    -------
    str
        A readable formatted-datetime string.
    """
   
    hours, remainder = divmod(t_delta, 3600)
    minutes, seconds = divmod(remainder, 60)
   
    return '%d:%02d:%02d' % (hours, minutes, seconds)


def progress_bar_gs(iteration, total, e_time, accuracy, best_acc, prefix='', \
                    suffix='', decimals=1, length=25, fill='#'):
    """
    It shows a customizable progress bar for grid search.
    
    Parameters
    ----------
    iteration : int
        Current iteration.
    
    total : int
        Maximumn number of iterations.
        
    e_time : str
        Elapsed time.

    accuracy : tuple
        The accuracy and its std at current iteration (acc, std).
        
    best_acc : tuple 
        The best accuracy and its std that were obtained at current iteration
        (best_acc, std).
        
    prefix : str, optional (default='') 
        Prefix string.
        
    suffix : str, optional (default='') 
        Suffix string.
        
    decimals : int, optinal (default=1)
        Number of decimal places for percentage of completion.
        
    length : int, optional (default=25) 
        Character length of the progress bar.
    
    fill : str, optional (default='#') 
        Bar fill character.
    """ 
    
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    fill_length = int(length * iteration // total)
    bar = fill * fill_length + '-' * (length - fill_length)

    output = "\r%sB-Acc:%.2f+-%.2f|Acc:%.2f+-%.2f |%s| %s%% %sElapsed:%s"
    print(output % (prefix, best_acc[0], best_acc[1], accuracy[0], accuracy[1], \
                    bar, percent, suffix, e_time), end='\r')

    if iteration == total:
        print()


#def validate_step_size(kernel, C1_range, C2_range, u_range, step_size):
#    """
#    Checks whether step size for generating search elements are valid or not.
#    
#    Parameters
#    ----------
#    kernel : str, {'linear', 'RBF'}
#        Type of the kernel function which is either 'linear' or 'RBF'.
#    
#    C1_range : tuple
#        Lower and upper bound for C1 penalty parameter.
#    
#    C2_range : tuple
#        Lower and upper bound for C2 penalty parameter.
#        
#    u_range : tuple
#        Lower and upper bound for gamma parameter.
#          
#    step_size : int, optinal (default=1)
#        Step size to increase power of 2. 
#    
#    Returns
#    -------
#    boolean
#        Whether step size is valid or not.
#    """
#    
#    return (step_size < abs(C1_range[1] - C1_range[0]) and step_size < \
#    abs(C2_range[1] - C2_range[0])) and (step_size < abs(u_range[1] - \
#       u_range[0]) if kernel == 'RBF' else True)
    
    
def validate_path(path):
    """
    Checks whether the specified path exists on a system or not.
    
    path : str
        The specified path.
    """
    
    return isdir(path)
    
    
    
    
    