# -*- coding: utf-8 -*-

# Developers: Mir, A.
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0

"""
In this module, several miscellaneous functions are defined for using
in other module, such as date time formatting.
"""


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
