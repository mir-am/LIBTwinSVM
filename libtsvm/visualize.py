# -*- coding: utf-8 -*-

# Developers: Mir, A.
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0

from PyQt5.QtCore import QObject


class VisualThread(QObject):
    """
    It runs the visualization in a separate thread.
    
    Parameters
    ----------
    usr_input : object
        An instance of :class:`UserInput` class which holds the user input.
    """
    
    def __init__(self, usr_input):
        
        super(VisualThread, self).__init__()
        self.usr_input = usr_input
        
        
    