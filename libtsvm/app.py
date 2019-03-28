# -*- coding: utf-8 -*-

# Developers: Mir, A. and Mahdi Rahbar
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0

from PyQt5 import QtWidgets
from libtsvm.ui import view
import sys


class LIBTwinSVMApp(view.Ui_MainWindow, QtWidgets.QMainWindow):
    
    def __init__(self):
        
        super(LIBTwinSVMApp, self).__init__()
        self.setupUi(self)
     
        
def main():
    
   app = QtWidgets.QApplication(sys.argv)
   libtsvm_app = LIBTwinSVMApp()
   libtsvm_app.show()
   sys.exit(app.exec_())
 
