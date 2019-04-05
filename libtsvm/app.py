# -*- coding: utf-8 -*-

# Developers: Mir, A. and Mahdi Rahbar
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QGridLayout, QTableWidgetItem
from sklearn.utils.multiclass import type_of_target
from libtsvm.ui import view
from libtsvm.model import UserInput
from libtsvm.preprocess import DataReader
import sys


class LIBTwinSVMApp(view.Ui_MainWindow, QMainWindow):
    
    def __init__(self):
        
        super(LIBTwinSVMApp, self).__init__()
        
        self.setupUi(self)
        self.user_in = UserInput() # Stores user's data and input
        self.init_GUI()
        
    def init_GUI(self):
        """
        Initialize the GUI of application
        """
        
        # Buttons
        self.open_btn.clicked.connect(self.get_data_path)
        self.load_btn.clicked.connect(self.load_data)
        self.run_btn.clicked.connect(self.run_gridsearch)
        
        # Enable widgets based on the user selection
        self.rect_kernel_rbtn.toggled.connect(self.rect_kernel_percent.setEnabled)
        self.rect_kernel_rbtn.toggled.connect(lambda checked: not checked and \
                                self.rect_kernel_percent.setEnabled(False))
        
        self.rect_kernel_rbtn.toggled.connect(self.u_lbound.setEnabled)
        self.rect_kernel_rbtn.toggled.connect(self.u_ubound.setEnabled)
        
        self.rbf_kernel_rbtn.toggled.connect(self.u_lbound.setEnabled)
        self.rbf_kernel_rbtn.toggled.connect(self.u_ubound.setEnabled)
        
        self.tt_split_rbtn.toggled.connect(self.tt_percentage.setEnabled)
        self.tt_split_rbtn.toggled.connect(lambda checked: not checked and \
                                self.tt_percentage.setEnabled(False))
        
        self.cv_rbtn.toggled.connect(self.cv_folds.setEnabled)
        
    def get_data_path(self):
        """is
        gets the dataset path from a user.
        """
        
        data_filename, _ = QFileDialog.getOpenFileName(self, "Import dataset",
                                                       "", "CSV files (*.csv)")
        
        if data_filename:
            
            self.path_box.setText(data_filename)
            self.read_box.setEnabled(True)
            
            # Enable widgets in Read group box
            for w in self.read_box.children():
                
                if not isinstance(w, QGridLayout):
                    
                    w.setEnabled(True)
         
    def load_data(self):
        """
        Loads a dataset
        """
        
        self.data_reader = DataReader(self.path_box.text(), self.sep_char_box.text(),
                                 True if self.header_check.isChecked() else False)
        
        self.data_reader.load_data(True if self.shuffle_box.isChecked() else False,
                              True if self.normalize_box.isChecked() else False)
        
        self.user_in.X_train, self.user_in.y_train, \
        self.user_in.data_filename = self.data_reader.get_data()
        
        print(self.user_in.X_train)
        print(self.user_in.y_train)
        # TODO: Handle exception when it fails to load dataset and show the error
        # in message box
        load_data_dialog(True)
        
        self.update_data_info(self.data_reader.hdr_names)
        self.enable_classify()
        
    def update_data_info(self, hdr_name):
        """
        Updates the data information like no. features, no. samples and etc.
        
        Parameters
        ----------
        hdr_name : list
            Header names of dataset.
        """
        
        data_info = self.data_reader.get_data_info()
        
        self.no_samples.setText(str(data_info.no_samples))
        self.no_features.setText(str(data_info.no_features))
        self.no_classes.setText(str(data_info.no_class))
        
        if len(data_info.header_names) != 0:
            
            self.feature_table.setRowCount(data_info.no_features)
                
            for i, hdr_name in enumerate(data_info.header_names):

                self.feature_table.setItem(i, 0, QTableWidgetItem(hdr_name))
                
    def enable_classify(self):
        """
        Enables classify tab after a dataset is successfully loaded.
        """
        
        self.user_in.class_type = type_of_target(self.user_in.y_train)
        
        print(self.user_in.class_type)
        
        enable_mc = True if self.user_in.class_type == 'multiclass' else False
            
        self.ova_rbtn.setEnabled(enable_mc)
        self.ovo_rbtn.setEnabled(enable_mc)
        
        # Enable gird search buttonsl
        self.run_btn.setEnabled(True)
        
    def run_gridsearch(self):
        """
        Runs grid search based on user choices.
        """
        
        if self.STSVM_rbtn.isChecked():
            
            self.user_in.clf_type = 'tsvm'
            
        elif self.LSTSVM_rbtn.isChecked():
            
            self.user_in.clf_type = 'lstsvm'
            
        if self.user_in.class_type == 'multiclass':
            
                if self.ova_rbtn.isChecked():
                    
                    self.user_in.mc_scheme = 'ova'
                    
                elif self.ovo_rbtn.isChecked():
                    
                    self.user_in.mc_scheme = 'ovo'
                    
        if self.lin_kernel_rbtn.isChecked():
            
            self.user_in.kernel_type = 'linear'
            
        elif self.rbf_kernel_rbtn.isChecked():
            
            self.user_in.kernel_type = 'RBF'
            
        elif self.rect_kernel_rbtn.isChecked():
            
            self.user_in.kernel_type = 'RBF'
            self.user_in.rect_kernel = self.rect_kernel_percent.value() / 100
            
        if self.cv_rbtn.isChecked():
            
            self.user_in.test_method_tuple = ('CV', self.cv_folds.value())
            
        elif self.tt_split_rbtn.isChecked():
            
            self.user_in.test_method_tuple = ('t_t_split',
                                              self.tt_percentage.value())
        
        self.user_in.C1_range = range(self.C1_lbound.value(),
                                      self.C1_ubound.value() + 1)
        self.user_in.C2_range = range(self.C2_lbound.value(),
                                      self.C2_ubound.value() + 1)
        
        if self.rbf_kernel_rbtn.isChecked() or self.rect_kernel_rbtn.isChecked():
        
            self.user_in.u_range = range(self.u_lbound.value(),
                                         self.u_ubound.value() + 1)
            
        
        

def load_data_dialog(status, error_msg=''):
    """
    A message box that shows whether dataset is loaded successfully or not.
    """
    
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    
    msg.setWindowTitle("Data Status")
    msg.setText("Loaded the dataset successfully!")
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()
  
        
def main():
    
   app = QApplication(sys.argv)
   libtsvm_app = LIBTwinSVMApp()
   libtsvm_app.show()
   sys.exit(app.exec_())
 
