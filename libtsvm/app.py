# -*- coding: utf-8 -*-

# Developers: Mir, A. and Mahdi Rahbar
# Version: 0.1 - 2019-03-20
# License: GNU General Public License v3.0

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QGridLayout, QTableWidgetItem, QDialog, QWidget
from PyQt5.QtCore import QThread, pyqtSlot
from sklearn.utils.multiclass import type_of_target
from libtsvm.ui import view
from libtsvm.ui import confirm_diag
from libtsvm.model import UserInput
from libtsvm.preprocess import DataReader
from libtsvm.model_selection import ThreadGS
from datetime import datetime
import numpy as np
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
        
        # Threads
        self.__threads = []
        
        # Buttons
        self.open_btn.clicked.connect(self.get_data_path)
        self.load_btn.clicked.connect(self.load_data)
        self.run_btn.clicked.connect(self.gather_usr_input)
        self.save_res_btn.clicked.connect(self.get_save_path)
        
        # Quit main window
        self.actionExit.triggered.connect(self.closeEvent)
        
        # Checkbox
        self.log_file_chk.clicked.connect(self.log_file_info)
        
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
        """
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
                    
    def get_save_path(self):
        """
        Gets save path for storing classification results.
        """
        
        results_path = QFileDialog.getExistingDirectory(self, "Select results' directory")
         
        if results_path:
            
            self.save_path_box.setText(results_path)
        
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
        
        print(self.user_in.X_train.shape)
        print(self.user_in.y_train.shape)
        # TODO: Handle exception when it fails to load dataset and show the error
        # in message box
        show_info_dialog("Data Status", "Loaded the dataset successfully.")
        
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
        
    def gather_usr_input(self):
        """
        It gathers all the input variables that set by a user.
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
        
        self.user_in.C1_range = (self.C1_lbound.value(), self.C1_ubound.value())
        self.user_in.C2_range = (self.C2_lbound.value(), self.C2_ubound.value())
        
        if self.rbf_kernel_rbtn.isChecked() or self.rect_kernel_rbtn.isChecked():
        
            self.user_in.u_range = (self.u_lbound.value(), self.u_ubound.value())
            
        self.user_in.step_size = self.step_box_val.value()

        # TODO: Check whether the path exists or not.    
        self.user_in.result_path = self.save_path_box.text()
                            
        # All the input variables are inserted.
        self.user_in.input_complete = True
        ConfrimDialog(self.user_in.get_current_selection(), self.run_gs_thread)
        
    def validate_usr_input(self):
        """
        Checks whether user's inputs such as save path, etc are valid or not.
        """
        
        pass
        
    def log_file_info(self):
        """
        Gets logging file state and also warns user about its usage.
        """
        
        if self.log_file_chk.isChecked():
            
            self.user_in.log_file = True
            
            show_info_dialog("Logging Results", "Note that logging classification"
                             " results may reduce the speed of the grid search "
                             "process due to the I/O operations. However, "
                             "results will be not lost in case of power failure, etc.")
            
        else:
            
            self.user_in.log_file = False
            
    def closeEvent(self, event):
        """
        Shows a message box to confirm exiting the program.
        """
        
        close_win = QMessageBox()
        close_win.setIcon(QMessageBox.Warning)
        
        close_win.setWindowTitle("Exit")
        close_win.setText("Are you sure you want to quit the program?")
        close_win.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        quit_state = close_win.exec_()
        
        if quit_state == QMessageBox.Yes:
            
            event.accept()
            
        else:
            
            event.ignore()
        
    def run_gs_thread(self):
        """
        Runs grid search based on user choices.
        """
        
        t = QThread()
        gs_t = ThreadGS(self.user_in)
        self.__threads.append((t, gs_t))
        
        gs_t.moveToThread(t)
        t.started.connect(gs_t.initialize)
        
        # Connect signals
        gs_t.sig_pbar_set.connect(self.set_pbar_range)
        gs_t.sig_gs_info_set.connect(self.update_gs_info)
        
        t.start()
     
    @pyqtSlot(int)
    def set_pbar_range(self, pbar_rng):
        """
        Sets range of the progress bar by the grid search thread.
        
        Parameters
        ----------
        pbar_rng : int
            range of the progress bar.
        """
        
        self.gs_progress_bar.setRange(0, pbar_rng)
    
    @pyqtSlot(int, str, str, str)    
    def update_gs_info(self, pbar_val, curr_acc, best_acc, elapsed_t):
        """
        Updates current value of the progress bar, current accuracy, and best
        accuract, elapsed time.
        
        Parameters
        ----------
        pbar_val : int
            value of progress bar.
            
        curr_acc
        """
        
        self.gs_progress_bar.setValue(pbar_val)
        self.acc.setText(curr_acc)
        self.best_acc.setText(best_acc)
        self.elapsed_time.setText(elapsed_t)
        
        
def show_info_dialog(title, msg_txt):
    """
    A message box that shows extra information to users.
    
    Parameters
    ----------
    title : str
        Title of the message box.
        
    msg_txt : str
        Message that shows information to users.
    """
    
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    
    msg.setWindowTitle(title)
    msg.setText(msg_txt)
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()
    
    
class ConfrimDialog(confirm_diag.Ui_confirm_diag, QDialog):
    """
    It shows a message box to confirm the currect user selection for running
    grid search.
    
    Parameters
    ----------
    usr_input : str
        Selected options by a user.
    """
    
    def __init__(self, usr_input, ok_func):
        
        super(ConfrimDialog, self).__init__()
        
        self.setupUi(self)
        
        self.extra_info.setText(usr_input)
        self.confirm_btn.accepted.connect(ok_func)
        
        self.show()
        self.exec_()
        
        

        
#def confirm_dialog(usr_input):
#    """
#    It shows a message box to confirm the currect user selection for running
#    grid search.
#    
#    Parameters
#    ----------
#    usr_input : str
#        Selected options by a user.
#    """
#  
#    msg = QMessageBox()
#    msg.setIcon(QMessageBox.Question)
#    
#    msg.setWindowTitle("Confirmation")
#    msg.setText("Do you confirm the following settings for running the classifier?")
#    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
#    msg.setDetailedText(usr_input)
#    msg.exec_()
    
  
def main():
    
   app = QApplication(sys.argv)
   libtsvm_app = LIBTwinSVMApp()
   libtsvm_app.show()
   sys.exit(app.exec_())
 
