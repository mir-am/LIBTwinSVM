# -*- coding: utf-8 -*-

# LIBTwinSVM: A Library for Twin Support Vector Machines
# Developers: Mir, A. and Mahdi Rahbar
# Version: 0.1.0 - 2019-03-20
# License: GNU General Public License v3.0

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox, QGridLayout, QTableWidgetItem, QDialog, QWidget, QActionGroup
from PyQt5.QtCore import QThread, pyqtSlot
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from sklearn.utils.multiclass import type_of_target
from libtsvm.ui import view, confirm_diag, about_diag
from libtsvm.model import UserInput
from libtsvm.preprocess import DataReader
from libtsvm.model_selection import ThreadGS
from libtsvm.model_eval import load_model, ModelThread
from libtsvm.visualize import VisualThread
from libtsvm.misc import validate_path
from datetime import datetime
import numpy as np
import sys
import webbrowser


class LIBTwinSVMApp(view.Ui_MainWindow, QMainWindow):
    
    def __init__(self):
        
        super(LIBTwinSVMApp, self).__init__()
        
        self.setupUi(self)
        
        self.user_in = UserInput() # Stores user's data and input
        self.data_info = None
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
        self.vis_plot_btn.clicked.connect(self.plot_figure)
        self.vis_select_btn.clicked.connect(self.get_save_path_fig)
        self.model_path_btn.clicked.connect(self.get_model_path)
        self.model_load_btn.clicked.connect(self.load_model_info)
        self.model_eval_btn.clicked.connect(self.eval_model_test)
        self.stop_btn.clicked.connect(self.stop_gs_thread)
        
        # Checkbox
        self.log_file_chk.clicked.connect(self.log_file_info)
        
        self.model_pred_chk.clicked.connect(self.get_save_pred_path)
#        self.model_pred_chk.toggled.connect(lambda checked: not checked and \
#                                            self.get_save_pred_path(False))
        
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
        
        self.vis_save_fig_chk.toggled.connect(self.vis_save_val.setEnabled)
        self.vis_save_fig_chk.toggled.connect(self.vis_select_btn.setEnabled)
        
        self.vis_lin_rbtn.toggled.connect(self.vis_db_chk.setEnabled)
        self.vis_lin_rbtn.toggled.connect(lambda checked: not checked and \
                                self.vis_db_chk.setEnabled(False))
        
        self.cv_rbtn.toggled.connect(self.cv_folds.setEnabled)
        
        
        # Device menu
        #self.ag = QActionGroup(self.device_menu, exclusive=True)
        #self.ag.addAction(self.cpu_chk_dev)
        #self.ag.addAction(self.gpu_chk_dev)
        #self.ag.triggered.connect(self.onTriggered)
        
        # Help menu
        self.actionDocumentation.triggered.connect(lambda: webbrowser.open("https://libtwinsvm.readthedocs.io/en/latest/"))
        self.actionAbout.triggered.connect(lambda: AboutDialog())
        
    def get_data_path(self):
        """
        Gets the dataset path from a user.
        """
        
        data_filename, _ = QFileDialog.getOpenFileName(self, "Import dataset",
                           "", "CSV files (*.csv);; LIBSVM datafiles (*.libsvm)")
        
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
        Loads a dataset.
        """
        
        self.data_reader = DataReader(self.path_box.text(), self.sep_char_box.text(),
                                 True if self.header_check.isChecked() else False)
        
        self.data_reader.load_data(True if self.shuffle_box.isChecked() else False,
                              True if self.normalize_box.isChecked() else False)
        
        self.user_in.X_train, self.user_in.y_train, \
        self.user_in.data_filename = self.data_reader.get_data()
        
        # TODO: Handle exception when it fails to load dataset and show the error
        # in message box
        show_dialog("Data Status", "Loaded the dataset successfully.", 
                         QMessageBox.Information)
        
        self.update_data_info()
        
        self.user_in.class_type = type_of_target(self.user_in.y_train)
        
        # Enable other tabs' functionalities
        self.enable_classify()
        self.enable_visualize()
        self.enable_model()
        
    def update_data_info(self):
        """
        Updates the data information like no. features, no. samples and etc.
        
        Parameters
        ----------
        hdr_name : list
            Header names of dataset.
        """
        
        self.data_info = self.data_reader.get_data_info()
        
        self.no_samples.setText(str(self.data_info.no_samples))
        self.no_features.setText(str(self.data_info.no_features))
        self.no_classes.setText(str(self.data_info.no_class))
        
        print(self.data_info.header_names)
        
        if len(self.data_info.header_names) != 0:
            
            self.feature_table.setRowCount(self.data_info.no_features)
                
            for i, hdr_name in enumerate(self.data_info.header_names):

                self.feature_table.setItem(i, 0, QTableWidgetItem(hdr_name))
                
    def enable_classify(self):
        """
        Enables classify tab after a dataset is successfully loaded.
        """
        
        enable_mc = True if self.user_in.class_type == 'multiclass' else False
            
        self.ova_rbtn.setEnabled(enable_mc)
        self.ovo_rbtn.setEnabled(enable_mc)
        
        # Enable gird search buttons
        self.run_btn.setEnabled(True)
        
    def enable_visualize(self):
        """
        Enables visualize tab after a dataset is successfully loaded.
        """
        
        if self.user_in.class_type == 'multiclass':
            
            self.vis_mc_cbox.setEnabled(True)
        
        if self.data_info.no_features <= 3:
            
            self.vis_plot_btn.setEnabled(True)
            self.vis_status_msg.setText("Ready to plot!")
            
        else:
            
            self.vis_status_msg.setText("Number of features must be equal or"
                                        " less than 3.")
            
    def enable_model(self):
        """
        Enables evaluation of pre-trained models.
        """
        
        self.model_eval_btn.setEnabled(True)
        self.model_pred_chk.setEnabled(True)
        self.model_status_msg.setText("Select a pre-trained model.")
        
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
                                              (100  - self.tt_percentage.value()) / 100)
            
        print(self.tt_percentage.value())
        
        self.user_in.C1_range = (self.C1_lbound.value(), self.C1_ubound.value())
        self.user_in.C2_range = (self.C2_lbound.value(), self.C2_ubound.value())
        
        if self.rbf_kernel_rbtn.isChecked() or self.rect_kernel_rbtn.isChecked():
        
            self.user_in.u_range = (self.u_lbound.value(), self.u_ubound.value())
            
        self.user_in.step_size = self.step_box_val.value()
 
        self.user_in.result_path = self.save_path_box.text()
            
        self.user_in.save_clf_results = True if self.clf_results_chk.isChecked() else False
        self.user_in.save_best_model = True if self.best_mode_chk.isChecked() else False
        
        # All the input variables are inserted.
        self.user_in.input_complete = True
        
        if self.validate_usr_input():
            
            ConfrimDialog(self.user_in.get_current_selection(), self.run_gs_thread)
        
    def validate_usr_input(self):
        """
        Checks whether user's inputs such as save path, etc are valid or not.
        
        Returns
        -------
        boolean
            The user's input is valid or not.
        """
        
        total_valid = []
        
        if self.user_in.validate_step_size():
            total_valid.append(True)
        else:
            total_valid.append(False)
            show_dialog("Invalid Step Size", "Step size exceeds the chosen range"
                        " for hyper-parameters.", QMessageBox.Warning)
           
        if any([self.user_in.save_clf_results, self.user_in.save_best_model,
               self.user_in.log_file]):
            
            if validate_path(self.user_in.result_path):
                
                total_valid.append(True)
            
            else:
                
                total_valid.append(False)
                show_dialog("Invalid Save Path", "The path for saving classification"
                            " results does not exist.", QMessageBox.Warning)
            
        return all(total_valid)
        
    def log_file_info(self):
        """
        Gets logging file state and also warns user about its usage.
        """
        
        if self.log_file_chk.isChecked():
            
            self.user_in.log_file = True
            
            show_dialog("Logging Results", "Note that logging classification"
                        " results may reduce the speed of the grid search "
                        "process due to the I/O operations. However, "
                        "results will be not lost in case of power failure, etc.",
                        QMessageBox.Information)
            
        else:
            
            self.user_in.log_file = False
            
    def closeEvent(self, event):
        """
        Shows a message box to confirm exiting the program.
        """
                
        if yes_no_dialog("Exit", "Are you sure you want"
                         "to quit the program?") == QMessageBox.Yes:
            
            event.accept()
            
        else:
            
            event.ignore()
            
#    def onTriggered(self, action):
#        """
#        Sets the current device that does the major compuation. CPU or GPU.
#        """
#        
#        # TODO: Set the current device.
#        if self.cpu_chk_dev.isChecked():
#            
#            pass
#        
#        elif self.gpu_chk_dev.isChecked():
#            
#            pass
        
    def run_gs_thread(self):
        """
        Runs grid search based on user choices.
        """
        
        # TODO: stop button
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        
        self.t = QThread()
        self.gs_t = ThreadGS(self.user_in)
        self.__threads.append((self.t, self.gs_t))
        
        self.gs_t.moveToThread(self.t)
        self.t.started.connect(self.gs_t.initialize)
        
        # Connect signals
        self.gs_t.sig_pbar_set.connect(self.set_pbar_range)
        self.gs_t.sig_gs_info_set.connect(self.update_gs_info)
        self.gs_t.sig_finished.connect(self.stop_gs_thread)
        
        self.t.start()
     
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
    
    @pyqtSlot(bool)
    def stop_gs_thread(self, finished=False):
        """
        It stops the grid search's thread.
        """
        
        if not finished:
        
            if yes_no_dialog("Stop grid search", "Are you sure you want "
                             "to stop the grid search?") == QMessageBox.Yes:
            
                self.gs_t.stop()
                
        else:
            
            show_path = ''
            if any([self.user_in.save_clf_results, self.user_in.save_best_model,
                    self.user_in.log_file]):
                    show_path = ("If selected, the results, model, and log file"
                                " are saved at:\n%s\n" % self.user_in.result_path)
            
            show_dialog("Done!", "The grid search is finished!\n%s" % (show_path),
                        QMessageBox.Information)
                
        self.stop_btn.setEnabled(False)
        self.run_btn.setEnabled(True)
        
    def plot_figure(self):
        """
        Plots a decision boundary based on users' input.
        """
        
        if self.vis_clf_cbox.currentIndex() == 0:
            
            self.user_in.clf_type = 'tsvm'
        
        elif self.vis_clf_cbox.currentIndex() == 1:
            
            self.user_in.clf_type = 'lstsvm'
            
        if self.vis_mc_cbox.currentIndex() == 0:
            
            self.user_in.mc_scheme = 'ova'
            
        elif self.vis_mc_cbox.currentIndex() == 1:
            
            self.user_in.mc_scheme = 'ovo'
            
        if self.vis_lin_rbtn.isChecked():
            
            self.user_in.kernel_type = 'linear'
            
        elif self.vis_non_lin_rbtn.isChecked():
            
            self.user_in.kernel_type = 'RBF'
            
        self.user_in.C1 = 2 ** self.vis_C1_val.value()
        self.user_in.C2 = 2 ** self.vis_C2_val.value()
        self.user_in.u = 2 ** self.vis_u_value.value()
        
        self.user_in.linear_db = True if self.vis_db_chk.isChecked() else False
    
        if self.vis_save_fig_chk.isChecked():
            
            self.user_in.fig_save = True
            self.user_in.fig_dpi =  self.vis_dpi_val.value()
            # TODO: Check whether the path is valid or not.
            self.user_in.fig_save_path = self.vis_save_val.text()
        
        self.run_plot_thread()
        
    def get_save_path_fig(self):
        """
        Gets save path for storing the figure of visualization.
        """
        
        fig_path = QFileDialog.getExistingDirectory(self, "Select a directory")
         
        if fig_path:
            
            self.vis_save_val.setText(fig_path)
            
    def run_plot_thread(self):
        """
        Runs visualization in a separate thread.
        """
        
        self.vis_status_msg.setText("Please wait...")        
        
        t = QThread()
        vis_t = VisualThread(self.user_in)
        self.__threads.append((t, vis_t))
        
        vis_t.moveToThread(t)
        t.started.connect(vis_t.plot)
        
        vis_t.sig_update_plt.connect(self.update_plot)
        
        t.start()
    
    @pyqtSlot(object)    
    def update_plot(self, fig_canvas):
        """
        Inserts the figure canvas to display the plot in the GUI.
        """
        
        print("Draw....")
        
        self.plot_frame_grid.addWidget(fig_canvas, 0, 0, 1, 1)
        
        self.vis_status_msg.setText("Done!")
        
    def get_model_path(self):
        """
        It gets the path at which the fitted model is stored.
        """
        
        model_filename, _ = QFileDialog.getOpenFileName(self, "Import pre-trained model",
                                                       "", "Model files (*.joblib)")
        print(model_filename)
        
        # TODO: Check whether the given path is valid
        if model_filename:
            
            self.model_path_box.setText(model_filename)
            self.model_load_btn.setEnabled(True)
            self.model_status_msg.setText("Load the pre-trained model.")
            
    def get_save_pred_path(self):
        """
        It gets a path to save the file of predicted labels.
        """
        
        if self.model_pred_chk.isChecked():
        
            pred_file_path = QFileDialog.getExistingDirectory(self, "Select a directory")
            
            # TODO: Check whether the path is valid or not.
            if pred_file_path:
                
                self.user_in.save_pred_path = pred_file_path
                self.user_in.save_pred = True
                print(self.user_in.save_pred_path)
                
        else:
            
            self.user_in.save_pred_path = ''
            self.user_in.save_pred = False
            print("Empty path for save pred")
        
            
    def load_model_info(self):
        """
        It loads a pre-trained model and displays its info.
        """
        
        self.user_in.pre_trained_model, model_info = load_model(self.model_path_box.text())
        
        print(model_info)
        self.user_in.kernel_type = model_info['kernel']
        self.user_in.rect_kernel = model_info['rect_kernel']
        
        self.model_clf_name.setText(model_info['model_name'])
        self.model_clf_type_val.setText(model_info['clf_type'])
        self.model_ker_name.setText(self.user_in._get_kernel_selection())
        self.model_n_p_val.setText(model_info['no_params'])
        self.model_C1_val.setText(str(model_info['h_params']['C1']))
        self.model_C2_val.setText(str(model_info['h_params']['C2']))
        self.model_u_val.setText(str(model_info['h_params']['gamma']))
        
        self.model_status_msg.setText("Ready to evaluate!")
        
    def eval_model_test(self):
        """
        Checks required fields and then runs model evaluation in its own thread.
        """
        
        if self.user_in.pre_trained_model is None:
            
                show_dialog("Model", "You must load a pre-trained model for"
                            " evaluation.", QMessageBox.Warning)
                
        else:
            
            self.run_eval_model_thread()
        
    def run_eval_model_thread(self):
        """
        It evaluates a pre-trained model on test samples in a separate thread.
        """
        
        t = QThread()
        model_t = ModelThread(self.user_in)
        self.__threads.append((t, model_t))
        
        model_t.moveToThread(t)
        t.started.connect(model_t.eval_model)
        
        # Signals
        model_t.sig_update_model_eval.connect(self.update_eval_model)
        
        t.start()
        
    @pyqtSlot(str, str)
    def update_eval_model(self, test_acc, test_time):
        """
        It shows the results of a pre-trained model evaluation on test samples 
        in the model tab.
        """
        
        self.model_acc_val.setText(test_acc)
        self.model_te_time_val.setText(test_time)
        
        self.model_status_msg.setText("Done!")
        
        
def show_dialog(title, msg_txt, diag_type):
    """
    A message box that shows extra information to users.
    
    Parameters
    ----------
    title : str
        Title of the message box.
        
    msg_txt : str
        Message that shows information to users.
    
    diag_type : object
        Type of message box, either information or warning.
    """
    
    msg = QMessageBox()
    msg.setIcon(diag_type)
    
    msg.setWindowTitle(title)
    msg.setText(msg_txt)
    msg.setStandardButtons(QMessageBox.Ok)
    msg.exec_()
    
    
def yes_no_dialog(title, msg_txt):
    """
    Asks users for confirmation of an event.
    
    Parameters
    ----------
    title : str
        Title of the message box.
        
    msg_txt : str
        Message that shows information to users.
        
    Returns
    -------
        object
           The yes or no answer. 
    """    
    
    close_win = QMessageBox()
    close_win.setIcon(QMessageBox.Warning)
    
    close_win.setWindowTitle(title)
    close_win.setText(msg_txt)
    close_win.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)

    return close_win.exec_()

      
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
        
   
class AboutDialog(about_diag.Ui_license_diag, QDialog):
    """
    It shows brief infomation about the LIBTwinSVM such as home page and its
    license.
    """
    
    def __init__(self):
        
        super(AboutDialog, self).__init__()

        self.setupUi(self)
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
 
