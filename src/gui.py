# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\GUI-LIBTwinSVM.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(640, 320)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 0, 151, 81))
        self.groupBox.setObjectName("groupBox")
        self.radioButton = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton.setGeometry(QtCore.QRect(10, 20, 111, 17))
        self.radioButton.setChecked(True)
        self.radioButton.setObjectName("radioButton")
        self.radioButton_2 = QtWidgets.QRadioButton(self.groupBox)
        self.radioButton_2.setGeometry(QtCore.QRect(10, 50, 141, 17))
        self.radioButton_2.setObjectName("radioButton_2")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(170, 0, 120, 80))
        self.groupBox_2.setObjectName("groupBox_2")
        self.radioButton_3 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_3.setGeometry(QtCore.QRect(10, 20, 82, 17))
        self.radioButton_3.setChecked(True)
        self.radioButton_3.setObjectName("radioButton_3")
        self.radioButton_4 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_4.setGeometry(QtCore.QRect(10, 40, 82, 17))
        self.radioButton_4.setObjectName("radioButton_4")
        self.radioButton_5 = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_5.setGeometry(QtCore.QRect(10, 60, 82, 17))
        self.radioButton_5.setObjectName("radioButton_5")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(300, 0, 121, 111))
        self.groupBox_3.setObjectName("groupBox_3")
        self.radioButton_6 = QtWidgets.QRadioButton(self.groupBox_3)
        self.radioButton_6.setGeometry(QtCore.QRect(10, 20, 101, 17))
        self.radioButton_6.setObjectName("radioButton_6")
        self.label = QtWidgets.QLabel(self.groupBox_3)
        self.label.setGeometry(QtCore.QRect(10, 40, 31, 16))
        self.label.setObjectName("label")
        self.spinBox = QtWidgets.QSpinBox(self.groupBox_3)
        self.spinBox.setGeometry(QtCore.QRect(40, 40, 42, 22))
        self.spinBox.setMinimum(1)
        self.spinBox.setMaximum(10)
        self.spinBox.setProperty("value", 5)
        self.spinBox.setObjectName("spinBox")
        self.radioButton_7 = QtWidgets.QRadioButton(self.groupBox_3)
        self.radioButton_7.setGeometry(QtCore.QRect(10, 70, 91, 17))
        self.radioButton_7.setObjectName("radioButton_7")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 640, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Classifiers"))
        self.radioButton.setText(_translate("MainWindow", "Standard TwinSVM"))
        self.radioButton_2.setText(_translate("MainWindow", "Least Squares TwinSVM"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Kernel"))
        self.radioButton_3.setText(_translate("MainWindow", "Linear"))
        self.radioButton_4.setText(_translate("MainWindow", "RBF"))
        self.radioButton_5.setText(_translate("MainWindow", "Rectangular"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Evaluation"))
        self.radioButton_6.setText(_translate("MainWindow", "Cross-validation"))
        self.label.setText(_translate("MainWindow", "Folds:"))
        self.radioButton_7.setText(_translate("MainWindow", "Train/Test split"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

