# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\about_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_license_diag(object):
    def setupUi(self, license_diag):
        license_diag.setObjectName("license_diag")
        license_diag.resize(478, 394)
        self.gridLayout = QtWidgets.QGridLayout(license_diag)
        self.gridLayout.setObjectName("gridLayout")
        self.license_diag_frame = QtWidgets.QFrame(license_diag)
        self.license_diag_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.license_diag_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.license_diag_frame.setObjectName("license_diag_frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.license_diag_frame)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.ld_diag_page_link = QtWidgets.QLabel(self.license_diag_frame)
        self.ld_diag_page_link.setTextFormat(QtCore.Qt.AutoText)
        self.ld_diag_page_link.setOpenExternalLinks(True)
        self.ld_diag_page_link.setObjectName("ld_diag_page_link")
        self.gridLayout_2.addWidget(self.ld_diag_page_link, 4, 1, 1, 1)
        self.ld_diag_devs = QtWidgets.QLabel(self.license_diag_frame)
        self.ld_diag_devs.setObjectName("ld_diag_devs")
        self.gridLayout_2.addWidget(self.ld_diag_devs, 3, 0, 1, 1)
        self.ld_diag_license_box = QtWidgets.QGroupBox(self.license_diag_frame)
        self.ld_diag_license_box.setObjectName("ld_diag_license_box")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.ld_diag_license_box)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.ld_dig_license_text = QtWidgets.QTextEdit(self.ld_diag_license_box)
        self.ld_dig_license_text.setReadOnly(True)
        self.ld_dig_license_text.setObjectName("ld_dig_license_text")
        self.gridLayout_3.addWidget(self.ld_dig_license_text, 0, 0, 1, 1)
        self.gridLayout_2.addWidget(self.ld_diag_license_box, 5, 0, 1, 2)
        self.ld_libtsvm_label = QtWidgets.QLabel(self.license_diag_frame)
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        self.ld_libtsvm_label.setFont(font)
        self.ld_libtsvm_label.setObjectName("ld_libtsvm_label")
        self.gridLayout_2.addWidget(self.ld_libtsvm_label, 0, 0, 1, 1)
        self.ld_libtsvm_ver = QtWidgets.QLabel(self.license_diag_frame)
        self.ld_libtsvm_ver.setObjectName("ld_libtsvm_ver")
        self.gridLayout_2.addWidget(self.ld_libtsvm_ver, 0, 1, 1, 1)
        self.ld_libtsvm_sub = QtWidgets.QLabel(self.license_diag_frame)
        self.ld_libtsvm_sub.setObjectName("ld_libtsvm_sub")
        self.gridLayout_2.addWidget(self.ld_libtsvm_sub, 1, 0, 1, 2)
        self.ld_diag_dev_names = QtWidgets.QLabel(self.license_diag_frame)
        self.ld_diag_dev_names.setOpenExternalLinks(False)
        self.ld_diag_dev_names.setObjectName("ld_diag_dev_names")
        self.gridLayout_2.addWidget(self.ld_diag_dev_names, 3, 1, 1, 1)
        self.ld_diag_page = QtWidgets.QLabel(self.license_diag_frame)
        self.ld_diag_page.setObjectName("ld_diag_page")
        self.gridLayout_2.addWidget(self.ld_diag_page, 4, 0, 1, 1)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        self.gridLayout_2.addItem(spacerItem, 2, 0, 1, 1)
        self.gridLayout.addWidget(self.license_diag_frame, 0, 0, 1, 1)

        self.retranslateUi(license_diag)
        QtCore.QMetaObject.connectSlotsByName(license_diag)

    def retranslateUi(self, license_diag):
        _translate = QtCore.QCoreApplication.translate
        license_diag.setWindowTitle(_translate("license_diag", "About"))
        self.ld_diag_page_link.setText(_translate("license_diag", "<html><head/><body><p><a href=\"https://github.com/mir-am/LIBTwinSVM\"><span style=\" text-decoration: underline; color:#0000ff;\">https://github.com/mir-am/LIBTwinSVM</span></a></p></body></html>"))
        self.ld_diag_devs.setText(_translate("license_diag", "Developers:"))
        self.ld_diag_license_box.setTitle(_translate("license_diag", "License"))
        self.ld_dig_license_text.setHtml(_translate("license_diag", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8pt; font-weight:400; font-style:normal;\">\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, version 3.</p>\n"
"<p align=\"justify\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.</p>\n"
"<p align=\"justify\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p align=\"justify\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">You should have received a copy of the GNU General Public License along with this program. If not, see &lt;http://www.gnu.org/licenses/&gt;.</p></body></html>"))
        self.ld_libtsvm_label.setText(_translate("license_diag", "LIBTwinSVM"))
        self.ld_libtsvm_ver.setText(_translate("license_diag", "(v0.1.0)"))
        self.ld_libtsvm_sub.setText(_translate("license_diag", "A Library for Twin Support Vector Machines"))
        self.ld_diag_dev_names.setText(_translate("license_diag", "A. Mir, Mahdi Rahbar"))
        self.ld_diag_page.setText(_translate("license_diag", "Project Page:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    license_diag = QtWidgets.QDialog()
    ui = Ui_license_diag()
    ui.setupUi(license_diag)
    license_diag.show()
    sys.exit(app.exec_())

