# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\dialog.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_confirm_diag(object):
    def setupUi(self, confirm_diag):
        confirm_diag.setObjectName("confirm_diag")
        confirm_diag.resize(477, 300)
        confirm_diag.setFocusPolicy(QtCore.Qt.StrongFocus)
        confirm_diag.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        confirm_diag.setModal(True)
        self.gridLayout = QtWidgets.QGridLayout(confirm_diag)
        self.gridLayout.setObjectName("gridLayout")
        self.diag_frame = QtWidgets.QFrame(confirm_diag)
        self.diag_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.diag_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.diag_frame.setObjectName("diag_frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.diag_frame)
        self.gridLayout_2.setObjectName("gridLayout_2")
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 0, 0, 1, 1)
        self.confirm_btn = QtWidgets.QDialogButtonBox(self.diag_frame)
        self.confirm_btn.setOrientation(QtCore.Qt.Horizontal)
        self.confirm_btn.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.confirm_btn.setObjectName("confirm_btn")
        self.gridLayout_2.addWidget(self.confirm_btn, 0, 1, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 0, 2, 1, 1)
        self.extra_info = QtWidgets.QTextEdit(self.diag_frame)
        self.extra_info.setReadOnly(True)
        self.extra_info.setObjectName("extra_info")
        self.gridLayout_2.addWidget(self.extra_info, 1, 0, 1, 3)
        self.gridLayout.addWidget(self.diag_frame, 1, 0, 1, 1)
        self.msg_label = QtWidgets.QLabel(confirm_diag)
        self.msg_label.setObjectName("msg_label")
        self.gridLayout.addWidget(self.msg_label, 0, 0, 1, 1)

        self.retranslateUi(confirm_diag)
        self.confirm_btn.accepted.connect(confirm_diag.accept)
        self.confirm_btn.rejected.connect(confirm_diag.reject)
        QtCore.QMetaObject.connectSlotsByName(confirm_diag)

    def retranslateUi(self, confirm_diag):
        _translate = QtCore.QCoreApplication.translate
        confirm_diag.setWindowTitle(_translate("confirm_diag", "Confirmation"))
        self.msg_label.setText(_translate("confirm_diag", "Do you confirm the following settings for running the classifier?"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    confirm_diag = QtWidgets.QDialog()
    ui = Ui_confirm_diag()
    ui.setupUi(confirm_diag)
    confirm_diag.show()
    sys.exit(app.exec_())

