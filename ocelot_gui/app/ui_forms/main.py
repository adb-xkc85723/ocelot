# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1159, 828)
        MainWindow.setMinimumSize(QtCore.QSize(1150, 0))
        self.central_widget = QtWidgets.QWidget(MainWindow)
        self.central_widget.setObjectName("central_widget")
        MainWindow.setCentralWidget(self.central_widget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1159, 22))
        self.menuBar.setObjectName("menuBar")
        self.menuFile = QtWidgets.QMenu(self.menuBar)
        self.menuFile.setObjectName("menuFile")
        self.menuEdit = QtWidgets.QMenu(self.menuBar)
        self.menuEdit.setObjectName("menuEdit")
        self.menuSimulations = QtWidgets.QMenu(self.menuBar)
        self.menuSimulations.setObjectName("menuSimulations")
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.action_Parameters = QtWidgets.QAction(MainWindow)
        self.action_Parameters.setObjectName("action_Parameters")
        self.actionLoad_Golden_Orbit = QtWidgets.QAction(MainWindow)
        self.actionLoad_Golden_Orbit.setObjectName("actionLoad_Golden_Orbit")
        self.actionSave_Golden_Orbit = QtWidgets.QAction(MainWindow)
        self.actionSave_Golden_Orbit.setObjectName("actionSave_Golden_Orbit")
        self.actionRead_BPMs_Corrs = QtWidgets.QAction(MainWindow)
        self.actionRead_BPMs_Corrs.setObjectName("actionRead_BPMs_Corrs")
        self.actionCalculate_RM = QtWidgets.QAction(MainWindow)
        self.actionCalculate_RM.setObjectName("actionCalculate_RM")
        self.actionCalculate_ORM = QtWidgets.QAction(MainWindow)
        self.actionCalculate_ORM.setObjectName("actionCalculate_ORM")
        self.actionAdaptive_Feedback = QtWidgets.QAction(MainWindow)
        self.actionAdaptive_Feedback.setObjectName("actionAdaptive_Feedback")
        self.actionLoad_GO_from_Orbit_Display = QtWidgets.QAction(MainWindow)
        self.actionLoad_GO_from_Orbit_Display.setObjectName("actionLoad_GO_from_Orbit_Display")
        self.actionSave_corrs = QtWidgets.QAction(MainWindow)
        self.actionSave_corrs.setObjectName("actionSave_corrs")
        self.actionLoad_corrs = QtWidgets.QAction(MainWindow)
        self.actionLoad_corrs.setObjectName("actionLoad_corrs")
        self.actionUncheck_Red = QtWidgets.QAction(MainWindow)
        self.actionUncheck_Red.setObjectName("actionUncheck_Red")
        self.actionGO_Adviser = QtWidgets.QAction(MainWindow)
        self.actionGO_Adviser.setObjectName("actionGO_Adviser")
        self.actionSend_orbit = QtWidgets.QAction(MainWindow)
        self.actionSend_orbit.setObjectName("actionSend_orbit")
        self.actionSend_all = QtWidgets.QAction(MainWindow)
        self.actionSend_all.setObjectName("actionSend_all")
        self.actionTake_Ref_Orbit_from_Server = QtWidgets.QAction(MainWindow)
        self.actionTake_Ref_Orbit_from_Server.setObjectName("actionTake_Ref_Orbit_from_Server")
        self.actionTake_GO_from_Server = QtWidgets.QAction(MainWindow)
        self.actionTake_GO_from_Server.setObjectName("actionTake_GO_from_Server")
        self.actionFile = QtWidgets.QAction(MainWindow)
        self.actionFile.setObjectName("actionFile")
        self.action_new_lattice = QtWidgets.QAction(MainWindow)
        self.action_new_lattice.setObjectName("action_new_lattice")
        self.action_open_lattice = QtWidgets.QAction(MainWindow)
        self.action_open_lattice.setObjectName("action_open_lattice")
        self.action_save_lattice = QtWidgets.QAction(MainWindow)
        self.action_save_lattice.setObjectName("action_save_lattice")
        self.action_exit = QtWidgets.QAction(MainWindow)
        self.action_exit.setObjectName("action_exit")
        self.action_edit_lattice = QtWidgets.QAction(MainWindow)
        self.action_edit_lattice.setObjectName("action_edit_lattice")
        self.action_calc_twiss = QtWidgets.QAction(MainWindow)
        self.action_calc_twiss.setObjectName("action_calc_twiss")
        self.action_calc_matching = QtWidgets.QAction(MainWindow)
        self.action_calc_matching.setObjectName("action_calc_matching")
        self.menuFile.addAction(self.action_new_lattice)
        self.menuFile.addAction(self.action_open_lattice)
        self.menuFile.addAction(self.action_save_lattice)
        self.menuFile.addAction(self.action_exit)
        self.menuEdit.addAction(self.action_edit_lattice)
        self.menuSimulations.addAction(self.action_calc_twiss)
        self.menuSimulations.addAction(self.action_calc_matching)
        self.menuBar.addAction(self.menuFile.menuAction())
        self.menuBar.addAction(self.menuEdit.menuAction())
        self.menuBar.addAction(self.menuSimulations.menuAction())
        self.mainToolBar.addSeparator()

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuEdit.setTitle(_translate("MainWindow", "Edit"))
        self.menuSimulations.setTitle(_translate("MainWindow", "Simulations"))
        self.action_Parameters.setText(_translate("MainWindow", "Settings"))
        self.actionLoad_Golden_Orbit.setText(_translate("MainWindow", "Load Golden Orbit"))
        self.actionSave_Golden_Orbit.setText(_translate("MainWindow", "Save Golden Orbit"))
        self.actionRead_BPMs_Corrs.setText(_translate("MainWindow", "Read BPMs and Corrs"))
        self.actionCalculate_RM.setText(_translate("MainWindow", "Calculate ORM and DRM"))
        self.actionCalculate_ORM.setText(_translate("MainWindow", "Calculate ORM"))
        self.actionAdaptive_Feedback.setText(_translate("MainWindow", "Adaptive Feedback"))
        self.actionLoad_GO_from_Orbit_Display.setText(_translate("MainWindow", "Load GO from Orbit Display"))
        self.actionSave_corrs.setText(_translate("MainWindow", "Save"))
        self.actionLoad_corrs.setText(_translate("MainWindow", "Load"))
        self.actionUncheck_Red.setText(_translate("MainWindow", "Uncheck Red"))
        self.actionGO_Adviser.setText(_translate("MainWindow", "GO Adviser"))
        self.actionSend_orbit.setText(_translate("MainWindow", "Send only orbit"))
        self.actionSend_all.setText(_translate("MainWindow", "Send all"))
        self.actionTake_Ref_Orbit_from_Server.setText(_translate("MainWindow", "DOOCS Ref Orbit -> GO"))
        self.actionTake_GO_from_Server.setText(_translate("MainWindow", "DOOCS GO -> GO"))
        self.actionFile.setText(_translate("MainWindow", "File"))
        self.action_new_lattice.setText(_translate("MainWindow", "New"))
        self.action_open_lattice.setText(_translate("MainWindow", "Open Lattice"))
        self.action_save_lattice.setText(_translate("MainWindow", "Save Lattice"))
        self.action_exit.setText(_translate("MainWindow", "Exit"))
        self.action_edit_lattice.setText(_translate("MainWindow", "Edit Lattice and Parameters"))
        self.action_calc_twiss.setText(_translate("MainWindow", "Main Parameters and Twiss Functions"))
        self.action_calc_matching.setText(_translate("MainWindow", "Matching"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

