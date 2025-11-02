from PyQt6.QtWidgets import QMessageBox, QMainWindow

from project_retail.connectors.employee_connector import EmployeeConnector
from project_retail.ui.EmployeeMainWindowEx import EmployeeMainWindowEx
from project_retail.ui.LoginMainWindow import Ui_MainWindow


class LoginMainWindowEx(Ui_MainWindow):
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.MainWindow=MainWindow
        self.setupSignalAndSlot()
    def showWindow(self):
        self.MainWindow.show()
    def closeWindow(self):
        self.MainWindow.close()
    def setupSignalAndSlot(self):
        self.pushButtonLogin.clicked.connect(self.process_login)
    def process_login(self):
        uid=self.lineEditUserName.text()
        pwd=self.lineEditPassword.text()
        ec = EmployeeConnector()
        ec.connect()
        em = ec.login(uid, pwd)
        if em == None:
            print("Login failed")
            msg=QMessageBox()
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("infor")
            msg.setText("Login failed, please contact to ADMIN")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()
        else:
            self.closeWindow()
            self.gui_emp=EmployeeMainWindowEx()
            self.gui_emp.setupUi(QMainWindow())
            self.gui_emp.showWindow()