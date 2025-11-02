from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTableWidgetItem, QMessageBox

from project_retail.connectors.employee_connector import EmployeeConnector
from project_retail.models.employee import Employee
from project_retail.ui.EmployeeMainWindow import Ui_MainWindow


class EmployeeMainWindowEx(Ui_MainWindow):
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.MainWindow=MainWindow
        self.setupSignalAndSlot()

        self.ec = EmployeeConnector()
        self.ec.connect()
        self.display_all_employees()
    def showWindow(self):
        self.MainWindow.show()
    def closeWindow(self):
        self.MainWindow.close()
    def setupSignalAndSlot(self):
        self.tableWidgetEmployee.itemSelectionChanged.connect(self.show_Detail)
        self.pushButtonNew.clicked.connect(self.clear_data)
        self.pushButtonInsert.clicked.connect(self.insert_data)
    def display_all_employees(self):
        #empty existing data:
        self.tableWidgetEmployee.setRowCount(0)
        #loop all employee, then show emp into table:
        employees = self.ec.get_list_employee()
        for emp in employees:
            #get lasted row
            row=self.tableWidgetEmployee.rowCount()
            #insert new row at the end of table:
            self.tableWidgetEmployee.insertRow(row)
            #assign value for column:
            column_id=QTableWidgetItem(str(emp.ID))
            column_name=QTableWidgetItem(emp.Name)
            column_email=QTableWidgetItem(emp.Email)
            column_phone=QTableWidgetItem(emp.Phone)
            if emp.IsDeleted==1:
                column_id.setBackground(Qt.GlobalColor.yellow)
                column_name.setBackground(Qt.GlobalColor.yellow)
                column_email.setBackground(Qt.GlobalColor.yellow)
                column_phone.setBackground(Qt.GlobalColor.yellow)

            #push column(s) into row:
            self.tableWidgetEmployee.setItem(row,0,column_id)
            self.tableWidgetEmployee.setItem(row,1,column_name)
            self.tableWidgetEmployee.setItem(row,2,column_email)
            self.tableWidgetEmployee.setItem(row,3,column_phone)
    def show_Detail(self):
        row_number=self.tableWidgetEmployee.currentIndex().row()
        id=self.tableWidgetEmployee.item(row_number,0).text()
        emp=self.ec.get_detail(id)
        if emp!=None:
            self.lineEditID.setText(str(emp.ID))
            self.lineEditName.setText(emp.Name)
            self.lineEditPhone.setText(emp.Phone)
            self.lineEditEmail.setText(emp.Email)
            self.lineEditPassword.setText(emp.Password)
    def clear_data(self):
        self.lineEditID.setText("")
        self.lineEditName.clear()
        self.lineEditPhone.setText("")
        self.lineEditEmail.setText("")
        self.lineEditName.setFocus()
    def insert_data(self):
        emp = Employee()
        emp.Name = self.lineEditName.text()
        emp.Email = self.lineEditEmail.text()
        emp.Phone = self.lineEditPhone.text()
        emp.Password = self.lineEditPassword.text()
        emp.IsDeleted = 0
        result = self.ec.insert_employee(emp)
        if result > 0:
            self.display_all_employees()
        else:
            #use msgbox for warning
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Critical)
            msg.setWindowTitle("infor")
            msg.setText("Can not insert a new employee")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec()