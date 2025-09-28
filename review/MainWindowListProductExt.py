from PIL.ImImagePlugin import number
from PyQt6.QtWidgets import QTableWidgetItem

from review.MainWindowListProduct import Ui_MainWindow


class MainWindowListProductExt(Ui_MainWindow):
        def setupUi(self,MainWindow):
            super().setupUi(MainWindow)
            self.MainWindow=MainWindow
        def showWindow(self):
            self.MainWindow.show()
        def load_products(self):
            self.tableWidgetProduct.setRowCount(0)
            for i in range (0,len(lp.products)):
                p=lp.products[i]
                number_row=self.tableWidgetProduct.rowCount()
                self.tableWidgetProduct.insertRow(number_row)
                self.tableWidgetProduct.setItem(number_row,0,QTableWidgetItem(p,id))
                self.tableWidgetProduct.setItem(number_row,1,QTableWidgetItem(p,name))
                self.tableWidgetProduct.setItem(number_row,2,QTableWidgetItem(p,quantity))
                self.tableWidgetProduct.setItem(number_row,3,QTableWidgetItem(p,price))