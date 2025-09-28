from PyQt6.QtWidgets import QMainWindow, QApplication # <-- Đã thêm QApplication

from review.MainWindowListProductExt import MainWindowListProductExt
from review.product import Product
from review.products import ListProduct

app=QApplication([]) # Dấu ngoặc vuông [] là bắt buộc cho QApplication
gmain=QMainWindow()
my_window=MainWindowListProductExt()
my_window.setupUi(gmain)

lp=ListProduct()
lp.add_product(Product(id="p1",name="coca",quantity=15,price=35))
lp.add_product(Product(id="p2",name="pepsi",quantity=14,price=25))
lp.add_product(Product(id="p3",name="sting",quantity=20,price=32))
lp.add_product(Product(id="p4",name="redbull",quantity=30,price=25))

my_window.load_products(lp)# Hoặ# c app.exec_() tùy phiên bản
my_window.showWindow()
app.exec()