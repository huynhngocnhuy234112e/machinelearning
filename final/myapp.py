from PyQt6.QtWidgets import QApplication, QMainWindow
from user_portalex import UserPortalEx

app = QApplication([])

win = QMainWindow()
ui  = UserPortalEx()
ui.setupUi(win)      # giờ đã tự tạo centralWidget
ui.showWindow()

app.exec()
