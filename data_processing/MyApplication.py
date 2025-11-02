from PyQt6.QtWidgets import QApplication

from data_processing.MainWindowEx import MainWindowEx

app = QApplication([])
mywindow = MainWindowEx()
mywindow.show()
app.exec()
