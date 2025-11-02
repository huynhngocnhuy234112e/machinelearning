import base64
import traceback
import mysql.connector
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QTableWidgetItem, QFileDialog, QMessageBox
from MainWindow import Ui_MainWindow

class MainWindowEx(Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.default_avatar="images/ic_no_avatar.png"
        self.id = None
        self.code = None
        self.name = None
        self.age = None
        self.avatar = None
        self.intro = None
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.MainWindow=MainWindow
        self.tableWidgetStudent.itemSelectionChanged.connect(self.processItemSelection)
        self.pushButtonAvatar.clicked.connect(self.pickAvatar)
        self.pushButtonRemoveAvatar.clicked.connect(self.removeAvatar)
        self.pushButtonInsert.clicked.connect(self.processInsert)
        self.pushButtonUpdate.clicked.connect(self.processUpdate)
        self.pushButtonRemove.clicked.connect(self.processRemove)
    def show(self):
        self.MainWindow.show()
    def connectMySQL(self):
        server = "localhost"
        port = 3306
        database = "student-management"
        username = "root"

        self.conn = mysql.connector.connect(
            host=server,
            port=port,
            database=database,
            user=username,
        )
    def selectAllStudent(self):
        cursor = self.conn.cursor()
        # query all students
        sql = "select * from student"
        cursor.execute(sql)
        dataset = cursor.fetchall()
        self.tableWidgetStudent.setRowCount(0)
        row=0
        for item in dataset:
            row = self.tableWidgetStudent.rowCount()
            self.tableWidgetStudent.insertRow(row)

            self.id = item[0]
            self.code = item[1]
            self.name = item[2]
            self.age = item[3]
            self.avatar = item[4]
            self.intro = item[5]

            self.tableWidgetStudent.setItem(row, 0, QTableWidgetItem(str(self.id)))
            self.tableWidgetStudent.setItem(row, 1, QTableWidgetItem(self.code))
            self.tableWidgetStudent.setItem(row, 2, QTableWidgetItem(self.name))
            self.tableWidgetStudent.setItem(row, 3, QTableWidgetItem(str(self.age)))

        cursor.close()

    def processItemSelection(self):
        row=self.tableWidgetStudent.currentRow()
        if row ==-1:
            return
        try:
            code = self.tableWidgetStudent.item(row, 1).text()
            cursor = self.conn.cursor()
            # query all students
            sql = "select * from student where code=%s"
            val = (code,)
            cursor.execute(sql, val)
            item = cursor.fetchone()
            if item != None:
                self.id = item[0]
                self.code = item[1]
                self.name = item[2]
                self.age = item[3]
                self.avatar = item[4]
                self.intro = item[5]
                self.lineEditId.setText(str(self.id))
                self.lineEditCode.setText(self.code)
                self.lineEditName.setText(self.name)
                self.lineEditAge.setText(str(self.age))
                self.lineEditIntro.setText(self.intro)
                # self.labelAvatar.setPixmap(None)
                if self.avatar != None:
                    imgdata = base64.b64decode(self.avatar)
                    pixmap = QPixmap()
                    pixmap.loadFromData(imgdata)
                    self.labelAvatar.setPixmap(pixmap)
                else:
                    pixmap = QPixmap("images/ic_no_avatar.png")

                    self.labelAvatar.setPixmap(pixmap)
            else:
                print("Not Found")
            cursor.close()
        except:
            traceback.print_exc()

    def pickAvatar(self):
        filters = "Picture PNG (*.png);;All files(*)"
        filename, selected_filter = QFileDialog.getOpenFileName(
            self.MainWindow,
            filter=filters,
        )
        if filename=='':
            return
        pixmap = QPixmap(filename)
        self.labelAvatar.setPixmap(pixmap)

        with open(filename, "rb") as image_file:
            self.avatar = base64.b64encode(image_file.read())
            print(self.avatar)
        pass
    def removeAvatar(self):
        self.avatar=None
        pixmap = QPixmap(self.default_avatar)
        self.labelAvatar.setPixmap(pixmap)
    def processInsert(self):
        try:
            cursor = self.conn.cursor()
            # query all students
            sql = "insert into student(Code,Name,Age,Avatar,Intro) values(%s,%s,%s,%s,%s)"

            self.code = self.lineEditCode.text()
            self.name = self.lineEditName.text()
            self.age = int(self.lineEditAge.text())
            if not hasattr(self, 'avatar'):
                avatar = None
            intro = self.lineEditIntro.text()
            val = (self.code, self.name, self.age, self.avatar, self.intro)

            cursor.execute(sql, val)

            self.conn.commit()

            print(cursor.rowcount, " record inserted")
            self.lineEditId.setText(str(cursor.lastrowid))

            cursor.close()
            self.selectAllStudent()
        except:
            traceback.print_exc()

    def processUpdate(self):
        cursor = self.conn.cursor()
        # query all students
        sql = "update student set Code=%s,Name=%s,Age=%s,Avatar=%s,Intro=%s" \
              " where Id=%s"
        self.id=int(self.lineEditId.text())
        self.code = self.lineEditCode.text()
        self.name = self.lineEditName.text()
        self.age = int(self.lineEditAge.text())
        if not hasattr(self, 'avatar'):
            self.avatar = None
        self.intro = self.lineEditIntro.text()

        val = (self.code,self.name,self.age,self.avatar ,self.intro,self.id )

        cursor.execute(sql, val)

        self.conn.commit()

        print(cursor.rowcount, " record updated")
        cursor.close()
        self.selectAllStudent()
    def processRemove(self):
        dlg = QMessageBox(self.MainWindow)
        dlg.setWindowTitle("Confirmation Deleting")
        dlg.setText("Are you sure you want to delete?")
        dlg.setIcon(QMessageBox.Icon.Question)
        buttons = QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        dlg.setStandardButtons(buttons)
        button = dlg.exec()
        if button == QMessageBox.StandardButton.No:
            return
        cursor = self.conn.cursor()
        # query all students
        sql = "delete from student "\
              " where Id=%s"

        val = (self.lineEditId.text(),)

        cursor.execute(sql, val)

        self.conn.commit()

        print(cursor.rowcount, " record removed")

        cursor.close()
        self.selectAllStudent()
        self.clearData()
    def clearData(self):
        self.lineEditId.setText("")
        self.lineEditCode.setText("")
        self.lineEditName.setText("")
        self.lineEditAge.setText("")
        self.lineEditIntro.setText("")
        self.avatar=None
