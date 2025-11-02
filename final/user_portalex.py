# user_portalex.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
import pandas as pd

from PyQt6.QtWidgets import (
    QMessageBox, QMainWindow, QWidget, QTableView
)
from PyQt6.QtCore import Qt, QAbstractTableModel

from user_portal import Ui_UserPortal   # <-- file UI do bạn gen ra (đúng tên, chữ thường)

# ----- Model nhỏ để đổ DataFrame vào QTableView -----
class _PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.df = df.reset_index(drop=True)

    def rowCount(self, parent=None): return len(self.df)
    def columnCount(self, parent=None): return self.df.shape[1]
    def data(self, idx, role=Qt.ItemDataRole.DisplayRole):
        if not idx.isValid() or role != Qt.ItemDataRole.DisplayRole: return None
        v = self.df.iat[idx.row(), idx.column()]
        return "" if pd.isna(v) else str(v)
    def headerData(self, sec, ori, role):
        if role == Qt.ItemDataRole.DisplayRole and ori == Qt.Orientation.Horizontal:
            return str(self.df.columns[sec])
        if role == Qt.ItemDataRole.DisplayRole and ori == Qt.Orientation.Vertical:
            return str(sec)
        return None

# ----- State demo (sau này thay bằng MySQL/REST) -----
@dataclass
class UserState:
    current_user: str | None = None
    users: Dict[str, str] = field(default_factory=lambda: {"demo@mlba.vn": "123"})
    bills: pd.DataFrame = field(default_factory=lambda: pd.DataFrame([
        {"invoice_id": 1001, "status": "DUE",     "amount": 120.0, "due_date": "2025-11-15"},
        {"invoice_id": 1002, "status": "PAID",    "amount":  80.0, "due_date": "2025-09-15"},
        {"invoice_id": 1003, "status": "OVERDUE", "amount":  60.0, "due_date": "2025-08-15"},
    ]))

class UserPortalEx(Ui_UserPortal):
    def __init__(self):
        super().__init__()
        self.MainWindow: QMainWindow | None = None
        self.state = UserState()

    def setupUi(self, MainWindow: QMainWindow | QWidget):
        # Nếu là QMainWindow thì tạo centralWidget rồi gắn UI vào đó
        if isinstance(MainWindow, QMainWindow):
            cw = QWidget(MainWindow)
            MainWindow.setCentralWidget(cw)
            super().setupUi(cw)     # nạp UI vào central widget
            self.MainWindow = MainWindow
        else:
            # trường hợp truyền QWidget (cách 1)
            super().setupUi(MainWindow)
            self.MainWindow = MainWindow

        self._post_build_fix()
        self.setupSignalAndSlot()
        self._first_render()

    def showWindow(self):
        if self.MainWindow: self.MainWindow.show()
    def closeWindow(self):
        if self.MainWindow: self.MainWindow.close()

    # ===== wire events =====
    def setupSignalAndSlot(self):
        # Các objectName dưới đây cần có trong file user_portal.ui.
        # Nếu khác tên, bạn đổi cho trùng: edUser, edPass, edPass2, lblAuthMsg,
        # lblWelcome, cmbStatus, tblBills, btnLogin, btnReg, btnLogout,
        # btnRefresh, btnMarkPaid, btnPay, edCard, edCVV.
        self.btnLogin.clicked.connect(self.process_login)
        self.btnReg.clicked.connect(self.process_register)
        self.btnLogout.clicked.connect(self.process_logout)
        self.btnRefresh.clicked.connect(self.refresh_billing)
        self.btnMarkPaid.clicked.connect(self.mark_selected_paid)
        self.btnPay.clicked.connect(self.pay_selected)
        self.cmbStatus.currentTextChanged.connect(lambda _s: self.refresh_billing())

    # ===== helpers =====
    def _post_build_fix(self):
        # Một số UI có thể thiếu sẵn items cho combo
        if self.cmbStatus.count() == 0:
            self.cmbStatus.addItems(["ALL", "DUE", "OVERDUE", "PAID"])

    def _first_render(self):
        self.lblWelcome.setText("—")
        self.refresh_billing()

    # ===== actions =====
    def process_login(self):
        uid = self.edUser.text().strip()
        pwd = self.edPass.text()
        if not uid or not pwd:
            self.lblAuthMsg.setText("Điền đủ user & password")
            return
        if self.state.users.get(uid) == pwd:
            self.state.current_user = uid
            self.lblWelcome.setText(f"Xin chào, {uid}")
            self.lblAuthMsg.setText("Đăng nhập thành công")
            self.refresh_billing()
        else:
            self.lblAuthMsg.setText("Sai tài khoản hoặc mật khẩu")

    def process_register(self):
        uid = self.edUser.text().strip()
        p1, p2 = self.edPass.text(), self.edPass2.text()
        if not uid or not p1:
            self.lblAuthMsg.setText("Nhập user & pass để đăng ký")
            return
        if p1 != p2:
            self.lblAuthMsg.setText("Xác nhận mật khẩu không khớp")
            return
        if uid in self.state.users:
            self.lblAuthMsg.setText("User đã tồn tại")
            return
        self.state.users[uid] = p1
        self.lblAuthMsg.setText("Đăng ký thành công — hãy đăng nhập")

    def process_logout(self):
        self.state.current_user = None
        self.lblWelcome.setText("—")
        self.lblAuthMsg.setText("Đã đăng xuất")
        self._set_table(pd.DataFrame())

    def refresh_billing(self):
        df = self.state.bills.copy()
        st = self.cmbStatus.currentText()
        if st and st != "ALL":
            df = df[df["status"] == st]
        self._set_table(df)
        # cập nhật KPI (nếu bạn có 3 QLabel: kDueVal, kOverVal, kNextVal)
        if hasattr(self, "kDueVal"):
            due_amt = float(self.state.bills.loc[self.state.bills.status == "DUE", "amount"].sum())
            over_amt = float(self.state.bills.loc[self.state.bills.status == "OVERDUE", "amount"].sum())
            next_due = self.state.bills.loc[self.state.bills.status == "DUE", "due_date"].min() if not self.state.bills.empty else "—"
            self.kDueVal.setText(f"{due_amt:,.0f}")
            self.kOverVal.setText(f"{over_amt:,.0f}")
            self.kNextVal.setText(str(next_due))

    def mark_selected_paid(self):
        idx = self.tblBills.currentIndex()
        if not idx.isValid():
            QMessageBox.information(self.MainWindow, "Pay", "Chọn 1 invoice trong bảng.")
            return
        inv_id = int(self.tblBills.model().df.iloc[idx.row()]["invoice_id"])
        self.state.bills.loc[self.state.bills.invoice_id == inv_id, "status"] = "PAID"
        self.refresh_billing()

    def pay_selected(self):
        if not self.state.current_user:
            QMessageBox.information(self.MainWindow, "Pay", "Hãy đăng nhập trước.")
            return
        card = self.edCard.text().strip() if hasattr(self, "edCard") else ""
        cvv = self.edCVV.text().strip() if hasattr(self, "edCVV") else ""
        if card and cvv and len(card) >= 8 and len(cvv) >= 3:
            self.mark_selected_paid()
        else:
            QMessageBox.information(self.MainWindow, "Pay", "Thông tin thẻ demo chưa hợp lệ.")

    # ===== small util =====
    def _set_table(self, df: pd.DataFrame):
        self.tblBills.setModel(_PandasModel(df.reset_index(drop=True)))
        self.tblBills.resizeColumnsToContents()
