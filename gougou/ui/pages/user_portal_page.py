from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGroupBox, QTableWidget, QTableWidgetItem, QLineEdit, QComboBox,
    QMessageBox, QScrollArea
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
import pandas as pd

from ui.theme import AppTheme
from ui.components.kpi_card import KPICard
from utils.models import UserState

class PandasTableModel:
    """Simple model to display pandas DataFrame in QTableWidget"""
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

class UserPortalPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.state = UserState()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(20, 20, 20, 20)
        body_layout.setSpacing(16)
        
        # Title
        title = QLabel("Billing & Payments")
        title.setFont(AppTheme.get_font(16, bold=True))
        title.setObjectName("lblTitle")
        body_layout.addWidget(title)
        
        # KPI Cards
        kpi_layout = QHBoxLayout()
        kpi_layout.setSpacing(12)
        
        self.kpi_due = KPICard("Amount Due", "$1,200.00", AppTheme.WARNING)
        self.kpi_overdue = KPICard("Overdue", "$450.00", AppTheme.DANGER)
        self.kpi_next = KPICard("Next Due Date", "2025-11-15", AppTheme.PRIMARY)
        
        kpi_layout.addWidget(self.kpi_due)
        kpi_layout.addWidget(self.kpi_overdue)
        kpi_layout.addWidget(self.kpi_next)
        
        body_layout.addLayout(kpi_layout)
        
        # Filter section
        filter_group = QGroupBox("Filters")
        filter_layout = QHBoxLayout(filter_group)
        
        filter_layout.addWidget(QLabel("Status:"))
        self.status_combo = QComboBox()
        self.status_combo.addItems(["All", "DUE", "OVERDUE", "PAID"])
        self.status_combo.currentTextChanged.connect(self.refresh_billing)
        filter_layout.addWidget(self.status_combo)
        
        filter_layout.addStretch()
        
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_billing)
        filter_layout.addWidget(refresh_btn)
        
        mark_paid_btn = QPushButton("Mark Selected as PAID")
        mark_paid_btn.setObjectName("btnSuccess")
        mark_paid_btn.clicked.connect(self.mark_selected_paid)
        filter_layout.addWidget(mark_paid_btn)
        
        body_layout.addWidget(filter_group)
        
        # Invoices table
        invoices_group = QGroupBox("Invoices")
        table_layout = QVBoxLayout(invoices_group)
        
        self.invoices_table = QTableWidget()
        self.invoices_table.setColumnCount(5)
        self.invoices_table.setHorizontalHeaderLabels(
            ["Invoice ID", "Status", "Amount", "Due Date", "Action"]
        )
        self.invoices_table.setMinimumHeight(300)
        self.invoices_table.setAlternatingRowColors(True)
        self.invoices_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        
        table_layout.addWidget(self.invoices_table)
        body_layout.addWidget(invoices_group)
        
        # Payment section
        payment_group = QGroupBox("Payment")
        payment_layout = QHBoxLayout(payment_group)
        
        payment_layout.addWidget(QLabel("Card Number:"))
        self.card_input = QLineEdit()
        self.card_input.setPlaceholderText("Card number (demo)")
        self.card_input.setMaximumWidth(200)
        payment_layout.addWidget(self.card_input)
        
        payment_layout.addWidget(QLabel("CVV:"))
        self.cvv_input = QLineEdit()
        self.cvv_input.setPlaceholderText("CVV")
        self.cvv_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.cvv_input.setMaximumWidth(100)
        payment_layout.addWidget(self.cvv_input)
        
        payment_layout.addStretch()
        
        pay_btn = QPushButton("Pay Selected Invoice")
        pay_btn.setObjectName("btnSuccess")
        pay_btn.clicked.connect(self.pay_selected)
        payment_layout.addWidget(pay_btn)
        
        body_layout.addWidget(payment_group)
        body_layout.addStretch()
        
        scroll.setWidget(body)
        layout.addWidget(scroll)
        
        # Load initial data
        self.refresh_billing()
    
    def load_demo_data(self):
        """Load demo billing data"""
        self.state.bills = pd.DataFrame([
            {"invoice_id": 1001, "status": "DUE",     "amount": 120.0, "due_date": "2025-11-15"},
            {"invoice_id": 1002, "status": "PAID",    "amount":  80.0, "due_date": "2025-09-15"},
            {"invoice_id": 1003, "status": "OVERDUE", "amount":  60.0, "due_date": "2025-08-15"},
        ])
    
    def refresh_billing(self):
        """Refresh billing table based on filter"""
        df = self.state.bills.copy()
        st = self.status_combo.currentText()
        if st and st != "All":
            df = df[df["status"] == st]
        
        self._set_table(df)
        
        # Update KPI
        due_amt = float(self.state.bills.loc[self.state.bills.status == "DUE", "amount"].sum())
        over_amt = float(self.state.bills.loc[self.state.bills.status == "OVERDUE", "amount"].sum())
        next_due = self.state.bills.loc[self.state.bills.status == "DUE", "due_date"].min() if not self.state.bills.empty else "—"
        
        self.kpi_due.update_value(f"${due_amt:,.2f}")
        self.kpi_overdue.update_value(f"${over_amt:,.2f}")
        self.kpi_next.update_value(str(next_due))
    
    def mark_selected_paid(self):
        """Mark selected invoice as paid"""
        idx = self.invoices_table.currentRow()
        if idx < 0:
            QMessageBox.information(self.main_window, "Pay", "Chọn 1 invoice trong bảng.")
            return
        inv_id = int(self.invoices_table.item(idx, 0).text())
        self.state.bills.loc[self.state.bills.invoice_id == inv_id, "status"] = "PAID"
        self.refresh_billing()
    
    def pay_selected(self):
        """Process payment"""
        if not self.main_window.current_user:
            QMessageBox.information(self.main_window, "Pay", "Hãy đăng nhập trước.")
            return
        card = self.card_input.text().strip()
        cvv = self.cvv_input.text().strip()
        if card and cvv and len(card) >= 8 and len(cvv) >= 3:
            self.mark_selected_paid()
            QMessageBox.information(self.main_window, "Success", "Thanh toán thành công!")
        else:
            QMessageBox.information(self.main_window, "Pay", "Thông tin thẻ demo chưa hợp lệ.")
    
    def _set_table(self, df: pd.DataFrame):
        """Populate table with dataframe"""
        self.invoices_table.setRowCount(len(df))
        for row, (_, record) in enumerate(df.iterrows()):
            self.invoices_table.setItem(row, 0, QTableWidgetItem(str(record["invoice_id"])))
            self.invoices_table.setItem(row, 1, QTableWidgetItem(str(record["status"])))
            self.invoices_table.setItem(row, 2, QTableWidgetItem(f"${record['amount']:.2f}"))
            self.invoices_table.setItem(row, 3, QTableWidgetItem(str(record["due_date"])))
            
            mark_btn = QPushButton("Mark Paid")
            mark_btn.setObjectName("btnSecondary")
            mark_btn.setMaximumWidth(100)
            self.invoices_table.setCellWidget(row, 4, mark_btn)
