
from __future__ import annotations
import sys
import os
import json
import math
import threading
import re
from datetime import date
from typing import Optional, List

import pandas as pd
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QFileDialog, QTableView, QComboBox,
    QGroupBox, QFormLayout, QSpinBox, QProgressBar, QMessageBox, QPlainTextEdit,
    QCheckBox, QScrollArea, QGridLayout, QSizePolicy   # <-- thêm cái này
)
from PyQt6.QtGui import QFont

import mysql.connector
from mysql.connector import Error


# -----------------------------
# Helpers: Pandas -> Qt Model
# -----------------------------
class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df.copy()

    def rowCount(self, parent=QModelIndex()):
        return len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return len(self._df.columns)

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            val = self._df.iat[index.row(), index.column()]
            if isinstance(val, float):
                if math.isnan(val):
                    return ""
                return f"{val:.4g}"
            return str(val)
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            return str(self._df.columns[section])
        else:
            return str(self._df.index[section])


# -----------------------------
# DB manager
# -----------------------------
class DB:
    def __init__(self):
        self.conn: Optional[mysql.connector.MySQLConnection] = None
        self.cfg = {
            "host": "127.0.0.1",
            "port": 3306,
            "user": "root",
            "password": "",
            "database": "final",
        }

    def connect(self, *, host: str, port: int, user: str, password: str, database: str) -> None:
        if self.conn:
            try:
                self.conn.close()
            except Exception:
                pass
        self.cfg.update(dict(host=host, port=port, user=user, password=password, database=database))
        self.conn = mysql.connector.connect(**self.cfg)

    def cursor(self):
        if not self.conn:
            raise RuntimeError("Not connected to MySQL")
        return self.conn.cursor()

    def execute(self, sql: str, params: Optional[tuple] = None):
        cur = self.cursor()
        cur.execute(sql, params or ())
        self.conn.commit()
        return cur

    def executemany(self, sql: str, rows: List[tuple]):
        cur = self.cursor()
        cur.executemany(sql, rows)
        self.conn.commit()
        return cur

    def query_df(self, sql: str) -> pd.DataFrame:
        cur = self.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)


DBM = DB()


# -----------------------------
# Global theme (improved styling)
# -----------------------------
def apply_theme(app: QApplication):
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(
        """
        QMainWindow { 
            background: #f5f5f5; 
        }
        
        QTabWidget::pane { 
            border: none;
            background: #ffffff; 
        }
        
        QTabBar::tab { 
            background: #e0e0e0; 
            color: #333333;
            padding: 10px 20px; 
            margin-right: 2px; 
            border: none;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            font-weight: 500;
        }
        
        QTabBar::tab:selected { 
            background: #FDD835; 
            color: #000000;
            font-weight: 600;
        }
        
        QTabBar::tab:hover:!selected {
            background: #eeeeee;
        }
        
        QPushButton { 
            background: #2196F3; 
            color: #ffffff; 
            border: none; 
            padding: 10px 20px; 
            border-radius: 6px; 
            font-weight: 600;
            min-width: 120px;
        }
        
        QPushButton:hover { 
            background: #1976D2; 
        }
        
        QPushButton:pressed {
            background: #0D47A1;
        }
        
        QPushButton:disabled { 
            background: #BDBDBD; 
            color: #757575; 
        }
        
        QPushButton#primaryButton {
            background: #4CAF50;
        }
        
        QPushButton#primaryButton:hover {
            background: #45a049;
        }
        
        QPushButton#dangerButton {
            background: #f44336;
        }
        
        QPushButton#dangerButton:hover {
            background: #da190b;
        }
                
                /* Tăng không gian phía trên cho title */
        QGroupBox {
            background: #ffffff;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            margin-top: 20px;
            /* trước: padding-top: 15px; */
            padding-top: 32px;  /* chừa đủ chỗ cho title */
            font-weight: 600;
        }
        
        /* Đặt title nằm TRONG padding thay vì “kéo” ra ngoài */
        QGroupBox::title {
            /* trước: subcontrol-origin: margin; left: 15px; top: -12px; */
            subcontrol-origin: padding;
            subcontrol-position: top left;
            left: 12px;
            top: 0px;            /* bỏ offset âm để tránh bị cắt */
            padding: 6px 10px;
            background: #FDD835;
            color: #000000;
            border-radius: 4px;
            font-weight: 700;
        }

        
        QLabel#headerTitle { 
            background: #FDD835; 
            color: #000000; 
            font-size: 24px; 
            font-weight: 700; 
            padding: 15px; 
            border-radius: 8px;
        }
        
        QLabel#statsLabel {
            background: #E3F2FD;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #90CAF9;
            font-weight: 600;
        }
        
        QLabel#footerLabel {
            background: #00BCD4;
            color: #ffffff;
            padding: 8px;
            border-radius: 4px;
            font-weight: 600;
        }
        
        QTableView { 
            background: #ffffff; 
            gridline-color: #e0e0e0; 
            border: 1px solid #e0e0e0; 
            border-radius: 6px; 
            selection-background-color: #FFF9C4; 
            selection-color: #000000;
        }
        
        QHeaderView::section { 
            background: #f5f5f5; 
            padding: 8px; 
            border: none; 
            border-bottom: 2px solid #FDD835; 
            font-weight: 700;
            color: #333333;
        }
        
        QProgressBar { 
            border: 2px solid #e0e0e0; 
            border-radius: 6px; 
            text-align: center; 
            background: #ffffff;
            height: 25px;
        }
        
        QProgressBar::chunk { 
            background-color: #4CAF50; 
            border-radius: 4px; 
        }
        
        QLineEdit, QComboBox, QSpinBox { 
            background: #ffffff; 
            border: 2px solid #e0e0e0; 
            border-radius: 6px; 
            padding: 8px;
            min-height: 25px;
        }
        
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
            border: 2px solid #2196F3;
        }
        
        QPlainTextEdit { 
            background: #ffffff; 
            border: 2px solid #e0e0e0; 
            border-radius: 6px; 
            padding: 8px;
        }
        
        QCheckBox {
            spacing: 8px;
        }
        
        QCheckBox::indicator {
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 2px solid #e0e0e0;
        }
        
        QCheckBox::indicator:checked {
            background: #4CAF50;
            border: 2px solid #4CAF50;
        }
        """
    )


# -----------------------------
# Tab: Home (improved design)
# -----------------------------
class HomeTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.df: Optional[pd.DataFrame] = None
        self.build_ui()

    def build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Header
        header = QLabel("Credit Risk — Late Payment Prediction System")
        header.setObjectName("headerTitle")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)

        # Quick Start Section
        quick_start = QGroupBox("Quick Start — Load & Preview Dataset")
        qs_layout = QVBoxLayout()
        qs_layout.setSpacing(12)

        # File selection row
        file_row = QHBoxLayout()
        self.btn_select_file = QPushButton("📁 Select Dataset (CSV)")
        self.btn_select_file.setObjectName("primaryButton")
        self.lbl_file_path = QLabel("No file selected")
        self.lbl_file_path.setStyleSheet("color: #757575; font-style: italic;")
        self.btn_load_preview = QPushButton("👁 Preview Data")

        file_row.addWidget(self.btn_select_file)
        file_row.addWidget(self.lbl_file_path, 1)
        file_row.addWidget(self.btn_load_preview)

        qs_layout.addLayout(file_row)

        # Statistics row
        stats_row = QHBoxLayout()
        self.lbl_rows = QLabel("Rows: -")
        self.lbl_rows.setObjectName("statsLabel")
        self.lbl_cols = QLabel("Columns: -")
        self.lbl_cols.setObjectName("statsLabel")
        self.lbl_defaults = QLabel("Default Rate: -")
        self.lbl_defaults.setObjectName("statsLabel")

        stats_row.addWidget(self.lbl_rows)
        stats_row.addWidget(self.lbl_cols)
        stats_row.addWidget(self.lbl_defaults)
        stats_row.addStretch()

        qs_layout.addLayout(stats_row)

        # Data preview table
        self.table_preview = QTableView()
        self.table_preview.setAlternatingRowColors(True)
        self.table_preview.setSortingEnabled(True)
        self.table_preview.setMinimumHeight(300)
        qs_layout.addWidget(self.table_preview)

        quick_start.setLayout(qs_layout)
        main_layout.addWidget(quick_start)

        # Database Connection Section (collapsible)
        db_section = QGroupBox("Database Connection (Optional)")
        db_layout = QGridLayout()
        db_layout.setSpacing(10)

        self.chk_use_db = QCheckBox("Use MySQL Database")
        db_layout.addWidget(self.chk_use_db, 0, 0, 1, 2)

        db_layout.addWidget(QLabel("Host:"), 1, 0)
        self.in_host = QLineEdit(DBM.cfg["host"])
        db_layout.addWidget(self.in_host, 1, 1)

        db_layout.addWidget(QLabel("Port:"), 1, 2)
        self.in_port = QSpinBox()
        self.in_port.setRange(1, 65535)
        self.in_port.setValue(DBM.cfg["port"])
        db_layout.addWidget(self.in_port, 1, 3)

        db_layout.addWidget(QLabel("User:"), 2, 0)
        self.in_user = QLineEdit(DBM.cfg["user"])
        db_layout.addWidget(self.in_user, 2, 1)

        db_layout.addWidget(QLabel("Password:"), 2, 2)
        self.in_pass = QLineEdit(DBM.cfg["password"])
        self.in_pass.setEchoMode(QLineEdit.EchoMode.Password)
        db_layout.addWidget(self.in_pass, 2, 3)

        db_layout.addWidget(QLabel("Database:"), 3, 0)
        self.in_db = QLineEdit(DBM.cfg["database"])
        db_layout.addWidget(self.in_db, 3, 1)

        self.btn_connect = QPushButton("🔌 Connect to Database")
        self.lbl_db_status = QLabel("Not connected")
        self.lbl_db_status.setStyleSheet("color: #f44336; font-weight: 600;")

        db_layout.addWidget(self.btn_connect, 3, 2)
        db_layout.addWidget(self.lbl_db_status, 3, 3)

        db_section.setLayout(db_layout)
        main_layout.addWidget(db_section)

        # Footer
        footer = QLabel("💡 Tip: Start by selecting a CSV file to preview your credit risk data")
        footer.setObjectName("footerLabel")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(footer)

        main_layout.addStretch()

        # Connect signals
        self.btn_select_file.clicked.connect(self.select_file)
        self.btn_load_preview.clicked.connect(self.load_preview)
        self.btn_connect.clicked.connect(self.connect_db)
        self.chk_use_db.stateChanged.connect(self.toggle_db_fields)

        # Initial state
        self.toggle_db_fields()

    def toggle_db_fields(self):
        enabled = self.chk_use_db.isChecked()
        self.in_host.setEnabled(enabled)
        self.in_port.setEnabled(enabled)
        self.in_user.setEnabled(enabled)
        self.in_pass.setEnabled(enabled)
        self.in_db.setEnabled(enabled)
        self.btn_connect.setEnabled(enabled)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Credit Card Dataset",
            os.getcwd(),
            "CSV files (*.csv)"
        )
        if path:
            self.lbl_file_path.setText(path)
            self.lbl_file_path.setStyleSheet("color: #2196F3; font-weight: 600;")

    def load_preview(self):
        path = self.lbl_file_path.text()
        if not path or path == "No file selected":
            QMessageBox.warning(self, "No File", "Please select a CSV file first.")
            return

        try:
            # Load full dataset for statistics
            df_full = pd.read_csv(path)
            self.df = df_full

            # Preview first 100 rows
            df_preview = df_full.head(100)
            self.table_preview.setModel(PandasModel(df_preview))

            # Update statistics
            self.lbl_rows.setText(f"Rows: {len(df_full):,}")
            self.lbl_cols.setText(f"Columns: {len(df_full.columns)}")

            # Calculate default rate if column exists
            default_col = None
            for col in df_full.columns:
                if 'default' in col.lower():
                    default_col = col
                    break

            if default_col:
                default_rate = df_full[default_col].mean() * 100
                self.lbl_defaults.setText(f"Default Rate: {default_rate:.2f}%")
            else:
                self.lbl_defaults.setText("Default Rate: N/A")

            QMessageBox.information(
                self,
                "Success",
                f"Loaded {len(df_full):,} rows with {len(df_full.columns)} columns.\nShowing first 100 rows."
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load CSV:\n{str(e)}")

    def connect_db(self):
        try:
            DBM.connect(
                host=self.in_host.text().strip(),
                port=int(self.in_port.value()),
                user=self.in_user.text().strip(),
                password=self.in_pass.text(),
                database=self.in_db.text().strip(),
            )
            self.lbl_db_status.setText("✓ Connected")
            self.lbl_db_status.setStyleSheet("color: #4CAF50; font-weight: 600;")
            QMessageBox.information(self, "Success", "Successfully connected to MySQL database!")
        except Error as e:
            QMessageBox.critical(self, "Connection Error", str(e))
            self.lbl_db_status.setText("✗ Failed")
            self.lbl_db_status.setStyleSheet("color: #f44336; font-weight: 600;")


# -----------------------------
# Tab: Data Intake (improved)
# -----------------------------
# -----------------------------
# Tab: Data Intake (improved with scroll)
# -----------------------------
class DataIntakeTab(QWidget):
    STAGING_TABLE = "stg_default_raw"
    SNAPSHOT_MONTH = date(2005, 9, 1)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.df: Optional[pd.DataFrame] = None
        self.build_ui()

    def build_ui(self):
        # Root layout of the tab
        root = QVBoxLayout(self)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        # Scroll area wraps the whole content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        root.addWidget(scroll)

        # Actual content widget inside the scroll area
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Step 1: Select CSV
        step1 = QGroupBox("Step 1: Select Dataset")
        s1_layout = QHBoxLayout()
        self.btn_pick = QPushButton("📁 Choose CSV File")
        self.btn_pick.setObjectName("primaryButton")
        self.lbl_path = QLabel("No file selected")
        self.lbl_path.setStyleSheet("color: #757575; font-style: italic;")
        self.btn_preview = QPushButton("👁 Preview")
        s1_layout.addWidget(self.btn_pick)
        s1_layout.addWidget(self.lbl_path, 1)
        s1_layout.addWidget(self.btn_preview)
        step1.setLayout(s1_layout)
        layout.addWidget(step1)

        # Step 2: Load to staging
        step2 = QGroupBox("Step 2: Load to Staging Table")
        s2_layout = QHBoxLayout()
        self.btn_load = QPushButton("⬆ Load to Database")
        self.btn_load.setObjectName("primaryButton")
        self.lbl_load_status = QLabel("Ready to load")
        s2_layout.addWidget(self.btn_load)
        s2_layout.addWidget(self.lbl_load_status)
        s2_layout.addStretch()
        step2.setLayout(s2_layout)
        layout.addWidget(step2)

        # Step 3: ETL
        step3 = QGroupBox("Step 3: Run ETL Pipeline")
        s3_layout = QHBoxLayout()
        self.btn_etl = QPushButton("⚙ Execute ETL")
        self.btn_etl.setObjectName("primaryButton")
        self.btn_etl.setEnabled(False)
        self.lbl_etl_status = QLabel("Waiting for data load")
        s3_layout.addWidget(self.btn_etl)
        s3_layout.addWidget(self.lbl_etl_status)
        s3_layout.addStretch()
        step3.setLayout(s3_layout)
        layout.addWidget(step3)

        # Data preview
        preview_group = QGroupBox("Data Preview")
        preview_layout = QVBoxLayout()
        self.table = QTableView()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.setMinimumHeight(250)
        preview_layout.addWidget(self.table)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setTextVisible(True)
        layout.addWidget(self.progress)

        # Process Log (bigger + expandable instead of capped height)
        log_group = QGroupBox("Process Log")
        log_layout = QVBoxLayout()
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(200)  # trước là setMaximumHeight(150)
        self.log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.log.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        log_layout.addWidget(self.log)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # Put content into the scroll area
        scroll.setWidget(content)

        # Connect signals
        self.btn_pick.clicked.connect(self.pick_csv)
        self.btn_preview.clicked.connect(self.preview)
        self.btn_load.clicked.connect(self.load_to_staging)
        self.btn_etl.clicked.connect(self.run_etl)

    def _append_log(self, msg: str):
        self.log.appendPlainText(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {msg}")

    def pick_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Dataset", os.getcwd(), "CSV files (*.csv)"
        )
        if path:
            self.lbl_path.setText(path)
            self.lbl_path.setStyleSheet("color: #2196F3; font-weight: 600;")
            self._append_log(f"Selected file: {os.path.basename(path)}")

    def preview(self):
        try:
            path = self.lbl_path.text()
            if not path or path == "No file selected":
                QMessageBox.warning(self, "No File", "Please select a CSV file first.")
                return
            self.df = pd.read_csv(path).head(100)
            self.table.setModel(PandasModel(self.df))
            self._append_log(f"Previewed 100 rows from {os.path.basename(path)}")
            self.lbl_load_status.setText("Ready to load")
            self.lbl_load_status.setStyleSheet("color: #4CAF50; font-weight: 600;")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self._append_log(f"Error: {str(e)}")

    def load_to_staging(self):
        try:
            path = self.lbl_path.text()
            if not path or path == "No file selected":
                QMessageBox.warning(self, "No File", "Please select a CSV file first.")
                return

            self._append_log("Starting data load...")
            df_full = pd.read_csv(path)

            # Auto-map headers
            def norm(s: str) -> str:
                return re.sub(r"[^A-Z0-9]+", "_", str(s).upper()).strip("_")

            required = [
                "ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARITAL", "AGE",
                "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
                "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
                "default_payment_next_month"
            ]

            colmap = {norm(c): c for c in df_full.columns}
            synonyms = {
                "MARITAL": ["MARRIAGE"],
                "DEFAULT_PAYMENT_NEXT_MONTH": [
                    "DEFAULT.PAYMENT.NEXT.MONTH", "DEFAULT PAYMENT NEXT MONTH",
                    "DEFAULTPAYMENTNEXTMONTH", "DEFAULT_PAYMENT_NEXTMONTH",
                    "DEFAULT_PAYMENT"
                ],
                "PAY_0": ["PAY0", "PAY_1"],
            }

            def find_col(canon: str) -> str | None:
                cands = [canon, canon.replace("_", "")]
                for alt in synonyms.get(canon.upper(), []):
                    cands.append(alt)
                for cand in cands:
                    k = norm(cand)
                    if k in colmap:
                        return colmap[k]
                return None

            rename_map: dict[str, str] = {}
            missing: list[str] = []

            for c in required:
                src = find_col(c)
                if src is not None:
                    rename_map[src] = c
                else:
                    missing.append(c)

            df_full = df_full.rename(columns=rename_map)

            if missing:
                for m in missing:
                    df_full[m] = 0
                self._append_log(f"Auto-filled missing columns: {missing}")

            columns = required
            placeholders = "( " + ",".join(["%s"] * len(columns)) + " )"
            insert_sql = (
                f"INSERT INTO {self.STAGING_TABLE} "
                f"({','.join(columns)}) VALUES {placeholders}"
            )

            df_full = df_full[columns]
            rows = [tuple(None if pd.isna(x) else x for x in r)
                    for r in df_full.itertuples(index=False, name=None)]

            batch = 1000
            total = len(rows)
            self.progress.setMaximum(total)

            for i in range(0, total, batch):
                chunk = rows[i:i + batch]
                DBM.executemany(insert_sql, chunk)
                self.progress.setValue(i + len(chunk))
                QApplication.processEvents()

            self._append_log(f"✓ Loaded {total:,} rows to {self.STAGING_TABLE}")
            self.lbl_load_status.setText(f"✓ Loaded {total:,} rows")
            self.lbl_load_status.setStyleSheet("color: #4CAF50; font-weight: 600;")
            self.btn_etl.setEnabled(True)
            self.lbl_etl_status.setText("Ready to run ETL")
            self.lbl_etl_status.setStyleSheet("color: #4CAF50; font-weight: 600;")

            QMessageBox.information(self, "Success", f"Successfully loaded {total:,} rows!")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self._append_log(f"✗ Error: {str(e)}")

    def run_etl(self):
        try:
            self._append_log("Starting ETL pipeline...")
            snap = self.SNAPSHOT_MONTH.isoformat()

            # A. core_clients
            self._append_log("Processing core_clients...")
            sql_clients = (
                "INSERT INTO core_clients (client_id, limit_bal, sex, education, marital, age) "
                "SELECT ID, LIMIT_BAL, SEX, EDUCATION, MARITAL, AGE FROM stg_default_raw "
                "ON DUPLICATE KEY UPDATE limit_bal=VALUES(limit_bal), sex=VALUES(sex), "
                "education=VALUES(education), marital=VALUES(marital), age=VALUES(age);"
            )
            DBM.execute(sql_clients)

            # B. fact_statements
            self._append_log("Processing fact_statements...")
            DBM.execute("SET @snap := DATE(%s);", (snap,))
            sql_stmt = f"""
            INSERT INTO fact_statements (client_id, snapshot_month, month_offset, bill_amt, pay_amt, pay_status)
            SELECT ID, @snap, 1, BILL_AMT1, PAY_AMT1, PAY_0 FROM stg_default_raw
            UNION ALL SELECT ID, @snap, 2, BILL_AMT2, PAY_AMT2, PAY_2 FROM stg_default_raw
            UNION ALL SELECT ID, @snap, 3, BILL_AMT3, PAY_AMT3, PAY_3 FROM stg_default_raw
            UNION ALL SELECT ID, @snap, 4, BILL_AMT4, PAY_AMT4, PAY_4 FROM stg_default_raw
            UNION ALL SELECT ID, @snap, 5, BILL_AMT5, PAY_AMT5, PAY_5 FROM stg_default_raw
            UNION ALL SELECT ID, @snap, 6, BILL_AMT6, PAY_AMT6, PAY_6 FROM stg_default_raw
            ON DUPLICATE KEY UPDATE bill_amt=VALUES(bill_amt), pay_amt=VALUES(pay_amt), pay_status=VALUES(pay_status);
            """
            DBM.execute(sql_stmt)

            # C. labels
            self._append_log("Processing fact_labels...")
            sql_lbl = (
                "INSERT INTO fact_labels (client_id, snapshot_month, default_next_month) "
                "SELECT ID, @snap, default_payment_next_month FROM stg_default_raw "
                "ON DUPLICATE KEY UPDATE default_next_month=VALUES(default_next_month);"
            )
            DBM.execute(sql_lbl)

            # D. Create view
            self._append_log("Creating features view...")
            sql_view = """
            CREATE OR REPLACE VIEW vw_features_baseline AS
            SELECT
              c.client_id,
              s.snapshot_month,
              MAX(CASE WHEN s.month_offset=1 AND c.limit_bal>0 THEN s.bill_amt / c.limit_bal ELSE 0 END) AS util_last,
              AVG(CASE WHEN c.limit_bal>0 THEN s.bill_amt / c.limit_bal ELSE 0 END) AS util_mean_6m,
              SUM(s.pay_status <= 0) AS on_time_count_6m,
              SUM(s.pay_status >= 1) AS late_1p_count_6m,
              SUM(s.pay_status >= 2) AS late_2p_count_6m,
              MAX(s.pay_status) AS max_delay_6m,
              AVG(CASE WHEN s.bill_amt>0 THEN LEAST(GREATEST(s.pay_amt/s.bill_amt,0),2) ELSE 0 END) AS avg_pay_ratio_6m,
              STDDEV_SAMP(s.bill_amt) AS bill_volatility_6m,
              c.limit_bal, c.sex, c.education, c.marital, c.age
            FROM core_clients c
            JOIN fact_statements s USING (client_id)
            GROUP BY c.client_id, s.snapshot_month;
            """
            DBM.execute(sql_view)

            # E. Show sample
            df = DBM.query_df("SELECT * FROM vw_features_baseline LIMIT 100;")
            self.table.setModel(PandasModel(df))

            self._append_log("✓ ETL completed successfully!")
            self.lbl_etl_status.setText("✓ ETL completed")
            self.lbl_etl_status.setStyleSheet("color: #4CAF50; font-weight: 600;")

            QMessageBox.information(
                self,
                "Success",
                f"ETL pipeline completed!\nSnapshot: {snap}\nFeatures are ready for training."
            )

        except Exception as e:
            QMessageBox.critical(self, "ETL Error", str(e))
            self._append_log(f"✗ ETL Error: {str(e)}")
class FeaturesTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QLabel("Feature Engineering")
        header.setObjectName("headerTitle")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        info = QGroupBox("Coming Soon")
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel("This module will include:"))
        info_layout.addWidget(QLabel("• View and select features from database"))
        info_layout.addWidget(QLabel("• Create custom features"))
        info_layout.addWidget(QLabel("• Feature importance analysis"))
        info_layout.addWidget(QLabel("• Feature correlation matrix"))
        info.setLayout(info_layout)
        layout.addWidget(info)
        layout.addStretch()


class TrainTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QLabel("Model Training & Tuning")
        header.setObjectName("headerTitle")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        info = QGroupBox("Coming Soon")
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel("This module will include:"))
        info_layout.addWidget(QLabel("• Select ML algorithms (Logistic Regression, Random Forest, XGBoost, etc.)"))
        info_layout.addWidget(QLabel("• Train/Test split configuration"))
        info_layout.addWidget(QLabel("• Hyperparameter tuning"))
        info_layout.addWidget(QLabel("• Model performance metrics (ROC-AUC, Precision, Recall)"))
        info_layout.addWidget(QLabel("• Save trained models to registry"))
        info.setLayout(info_layout)
        layout.addWidget(info)
        layout.addStretch()


class ThresholdTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QLabel("Decision Threshold Optimization")
        header.setObjectName("headerTitle")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        info = QGroupBox("Coming Soon")
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel("This module will include:"))
        info_layout.addWidget(QLabel("• Interactive threshold slider"))
        info_layout.addWidget(QLabel("• Cost matrix configuration"))
        info_layout.addWidget(QLabel("• Confusion matrix visualization"))
        info_layout.addWidget(QLabel("• Precision-Recall trade-off analysis"))
        info_layout.addWidget(QLabel("• Business impact calculator"))
        info.setLayout(info_layout)
        layout.addWidget(info)
        layout.addStretch()


class ScoreExplainTab(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        header = QLabel("Scoring & Model Explainability")
        header.setObjectName("headerTitle")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(header)

        info = QGroupBox("Coming Soon")
        info_layout = QVBoxLayout()
        info_layout.addWidget(QLabel("This module will include:"))
        info_layout.addWidget(QLabel("• Batch scoring for new customers"))
        info_layout.addWidget(QLabel("• SHAP values for model interpretation"))
        info_layout.addWidget(QLabel("• Top-5 feature contributions per prediction"))
        info_layout.addWidget(QLabel("• Export predictions to database"))
        info_layout.addWidget(QLabel("• Risk score distribution analysis"))
        info.setLayout(info_layout)
        layout.addWidget(info)
        layout.addStretch()


# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Credit Risk — Late Payment Prediction System")
        self.resize(1400, 900)

        tabs = QTabWidget()
        tabs.addTab(HomeTab(), "🏠 Home")
        tabs.addTab(DataIntakeTab(), "📊 Data Intake")
        tabs.addTab(FeaturesTab(), "🔧 Features")
        tabs.addTab(TrainTab(), "🤖 Train Model")
        tabs.addTab(ThresholdTab(), "⚖ Threshold")
        tabs.addTab(ScoreExplainTab(), "📈 Score & Explain")
        self.setCentralWidget(tabs)


# -----------------------------
# App entry
# -----------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_theme(app)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
