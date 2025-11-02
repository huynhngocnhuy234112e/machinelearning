from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGroupBox, QFileDialog, QSlider, QTableWidget, QTableWidgetItem,
    QDoubleSpinBox, QMessageBox, QScrollArea, QLineEdit, QCheckBox,
    QHeaderView, QPlainTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import pandas as pd
import numpy as np
import os

from ui.theme import AppTheme
from ui.components.kpi_card import KPICard
from utils.models import AppState
from utils.metrics import (
    confusion_at_threshold, precision, recall, f1_score,
    parse_binary_labels, demo_score
)

class AdminInputPage(QWidget):
    results_ready = pyqtSignal()
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.state = AppState()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(20, 20, 20, 20)
        body_layout.setSpacing(16)
        
        # Title
        title = QLabel("Data Input & Risk Scoring")
        title.setFont(AppTheme.get_font(16, bold=True))
        title.setObjectName("lblTitle")
        body_layout.addWidget(title)
        
        # A) Data & Schema section
        grpA = QGroupBox("A) Data & Schema")
        a_layout = QVBoxLayout(grpA)
        
        row = QHBoxLayout()
        self.file_label = QLabel("No file selected")
        self.file_label.setObjectName("lblSecondary")
        row.addWidget(self.file_label, 1)
        
        btn_pick = QPushButton("Choose CSV…")
        btn_pick.clicked.connect(self.pick_csv)
        row.addWidget(btn_pick)
        
        btn_preview = QPushButton("Preview 100")
        btn_preview.clicked.connect(self.preview)
        row.addWidget(btn_preview)
        
        self.chk_preview = QCheckBox("Show preview")
        self.chk_preview.setChecked(True)
        self.chk_preview.toggled.connect(lambda on: self.tbl_preview.setVisible(on))
        row.addWidget(self.chk_preview)
        
        a_layout.addLayout(row)
        
        self.lbl_schema = QLabel("Schema: —")
        self.lbl_schema.setWordWrap(True)
        self.lbl_schema.setMinimumHeight(40)
        a_layout.addWidget(self.lbl_schema)
        
        # Preview table
        self.tbl_preview = QTableWidget()
        self.tbl_preview.setAlternatingRowColors(True)
        self.tbl_preview.setMaximumHeight(250)
        self.tbl_preview.verticalHeader().setVisible(False)
        hh = self.tbl_preview.horizontalHeader()
        hh.setStretchLastSection(True)
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        a_layout.addWidget(self.tbl_preview)
        
        body_layout.addWidget(grpA)
        
        # B) Scoring section
        grpB = QGroupBox("B) Scoring")
        b_layout = QVBoxLayout(grpB)
        
        row_score = QHBoxLayout()
        self.txt_score = QLineEdit()
        self.txt_score.setReadOnly(True)
        row_score.addWidget(self.txt_score)
        
        btn_pick_score = QPushButton("Choose file…")
        btn_pick_score.clicked.connect(self.pick_score_file)
        row_score.addWidget(btn_pick_score)
        
        self.btn_score = QPushButton("Run Scoring")
        self.btn_score.clicked.connect(self.run_scoring)
        row_score.addWidget(self.btn_score)
        
        self.chk_flagged = QCheckBox("Flagged only (≥ t)")
        self.chk_flagged.stateChanged.connect(self.update_scored_view)
        row_score.addWidget(self.chk_flagged)
        
        b_layout.addLayout(row_score)
        
        self.lbl_summary = QLabel("—")
        b_layout.addWidget(self.lbl_summary)
        
        self.tbl_scored = QTableWidget()
        self.tbl_scored.setAlternatingRowColors(True)
        self.tbl_scored.setMinimumHeight(220)
        self.tbl_scored.verticalHeader().setVisible(False)
        self.tbl_scored.horizontalHeader().setStretchLastSection(True)
        b_layout.addWidget(self.tbl_scored)
        
        body_layout.addWidget(grpB)
        
        # C) Threshold & Cost section
        grpC = QGroupBox("C) Threshold & Cost (Live)")
        c_grid = QVBoxLayout(grpC)
        
        # Cost parameters
        cost_row = QHBoxLayout()
        cost_row.addWidget(QLabel("C_FN:"))
        self.sp_cfn = QDoubleSpinBox()
        self.sp_cfn.setMaximum(1e9)
        self.sp_cfn.setValue(self.state.cfn)
        cost_row.addWidget(self.sp_cfn)
        
        cost_row.addWidget(QLabel("C_FP:"))
        self.sp_cfp = QDoubleSpinBox()
        self.sp_cfp.setMaximum(1e9)
        self.sp_cfp.setValue(self.state.cfp)
        cost_row.addWidget(self.sp_cfp)
        cost_row.addStretch()
        c_grid.addLayout(cost_row)
        
        # Threshold slider
        threshold_row = QHBoxLayout()
        threshold_row.addWidget(QLabel("Threshold:"))
        
        self.sl_t = QSlider(Qt.Orientation.Horizontal)
        self.sl_t.setRange(0, 1000)
        self.sl_t.setValue(int(self.state.threshold_active * 1000))
        self.sl_t.valueChanged.connect(self.on_slider_t)
        threshold_row.addWidget(self.sl_t)
        
        self.txt_t = QDoubleSpinBox()
        self.txt_t.setDecimals(3)
        self.txt_t.setRange(0, 1)
        self.txt_t.setSingleStep(0.005)
        self.txt_t.setValue(self.state.threshold_active)
        self.txt_t.valueChanged.connect(self.on_text_t)
        threshold_row.addWidget(self.txt_t)
        
        c_grid.addLayout(threshold_row)
        
        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        
        btn_find_star = QPushButton("Find t* (Min Cost)")
        btn_find_star.clicked.connect(self.find_star_threshold)
        btn_row.addWidget(btn_find_star)
        
        btn_set_active = QPushButton("Set Active")
        btn_set_active.clicked.connect(self.save_active_threshold)
        btn_row.addWidget(btn_set_active)
        
        c_grid.addLayout(btn_row)
        
        # KPI cards
        kpi_row = QHBoxLayout()
        self.card_conf = KPICard("Confusion @t", "N/A", AppTheme.PRIMARY)
        self.card_cost = KPICard("Expected Cost @t", "N/A", AppTheme.PRIMARY)
        kpi_row.addWidget(self.card_conf)
        kpi_row.addWidget(self.card_cost)
        c_grid.addLayout(kpi_row)
        
        body_layout.addWidget(grpC)
        
        # Export button
        export_row = QHBoxLayout()
        export_row.addStretch()
        btn_export = QPushButton("Export predictions.csv")
        btn_export.clicked.connect(self.export_predictions)
        export_row.addWidget(btn_export)
        body_layout.addLayout(export_row)
        
        body_layout.addStretch()
        scroll.setWidget(body)
        layout.addWidget(scroll)
        
        self.update_threshold_cards()
    
    def pick_csv(self):
        """Choose CSV file"""
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "CSV Error", f"Cannot read CSV:\n{e}")
            return
        
        self.state.df_loaded = df
        self.file_label.setText(f"✓ {os.path.basename(path)}")
        self.check_schema()
        self.preview()
    
    def preview(self):
        """Show preview of first 100 rows"""
        if self.state.df_loaded is None:
            return
        df = self.state.df_loaded.head(100)
        self._set_table(self.tbl_preview, df)
    
    def check_schema(self):
        """Validate schema"""
        df = self.state.df_loaded
        if df is None:
            self.lbl_schema.setText("Schema: —")
            return
        
        target = self.state.schema.get("target", "default.payment.next.month")
        ok = target in df.columns
        pos_rate = None
        try:
            if ok and set(pd.unique(df[target])) <= {0, 1}:
                pos_rate = float(df[target].mean()) * 100.0
        except Exception:
            pos_rate = None
        
        msg = f"Target='{target}' → {'OK' if ok else 'MISSING'} • Rows={len(df):,}, Cols={df.shape[1]}"
        if pos_rate is not None:
            msg += f" • Positive rate={pos_rate:.2f}%"
        self.lbl_schema.setText(msg)
    
    def pick_score_file(self):
        """Choose file to score"""
        path, _ = QFileDialog.getOpenFileName(self, "Choose file to score", "", "CSV Files (*.csv)")
        if path:
            self.txt_score.setText(path)
    
    def run_scoring(self):
        """Run scoring on CSV file"""
        try:
            path = self.txt_score.text().strip()
            if not path and self.state.df_loaded is not None:
                tmp = os.path.join(os.getcwd(), "_tmp_to_score.csv")
                self.state.df_loaded.to_csv(tmp, index=False)
                path = tmp
                self.txt_score.setText(path)
            
            if not path or not os.path.exists(path):
                QMessageBox.warning(self, "Scoring", "Please select a valid file.")
                return
            
            df = pd.read_csv(path)
            
            # Clean numeric columns
            for c in [col for col in df.columns if col.startswith("PAY_")]:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            if "LIMIT_BAL" in df.columns:
                df["LIMIT_BAL"] = pd.to_numeric(df["LIMIT_BAL"], errors="coerce").fillna(0)
            
            # Calculate probabilities
            proba = demo_score(df)
            proba = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Get labels if available
            target = self.state.schema.get("target", "default.payment.next.month")
            if target in df.columns:
                y_true = parse_binary_labels(df[target])
            else:
                y_true = None
            
            out = df.copy()
            out["proba"] = proba
            out["flag"] = (out["proba"] >= self.state.threshold_active).astype(int)
            
            self.state.scored_df = out
            self.state.proba = proba
            self.state.y_true = y_true
            
            self.update_scored_view()
            self.update_threshold_cards()
            self.results_ready.emit()
            
        except Exception as e:
            QMessageBox.critical(self, "Scoring Error", f"Error during scoring:\n{e}")
    
    def update_scored_view(self):
        """Update scored results table"""
        if self.state.scored_df is None:
            return
        
        df = self.state.scored_df.copy()
        if self.chk_flagged.isChecked():
            df = df[df["flag"] == 1]
        
        self._set_table(self.tbl_scored, df.head(1000))
        
        flagged = int((self.state.scored_df["flag"] == 1).sum())
        total = len(self.state.scored_df)
        msg = f"Flagged: {flagged}/{total} ({(flagged/total*100):.2f}%) @t={self.state.threshold_active:.3f}"
        
        if self.state.y_true is not None:
            cm = confusion_at_threshold(self.state.y_true, self.state.proba, self.state.threshold_active)
            p = precision(cm["TP"], cm["FP"])
            r = recall(cm["TP"], cm["FN"])
            msg += f" • P={p:.3f}, R={r:.3f}"
        
        self.lbl_summary.setText(msg)
    
    def on_slider_t(self, val: int):
        """Handle slider change"""
        t = val / 1000.0
        self.txt_t.blockSignals(True)
        self.txt_t.setValue(t)
        self.txt_t.blockSignals(False)
        
        self.state.threshold_active = t
        if self.state.scored_df is not None:
            self.state.scored_df["flag"] = (self.state.scored_df["proba"] >= t).astype(int)
            self.update_scored_view()
        self.update_threshold_cards()
    
    def on_text_t(self, t: float):
        """Handle text input change"""
        self.sl_t.blockSignals(True)
        self.sl_t.setValue(int(t * 1000))
        self.sl_t.blockSignals(False)
        
        self.state.threshold_active = t
        if self.state.scored_df is not None:
            self.state.scored_df["flag"] = (self.state.scored_df["proba"] >= t).astype(int)
            self.update_scored_view()
        self.update_threshold_cards()
    
    def find_star_threshold(self):
        """Find optimal threshold minimizing cost"""
        if self.state.y_true is None or self.state.proba is None:
            QMessageBox.information(self, "Find t*", "Need y_true & proba (use file with target).")
            return
        
        cfn = float(self.sp_cfn.value())
        cfp = float(self.sp_cfp.value())
        best_t, best_cost = 0.5, float("inf")
        
        grid = np.linspace(0, 1, 401)
        for t in grid:
            cm = confusion_at_threshold(self.state.y_true, self.state.proba, t)
            cost = cm["FN"] * cfn + cm["FP"] * cfp
            if cost < best_cost:
                best_cost, best_t = cost, t
        
        self.txt_t.setValue(float(best_t))
        QMessageBox.information(self, "Find t*", f"t* ≈ {best_t:.3f} (min cost ≈ {best_cost:,.0f})")
    
    def save_active_threshold(self):
        """Save active threshold"""
        self.state.threshold_active = self.txt_t.value()
        QMessageBox.information(self, "Active threshold", f"Set threshold={self.state.threshold_active:.3f}")
        self.update_threshold_cards()
    
    def export_predictions(self):
        """Export predictions to CSV"""
        if self.state.scored_df is None:
            QMessageBox.warning(self, "Export", "No scored data available.")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Save predictions", "predictions.csv", "CSV Files (*.csv)")
        if not path:
            return
        
        self.state.scored_df.to_csv(path, index=False)
        QMessageBox.information(self, "Export", f"Exported to {path}")
    
    def update_threshold_cards(self):
        """Update KPI cards"""
        if self.state.y_true is None or self.state.proba is None:
            self.card_conf.update_value("N/A")
            self.card_cost.update_value("N/A")
            return
        
        t = self.state.threshold_active
        cm = confusion_at_threshold(self.state.y_true, self.state.proba, t)
        p = precision(cm["TP"], cm["FP"])
        r = recall(cm["TP"], cm["FN"])
        f1 = f1_score(p, r)
        
        conf_text = f"TP:{cm['TP']} FP:{cm['FP']} FN:{cm['FN']} TN:{cm['TN']}\nP={p:.3f} R={r:.3f} F1={f1:.3f}"
        self.card_conf.update_value(conf_text)
        
        cfn = float(self.sp_cfn.value())
        cfp = float(self.sp_cfp.value())
        exp_cost = cm["FN"] * cfn + cm["FP"] * cfp
        self.card_cost.update_value(f"{exp_cost:,.0f}")
    
    def _set_table(self, table: QTableWidget, df: pd.DataFrame):
        """Populate table with dataframe"""
        table.setColumnCount(len(df.columns))
        table.setHorizontalHeaderLabels(df.columns.tolist())
        table.setRowCount(len(df))
        
        for row, (_, record) in enumerate(df.iterrows()):
            for col, value in enumerate(record):
                item = QTableWidgetItem(str(value))
                table.setItem(row, col, item)
