from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGroupBox, QTableWidget, QTableWidgetItem, QComboBox,
    QScrollArea, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
import pandas as pd
import numpy as np

from ui.theme import AppTheme
from ui.components.kpi_card import KPICard
from ui.components.chart_placeholder import ChartPlaceholder
from utils.models import AppState
from utils.metrics import (
    confusion_at_threshold, precision, recall, f1_score,
    gains_curve_points, apply_rules
)
from utils.chart_renderer import (
    MatplotlibCanvas, plot_probability_distribution, plot_gains_curve,
    plot_calibration, plot_confusion_matrix
)

class AdminDashboardPage(QWidget):
    log_signal = pyqtSignal(str)
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.state = AppState()
        self.canvas_hist = None
        self.canvas_gains = None
        self.canvas_calib = None
        self.canvas_conf = None
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
        title = QLabel("Dashboard & Risk Analysis")
        title.setFont(AppTheme.get_font(16, bold=True))
        title.setObjectName("lblTitle")
        body_layout.addWidget(title)
        
        # KPI Summary
        kpi_layout = QHBoxLayout()
        kpi_layout.setSpacing(12)
        
        self.kpi_late_rate = KPICard("Avg PD (mean proba)", "N/A", AppTheme.PRIMARY)
        self.kpi_precision = KPICard("Precision (Hit rate)", "N/A", AppTheme.PRIMARY)
        self.kpi_recall = KPICard("Recall (Capture)", "N/A", AppTheme.SECONDARY)
        self.kpi_cost = KPICard("Expected Cost", "N/A", AppTheme.PRIMARY)
        self.kpi_flagged = KPICard("Declined / Total", "N/A", AppTheme.WARNING)
        
        kpi_layout.addWidget(self.kpi_late_rate)
        kpi_layout.addWidget(self.kpi_precision)
        kpi_layout.addWidget(self.kpi_recall)
        kpi_layout.addWidget(self.kpi_cost)
        kpi_layout.addWidget(self.kpi_flagged)
        
        body_layout.addLayout(kpi_layout)
        
        # Charts section
        charts_layout = QHBoxLayout()
        charts_layout.setSpacing(12)
        
        # Probability distribution
        hist_group = QGroupBox("Probability Distribution")
        hist_layout = QVBoxLayout(hist_group)
        self.canvas_hist = MatplotlibCanvas(width=5, height=4, dpi=80)
        hist_layout.addWidget(self.canvas_hist)
        charts_layout.addWidget(hist_group)
        
        # Gains curve
        gains_group = QGroupBox("Gains Curve (Cumulative capture)")
        gains_layout = QVBoxLayout(gains_group)
        self.canvas_gains = MatplotlibCanvas(width=5, height=4, dpi=80)
        gains_layout.addWidget(self.canvas_gains)
        charts_layout.addWidget(gains_group)
        
        body_layout.addLayout(charts_layout)
        
        # Second row of charts
        charts_layout2 = QHBoxLayout()
        charts_layout2.setSpacing(12)
        
        # Calibration
        calib_group = QGroupBox("Calibration (Reliability)")
        calib_layout = QVBoxLayout(calib_group)
        self.canvas_calib = MatplotlibCanvas(width=5, height=4, dpi=80)
        calib_layout.addWidget(self.canvas_calib)
        charts_layout2.addWidget(calib_group)
        
        # Confusion matrix
        conf_group = QGroupBox("Confusion @ cut-off")
        conf_layout = QVBoxLayout(conf_group)
        self.canvas_conf = MatplotlibCanvas(width=5, height=4, dpi=80)
        conf_layout.addWidget(self.canvas_conf)
        charts_layout2.addWidget(conf_group)
        
        body_layout.addLayout(charts_layout2)
        
        # Group analysis
        group_group = QGroupBox("Segment Analysis & Top-K")
        group_layout = QVBoxLayout(group_group)
        
        # Group selector
        group_row = QHBoxLayout()
        group_row.addWidget(QLabel("Group by:"))
        self.cmb_group = QComboBox()
        self.cmb_group.addItems(["PAY_0", "EDUCATION", "MARRIAGE", "SEX", "AGE_band", "LIMIT_BAL_band"])
        group_row.addWidget(self.cmb_group)
        
        btn_render = QPushButton("Render group chart")
        btn_render.clicked.connect(self.render_group_chart)
        group_row.addWidget(btn_render)
        group_row.addStretch()
        
        group_layout.addLayout(group_row)
        
        # Group chart placeholder
        group_layout.addWidget(ChartPlaceholder("Group Analysis Chart"))
        
        # Top-K table
        group_layout.addWidget(QLabel("Top-K High Risk Customers (by proba)"))
        self.tbl_top = QTableWidget()
        self.tbl_top.setColumnCount(7)
        self.tbl_top.setHorizontalHeaderLabels(
            ["Rank", "Customer ID", "Risk Score", "Default Prob", "Credit Limit", "Suggestions", "Action"]
        )
        self.tbl_top.setMinimumHeight(250)
        self.tbl_top.setAlternatingRowColors(True)
        group_layout.addWidget(self.tbl_top)
        
        # Suggestions/SHAP
        group_layout.addWidget(QLabel("Risk Mitigation Suggestions (Rule Engine)"))
        self.txt_explain = QTextEdit()
        self.txt_explain.setReadOnly(True)
        self.txt_explain.setPlainText(
            "Suggestions generated by rule engine. When SHAP is integrated, replace with local/global explanations."
        )
        self.txt_explain.setMaximumHeight(150)
        group_layout.addWidget(self.txt_explain)
        
        body_layout.addWidget(group_group)
        
        # Export button
        export_row = QHBoxLayout()
        export_row.addStretch()
        btn_export = QPushButton("Export CSV (flagged + suggestions)")
        btn_export.clicked.connect(self.export_flagged_with_suggestions)
        export_row.addWidget(btn_export)
        body_layout.addLayout(export_row)
        
        body_layout.addStretch()
        scroll.setWidget(body)
        layout.addWidget(scroll)
    
    def refresh_all(self, state: AppState):
        """Refresh dashboard with new state"""
        self.state = state
        
        if self.state.scored_df is None:
            for c in [self.kpi_late_rate, self.kpi_precision, self.kpi_recall, self.kpi_cost, self.kpi_flagged]:
                c.update_value("N/A")
            self.tbl_top.setRowCount(0)
            return
        
        df = self.state.scored_df.copy()
        proba = df["proba"].values
        t = self.state.threshold_active
        
        # Update KPIs
        late_rate = float(np.mean(proba))
        flagged = int((proba >= t).sum())
        total = len(df)
        approval = 1 - flagged / total if total > 0 else 0.0
        
        self.kpi_late_rate.update_value(f"{late_rate*100:.2f}%")
        self.kpi_flagged.update_value(f"{flagged}/{total}\n({flagged/total*100:.1f}% decline)")
        
        # If labels available
        if self.state.y_true is not None:
            y = self.state.y_true
            cm = confusion_at_threshold(y, proba, t)
            p = precision(cm["TP"], cm["FP"])
            r = recall(cm["TP"], cm["FN"])
            
            self.kpi_precision.update_value(f"{p:.3f}")
            self.kpi_recall.update_value(f"{r:.3f}")
            
            exp_cost = cm["FN"] * self.state.cfn + cm["FP"] * self.state.cfp
            self.kpi_cost.update_value(f"{exp_cost:,.0f}")
        else:
            self.kpi_precision.update_value("N/A")
            self.kpi_recall.update_value("N/A")
            self.kpi_cost.update_value("N/A")
        
        # Populate top-K table
        self.populate_topk_table()
    
    def render_charts(self):
        """Render all charts with current data"""
        if self.state.scored_df is None or self.state.proba is None:
            return
        
        try:
            from utils.chart_renderer import (
                plot_probability_distribution, plot_gains_curve,
                plot_calibration, plot_confusion_matrix
            )
            
            proba = self.state.proba
            y_true = self.state.y_true
            t = self.state.threshold_active
            
            # Render probability distribution
            if self.canvas_hist and self.canvas_hist.figure:
                ax = self.canvas_hist.figure.add_subplot(111)
                plot_probability_distribution(ax, proba, y_true)
                self.canvas_hist.draw()
            
            # Render gains curve
            if y_true is not None and self.canvas_gains and self.canvas_gains.figure:
                ax = self.canvas_gains.figure.add_subplot(111)
                plot_gains_curve(ax, y_true, proba)
                self.canvas_gains.draw()
            
            # Render calibration
            if y_true is not None and self.canvas_calib and self.canvas_calib.figure:
                ax = self.canvas_calib.figure.add_subplot(111)
                plot_calibration(ax, y_true, proba)
                self.canvas_calib.draw()
            
            # Render confusion matrix
            if y_true is not None and self.canvas_conf and self.canvas_conf.figure:
                cm = confusion_at_threshold(y_true, proba, t)
                ax = self.canvas_conf.figure.add_subplot(111)
                plot_confusion_matrix(ax, cm["TP"], cm["FP"], cm["FN"], cm["TN"])
                self.canvas_conf.draw()
        
        except Exception as e:
            print(f"[v0] Error rendering charts: {e}")
    
    def render_group_chart(self):
        """Render group analysis chart"""
        if self.state.scored_df is None:
            return
        
        df = self.state.scored_df.copy()
        
        # Create age bands if needed
        if "AGE" in df.columns:
            df["AGE_band"] = pd.cut(df["AGE"], bins=[0, 25, 35, 45, 60, 120], include_lowest=True).astype(str)
        else:
            df["AGE_band"] = "NA"
        
        # Create limit bands if needed
        if "LIMIT_BAL" in df.columns:
            q = df["LIMIT_BAL"].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
            q[0] = max(1.0, q[0])
            df["LIMIT_BAL_band"] = pd.cut(df["LIMIT_BAL"], bins=q, include_lowest=True, duplicates="drop").astype(str)
        else:
            df["LIMIT_BAL_band"] = "NA"
        
        feat = self.cmb_group.currentText()
        if feat not in df.columns:
            return
        
        grp = df.groupby(feat)["proba"].mean().sort_values(ascending=False)
        # Chart rendering would go here (matplotlib integration)
    
    def populate_topk_table(self, k: int = 50):
        """Populate top-K high risk customers table"""
        if self.state.scored_df is None:
            return
        
        df = self.state.scored_df.copy().sort_values("proba", ascending=False).head(k)
        df["suggestions"] = [" | ".join(apply_rules(row, float(row["proba"]), self.state.rules)) for _, row in df.iterrows()]
        
        # Select columns to display
        cols_to_show = ["proba", "suggestions"] + [c for c in df.columns if c not in ["proba", "suggestions", "flag"]]
        df_display = df[cols_to_show[:7]]  # Limit to 7 columns
        
        self.tbl_top.setColumnCount(len(df_display.columns))
        self.tbl_top.setHorizontalHeaderLabels(df_display.columns.tolist())
        self.tbl_top.setRowCount(len(df_display))
        
        for row, (_, record) in enumerate(df_display.iterrows()):
            for col, value in enumerate(record):
                item = QTableWidgetItem(str(value)[:50])  # Truncate long values
                self.tbl_top.setItem(row, col, item)
    
    def export_flagged_with_suggestions(self):
        """Export flagged customers with suggestions"""
        if self.state.scored_df is None:
            return
        
        from PyQt6.QtWidgets import QFileDialog
        
        df = self.state.scored_df.copy()
        df = df[df["proba"] >= self.state.threshold_active].copy()
        df["suggestions"] = [" | ".join(apply_rules(row, float(row["proba"]), self.state.rules)) for _, row in df.iterrows()]
        
        path, _ = QFileDialog.getSaveFileName(self, "Save flagged report", "flagged_report.csv", "CSV Files (*.csv)")
        if not path:
            return
        
        df.to_csv(path, index=False)
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Export", f"Exported to {path}")
