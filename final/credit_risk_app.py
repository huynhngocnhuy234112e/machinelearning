# credit_risk_app.py
# Run:
#   pip install PyQt6 pandas numpy matplotlib
#   python credit_risk_app.py

from __future__ import annotations

import os, sys, json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

# Matplotlib: dùng backend Qt tương thích PyQt6
import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt6.QtCore import Qt, QAbstractTableModel, pyqtSignal
from PyQt6.QtGui import QAction   # <-- QAction nằm ở đây trong PyQt6
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QTableView, QGroupBox, QGridLayout, QPlainTextEdit,
    QLineEdit, QDoubleSpinBox, QSlider, QCheckBox, QDialog, QTextEdit, QSizePolicy,
    QComboBox, QMessageBox, QScrollArea, QHeaderView

)

APP_TITLE = "Credit Risk — Compute & Dashboard (2 Tabs)"

# ===========================
# Utilities & Metrics
# ===========================

class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df.reset_index(drop=True)

    def rowCount(self, parent=None):
        return len(self._df)

    def columnCount(self, parent=None):
        return self._df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            val = self._df.iat[index.row(), index.column()]
            return "" if pd.isna(val) else str(val)
        return None

    def headerData(self, section, orientation, role):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._df.columns[section])
            else:
                return str(section)
        return None


def confusion_at_threshold(y_true: np.ndarray, proba: np.ndarray, t: float) -> Dict[str, int]:
    y_pred = (proba >= t).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def precision(tp, fp):
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(tp, fn):
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(p, r):
    return 2*p*r/(p+r) if (p+r) > 0 else 0.0


def brier_score(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float(np.mean((proba - y_true)**2))


def pr_curve(y_true: np.ndarray, proba: np.ndarray) -> List[Tuple[float, float, float]]:
    """
    Trả về danh sách (threshold, precision, recall) trên các ngưỡng unique của proba (sắp giảm dần).
    """
    order = np.argsort(-proba)
    y_sorted = y_true[order]
    proba_sorted = proba[order]
    tp = 0
    fp = 0
    total_pos = int(np.sum(y_true == 1))
    seen = set()
    curve = []
    for i in range(len(proba_sorted)):
        prob = float(proba_sorted[i])
        if prob not in seen:
            p = precision(tp, fp)
            r = tp / total_pos if total_pos > 0 else 0.0
            curve.append((prob, p, r))
            seen.add(prob)
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
    p = precision(tp, fp)
    r = tp / total_pos if total_pos > 0 else 0.0
    curve.append((0.0, p, r))
    return curve
def gains_curve_points(y_true: np.ndarray, proba: np.ndarray):
    """Trả về (x%, y%) cho đường Gains: %population vs %positives captured."""
    order = np.argsort(-proba)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted)
    total_pos = np.sum(y_true)
    n = len(y_true)
    x = np.arange(1, n+1) / n
    y = cum_pos / max(total_pos, 1)
    return x, y

def plot_confusion_heatmap(ax, cm: Dict[str, int]):
    """Vẽ heatmap 2x2 kèm số liệu lên axes đã cho."""
    mat = np.array([[cm["TP"], cm["FP"]],
                    [cm["FN"], cm["TN"]]], dtype=float)
    ax.clear()
    im = ax.imshow(mat, cmap="Blues")
    for (i, j), v in np.ndenumerate(mat):
        ax.text(j, i, f"{int(v):,}", ha="center", va="center", fontsize=11)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred=1", "Pred=0"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True=1", "True=0"])
    ax.set_title("Confusion Matrix @ cut-off")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
# --- thêm helper (đặt gần các util) ---
def parse_binary_labels(series: pd.Series) -> np.ndarray | None:
    """
    Cố gắng chuyển chuỗi nhãn bất kỳ về mảng 0/1.
    Chấp nhận: 0/1, True/False, Y/N, Yes/No, Default/Non-default, Paid/Unpaid...
    Trả về None nếu không thể suy ra.
    """
    if series is None:
        return None
    s = series.copy()

    # Chuẩn hoá chuỗi
    if s.dtype == object:
        s = s.astype(str).str.strip().str.lower()
        mapping = {
            "1": 1, "0": 0, "true": 1, "false": 0, "y": 1, "n": 0, "yes": 1, "no": 0,
            "default": 1, "non-default": 0, "bad": 1, "good": 0, "overdue": 1, "paid": 0,
            "defaulter": 1, "non defaulter": 0, "delinquent": 1, "non-delinquent": 0
        }
        s = s.map(mapping).astype("float64")
    # Ép số (nếu đã là số)
    s = pd.to_numeric(s, errors="coerce")
    # Nếu có giá trị >1, quy về (>=1 → 1; còn lại 0)
    # Nếu chỉ 0/1 thì giữ nguyên
    if s.notna().sum() == 0:
        return None
    if (s.dropna().isin([0, 1]).all()):
        return s.fillna(0).astype(int).values
    return (s.fillna(0) >= 1).astype(int).values



def demo_score(df: pd.DataFrame) -> np.ndarray:
    """
    Fallback xác suất đơn giản nếu chưa tích hợp model thật:
    - Cộng các PAY_* (cắt dưới 0), thêm 1 - LIMIT_BAL/max(LIMIT_BAL).
    - Chuẩn hóa [0,1].
    """
    cols = df.columns
    s = np.zeros(len(df), dtype=float)
    pays = [c for c in cols if c.startswith("PAY_")]
    if pays:
        for c in pays:
            try:
                s += df[c].clip(lower=0).astype(float).values
            except Exception:
                pass
    if "LIMIT_BAL" in cols:
        lb = pd.to_numeric(df["LIMIT_BAL"], errors="coerce").fillna(0).values
        s += (1.0 - (lb / (np.max(lb) + 1e-9)))
    # Normalize
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    return s


# ===========================
# Rule-based Suggestions
# ===========================

def default_rules() -> Dict[str, Any]:
    return {
        "rules": [
            {"if": {"PAY_0_ge": 2}, "then": "Nhắc nợ sớm (call trong 48h)."},
            {"if": {"PAY_2_plus_count_ge1": 2}, "then": "Theo dõi sát & chặn tăng hạn mức."},
            {"if": {"utilization_ge": 0.85}, "then": "Đề xuất điều chỉnh hạn mức / yêu cầu trả thêm."},
            {"if": {"AGE_le": 25, "proba_ge": 0.7}, "then": "Yêu cầu bảo đảm bổ sung / đồng bảo lãnh."}
        ]
    }


def compute_utilization(row: pd.Series) -> float | None:
    bills = [c for c in row.index if c.startswith("BILL_AMT")]
    pays = [c for c in row.index if c.startswith("PAY_AMT")]
    if "LIMIT_BAL" not in row.index:
        return None
    try:
        limit_bal = float(row["LIMIT_BAL"])
        total_bill = float(sum([row[c] for c in bills])) if bills else 0.0
        total_pay = float(sum([row[c] for c in pays])) if pays else 0.0
        util = (total_bill - total_pay) / (limit_bal * max(len(bills), 1))
        util = max(0.0, min(1.5, util))  # clamp
        return float(util)
    except Exception:
        return None


def apply_rules(row: pd.Series, proba: float, rules: Dict[str, Any]) -> List[str]:
    """
    Đánh giá các rule đơn giản, trả về tối đa 3 gợi ý.
    """
    suggestions = []
    util = compute_utilization(row)
    pay_cols = [c for c in row.index if c.startswith("PAY_")]
    pay_ge1_count = 0
    for c in pay_cols:
        try:
            if float(pd.to_numeric(row[c], errors="coerce")) >= 1:
                pay_ge1_count += 1
        except Exception:
            pass

    for r in rules.get("rules", []):
        cond = r.get("if", {})
        ok = True
        for k, v in cond.items():
            if k == "utilization_ge":
                if util is None or not (util >= float(v)): ok = False; break
            elif k == "PAY_2_plus_count_ge1":
                if not (pay_ge1_count >= int(v)): ok = False; break
            elif k == "proba_ge":
                if not (proba >= float(v)): ok = False; break
            elif k.endswith("_ge"):
                key = k[:-3]
                if key not in row.index or not (pd.to_numeric(row[key], errors="coerce") >= float(v)):
                    ok = False; break
            elif k.endswith("_le"):
                key = k[:-3]
                if key not in row.index or not (pd.to_numeric(row[key], errors="coerce") <= float(v)):
                    ok = False; break
            else:  # equality as fallback
                if key not in row.index or not (pd.to_numeric(row[key], errors="coerce") == float(v)):
                    ok = False; break
        if ok:
            suggestions.append(r.get("then", ""))

    return suggestions[:3]


# ===========================
# App State
# ===========================

DEFAULT_SCHEMA = {"target": "default.payment.next.month"}

@dataclass
class AppState:
    artifacts_dir: Optional[str] = None
    schema: Dict[str, Any] = field(default_factory=lambda: DEFAULT_SCHEMA.copy())
    threshold_active: float = 0.5
    cfn: float = 10.0
    cfp: float = 1.0

    df_loaded: Optional[pd.DataFrame] = None
    scored_df: Optional[pd.DataFrame] = None
    y_true: Optional[np.ndarray] = None
    proba: Optional[np.ndarray] = None

    rules: Dict[str, Any] = field(default_factory=default_rules)


# ===========================
# Reusable UI
# ===========================

class KpiCard(QGroupBox):
    def __init__(self, title: str):
        super().__init__(title)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout()
        self.value = QLabel("N/A")
        self.value.setStyleSheet("font-size:18px; font-weight:600;")
        self.sub = QLabel("—")
        self.sub.setStyleSheet("color:#666;")
        layout.addWidget(self.value); layout.addWidget(self.sub)
        self.setLayout(layout)

    def set_values(self, main: str, sub: str):
        self.value.setText(main)
        self.sub.setText(sub)

class MplCanvas(FigureCanvas):
        def __init__(self, width=6, height=3.6, dpi=100):
            fig = Figure(figsize=(width, height), dpi=dpi)  # <- bỏ layout='constrained'
            super().__init__(fig)
            self.ax = fig.add_subplot(111)
            self.setStyleSheet("background:white; border:1px solid #e5e7eb; border-radius:8px;")
            self.setMinimumHeight(int(height * 80))

        def clear(self):
            self.ax.clear()
            self.draw()



# ===========================
# Tab 1 — Compute & Raw Results
# ===========================
# ===========================
# Tab 1 — Compute & Raw Results (re-ordered)
# ===========================

class ComputeTab(QWidget):
    log_signal = pyqtSignal(str)
    results_ready = pyqtSignal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.build()

    def build(self):
        # ----- bọc toàn bộ nội dung tab trong QScrollArea -----
        outer = QVBoxLayout(self)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll)

        self.body = QWidget()
        self.scroll.setWidget(self.body)
        layout = QVBoxLayout(self.body)  # <— dùng layout này để add các group box

        # ===== A) Data & Schema =====
        grpA = QGroupBox("A) Data & Schema")
        a = QVBoxLayout(grpA)

        # thanh trên
        row = QHBoxLayout()
        self.txtPath = QLineEdit();
        self.txtPath.setReadOnly(True)
        btnPick = QPushButton("Chọn CSV…");
        btnPick.clicked.connect(self.pick_csv)
        self.btnPreview = QPushButton("Preview 100");
        self.btnPreview.clicked.connect(self.preview)
        # nút thu gọn preview
        self.chkPreview = QCheckBox("Hiện preview");
        self.chkPreview.setChecked(True)
        self.chkPreview.toggled.connect(lambda on: self.tblPreview.setVisible(on))

        row.addWidget(QLabel("Dataset:"))
        row.addWidget(self.txtPath)
        row.addWidget(btnPick)
        row.addWidget(self.btnPreview)
        row.addWidget(self.chkPreview)

        self.lblSchema = QLabel("Schema: —")

        # bảng preview (giới hạn chiều cao)
        self.tblPreview = QTableView()
        self.tblPreview.setAlternatingRowColors(True)
        self.tblPreview.setWordWrap(False)
        self.tblPreview.setMaximumHeight(300)  # <<< quan trọng
        self.tblPreview.verticalHeader().setVisible(False)
        hh = self.tblPreview.horizontalHeader()
        hh.setStretchLastSection(True)
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        a.addLayout(row)
        a.addWidget(self.lblSchema)
        a.addWidget(self.tblPreview)

        # ===== B) Scoring =====
        grpB = QGroupBox("B) Scoring")
        b = QVBoxLayout(grpB)
        r = QHBoxLayout()
        self.txtScore = QLineEdit();
        self.txtScore.setReadOnly(True)
        pickScore = QPushButton("Chọn file để chấm…");
        pickScore.clicked.connect(self.pick_score_file)
        self.btnScore = QPushButton("Run Scoring");
        self.btnScore.clicked.connect(self.run_scoring)
        self.chkFlaggedOnly = QCheckBox("Chỉ hiển thị flagged (≥ t)");
        self.chkFlaggedOnly.stateChanged.connect(self.update_scored_view)
        r.addWidget(self.txtScore);
        r.addWidget(pickScore);
        r.addWidget(self.btnScore);
        r.addWidget(self.chkFlaggedOnly)
        self.lblSummary = QLabel("—")

        self.tblScored = QTableView()
        self.tblScored.setAlternatingRowColors(True)
        self.tblScored.setMinimumHeight(220)  # cao tối thiểu vừa mắt
        self.tblScored.verticalHeader().setVisible(False)
        self.tblScored.horizontalHeader().setStretchLastSection(True)

        b.addLayout(r);
        b.addWidget(self.lblSummary);
        b.addWidget(self.tblScored)

        # ===== C) Threshold & Cost + Cost Curve =====
        grpC = QGroupBox("C) Threshold & Cost (Live)")
        c_grid = QGridLayout(grpC)
        self.spCFN = QDoubleSpinBox();
        self.spCFN.setMaximum(1e9);
        self.spCFN.setValue(self.state.cfn)
        self.spCFP = QDoubleSpinBox();
        self.spCFP.setMaximum(1e9);
        self.spCFP.setValue(self.state.cfp)
        self.slT = QSlider(Qt.Orientation.Horizontal);
        self.slT.setRange(0, 1000);
        self.slT.setValue(int(self.state.threshold_active * 1000))
        self.txtT = QDoubleSpinBox();
        self.txtT.setDecimals(3);
        self.txtT.setRange(0, 1);
        self.txtT.setSingleStep(0.005);
        self.txtT.setValue(self.state.threshold_active)
        self.btnFindStar = QPushButton("Find t* (Min Cost)")
        self.btnSetActive = QPushButton("Set Active")
        c_grid.addWidget(QLabel("C_FN"), 0, 0);
        c_grid.addWidget(self.spCFN, 0, 1)
        c_grid.addWidget(QLabel("C_FP"), 0, 2);
        c_grid.addWidget(self.spCFP, 0, 3)
        c_grid.addWidget(QLabel("Threshold"), 1, 0);
        c_grid.addWidget(self.slT, 1, 1, 1, 2);
        c_grid.addWidget(self.txtT, 1, 3)
        c_grid.addWidget(self.btnFindStar, 2, 2);
        c_grid.addWidget(self.btnSetActive, 2, 3)

        self.cardConf = KpiCard("Confusion @t")
        self.cardCost = KpiCard("Expected Cost @t")
        kpi = QHBoxLayout();
        kpi.setSpacing(10);
        kpi.addWidget(self.cardConf);
        kpi.addWidget(self.cardCost)

        self.figCost = MplCanvas(width=5, height=2.8)
        self.figCost.setMinimumHeight(240)

        boxC = QVBoxLayout();
        boxC.addLayout(c_grid);
        boxC.addLayout(kpi);
        boxC.addWidget(self.figCost);
        grpC.setLayout(boxC)

        # Export
        rowx = QHBoxLayout()
        self.btnExport = QPushButton("Export predictions.csv");
        self.btnExport.clicked.connect(self.export_predictions)
        rowx.addStretch(1);
        rowx.addWidget(self.btnExport)

        # add vào body (để được scroll)
        layout.addWidget(grpA)
        layout.addWidget(grpB)
        layout.addWidget(grpC)
        layout.addLayout(rowx)

        # signals
        self.slT.valueChanged.connect(self.on_slider_t)
        self.txtT.valueChanged.connect(self.on_text_t)
        self.btnFindStar.clicked.connect(self.find_star_threshold)
        self.btnSetActive.clicked.connect(self.save_active_threshold)

        self.update_threshold_cards()
        self.render_cost_curve()

    # ---- Handlers (giữ nguyên phần còn lại: pick_csv/preview/check_schema/...) ----
    def pick_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", "", "CSV Files (*.csv)")
        if not path: return
        try:
            df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "CSV", f"Không đọc được CSV:\n{e}")
            return
        self.state.df_loaded = df
        self.txtPath.setText(path)
        self.check_schema(); self.preview()
        self.log(f"Loaded dataset: {os.path.basename(path)} ({df.shape[0]}x{df.shape[1]})")

    def preview(self):
        if self.state.df_loaded is None: return
        self.tblPreview.setModel(PandasModel(self.state.df_loaded.head(100)))

    def check_schema(self):
        df = self.state.df_loaded
        if df is None:
            self.lblSchema.setText("Schema: —"); return
        target = self.state.schema.get("target","default.payment.next.month")
        ok = target in df.columns
        pos_rate = None
        try:
            if ok and set(pd.unique(df[target])) <= {0,1}:
                pos_rate = float(df[target].mean())*100.0
        except Exception:
            pos_rate = None
        msg = f"Target='{target}' → {'OK' if ok else 'MISSING'} • Rows={len(df):,}, Cols={df.shape[1]}"
        if pos_rate is not None: msg += f" • Positive rate={pos_rate:.2f}%"
        self.lblSchema.setText(msg)

    def pick_score_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Chọn file để chấm", "", "CSV Files (*.csv)")
        if path: self.txtScore.setText(path)


    # --- thay thế nội dung hàm run_scoring trong ComputeTab ---
    def run_scoring(self):
        try:
            path = self.txtScore.text().strip()
            if not path and self.state.df_loaded is not None:
                tmp = os.path.join(os.getcwd(), "_tmp_to_score.csv")
                # đảm bảo không lỗi mã hóa
                self.state.df_loaded.to_csv(tmp, index=False)
                path = tmp
                self.txtScore.setText(path)
            if not path or not os.path.exists(path):
                QMessageBox.warning(self, "Scoring", "Chưa chọn file hợp lệ.");
                return

            # Đọc CSV an toàn
            df = pd.read_csv(path)

            # Làm sạch tối thiểu các cột dùng để tính demo_score
            for c in [col for col in df.columns if col.startswith("PAY_")]:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            if "LIMIT_BAL" in df.columns:
                df["LIMIT_BAL"] = pd.to_numeric(df["LIMIT_BAL"], errors="coerce").fillna(0)

            # Tính proba (demo)
            proba = demo_score(df)
            # Thay NaN (nếu có) để không làm hỏng plot/metric
            proba = np.nan_to_num(proba, nan=0.0, posinf=1.0, neginf=0.0)

            # Lấy y_true nếu có target
            target = self.state.schema.get("target", "default.payment.next.month")
            if target in df.columns:
                y_true = parse_binary_labels(df[target])
            else:
                y_true = None  # không có nhãn thì vẫn cho chạy dashboard cơ bản

            out = df.copy()
            out["proba"] = proba
            out["flag"] = (out["proba"] >= self.state.threshold_active).astype(int)

            self.state.scored_df = out
            self.state.proba = proba
            self.state.y_true = y_true  # có thể là None

            self.update_scored_view()
            self.update_threshold_cards()
            self.render_cost_curve()
            self.results_ready.emit()
            self.log(f"Scored: {os.path.basename(path)} (rows={len(out)})")
        except Exception as e:
            # Bắt mọi lỗi → không out app
            QMessageBox.critical(self, "Run Scoring error", f"Đã xảy ra lỗi khi chấm điểm:\n{e}")

    def update_scored_view(self):
        if self.state.scored_df is None: return
        df = self.state.scored_df
        if self.chkFlaggedOnly.isChecked(): df = df[df["flag"]==1]
        self.tblScored.setModel(PandasModel(df.head(1000)))
        flagged = int((self.state.scored_df["flag"]==1).sum()); total = len(self.state.scored_df)
        msg = f"Flagged: {flagged}/{total} ({(flagged/total*100):.2f}%) @t={self.state.threshold_active:.3f}"
        if self.state.y_true is not None:
            cm = confusion_at_threshold(self.state.y_true, self.state.proba, self.state.threshold_active)
            p = precision(cm["TP"], cm["FP"]); r = recall(cm["TP"], cm["FN"])
            msg += f" • P={p:.3f}, R={r:.3f}"
        self.lblSummary.setText(msg)

    def on_slider_t(self, val: int):
        t = val/1000.0
        self.txtT.blockSignals(True); self.txtT.setValue(t); self.txtT.blockSignals(False)
        self.state.threshold_active = t
        if self.state.scored_df is not None:
            self.state.scored_df["flag"] = (self.state.scored_df["proba"] >= t).astype(int)
            self.update_scored_view()
        self.update_threshold_cards()
        self.render_cost_curve()

    def on_text_t(self, t: float):
        self.slT.blockSignals(True); self.slT.setValue(int(t*1000)); self.slT.blockSignals(False)
        self.state.threshold_active = t
        if self.state.scored_df is not None:
            self.state.scored_df["flag"] = (self.state.scored_df["proba"] >= t).astype(int)
            self.update_scored_view()
        self.update_threshold_cards()
        self.render_cost_curve()

    def find_star_threshold(self):
        if self.state.y_true is None or self.state.proba is None:
            QMessageBox.information(self, "Find t*", "Cần y_true & proba (hãy dùng file có target)."); return
        cfn = float(self.spCFN.value()); cfp = float(self.spCFP.value())
        best_t, best_cost = 0.5, float("inf")
        grid = np.linspace(0,1,401)
        for t in grid:
            cm = confusion_at_threshold(self.state.y_true, self.state.proba, t)
            cost = cm["FN"]*cfn + cm["FP"]*cfp
            if cost < best_cost: best_cost, best_t = cost, t
        self.txtT.setValue(float(best_t))
        QMessageBox.information(self, "Find t*", f"t* ≈ {best_t:.3f} (min cost ≈ {best_cost:,.0f})")
        self.render_cost_curve()

    def save_active_threshold(self):
        self.state.threshold_active = self.txtT.value()
        if self.state.artifacts_dir:
            outp = os.path.join(self.state.artifacts_dir, "threshold.json")
            try:
                with open(outp, "w", encoding="utf-8") as f:
                    json.dump({"threshold": float(self.state.threshold_active)}, f, ensure_ascii=False, indent=2)
                self.log(f"Saved threshold.json to {outp}")
            except Exception as e:
                QMessageBox.warning(self, "Save threshold", f"Không thể lưu: {e}")
        QMessageBox.information(self, "Active threshold", f"Đã đặt threshold={self.state.threshold_active:.3f}")
        self.update_threshold_cards()
        self.render_cost_curve()

    def export_predictions(self):
        if self.state.scored_df is None:
            QMessageBox.warning(self, "Export", "Chưa có dữ liệu đã score."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save predictions", "predictions.csv", "CSV Files (*.csv)")
        if not path: return
        self.state.scored_df.to_csv(path, index=False)
        self.log(f"Exported predictions to {path}")
        QMessageBox.information(self, "Export", "Đã xuất predictions.csv")

    def update_threshold_cards(self):
        if self.state.y_true is None or self.state.proba is None:
            self.cardConf.set_values("N/A", "Score file with target to view metrics")
            self.cardCost.set_values("N/A", "FN·C_FN + FP·C_FP"); return
        t = self.state.threshold_active
        cm = confusion_at_threshold(self.state.y_true, self.state.proba, t)
        p = precision(cm["TP"], cm["FP"]); r = recall(cm["TP"], cm["FN"]); f1 = f1_score(p, r)
        self.cardConf.set_values(f"TP:{cm['TP']} FP:{cm['FP']} FN:{cm['FN']} TN:{cm['TN']}", f"P={p:.3f} • R={r:.3f} • F1={f1:.3f}")
        cfn = float(self.spCFN.value()); cfp = float(self.spCFP.value())
        exp_cost = cm["FN"]*cfn + cm["FP"]*cfp
        self.cardCost.set_values(f"{exp_cost:,.0f}", f"@t={t:.3f} (C_FN={cfn:.0f}, C_FP={cfp:.0f})")

    # --- NEW: cost curve renderer with màu sắc
    def render_cost_curve(self):
        ax = self.figCost.ax
        ax.clear()
        if self.state.y_true is None or self.state.proba is None:
            ax.text(0.5,0.5,"Cần y_true & proba (Run Scoring)", ha="center", va="center")
            self.figCost.draw(); return

        cfn = float(self.spCFN.value()); cfp = float(self.spCFP.value())
        grid = np.linspace(0,1,401)
        costs = []
        for t in grid:
            cm = confusion_at_threshold(self.state.y_true, self.state.proba, t)
            costs.append(cm["FN"]*cfn + cm["FP"]*cfp)
        ax.plot(grid, costs, linewidth=2.0, color="#2563eb")                    # line xanh
        ax.fill_between(grid, costs, color="#93c5fd", alpha=0.25)               # nền nhạt
        # vẽ vạch ngưỡng hiện tại
        t = float(self.state.threshold_active)
        ax.axvline(t, color="#ef4444", linestyle="--", linewidth=1.8, label=f"t={t:.3f}")
        ax.set_xlabel("Threshold (t)"); ax.set_ylabel("Expected Cost")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.25)
        self.figCost.draw()

    def log(self, msg: str):
        self.log_signal.emit(msg)



# ===========================
# Tab 2 — Dashboard & Analysis
# ===========================

class AnalysisTab(QWidget):
    log_signal = pyqtSignal(str)

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.build()

    def build(self):
        # ---- khung ngoài có scroll ----
        outer = QVBoxLayout(self)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll)

        body = QWidget()
        self.scroll.setWidget(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(18)

        # ---------- KPI hàng trên ----------
        kgrid = QGridLayout()
        kgrid.setHorizontalSpacing(12)
        kgrid.setVerticalSpacing(8)
        self.kpiLateRate = KpiCard("Avg PD (mean proba)")
        self.kpiPrecision = KpiCard("Precision (Hit rate)")
        self.kpiRecall = KpiCard("Recall (Capture)")
        self.kpiCost = KpiCard("Expected Cost @ cut-off")
        self.kpiFlagged = KpiCard("Declined / Total")
        for i, card in enumerate([self.kpiLateRate, self.kpiPrecision, self.kpiRecall, self.kpiCost, self.kpiFlagged]):
            kgrid.addWidget(card, 0, i)
        layout.addLayout(kgrid)

        # ---------- Lưới 2×2 cho chart ----------
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)

        # (1) Histogram
        lbl = QLabel("Probability Distribution");
        lbl.setProperty("role", "section")
        grid.addWidget(lbl, 0, 0)
        self.figHist = MplCanvas(width=5.6, height=3.2);
        self.figHist.setMinimumHeight(260)
        grid.addWidget(self.figHist, 1, 0)

        # (2) Gains
        lbl = QLabel("Gains Curve (Cumulative capture)");
        lbl.setProperty("role", "section")
        grid.addWidget(lbl, 0, 1)
        self.figGains = MplCanvas(width=5.6, height=3.2);
        self.figGains.setMinimumHeight(260)
        grid.addWidget(self.figGains, 1, 1)

        # (3) Calibration
        lbl = QLabel("Calibration (Reliability)");
        lbl.setProperty("role", "section")
        grid.addWidget(lbl, 2, 0)
        self.figRel = MplCanvas(width=5.6, height=3.2);
        self.figRel.setMinimumHeight(260)
        grid.addWidget(self.figRel, 3, 0)

        # (4) Confusion heatmap
        lbl = QLabel("Confusion @ cut-off");
        lbl.setProperty("role", "section")
        grid.addWidget(lbl, 2, 1)
        self.figConf = MplCanvas(width=5.6, height=3.2);
        self.figConf.setMinimumHeight(260)
        grid.addWidget(self.figConf, 3, 1)

        layout.addLayout(grid)

        # ---------- Group Analysis & Top-K ----------
        grp = QGroupBox("Segment analysis & Top-K")
        g = QVBoxLayout(grp)
        g.setContentsMargins(12, 12, 12, 12)
        g.setSpacing(12)

        row = QHBoxLayout()
        self.cmbGroup = QComboBox()
        self.cmbGroup.addItems(["PAY_0", "EDUCATION", "MARRIAGE", "SEX", "AGE_band", "LIMIT_BAL_band"])
        self.btnRenderGroup = QPushButton("Render group chart")
        self.btnRenderGroup.clicked.connect(self.render_group_chart)
        row.addWidget(QLabel("Group by:"));
        row.addWidget(self.cmbGroup);
        row.addWidget(self.btnRenderGroup);
        row.addStretch(1)
        g.addLayout(row)

        self.figGroup = MplCanvas(width=11, height=3.2)
        self.figGroup.setMinimumHeight(260)
        g.addWidget(self.figGroup)

        self.tblTop = QTableView()
        g.addWidget(QLabel("Top-K high risk (by proba)"))
        g.addWidget(self.tblTop)

        self.txtExplainNote = QTextEdit()
        self.txtExplainNote.setReadOnly(True)
        self.txtExplainNote.setPlainText(
            "Gợi ý biện pháp được sinh bởi rule engine. Khi có SHAP, thay phần này bằng giải thích local/global."
        )
        g.addWidget(self.txtExplainNote)

        layout.addWidget(grp)

        # ---------- Export ----------
        rowx = QHBoxLayout()
        self.btnExportReport = QPushButton("Export CSV (flagged + suggestions)")
        self.btnExportReport.clicked.connect(self.export_flagged_with_suggestions)
        rowx.addStretch(1);
        rowx.addWidget(self.btnExportReport)
        layout.addLayout(rowx)

    def _section_label(self, text):
        lbl = QLabel(text); lbl.setProperty("role", "section"); return lbl

    # ---------- lifecycle ----------
    def refresh_all(self):
        if self.state.scored_df is None:
            for c in [self.kpiLateRate, self.kpiPrecision, self.kpiRecall, self.kpiCost, self.kpiFlagged]:
                c.set_values("N/A","Run scoring in Tab 1")
            for fig in [self.figHist, self.figGains, self.figRel, self.figConf, self.figGroup]:
                fig.clear()
            self.tblTop.setModel(PandasModel(pd.DataFrame()))
            return

        df = self.state.scored_df.copy()
        proba = df["proba"].values
        t = self.state.threshold_active

        # KPI luôn có
        late_rate = float(np.mean(proba))
        flagged = int((proba >= t).sum()); total = len(df)
        approval = 1 - flagged/total if total>0 else 0.0
        self.kpiLateRate.set_values(f"{late_rate*100:.2f}%", "Mean PD")
        self.kpiFlagged.set_values(f"{flagged}/{total}", f"Decline={flagged/total*100:.1f}%, Approve={approval*100:.1f}%")

        # Histogram
        self.render_hist(proba)

        # Nếu có nhãn, render phần còn lại
        if self.state.y_true is not None:
            y = self.state.y_true
            # Precision/Recall tại cut-off
            cm = confusion_at_threshold(y, proba, t)
            p = precision(cm["TP"], cm["FP"]); r = recall(cm["TP"], cm["FN"])
            self.kpiPrecision.set_values(f"{p:.3f}", "Hit rate of declines")
            self.kpiRecall.set_values(f"{r:.3f}", "Capture of bads")
            exp_cost = cm["FN"]*self.state.cfn + cm["FP"]*self.state.cfp
            self.kpiCost.set_values(f"{exp_cost:,.0f}", f"C_FN={self.state.cfn:.0f}, C_FP={self.state.cfp:.0f}")

            # Gains & Calibration & Confusion
            self.render_gains(y, proba, t)
            self.render_reliability(y, proba)
            self.render_confusion(cm)
        else:
            self.kpiPrecision.set_values("N/A","Need labels")
            self.kpiRecall.set_values("N/A","Need labels")
            self.kpiCost.set_values("N/A","Need labels")
            for fig in [self.figGains, self.figRel, self.figConf]:
                fig.clear()

        # Group & Top-K
        self.render_group_chart()
        self.populate_topk_table()

    # ---------- charts ----------
    def render_hist(self, proba):
        ax = self.figHist.ax;
        ax.clear()
        ax.hist(proba, bins=30)
        ax.set_xlabel("Predicted probability (PD)");
        ax.set_ylabel("Count")
        self.figHist.draw()  # bỏ tight_layout

    def render_gains(self, y, proba, t):
        ax = self.figGains.ax;
        ax.clear()
        x, ycap = gains_curve_points(y, proba)
        ax.plot(x * 100, ycap * 100, linewidth=2.0)
        ax.plot([0, 100], [0, 100], linestyle="--")
        flag_rate = (proba >= t).mean() * 100
        idx = int(min(max(round(flag_rate / 100 * (len(y) - 1)), 0), len(ycap) - 1))
        ax.scatter([flag_rate], [ycap[idx] * 100])
        ax.set_xlabel("% population (sorted by PD)");
        ax.set_ylabel("% bads captured")
        self.figGains.draw()  # bỏ tight_layout

    def render_reliability(self, y, proba):
        ax = self.figRel.ax;
        ax.clear()
        bins = np.linspace(0, 1, 11);
        inds = np.digitize(proba, bins) - 1
        xs, ys = [], []
        for i in range(10):
            mask = inds == i
            if np.sum(mask) == 0: continue
            xs.append(float(np.mean(proba[mask])));
            ys.append(float(np.mean(y[mask])))
        ax.plot([0, 1], [0, 1], linestyle="--");
        ax.scatter(xs, ys)
        ax.set_xlabel("Predicted probability (bin mean)");
        ax.set_ylabel("Observed default rate")
        self.figRel.draw()  # bỏ tight_layout

    def render_confusion(self, cm):
        plot_confusion_heatmap(self.figConf.ax, cm)
        self.figConf.figure.tight_layout(pad=1.0); self.figConf.draw()

    def render_group_chart(self):
        ax = self.figGroup.ax; ax.clear()
        if self.state.scored_df is None:
            ax.text(0.5,0.5,"No data", ha="center"); self.figGroup.draw(); return
        df = self.state.scored_df.copy()
        if "AGE" in df.columns:
            df["AGE_band"] = pd.cut(df["AGE"], bins=[0,25,35,45,60,120], include_lowest=True).astype(str)
        else:
            df["AGE_band"] = "NA"
        if "LIMIT_BAL" in df.columns:
            q = df["LIMIT_BAL"].quantile([0,0.25,0.5,0.75,1.0]).values; q[0] = max(1.0, q[0])
            df["LIMIT_BAL_band"] = pd.cut(df["LIMIT_BAL"], bins=q, include_lowest=True, duplicates="drop").astype(str)
        else:
            df["LIMIT_BAL_band"] = "NA"
        feat = self.cmbGroup.currentText()
        if feat not in df.columns:
            ax.text(0.5,0.5,f"Missing: {feat}", ha="center"); self.figGroup.draw(); return
        grp = df.groupby(feat)["proba"].mean().sort_values(ascending=False)
        ax.bar(range(len(grp.index)), grp.values)
        ax.set_xticks(range(len(grp.index))); ax.set_xticklabels(grp.index, rotation=45, ha="right")
        ax.set_ylabel("Mean PD")
        self.figGroup.figure.tight_layout(pad=1.0); self.figGroup.draw()

    def populate_topk_table(self, k: int = 50):
        df = self.state.scored_df.copy().sort_values("proba", ascending=False).head(k)
        df["suggestions"] = [" | ".join(apply_rules(row, float(row["proba"]), self.state.rules)) for _, row in df.iterrows()]
        cols = ["proba","flag","suggestions"] + [c for c in df.columns if c not in ["proba","flag","suggestions"]]
        self.tblTop.setModel(PandasModel(df[cols]))

    def export_flagged_with_suggestions(self):
        if self.state.scored_df is None:
            QMessageBox.information(self, "Export", "Không có dữ liệu."); return
        df = self.state.scored_df.copy()
        df = df[df["proba"] >= self.state.threshold_active].copy()
        df["suggestions"] = [" | ".join(apply_rules(row, float(row["proba"]), self.state.rules)) for _, row in df.iterrows()]
        path, _ = QFileDialog.getSaveFileName(self, "Save flagged report", "flagged_report.csv", "CSV Files (*.csv)")
        if not path: return
        df.to_csv(path, index=False)
        self.log_signal.emit(f"Exported flagged report to {path}")
        QMessageBox.information(self, "Export", "Đã xuất flagged_report.csv")

# ===========================
# Settings Dialog
# ===========================

class SettingsDialog(QDialog):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.setWindowTitle("Settings"); self.resize(640, 480)
        lay = QVBoxLayout(self)

        row = QHBoxLayout()
        self.txtArtifacts = QLineEdit(self.state.artifacts_dir or "")
        btnPick = QPushButton("Chọn thư mục artifacts…"); btnPick.clicked.connect(self.pick_artifacts)
        row.addWidget(QLabel("Artifacts dir:")); row.addWidget(self.txtArtifacts); row.addWidget(btnPick)

        self.rulesEdit = QPlainTextEdit(json.dumps(self.state.rules, ensure_ascii=False, indent=2))
        grpRules = QGroupBox("rules.json")
        vr = QVBoxLayout(grpRules); vr.addWidget(self.rulesEdit)

        rowb = QHBoxLayout()
        btnSave = QPushButton("Lưu"); btnSave.clicked.connect(self.save)
        btnCancel = QPushButton("Đóng"); btnCancel.clicked.connect(self.close)
        rowb.addStretch(1); rowb.addWidget(btnSave); rowb.addWidget(btnCancel)

        lay.addLayout(row); lay.addWidget(grpRules); lay.addLayout(rowb)

    def pick_artifacts(self):
        p = QFileDialog.getExistingDirectory(self, "Chọn thư mục artifacts", "")
        if p: self.txtArtifacts.setText(p)

    def save(self):
        self.state.artifacts_dir = self.txtArtifacts.text().strip() or None
        try:
            self.state.rules = json.loads(self.rulesEdit.toPlainText())
            QMessageBox.information(self, "Settings", "Đã lưu.")
        except Exception as e:
            QMessageBox.warning(self, "Settings", f"Lỗi parse rules.json: {e}")
        self.close()


# ===========================
# Main Window
# ===========================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE); self.resize(1280, 860)
        self.state = AppState()

        self.tabs = QTabWidget()
        self.compute = ComputeTab(self.state)
        self.analysis = AnalysisTab(self.state)
        self.tabs.addTab(self.compute, "Nhập & Tính toán")
        self.tabs.addTab(self.analysis, "Dashboard & Phân tích")

        # App menu
        menu = self.menuBar().addMenu("App")
        actSettings = QAction("Settings…", self); actSettings.triggered.connect(self.open_settings)
        actQuit = QAction("Quit", self); actQuit.triggered.connect(self.close)
        menu.addAction(actSettings); menu.addAction(actQuit)

        # Layout + Activity Log
        wrapper = QWidget(); wlay = QVBoxLayout(wrapper)
        wlay.addWidget(self.tabs)
        self.logView = QPlainTextEdit(); self.logView.setReadOnly(True)
        grp = QGroupBox("Activity Log"); v = QVBoxLayout(grp); v.addWidget(self.logView)
        wlay.addWidget(grp)
        self.setCentralWidget(wrapper)
        # Theme màu nhẹ
        self.setStyleSheet("""
            QWidget { font-size: 13px; }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #e2e8f0; border-radius: 10px; margin-top: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #2b6cb0; }
            QPushButton {
                background: #2563eb; color: white; border: none; padding: 6px 12px; border-radius: 8px;
            }
            QPushButton:hover { background: #1d4ed8; }
            QSlider::groove:horizontal { height: 6px; background: #e5e7eb; border-radius: 3px; }
            QSlider::handle:horizontal {
                width: 16px; background: #2563eb; border-radius: 8px; margin: -5px 0;
                border: 1px solid rgba(0,0,0,0.08);
            }
            QLabel { color: #111827; }
        """)

        self.compute.log_signal.connect(self.logView.appendPlainText)
        self.analysis.log_signal.connect(self.logView.appendPlainText)
        self.compute.results_ready.connect(self.analysis.refresh_all)

    def open_settings(self):
        dlg = SettingsDialog(self.state)
        dlg.exec()
        self.analysis.refresh_all()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
