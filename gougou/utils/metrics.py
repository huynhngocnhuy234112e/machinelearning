import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

def confusion_at_threshold(y_true: np.ndarray, proba: np.ndarray, t: float) -> Dict[str, int]:
    """Calculate confusion matrix at given threshold"""
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

def parse_binary_labels(series: pd.Series) -> Optional[np.ndarray]:
    """Convert various label formats to 0/1 array"""
    if series is None:
        return None
    s = series.copy()
    
    if s.dtype == object:
        s = s.astype(str).str.strip().str.lower()
        mapping = {
            "1": 1, "0": 0, "true": 1, "false": 0, "y": 1, "n": 0, "yes": 1, "no": 0,
            "default": 1, "non-default": 0, "bad": 1, "good": 0, "overdue": 1, "paid": 0,
            "defaulter": 1, "non defaulter": 0, "delinquent": 1, "non-delinquent": 0
        }
        s = s.map(mapping).astype("float64")
    
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return None
    if (s.dropna().isin([0, 1]).all()):
        return s.fillna(0).astype(int).values
    return (s.fillna(0) >= 1).astype(int).values

def demo_score(df: pd.DataFrame) -> np.ndarray:
    """Generate demo probability scores"""
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
    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
    return s

def compute_utilization(row: pd.Series) -> Optional[float]:
    """Calculate credit utilization ratio"""
    bills = [c for c in row.index if c.startswith("BILL_AMT")]
    pays = [c for c in row.index if c.startswith("PAY_AMT")]
    if "LIMIT_BAL" not in row.index:
        return None
    try:
        limit_bal = float(row["LIMIT_BAL"])
        total_bill = float(sum([row[c] for c in bills])) if bills else 0.0
        total_pay = float(sum([row[c] for c in pays])) if pays else 0.0
        util = (total_bill - total_pay) / (limit_bal * max(len(bills), 1))
        return float(max(0.0, min(1.5, util)))
    except Exception:
        return None

def apply_rules(row: pd.Series, proba: float, rules: Dict) -> List[str]:
    """Apply rule engine to generate suggestions"""
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
        if ok:
            suggestions.append(r.get("then", ""))
    
    return suggestions[:3]

def gains_curve_points(y_true: np.ndarray, proba: np.ndarray):
    """Calculate gains curve points"""
    order = np.argsort(-proba)
    y_sorted = y_true[order]
    cum_pos = np.cumsum(y_sorted)
    total_pos = np.sum(y_true)
    n = len(y_true)
    x = np.arange(1, n+1) / n
    y = cum_pos / max(total_pos, 1)
    return x, y
