from dataclasses import dataclass, field
from typing import Dict, Optional
import pandas as pd
import numpy as np

@dataclass
class UserState:
    """User authentication and billing state"""
    current_user: Optional[str] = None
    users: Dict[str, str] = field(default_factory=lambda: {"demo@mlba.vn": "123", "admin": "admin"})
    bills: pd.DataFrame = field(default_factory=lambda: pd.DataFrame([
        {"invoice_id": 1001, "status": "DUE",     "amount": 120.0, "due_date": "2025-11-15"},
        {"invoice_id": 1002, "status": "PAID",    "amount":  80.0, "due_date": "2025-09-15"},
        {"invoice_id": 1003, "status": "OVERDUE", "amount":  60.0, "due_date": "2025-08-15"},
    ]))

@dataclass
class AppState:
    """Global application state for credit risk analysis"""
    artifacts_dir: Optional[str] = None
    schema: Dict = field(default_factory=lambda: {"target": "default.payment.next.month"})
    threshold_active: float = 0.5
    cfn: float = 10.0
    cfp: float = 1.0
    
    df_loaded: Optional[pd.DataFrame] = None
    scored_df: Optional[pd.DataFrame] = None
    y_true: Optional[np.ndarray] = None
    proba: Optional[np.ndarray] = None
    
    rules: Dict = field(default_factory=lambda: {
        "rules": [
            {"if": {"PAY_0_ge": 2}, "then": "Nhắc nợ sớm (call trong 48h)."},
            {"if": {"PAY_2_plus_count_ge1": 2}, "then": "Theo dõi sát & chặn tăng hạn mức."},
            {"if": {"utilization_ge": 0.85}, "then": "Đề xuất điều chỉnh hạn mức / yêu cầu trả thêm."},
            {"if": {"AGE_le": 25, "proba_ge": 0.7}, "then": "Yêu cầu bảo đảm bổ sung / đồng bảo lãnh."}
        ]
    })
