import io
import json
import zipfile
from typing import List, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


def train_linear_regression(X, y, test_size: float = 0.2, random_state: int = 42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return model, (mae, mse, rmse), (X_train, X_test, y_train, y_test, y_pred)


def model_summary(model: LinearRegression, feature_names: List[str]) -> str:
    coefs = {fname: float(c) for fname, c in zip(feature_names, model.coef_)}
    lines = [
        f"Intercept: {model.intercept_}",
        "Coefficients:",
    ]
    for k, v in coefs.items():
        lines.append(f"  - {k}: {v}")
    return "\n".join(lines)


def save_model_zip(model: LinearRegression, feature_names: List[str], out_zip_path: str):
    with zipfile.ZipFile(out_zip_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        # model pickle
        import pickle
        model_bytes = pickle.dumps(model)
        zf.writestr('model.pkl', model_bytes)
        # feature names
        zf.writestr('feature_names.json', json.dumps(feature_names))


def load_model_zip(zip_path: str) -> Tuple[LinearRegression, List[str]]:
    with zipfile.ZipFile(zip_path, mode='r') as zf:
        import pickle
        model = pickle.loads(zf.read('model.pkl'))
        feature_names = json.loads(zf.read('feature_names.json').decode('utf-8'))
        return model, feature_names