import os
import pandas as pd

EXPECTED_COLUMNS = [
    'Avg Area Income',
    'Avg Area House Age',
    'Avg Area Number of Rooms',
    'Avg Area Number of Bedrooms',
    'Area Population',
    'Price',
]


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    return df


def ensure_columns(df: pd.DataFrame):
    # Support official USA_Housing.csv headers that include dots after 'Avg.'
    rename_map = {
        'Avg. Area Income': 'Avg Area Income',
        'Avg. Area House Age': 'Avg Area House Age',
        'Avg. Area Number of Rooms': 'Avg Area Number of Rooms',
        'Avg. Area Number of Bedrooms': 'Avg Area Number of Bedrooms',
        'Area Population': 'Area Population',
        'Price': 'Price',
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing:
        df.rename(columns=existing, inplace=True)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")


def split_features_target(df: pd.DataFrame):
    ensure_columns(df)
    X = df[[
        'Avg Area Income',
        'Avg Area House Age',
        'Avg Area Number of Rooms',
        'Avg Area Number of Bedrooms',
        'Area Population',
    ]]
    y = df['Price']
    return X, y