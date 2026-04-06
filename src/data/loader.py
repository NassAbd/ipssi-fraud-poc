import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load BankSim dataset and perform basic string cleaning.
    """
    df = pd.read_csv(file_path)
    # Clean quotes from string columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace("'", "")

    # Drop columns that are constant in BankSim
    to_drop = [c for c in ['zipcodeOri', 'zipMerchant'] if c in df.columns]
    df = df.drop(to_drop, axis=1)
    return df

def preprocess_data(df: pd.DataFrame, features: List[str], target: str = 'fraud') -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Split data then apply encoding and scaling to prevent data leakage.
    Returns: X_train, X_test, y_train, y_test, X_test_raw (for legacy model)
    """
    X_raw = df[features]
    y = df[target]

    # Split 70/30 with stratification
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.3, stratify=y, random_state=42
    )

    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()

    # Categorical encoding
    categorical_cols = [c for c in ['age', 'gender', 'category'] if c in features]
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train_raw[col])
        X_test[col] = X_test_raw[col].map(
            lambda s: le.transform([s])[0] if s in le.classes_ else -1
        )

    # Scaling
    if 'amount' in features:
        scaler = StandardScaler()
        X_train['amount'] = scaler.fit_transform(X_train[['amount']])
        X_test['amount'] = scaler.transform(X_test[['amount']])

    return X_train, X_test, y_train, y_test, X_test_raw
