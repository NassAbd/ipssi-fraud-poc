import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load BankSim dataset, clean, and split into train/test.
    """
    # Loading data
    df = pd.read_csv(file_path)

    # Cleaning: remove quotes and unnecessary columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.replace("'", "")

    # Drop non-informative columns (all constant in this dataset)
    df = df.drop(['zipcodeOri', 'zipMerchant'], axis=1)

    # Feature Engineering
    # Label encoding for category and gender
    le = LabelEncoder()
    df['category'] = le.fit_transform(df['category'])
    df['gender'] = le.fit_transform(df['gender'])

    # Normalization of amount
    scaler = StandardScaler()
    df['amount_scaled'] = scaler.fit_transform(df[['amount']])

    # Split: 70% train / 30% test with stratification
    X = df.drop(['fraud', 'customer', 'merchant', 'age'], axis=1) # Minimal features for the POC
    y = df['fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test
