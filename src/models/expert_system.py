import pandas as pd
import numpy as np

def predict_legacy(row: pd.Series) -> int:
    """
    Expert system for fraud detection (legacy rule-based).
    Logic based on:
    - Amount > 1000 -> Fraud
    - Category in ['es_leisure', 'es_travel'] AND Amount > 500 -> Fraud
    - Else Legitimate
    """
    # Follow notebook logic (es_ prefix)
    fraud_cats = ['es_leisure', 'es_travel', 'leisure', 'travel'] # Support both
    if row['amount'] > 1000:
        return 1
    if row['amount'] > 500 and row['category'] in fraud_cats:
        return 1
    return 0

def predict_expert_system_on_df(df: pd.DataFrame) -> np.ndarray:
    """
    Apply expert system prediction on a whole data frame.
    """
    # Vectorized for efficiency
    fraud_cats = ['es_leisure', 'es_travel', 'leisure', 'travel']
    rule1 = df['amount'] > 1000
    rule2 = (df['amount'] > 500) & (df['category'].isin(fraud_cats))
    return (rule1 | rule2).astype(int).values
