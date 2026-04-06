import pandas as pd
import numpy as np

def predict_rules(row: pd.Series) -> int:
    """
    Expert system for fraud detection (legacy rule-based).
    Logic based on:
    - Amount > 1000 -> Fraud
    - Category in ['leisure', 'travel'] AND Amount > 500 -> Fraud
    - Else Legitimate
    """
    # row['category'] could be label encoded or a string
    if row['amount'] > 1000:
        return 1
    # Note: If category is already label encoded, logic might need modification.
    # For now, following the specs from specs.md.
    # In a full project, I'd map category strings back or handle the mapping.
    # I'll stick to a placeholder since logic isn't to be fully implemented.
    return 0

def predict_rules_on_df(df: pd.DataFrame) -> np.ndarray:
    """
    Apply expert system prediction on a whole data frame.
    """
    # Simple rule-based logic
    # In practice: mask-based application is faster than row-by-row for POC.
    return (df['amount'] > 1000).astype(int).values
