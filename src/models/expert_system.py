import pandas as pd
import numpy as np

def predict_legacy(row: pd.Series) -> int:
    """
    Expert system for fraud detection (rule-based).
    Optimized logic with progressive thresholds based on risk categories.
    """
    cat = row['category']
    amt = row['amount']
    
    extreme_risk = ['es_leisure', 'es_travel', 'leisure', 'travel']
    high_risk = ['es_sportsandtoys', 'es_hotelservices', 'es_otherservices', 'sportsandtoys', 'hotelservices', 'otherservices']
    moderate_risk = ['es_health', 'es_wellnessandbeauty', 'health', 'wellnessandbeauty']
    
    if amt > 800:
        return 1
    if amt > 100 and cat in extreme_risk:
        return 1
    if amt > 250 and cat in high_risk:
        return 1
    if amt > 400 and cat in moderate_risk:
        return 1
        
    return 0

def predict_expert_system_on_df(df: pd.DataFrame) -> np.ndarray:
    """
    Apply expert system prediction on a whole data frame.
    """
    extreme_risk = ['es_leisure', 'es_travel', 'leisure', 'travel']
    high_risk = ['es_sportsandtoys', 'es_hotelservices', 'es_otherservices', 'sportsandtoys', 'hotelservices', 'otherservices']
    moderate_risk = ['es_health', 'es_wellnessandbeauty', 'health', 'wellnessandbeauty']
    
    rule1 = df['amount'] > 800
    rule2 = (df['amount'] > 100) & (df['category'].isin(extreme_risk))
    rule3 = (df['amount'] > 250) & (df['category'].isin(high_risk))
    rule4 = (df['amount'] > 400) & (df['category'].isin(moderate_risk))
    
    return (rule1 | rule2 | rule3 | rule4).astype(int).values
