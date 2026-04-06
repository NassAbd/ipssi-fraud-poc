from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

def train_rf_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    """
    Train a Random Forest classifier with 100 estimators and balanced weights.
    """
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    return rf

def predict_rf_model(model: RandomForestClassifier, X_test: pd.DataFrame) -> np.ndarray:
    """
    Make predictions on the test set using a trained Random Forest model.
    """
    return model.predict(X_test)
