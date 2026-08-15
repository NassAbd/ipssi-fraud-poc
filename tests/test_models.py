import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from src.models.ml_model import train_rf_model, predict_rf_model

def test_rf_model_training_and_prediction():
    """
    Test that the model trains and predicts correctly on dummy data, 
    verifying shape and output type stability.
    """
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.rand(10, 5), columns=["step", "age", "gender", "category", "amount"])
    y_train = pd.Series(np.random.randint(0, 2, 10))
    
    # Train
    model = train_rf_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    
    # Predict
    X_test = pd.DataFrame(np.random.rand(2, 5), columns=["step", "age", "gender", "category", "amount"])
    # ML Shape TDD constraint checking
    assert X_test.shape[1] == 5, f"Expected 5 features, got {X_test.shape[1]}"
    
    preds = predict_rf_model(model, X_test)
    
    assert len(preds) == 2
    assert set(np.unique(preds)).issubset({0, 1})
