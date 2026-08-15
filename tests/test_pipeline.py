import pandas as pd
from src.models.expert_system import predict_expert_system_on_df
from src.schemas.fraud_detection import Transaction

def test_transaction_schema():
    """
    Test it can instantiate a transaction from the BankSim dataset with the pydantic model.
    """
    dummy_data = {
        "step": 1,
        "customer": "C123456789",
        "age": "20-30",
        "gender": "M",
        "zipcodeOri": "28001",
        "merchant": "M123456789",
        "zipMerchant": "28001",
        "category": "leisure",
        "amount": 100.50,
        "fraud": 0
    }
    tx = Transaction(**dummy_data)
    assert tx.amount == 100.50
    assert tx.gender == "M"

def test_expert_system_logic():
    """
    Test the rule-based legacy model logic for a few rows with notebook rules.
    """
    df = pd.DataFrame([
        {"amount": 1200, "category": "food"},       # Fraud rule 1 (>1000)
        {"amount": 600, "category": "leisure"},     # Fraud rule 2 (>500 and risk cat)
        {"amount": 600, "category": "food"},        # Legit (no risk cat)
        {"amount": 100, "category": "leisure"}      # Legit (amt < 500)
    ])
    preds = predict_expert_system_on_df(df)
    assert preds[0] == 1
    assert preds[1] == 1
    assert preds[2] == 0
    assert preds[3] == 0

def test_preprocess_split_stratification():
    """
    Test preprocessing function pipeline logic with a dummy dataframe.
    """
    from src.data.loader import preprocess_data
    df = pd.DataFrame({
        "step": [1]*10,
        "age": ["20-30"]*10,
        "gender": ["M", "F"]*5,
        "category": ["food", "leisure"]*5,
        "amount": [10, 50, 100, 200, 300, 400, 500, 600, 700, 800],
        "fraud": [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
    })
    
    X_train, X_test, y_train, y_test, X_test_raw = preprocess_data(
        df, features=["step", "age", "gender", "category", "amount"]
    )
    
    assert len(X_train) == 7
    assert len(X_test) == 3
    assert "amount" in X_train.columns
    # Check that scale transformed the amount (not identical)
    assert X_train["amount"].iloc[0] != 10
    
    # Check that test shape is correct
    assert X_train.shape[1] == 5
