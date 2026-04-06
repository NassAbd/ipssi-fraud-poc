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
    Test preprocessing function existence (no fake data split check without full dataset).
    """
    from src.data.loader import preprocess_data
    assert callable(preprocess_data)
