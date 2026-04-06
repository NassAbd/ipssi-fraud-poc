import pandas as pd
from src.models.expert_system import predict_rules_on_df
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

def test_expert_system_simple_logic():
    """
    Test the rule-based legacy model logic for a single row.
    """
    # Create a simple df
    df = pd.DataFrame([{
        "amount": 1200,
        "category": 1 # Representing encoded info for now
    }, {
        "amount": 100,
        "category": 1
    }])
    preds = predict_rules_on_df(df)
    assert preds[0] == 1 # 1200 is fraud (> 1000)
    assert preds[1] == 0 # 100 is legitimate
