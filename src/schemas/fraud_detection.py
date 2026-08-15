from pydantic import BaseModel, Field, ConfigDict
from typing import Literal

class Transaction(BaseModel):
    """
    Schema for a single transaction from the BankSim dataset.
    """
    model_config = ConfigDict(populate_by_name=True)

    step: int = Field(..., description="The time step of the transaction")
    customer: str = Field(..., description="Customer ID")
    age: str = Field(..., description="Age category of the customer")
    gender: Literal['M', 'F', 'E', 'U'] = Field(..., description="Gender: Male, Female, Enterprise, Unknown")
    zipcode_ori: str = Field(..., alias="zipcodeOri", description="Zip code of the customer's origin")
    merchant: str = Field(..., description="Merchant ID")
    zip_merchant: str = Field(..., alias="zipMerchant", description="Zip code of the merchant")
    category: str = Field(..., description="Transaction category (e.g., leisure, travel, food)")
    amount: float = Field(..., gt=0, description="Amount of the transaction")
    fraud: int = Field(..., ge=0, le=1, description="Binary indicator of fraud: 1 for fraud, 0 for legitimate")

class ModelMetrics(BaseModel):
    """
    Schema for evaluating model performance.
    """
    precision: float = Field(..., ge=0, le=1)
    recall: float = Field(..., ge=0, le=1)
    f1_score: float = Field(..., ge=0, le=1)
    accuracy: float = Field(..., ge=0, le=1)

class TransactionInput(BaseModel):
    """
    Schema for real-time transaction simulation input.
    """
    step: int = Field(..., gt=0)
    age: str = Field(...)
    gender: Literal['M', 'F', 'E', 'U'] = Field(...)
    category: str = Field(...)
    amount: float = Field(..., gt=0)

class RiskScoreOutput(BaseModel):
    """
    Schema for the real-time API response.
    """
    risk_score: float = Field(..., ge=0, le=1, description="Probability of fraud (0 to 1)")
    is_fraud: bool = Field(..., description="True if risk_score >= threshold")
    explanation: dict = Field(default_factory=dict, description="SHAP feature contributions")
