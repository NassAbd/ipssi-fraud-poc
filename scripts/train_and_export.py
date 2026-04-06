"""
Script: train_and_export.py
Usage: uv run python scripts/train_and_export.py --data notebook/bs140513_032310.csv
Outputs: models/rf_model.joblib, models/scaler.joblib, models/label_encoders.joblib
"""
import argparse
import os
import sys

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


FEATURES = ["step", "age", "gender", "category", "amount"]
TARGET = "fraud"
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def main(data_path: str) -> None:
    print(f"[1/4] Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.replace("'", "")
    cols_to_drop = [c for c in ["zipcodeOri", "zipMerchant"] if c in df.columns]
    df = df.drop(cols_to_drop, axis=1)

    X_raw = df[FEATURES]
    y = df[TARGET]

    print("[2/4] Splitting train/test (70/30, stratified)...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.3, stratify=y, random_state=42
    )

    print("[3/4] Encoding and scaling...")
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()

    categorical_cols = ["age", "gender", "category"]
    label_encoders: dict[str, LabelEncoder] = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train_raw[col])
        X_test[col] = X_test_raw[col].map(
            lambda s, _le=le: _le.transform([s])[0] if s in _le.classes_ else -1
        )
        label_encoders[col] = le

    scaler = StandardScaler()
    X_train["amount"] = scaler.fit_transform(X_train[["amount"]])
    X_test["amount"] = scaler.transform(X_test[["amount"]])

    print("[4/4] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=42,
        max_depth=10,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = rf.predict(X_test)
    print("\n--- Classification Report (Test Set) ---")
    print(classification_report(y_test, y_pred))

    # Export artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf, os.path.join(MODEL_DIR, "rf_model.joblib"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    joblib.dump(label_encoders, os.path.join(MODEL_DIR, "label_encoders.joblib"))
    # Also store test data for the dashboard page
    joblib.dump(
        {"X_test_raw": X_test_raw, "X_test": X_test, "y_test": y_test},
        os.path.join(MODEL_DIR, "test_data.joblib"),
    )

    print(f"\n✅ Artifacts exported to: {MODEL_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and export fraud detection model")
    parser.add_argument("--data", required=True, help="Path to BankSim CSV file")
    args = parser.parse_args()
    main(args.data)
