"""
Scikit-Learn Preprocessing & Feature Engineering Pipeline for CreditRisk AI.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

PROCESSED_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "processed"))
PIPELINE_SAVE_PATH = os.path.join(PROCESSED_DATA_DIR, "preprocessor.joblib")


class CreditFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom Transformer adding domain-specific credit risk ratio features."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        if isinstance(X_out, np.ndarray):
            # If array, convert to DataFrame for feature math
            cols = [
                "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
                "PAY_0", "PAY_2", "PAY_3",
                "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                "PAY_AMT1", "PAY_AMT2", "PAY_AMT3"
            ]
            X_out = pd.DataFrame(X_out, columns=cols[:X_out.shape[1]])

        # 1. Credit Utilization Ratio
        limit = X_out["LIMIT_BAL"].replace(0, 1.0)
        X_out["UTILIZATION_1"] = (X_out["BILL_AMT1"] / limit).clip(0.0, 2.0)

        # 2. Payment-to-Bill Ratio
        bill1 = X_out["BILL_AMT1"].clip(lower=1.0)
        X_out["PAY_RATIO_1"] = (X_out["PAY_AMT1"] / bill1).clip(0.0, 5.0)

        # 3. Delinquency Count (Number of late payment months)
        pay_cols = [c for c in ["PAY_0", "PAY_2", "PAY_3"] if c in X_out.columns]
        if pay_cols:
            X_out["DELINQUENCY_COUNT"] = (X_out[pay_cols] > 0).sum(axis=1)
        else:
            X_out["DELINQUENCY_COUNT"] = 0

        return X_out


def build_preprocessor_pipeline():
    """Build Scikit-Learn ColumnTransformer pipeline."""
    num_features = [
        "LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "UTILIZATION_1", "PAY_RATIO_1", "DELINQUENCY_COUNT"
    ]
    cat_features = ["SEX", "EDUCATION", "MARRIAGE"]
    pay_features = ["PAY_0", "PAY_2", "PAY_3"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
            ("pay", StandardScaler(), pay_features),
        ],
        remainder="passthrough"
    )

    full_pipeline = Pipeline([
        ("feature_engineering", CreditFeatureEngineer()),
        ("preprocessor", preprocessor)
    ])

    return full_pipeline, num_features, cat_features, pay_features


def prepare_train_test_data(raw_csv_path: str, test_size: float = 0.2, random_state: int = 42):
    """
    Load raw CSV data, fit the preprocessing pipeline, and split into train/test sets.
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    df = pd.read_csv(raw_csv_path)

    # Drop ID column if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Dynamically find target column
    target_col = None
    for candidate in ["default", "default payment next month", "default.payment.next.month"]:
        if candidate in df.columns:
            target_col = candidate
            break

    if target_col is None:
        raise ValueError(f"Target default column not found in {df.columns.tolist()}")

    df = df.rename(columns={target_col: "default"})

    X = df.drop(columns=["default"])
    y = df["default"].values

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit and transform training features
    pipeline, num_cols, cat_cols, pay_cols = build_preprocessor_pipeline()
    X_train_transformed = pipeline.fit_transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    # Save fitted pipeline transformer for inference & API serving
    joblib.dump(pipeline, PIPELINE_SAVE_PATH)
    print(f"[PIPELINE] Saved fitted preprocessing pipeline to {PIPELINE_SAVE_PATH}")

    return X_train, X_test, X_train_transformed, X_test_transformed, y_train, y_test, pipeline
