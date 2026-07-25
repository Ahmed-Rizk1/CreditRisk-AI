"""
Data Ingestion and Synthetic Drift Data Generator for CreditRisk AI.

This module fetches the UCI Credit Card Default dataset or generates synthetic
credit applicant data with realistic feature distributions and controlled data drift.
"""

import os
import urllib.request
import numpy as np
import pandas as pd

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data"))
RAW_DATA_PATH = os.path.join(DATA_DIR, "raw", "credit_card_default.csv")
DRIFT_DATA_PATH = os.path.join(DATA_DIR, "drift", "credit_card_drift.csv")

# Direct public URL for UCI Credit Card dataset
UCI_CREDIT_URL = "https://raw.githubusercontent.com/selva86/datasets/master/UCI_Credit_Card.csv"
# Official UCI Repository URL
UCI_CREDIT_ZIP_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"


def ensure_directories():
    """Ensure data subdirectories exist."""
    os.makedirs(os.path.join(DATA_DIR, "raw"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "processed"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "drift"), exist_ok=True)


def download_uci_dataset() -> bool:
    """Attempt to download and parse official UCI Credit Card Default dataset."""
    import zipfile
    import io

    ensure_directories()
    print(f"[DATA] Attempting to download official UCI Credit Card dataset from UCI Repository...")
    try:
        req = urllib.request.urlopen(UCI_CREDIT_ZIP_URL)
        zip_bytes = io.BytesIO(req.read())
        with zipfile.ZipFile(zip_bytes) as z:
            # Excel file inside zip: 'default of credit card clients.xls'
            with z.open("default of credit card clients.xls") as excel_file:
                df = pd.read_excel(excel_file, header=1)
                
        # Clean column names
        if "default payment next month" in df.columns:
            df.rename(columns={"default payment next month": "default"}, inplace=True)
        if "PAY_0" not in df.columns and "PAY_1" in df.columns:
            df.rename(columns={"PAY_1": "PAY_0"}, inplace=True)
        if "ID" in df.columns:
            df.drop(columns=["ID"], inplace=True)

        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"[DATA] Successfully downloaded real UCI dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
        return True
    except Exception as e:
        print(f"[DATA] Download from UCI repository failed ({e}). Falling back to synthetic data generation.")
        return False


def generate_synthetic_credit_data(n_samples: int = 10000, seed: int = 42, drift: bool = False) -> pd.DataFrame:
    """
    Generate realistic synthetic credit application data.

    Parameters:
    -----------
    n_samples : int
        Number of synthetic applicant records to generate.
    seed : int
        Random seed for reproducibility.
    drift : bool
        If True, injects macroeconomic inflation drift (higher debt, lower payment ratios, higher delinquency).

    Returns:
    --------
    pd.DataFrame
        DataFrame with credit applicant features and binary target 'default'.
    """
    np.random.seed(seed)

    # Demographic & Base Attributes
    age = np.random.randint(21, 70, size=n_samples)
    sex = np.random.choice([1, 2], size=n_samples, p=[0.45, 0.55])  # 1=Male, 2=Female
    education = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.35, 0.45, 0.15, 0.05])  # Grad, Univ, HighSchool, Other
    marriage = np.random.choice([1, 2, 3], size=n_samples, p=[0.4, 0.55, 0.05])  # Married, Single, Other

    # Income & Credit Limit (Log-normal distribution)
    base_limit = np.random.lognormal(mean=11.5, sigma=0.8, size=n_samples)
    limit_bal = np.round(base_limit / 5000) * 5000  # Round to nearest 5k

    # Drift Shift Multipliers
    drift_debt_mult = 1.35 if drift else 1.0
    drift_pay_shift = 1.2 if drift else 0.0

    # Past Payment Repayment Status (-1=Pay duly, 0=Revolving credit, 1=Delay 1mo, 2=Delay 2mo, 3=Delay 3+mo)
    pay_probs_baseline = [0.45, 0.35, 0.12, 0.05, 0.03]
    if drift:
        # Shift probabilities towards payment delays
        pay_probs = [0.25, 0.30, 0.25, 0.12, 0.08]
    else:
        pay_probs = pay_probs_baseline

    pay_0 = np.random.choice([-1, 0, 1, 2, 3], size=n_samples, p=pay_probs)
    pay_2 = np.random.choice([-1, 0, 1, 2, 3], size=n_samples, p=pay_probs)
    pay_3 = np.random.choice([-1, 0, 1, 2, 3], size=n_samples, p=pay_probs)

    # Bill Statement Amounts & Payments
    utilization = np.random.uniform(0.1, 0.9, size=n_samples) * drift_debt_mult
    utilization = np.clip(utilization, 0.05, 1.2)

    bill_amt1 = np.round(limit_bal * utilization)
    bill_amt2 = np.round(bill_amt1 * np.random.uniform(0.8, 1.05, size=n_samples))
    bill_amt3 = np.round(bill_amt2 * np.random.uniform(0.8, 1.05, size=n_samples))

    # Payments made
    pay_ratio = np.random.uniform(0.05, 0.4, size=n_samples) / drift_debt_mult
    pay_amt1 = np.round(bill_amt1 * pay_ratio)
    pay_amt2 = np.round(bill_amt2 * pay_ratio)
    pay_amt3 = np.round(bill_amt3 * pay_ratio)

    # Calculate Probability of Default based on non-linear risk factors
    log_odds = (
        -2.2
        + (pay_0 * 0.85)
        + (pay_2 * 0.45)
        + (pay_3 * 0.30)
        + (utilization * 1.5)
        - (limit_bal / 250000)
        - (pay_ratio * 2.0)
        + (drift_pay_shift)
    )

    default_prob = 1.0 / (1.0 + np.exp(-log_odds))
    default = (np.random.rand(n_samples) < default_prob).astype(int)

    data = pd.DataFrame({
        "LIMIT_BAL": limit_bal,
        "SEX": sex,
        "EDUCATION": education,
        "MARRIAGE": marriage,
        "AGE": age,
        "PAY_0": pay_0,
        "PAY_2": pay_2,
        "PAY_3": pay_3,
        "BILL_AMT1": bill_amt1,
        "BILL_AMT2": bill_amt2,
        "BILL_AMT3": bill_amt3,
        "PAY_AMT1": pay_amt1,
        "PAY_AMT2": pay_amt2,
        "PAY_AMT3": pay_amt3,
        "default": default
    })

    return data


def run():
    """Main execution function to prepare raw & drift datasets."""
    ensure_directories()

    # Step 1: Raw Baseline Training Dataset
    if not download_uci_dataset():
        print("[DATA] Generating baseline training dataset...")
        df_raw = generate_synthetic_credit_data(n_samples=15000, seed=42, drift=False)
        df_raw.to_csv(RAW_DATA_PATH, index=False)
        print(f"[DATA] Saved baseline synthetic dataset to {RAW_DATA_PATH} ({df_raw.shape[0]} rows).")

    # Standardize column name if UCI dataset was downloaded
    df_check = pd.read_csv(RAW_DATA_PATH)
    if "default.payment.next.month" in df_check.columns:
        df_check.rename(columns={"default.payment.next.month": "default"}, inplace=True)
        df_check.to_csv(RAW_DATA_PATH, index=False)
        print("[DATA] Standardized target column to 'default'.")

    # Step 2: Synthetic Drift Dataset (for Evidently AI drift monitoring)
    print("[DATA] Generating production drift dataset...")
    df_drift = generate_synthetic_credit_data(n_samples=3000, seed=99, drift=True)
    df_drift.to_csv(DRIFT_DATA_PATH, index=False)
    print(f"[DATA] Saved drift dataset to {DRIFT_DATA_PATH} ({df_drift.shape[0]} rows).")

    print("[DATA] Step 1 Complete! Baseline & Drift Datasets are ready.")


if __name__ == "__main__":
    run()
