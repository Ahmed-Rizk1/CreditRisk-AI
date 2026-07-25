"""
Data Drift Telemetry Service for CreditRisk AI using Kolmogorov-Smirnov Statistical Testing.

Calculates statistical feature drift and distribution shifts between baseline
training data and production incoming credit application streams.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

RAW_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "credit_card_default.csv"))
DRIFT_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "drift", "credit_card_drift.csv"))


def compute_data_drift(
    reference_path: str = RAW_DATA_PATH,
    current_path: str = DRIFT_DATA_PATH,
    p_value_threshold: float = 0.05
) -> dict:
    """
    Compute statistical feature drift using 2-sample Kolmogorov-Smirnov test.

    Returns:
    --------
    dict:
        - dataset_drift: bool
        - number_of_drifted_columns: int
        - total_columns: int
        - drift_share: float
        - drift_by_columns: dict of feature-level p-values and status
    """
    if not os.path.exists(reference_path) or not os.path.exists(current_path):
        return {
            "dataset_drift": False,
            "error": "Reference or current dataset not found."
        }

    df_ref = pd.read_csv(reference_path)
    df_curr = pd.read_csv(current_path)

    # Standardize target column names
    for df in [df_ref, df_curr]:
        if "default.payment.next.month" in df.columns:
            df.rename(columns={"default.payment.next.month": "default"}, inplace=True)

    target_col = "default"
    common_cols = [c for c in df_ref.columns if c in df_curr.columns and c not in ["ID", target_col]]

    drift_by_cols = {}
    drifted_count = 0

    for col in common_cols:
        ref_vals = df_ref[col].dropna().values
        curr_vals = df_curr[col].dropna().values

        # Perform 2-sample KS test
        stat_res = ks_2samp(ref_vals, curr_vals)
        p_val = float(stat_res.pvalue)
        ks_stat = float(stat_res.statistic)
        is_drifted = bool(p_val < p_value_threshold)

        if is_drifted:
            drifted_count += 1

        drift_by_cols[col] = {
            "column_name": col,
            "drift_detected": is_drifted,
            "p_value": round(p_val, 6),
            "ks_statistic": round(ks_stat, 4),
            "threshold": p_value_threshold
        }

    drift_share = float(drifted_count / len(common_cols)) if common_cols else 0.0
    dataset_drift = bool(drift_share >= 0.3)  # Dataset drift flagged if >=30% features drifted

    return {
        "dataset_drift": dataset_drift,
        "number_of_drifted_columns": int(drifted_count),
        "total_columns": int(len(common_cols)),
        "drift_share": round(drift_share, 4),
        "drift_by_columns": drift_by_cols
    }


if __name__ == "__main__":
    print("[DRIFT] Computing Statistical Feature Drift report...")
    res = compute_data_drift()
    print(f"[DRIFT] Dataset Drift Detected: {res['dataset_drift']}")
    print(f"[DRIFT] Drifted Features: {res['number_of_drifted_columns']} / {res['total_columns']} ({res['drift_share']*100:.1f}%)")
