"""
Pytest Verification Suite for CreditRisk AI Pipeline & ML Engine.
"""

import os
import sys
import joblib
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.ml.data.download_or_generate import RAW_DATA_PATH, DRIFT_DATA_PATH
from backend.ml.data.pipeline import PIPELINE_SAVE_PATH
from backend.ml.explainer import EXPLAINER_SAVE_PATH, CreditRiskExplainer
from backend.ml.drift import compute_data_drift


def test_data_files_exist():
    """Verify raw UCI dataset and drift dataset files exist and are non-empty."""
    assert os.path.exists(RAW_DATA_PATH), "Raw UCI dataset file missing!"
    assert os.path.exists(DRIFT_DATA_PATH), "Drift dataset file missing!"

    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_drift = pd.read_csv(DRIFT_DATA_PATH)

    assert len(df_raw) >= 10000, f"Expected at least 10,000 raw rows, got {len(df_raw)}"
    assert len(df_drift) >= 1000, f"Expected at least 1,000 drift rows, got {len(df_drift)}"


def test_preprocessor_artifact():
    """Verify Scikit-Learn preprocessor artifact exists and transforms sample data."""
    assert os.path.exists(PIPELINE_SAVE_PATH), "Preprocessor joblib artifact missing!"
    pipeline = joblib.load(PIPELINE_SAVE_PATH)

    df_raw = pd.read_csv(RAW_DATA_PATH)
    target_col = "default" if "default" in df_raw.columns else "default payment next month"
    X = df_raw.drop(columns=[c for c in ["ID", target_col] if c in df_raw.columns])

    X_transformed = pipeline.transform(X.iloc[:5])
    assert X_transformed.shape[0] == 5, "Pipeline transformation output row count mismatch!"
    assert X_transformed.shape[1] >= 15, "Pipeline feature column output mismatch!"


def test_shap_explainer_artifact():
    """Verify SHAP explainer artifact loads and calculates feature attributions."""
    assert os.path.exists(EXPLAINER_SAVE_PATH), "SHAP explainer joblib artifact missing!"
    explainer = joblib.load(EXPLAINER_SAVE_PATH)

    df_raw = pd.read_csv(RAW_DATA_PATH)
    pipeline = joblib.load(PIPELINE_SAVE_PATH)

    target_col = "default" if "default" in df_raw.columns else "default payment next month"
    X = df_raw.drop(columns=[c for c in ["ID", target_col] if c in df_raw.columns])
    X_trans = pd.DataFrame(pipeline.transform(X.iloc[:1]))

    explanation = explainer.explain_instance(X_trans)
    assert "base_value" in explanation
    assert "feature_contributions" in explanation
    assert len(explanation["feature_contributions"]) > 0


def test_data_drift_engine():
    """Verify Evidently AI data drift detection executes and returns valid metrics."""
    drift_result = compute_data_drift(reference_path=RAW_DATA_PATH, current_path=DRIFT_DATA_PATH)

    assert "dataset_drift" in drift_result
    assert "drift_share" in drift_result
    assert isinstance(drift_result["dataset_drift"], bool)
    assert 0.0 <= drift_result["drift_share"] <= 1.0
