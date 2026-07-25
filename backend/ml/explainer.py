"""
SHAP Explainability Engine for CreditRisk AI.

Provides feature attribution values, waterfall explanations, and regulatory compliance
adverse action notice reasons for individual applicant risk evaluations.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap

PROCESSED_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
EXPLAINER_SAVE_PATH = os.path.join(PROCESSED_DATA_DIR, "shap_explainer.joblib")


class CreditRiskExplainer:
    """Wrapper class around SHAP TreeExplainer / Explainer for Credit Risk model decision attribution."""

    def __init__(self, model=None, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        if model is not None:
            self._init_explainer(model)

    def _init_explainer(self, model):
        """Initialize SHAP explainer targeting tree-based models or general estimators."""
        try:
            self.explainer = shap.TreeExplainer(model)
        except Exception:
            # Fallback to general SHAP explainer
            self.explainer = shap.Explainer(model)

    def explain_instance(self, X_instance: pd.DataFrame) -> dict:
        """
        Calculate SHAP feature attributions for a single applicant instance.

        Returns:
        --------
        dict:
            - base_value: Expected baseline model logit or probability
            - shap_values: Feature attribution scores
            - feature_contributions: List of {feature, value, shap_value, impact} dicts
            - top_risk_drivers: Top 3 reasons increasing credit default risk
            - top_risk_reducers: Top 3 reasons lowering credit default risk
        """
        if self.explainer is None:
            raise ValueError("SHAP explainer has not been initialized with a trained model.")

        shap_vals = self.explainer(X_instance)
        
        # Handle 1D vs 2D SHAP output arrays
        values = shap_vals.values[0] if len(shap_vals.values.shape) > 1 else shap_vals.values
        if len(values.shape) > 1:  # Binary classification binary class select
            values = values[:, 1] if values.shape[1] > 1 else values[:, 0]

        base_val = shap_vals.base_values[0] if hasattr(shap_vals, "base_values") and len(shap_vals.base_values) > 0 else 0.0
        if isinstance(base_val, np.ndarray):
            base_val = float(base_val[0])

        feature_names = self.feature_names if self.feature_names else list(X_instance.columns)
        instance_vals = X_instance.iloc[0].values if isinstance(X_instance, pd.DataFrame) else X_instance[0]

        contributions = []
        for feat_name, raw_val, s_val in zip(feature_names, instance_vals, values):
            contributions.append({
                "feature": str(feat_name),
                "feature_value": float(raw_val) if isinstance(raw_val, (int, float, np.number)) else str(raw_val),
                "shap_value": float(s_val),
                "impact": "INCREASES_RISK" if s_val > 0 else "REDUCES_RISK"
            })

        # Sort by absolute SHAP impact
        contributions_sorted = sorted(contributions, key=lambda x: abs(x["shap_value"]), reverse=True)
        top_risk_drivers = [c for c in contributions_sorted if c["shap_value"] > 0][:3]
        top_risk_reducers = [c for c in contributions_sorted if c["shap_value"] < 0][:3]

        return {
            "base_value": float(base_val),
            "feature_contributions": contributions,
            "top_risk_drivers": top_risk_drivers,
            "top_risk_reducers": top_risk_reducers
        }

    def save(self, filepath: str = EXPLAINER_SAVE_PATH):
        """Serialize explainer artifact."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[EXPLAINER] Saved SHAP explainer to {filepath}")

    @classmethod
    def load(cls, filepath: str = EXPLAINER_SAVE_PATH):
        """Deserialize explainer artifact."""
        return joblib.load(filepath)
