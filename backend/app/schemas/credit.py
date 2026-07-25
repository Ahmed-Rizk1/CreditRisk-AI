"""
Pydantic v2 Data Validation Schemas for CreditRisk AI.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class CreditApplicantInput(BaseModel):
    """Input features for single borrower credit risk evaluation."""
    LIMIT_BAL: float = Field(..., example=50000.0, description="Amount of given credit in NT dollars")
    SEX: int = Field(..., example=2, description="1=Male, 2=Female")
    EDUCATION: int = Field(..., example=2, description="1=Graduate School, 2=University, 3=High School, 4=Others")
    MARRIAGE: int = Field(..., example=1, description="1=Married, 2=Single, 3=Others")
    AGE: int = Field(..., example=35, description="Age in years")
    PAY_0: int = Field(0, json_schema_extra={"example": 0}, description="Repayment status in Sept (-1=Duly, 0=Revolving, 1..8=Delay months)")
    PAY_2: int = Field(0, json_schema_extra={"example": 0}, description="Repayment status in Aug")
    PAY_3: int = Field(0, json_schema_extra={"example": 0}, description="Repayment status in July")
    PAY_4: int = Field(0, json_schema_extra={"example": 0}, description="Repayment status in June")
    PAY_5: int = Field(0, json_schema_extra={"example": 0}, description="Repayment status in May")
    PAY_6: int = Field(0, json_schema_extra={"example": 0}, description="Repayment status in April")
    BILL_AMT1: float = Field(0.0, json_schema_extra={"example": 20000.0}, description="Bill statement amount in Sept")
    BILL_AMT2: float = Field(0.0, json_schema_extra={"example": 19000.0}, description="Bill statement amount in Aug")
    BILL_AMT3: float = Field(0.0, json_schema_extra={"example": 18000.0}, description="Bill statement amount in July")
    BILL_AMT4: float = Field(0.0, json_schema_extra={"example": 17000.0}, description="Bill statement amount in June")
    BILL_AMT5: float = Field(0.0, json_schema_extra={"example": 16000.0}, description="Bill statement amount in May")
    BILL_AMT6: float = Field(0.0, json_schema_extra={"example": 15000.0}, description="Bill statement amount in April")
    PAY_AMT1: float = Field(0.0, json_schema_extra={"example": 2000.0}, description="Previous payment amount in Sept")
    PAY_AMT2: float = Field(0.0, json_schema_extra={"example": 2000.0}, description="Previous payment amount in Aug")
    PAY_AMT3: float = Field(0.0, json_schema_extra={"example": 2000.0}, description="Previous payment amount in July")
    PAY_AMT4: float = Field(0.0, json_schema_extra={"example": 2000.0}, description="Previous payment amount in June")
    PAY_AMT5: float = Field(0.0, json_schema_extra={"example": 2000.0}, description="Previous payment amount in May")
    PAY_AMT6: float = Field(0.0, json_schema_extra={"example": 2000.0}, description="Previous payment amount in April")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "LIMIT_BAL": 50000.0,
            "SEX": 2,
            "EDUCATION": 2,
            "MARRIAGE": 1,
            "AGE": 35,
            "PAY_0": 0,
            "PAY_2": 0,
            "PAY_3": 0,
            "BILL_AMT1": 20000.0,
            "BILL_AMT2": 19000.0,
            "BILL_AMT3": 18000.0,
            "PAY_AMT1": 2000.0,
            "PAY_AMT2": 2000.0,
            "PAY_AMT3": 2000.0
        }
    })


class FeatureContribution(BaseModel):
    feature: str
    feature_value: Any
    shap_value: float
    impact: str  # "INCREASES_RISK" or "REDUCES_RISK"


class SHAPExplanation(BaseModel):
    base_value: float
    feature_contributions: List[FeatureContribution]
    top_risk_drivers: List[FeatureContribution]
    top_risk_reducers: List[FeatureContribution]


class PredictionResponse(BaseModel):
    applicant_id: str
    risk_score: float  # Percentage score (e.g. 14.5%)
    default_probability: float  # Prob (0.0 to 1.0)
    risk_category: str  # "LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"
    decision: str  # "APPROVED", "MANUAL_REVIEW", "DECLINED"
    recommended_tier: str
    adverse_action_reasons: List[str]
    shap_explanation: SHAPExplanation
    latency_ms: float


class BatchPredictionSummary(BaseModel):
    total_applicants: int
    approved_count: int
    review_count: int
    declined_count: int
    average_risk_score: float


class BatchPredictionResponse(BaseModel):
    summary: BatchPredictionSummary
    predictions: List[PredictionResponse]


class DriftColumnDetail(BaseModel):
    column_name: str
    drift_detected: bool
    p_value: float
    ks_statistic: float
    threshold: float


class DriftReportResponse(BaseModel):
    dataset_drift: bool
    number_of_drifted_columns: int
    total_columns: int
    drift_share: float
    drift_by_columns: Dict[str, DriftColumnDetail]


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    stage: str
    metrics: Dict[str, float]
    registered_at: Optional[str] = None
