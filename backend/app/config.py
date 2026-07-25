"""
Centralized Configuration Settings for CreditRisk AI FastAPI Service.
"""

import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "CreditRisk AI — Enterprise Credit Risk Scoring & Explainable Decision Platform"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = True

    # Base Paths
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    @property
    def RAW_DATA_PATH(self) -> str:
        return os.path.join(self.BASE_DIR, "data", "raw", "credit_card_default.csv")

    @property
    def DRIFT_DATA_PATH(self) -> str:
        return os.path.join(self.BASE_DIR, "data", "drift", "credit_card_drift.csv")

    @property
    def PROCESSED_DATA_DIR(self) -> str:
        return os.path.join(self.BASE_DIR, "data", "processed")

    @property
    def PIPELINE_SAVE_PATH(self) -> str:
        return os.path.join(self.PROCESSED_DATA_DIR, "preprocessor.joblib")

    @property
    def EXPLAINER_SAVE_PATH(self) -> str:
        return os.path.join(self.PROCESSED_DATA_DIR, "shap_explainer.joblib")

    @property
    def MLRUNS_DIR(self) -> str:
        return os.path.join(self.BASE_DIR, "mlruns")

    @property
    def MLFLOW_TRACKING_URI(self) -> str:
        db_path = os.path.join(self.MLRUNS_DIR, "mlflow.db").replace("\\", "/")
        return f"sqlite:///{db_path}"

    MODEL_REGISTRY_NAME: str = "CreditRisk_Production_Model"

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True, extra="allow")


# Instantiate global settings singleton
settings = Settings()
