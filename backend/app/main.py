"""
FastAPI REST Application for CreditRisk AI.

Serves single applicant risk evaluation + SHAP explanations, batch CSV scoring,
model performance metadata, and statistical data drift reports.
"""

import sys
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.app.config import settings
from backend.app.schemas.credit import (
    CreditApplicantInput, PredictionResponse, BatchPredictionResponse,
    DriftReportResponse, ModelInfoResponse
)
from backend.app.services.model_service import model_service
from backend.ml.drift import compute_data_drift

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up ML model & SHAP explainer on server boot to avoid JIT latency on first request."""
    try:
        dummy_input = CreditApplicantInput(
            LIMIT_BAL=50000, SEX=2, EDUCATION=2, MARRIAGE=1, AGE=35,
            PAY_0=0, PAY_2=0, PAY_3=0, PAY_4=0, PAY_5=0, PAY_6=0,
            BILL_AMT1=20000, BILL_AMT2=19000, BILL_AMT3=18000, BILL_AMT4=17000, BILL_AMT5=16000, BILL_AMT6=15000,
            PAY_AMT1=2000, PAY_AMT2=2000, PAY_AMT3=2000, PAY_AMT4=2000, PAY_AMT5=2000, PAY_AMT6=2000
        )
        model_service.predict_single(dummy_input)
        print("[STARTUP WARMUP] Model & SHAP explainer successfully pre-warmed.")
    except Exception as ex:
        print(f"[STARTUP WARMUP WARNING] Pre-warm failed: {ex}")
    yield

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Enterprise Credit Risk Scoring & Explainable Decision Platform API",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for React SPA frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r"https://.*\.onrender\.com",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok", "service": "CreditRisk AI Backend"}


@app.get(f"{settings.API_V1_PREFIX}/model-info", response_model=ModelInfoResponse, tags=["Model Intelligence"])
def get_active_model_info():
    """Retrieve active production model version, stage, and benchmark metrics."""
    return model_service.get_model_info()


@app.post(f"{settings.API_V1_PREFIX}/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_applicant_risk(applicant: CreditApplicantInput):
    """
    Evaluate single borrower credit default risk probability.
    
    Returns:
    - Probability of Default %
    - Risk Category (LOW_RISK, MEDIUM_RISK, HIGH_RISK)
    - Underwriting Decision (APPROVED, MANUAL_REVIEW, DECLINED)
    - Interactive SHAP Feature Attribution Explanation
    - Adverse Action Notice Reasons
    """
    try:
        return model_service.predict_single(applicant)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post(f"{settings.API_V1_PREFIX}/predict-batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch_csv(file: UploadFile = File(...)):
    """
    Asynchronously process a bulk CSV file of credit applications.
    
    Accepts: Multipart CSV file upload
    Returns: Portfolio summary statistics and array of applicant risk decisions
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        return model_service.predict_batch(contents)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")


@app.get(f"{settings.API_V1_PREFIX}/drift-report", response_model=DriftReportResponse, tags=["MLOps & Telemetry"])
def get_data_drift_report():
    """
    Calculate statistical feature drift between baseline training data and incoming live stream.
    
    Returns Kolmogorov-Smirnov test p-values, KS statistics, and dataset drift alert status.
    """
    try:
        return compute_data_drift()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift computation error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
