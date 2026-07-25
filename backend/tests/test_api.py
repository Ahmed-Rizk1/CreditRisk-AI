"""
FastAPI REST API Pytest Verification Suite for CreditRisk AI.
"""

import os
import sys
import io
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint():
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "metrics" in data
    assert "roc_auc" in data["metrics"]


def test_single_applicant_prediction():
    payload = {
        "LIMIT_BAL": 50000.0,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 35,
        "PAY_0": 0,
        "PAY_2": 0,
        "PAY_3": 0,
        "PAY_4": 0,
        "PAY_5": 0,
        "PAY_6": 0,
        "BILL_AMT1": 20000.0,
        "BILL_AMT2": 19000.0,
        "BILL_AMT3": 18000.0,
        "BILL_AMT4": 17000.0,
        "BILL_AMT5": 16000.0,
        "BILL_AMT6": 15000.0,
        "PAY_AMT1": 2000.0,
        "PAY_AMT2": 2000.0,
        "PAY_AMT3": 2000.0,
        "PAY_AMT4": 2000.0,
        "PAY_AMT5": 2000.0,
        "PAY_AMT6": 2000.0
    }
    response = client.post("/api/v1/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "risk_score" in data
    assert "default_probability" in data
    assert "risk_category" in data
    assert "decision" in data
    assert "shap_explanation" in data
    assert "latency_ms" in data
    assert 0.0 <= data["default_probability"] <= 1.0


def test_batch_prediction_csv():
    csv_data = (
        "LIMIT_BAL,SEX,EDUCATION,MARRIAGE,AGE,PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6,BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6,PAY_AMT1,PAY_AMT2,PAY_AMT3,PAY_AMT4,PAY_AMT5,PAY_AMT6\n"
        "50000,2,2,1,35,0,0,0,0,0,0,20000,19000,18000,17000,16000,15000,2000,2000,2000,2000,2000,2000\n"
        "30000,1,3,2,45,2,2,1,0,0,0,28000,27000,26000,25000,24000,23000,500,500,500,500,500,500\n"
    )
    files = {"file": ("test_batch.csv", io.BytesIO(csv_data.encode("utf-8")), "text/csv")}
    response = client.post("/api/v1/predict-batch", files=files)
    assert response.status_code == 200
    data = response.json()

    assert "summary" in data
    assert data["summary"]["total_applicants"] == 2
    assert len(data["predictions"]) == 2


def test_data_drift_report_endpoint():
    response = client.get("/api/v1/drift-report")
    assert response.status_code == 200
    data = response.json()

    assert "dataset_drift" in data
    assert "number_of_drifted_columns" in data
    assert "drift_share" in data
