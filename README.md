# CreditRisk AI — Enterprise Credit Risk Scoring & Explainable Decision Platform

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![MLflow](https://img.shields.io/badge/MLflow-3.14-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![SHAP](https://img.shields.io/badge/SHAP-XAI-ff69b4?style=for-the-badge)](https://shap.readthedocs.io)
[![uv](https://img.shields.io/badge/package__manager-uv-de5b49?style=for-the-badge)](https://github.com/astral-sh/uv)

An enterprise-grade, end-to-end Machine Learning platform built for financial institutions to automate credit card default risk evaluation, provide real-time **SHAP feature attributions**, track experiment pipelines with **MLflow**, monitor statistical **data drift**, and serve low-latency decisions to an interactive **React SPA dashboard**.

---

## 📐 System Architecture

```mermaid
flowchart TD
    subgraph Data & Pipeline Layer
        A[UCI Credit Dataset\n30,000 Records] --> B[Scikit-Learn Feature Pipeline\nStandardScaler + OneHot + Ratios]
        C[Synthetic Drift Generator] --> B
    end

    subgraph MLOps & Experiment Tracking
        B --> D[Optuna Hyperparameter Tuning]
        D --> E[Model Benchmark Engine\nXGBoost | LightGBM | CatBoost | LogisticReg]
        E --> F[MLflow Experiment Tracking & SQLite Store]
        F --> G[MLflow Model Registry\nstage=Production]
    end

    subgraph FastAPI Serving & Telemetry
        G --> H[FastAPI Inference Engine\nLow Latency <50ms]
        H --> I[SHAP Explainer Module\nFeature Contributions & Adverse Actions]
        H --> J[KS Statistical Data Drift Engine]
    end

    subgraph User Experience Layer
        H <--> K[React + TypeScript + Vite SPA]
        K --> L[Single Applicant Risk Gauge & SHAP Chart]
        K --> M[Batch CSV Portfolio Processor]
        K --> N[Model Health & Telemetry Dashboard]
    end
```

---

## 🏆 Model Performance Benchmark

All models were evaluated on an 80/20 stratified split of the 30,000-record UCI Credit Card Default benchmark dataset.

| Model | ROC-AUC | PR-AUC | F1 Score | Accuracy | Avg Latency | MLflow Registry Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CatBoost Classifier** | **0.7817** | **0.5629** | **0.4646** | **81.90%** | **2.4 ms** | 🟢 **Production (Active)** |
| **LightGBM (Optuna Tuned)** | 0.7815 | 0.5586 | 0.4650 | 81.85% | 1.8 ms | Registered Version 1 |
| **XGBoost (Optuna Tuned)** | 0.7815 | 0.5613 | 0.4652 | 81.70% | 2.1 ms | Tracked in MLflow |
| **Logistic Regression** | 0.7351 | 0.5078 | 0.4033 | 78.10% | 0.5 ms | Baseline |

---

## ✨ Key Platform Features

1. **Explainable AI (SHAP & Adverse Action Notices):**
   - Calculates exact local SHAP feature attributions for every prediction.
   - Generates regulatory adverse action reasons (e.g., *"History of recent payment delinquencies"*, *"High credit utilization ratio"*).

2. **Automated MLflow MLOps Pipeline:**
   - Logs hyper-parameters, metrics (ROC-AUC, PR-AUC, F1, Log-Loss), confusion matrices, and ROC curves to MLflow.
   - Dynamically registers winning models to MLflow Model Registry (`stage=Production`).

3. **Data Drift Telemetry (Kolmogorov-Smirnov Test):**
   - Monitors production application streams against baseline training distributions.
   - Calculates p-values and flags dataset drift when macroeconomic shifts occur.

4. **Production FastAPI Backend:**
   - Single applicant scoring (`/api/v1/predict`).
   - Asynchronous batch CSV portfolio upload (`/api/v1/predict-batch`).
   - OpenAPI Swagger documentation (`/docs`).

5. **Modern React SPA Dashboard:**
   - Built with Vite, TypeScript, and Glassmorphism design system.
   - Interactive SHAP waterfall charts (Recharts) and batch CSV export.

---

## ⚡ Quickstart Guide

### Prerequisites
- Python 3.11
- Node.js 18+
- [`uv`](https://github.com/astral-sh/uv) (Fast Python Package Manager)

### 1. Environment & Dependencies Setup
```powershell
# Create isolated Python 3.11 virtual environment with uv
uv venv --python 3.11 .venv

# Install backend dependencies
uv pip install -r backend/requirements.txt --python .venv\Scripts\python.exe

# Install frontend npm dependencies
cd frontend
npm install
cd ..
```

### 2. Ingest Dataset & Run MLflow Training
```powershell
# Download UCI credit card dataset & generate drift stream
.venv\Scripts\python.exe backend/ml/data/download_or_generate.py

# Train models, run Optuna tuning & promote winner to MLflow Registry
.venv\Scripts\python.exe backend/ml/train.py
```

### 3. Launch Services
```powershell
# Launch FastAPI Inference Server (Port 8000)
.venv\Scripts\uvicorn.exe backend.app.main:app --reload --port 8000

# Launch MLflow Tracking UI (Port 5000)
.venv\Scripts\mlflow.exe ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

# Launch React SPA Frontend (Port 5173 / 5174)
cd frontend
npm run dev
```

---

## 🐳 Docker Deployment

Run the complete multi-container stack with Docker Compose:

```powershell
docker-compose up --build
```
- **React Dashboard:** `http://localhost:5173`
- **FastAPI Documentation:** `http://localhost:8000/docs`
- **MLflow Tracking UI:** `http://localhost:5000`

---

## 📄 Model Card Summary

- **Task:** Binary Classification (Default payment next month: `0` = No Default, `1` = Default).
- **Target Audience:** Financial Loan Officers, Risk Underwriters, MLOps Engineers.
- **Fairness & Compliance:** Explanations provided via SHAP to comply with the Fair Credit Reporting Act (FCRA) and EU AI Act transparent AI requirements.
