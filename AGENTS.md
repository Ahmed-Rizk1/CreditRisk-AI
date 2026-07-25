# AGENTS.md

This file is the source of truth for any AI coding agent (Antigravity, Cursor, Claude Code, Copilot, etc.) working in this repository. Read this file before creating or modifying code.

---

## 1. Project Overview

**CreditRisk AI** is an enterprise-grade Credit Risk Scoring & Explainable Decision Platform. It processes credit card applicant data to evaluate default risk, provides real-time SHAP feature attributions, logs experiments and models to an MLflow Model Registry, detects data drift via Evidently AI, and serves predictions to a modern React SPA frontend dashboard.

---

## 2. Tech Stack & Environment Rules

### Stack Specifications
* **Language & Runtime:** Python 3.10+ (Backend) & Node.js / TypeScript (Frontend).
* **Package Manager:** **`uv`** is the mandatory Python package manager (`uv pip install -r backend/requirements.txt`).
* **ML Engines:** XGBoost, LightGBM, CatBoost, Scikit-Learn baseline.
* **Hyperparameter Optimization:** Optuna.
* **MLOps & Tracking:** MLflow (Experiment tracking, artifacts, Model Registry with `stage=Production`).
* **Explainable AI (XAI):** SHAP (TreeExplainer & Waterfall explanations).
* **Drift Telemetry:** Evidently AI (Dataset drift analysis).
* **Backend API:** FastAPI (Async, Pydantic v2 schemas, CORS enabled).
* **Frontend SPA:** Vite + React + TypeScript + Vanilla CSS / Modern UI components.
* **Orchestration:** Docker & Docker Compose.

---

## 3. Repository Layout

```text
d:\machineLearning_project/
├── AGENTS.md                  # System instructions & context for AI agents (This file)
├── README.md                  # Recruiter-ready project overview & benchmarks
├── docker-compose.yml         # Container orchestration
├── data/                      # Local datasets (Gitignored raw payloads)
│   ├── raw/                   # Official 30,000-row UCI Credit Card dataset
│   ├── processed/             # Cleaned feature-engineered arrays
│   └── drift/                 # 3,000-row synthetic production drift dataset
├── mlruns/                    # MLflow local tracking store & model artifacts
├── backend/                   # Python FastAPI & ML pipeline (see backend/AGENTS.md)
│   ├── AGENTS.md
│   ├── requirements.txt
│   ├── app/                   # FastAPI service layer
│   │   ├── api/               # API endpoints (/predict, /predict-batch, /drift, /model-info)
│   │   ├── core/              # Config & settings module
│   │   ├── schemas/           # Pydantic data validation schemas
│   │   └── services/          # Dynamic MLflow loader & SHAP explainer
│   ├── ml/                    # Machine Learning pipeline
│   │   ├── data/              # Ingestion & synthetic drift generator
│   │   ├── pipeline.py        # Scikit-learn preprocessing & feature engineering
│   │   ├── train.py           # MLflow training engine with Optuna tuning
│   │   ├── explainer.py       # SHAP attributions engine
│   │   └── drift.py           # Evidently AI drift detector
│   └── tests/                 # Pytest suite
└── frontend/                  # React SPA (see frontend/AGENTS.md)
    ├── AGENTS.md
    ├── package.json
    └── src/                   # React components & API client
```

---

## 4. Workflow Rules for AI Agents

1. **Use `uv` for Python Package Management:**
   Always prefer `uv pip install <package>` or `uv pip install -r backend/requirements.txt` over plain `pip`.
2. **Never Call `os.getenv` Directly in Application Code:**
   Use `backend/app/core/config.py` as the single source of truth for configuration parameters.
3. **No Silent Fallbacks:**
   Validate inputs at boundaries using Pydantic schemas. Raise explicit HTTP errors when requests are invalid.
4. **Preserve Reproducibility:**
   Random seeds must be explicitly set (`seed=42`) across NumPy, Scikit-learn, XGBoost, LightGBM, and CatBoost.
5. **No Speculative Feature Flags:**
   Keep modules small, focused, and free of premature abstractions.

---

## 5. Verification Commands

Before declaring any task complete, run the corresponding verification:

```powershell
# 1. Regenerate or verify raw & drift data pipelines
python backend/ml/data/download_or_generate.py

# 2. Run backend pytest suite
pytest backend/tests/
```
