# Backend Guidelines - CreditRisk AI

This document provides instructions for AI agents working in the `backend/` directory.

## Backend Stack
- **Framework:** FastAPI
- **Python Package Manager:** `uv` (`uv pip install -r requirements.txt`)
- **ML Libraries:** Scikit-Learn, XGBoost, LightGBM, CatBoost, Optuna, MLflow, SHAP, Evidently AI
- **Testing:** Pytest

## Core Principles
1. **Pydantic Schemas:** All incoming request bodies and API response structures must be strictly typed in `backend/app/schemas/`.
2. **MLflow Registry:** The FastAPI model service (`backend/app/services/model_service.py`) must load models dynamically from the MLflow tracking store (`mlruns/`) targeting the `Production` model tag.
3. **Low-Latency Inference:** Keep preprocessing pipelines vectorized using Pandas/NumPy to maintain single prediction latency under 50ms on CPU.
4. **SHAP Integration:** The SHAP explainer module (`backend/ml/explainer.py`) must return both global base values and local feature attribution dictionaries to feed the frontend waterfall chart.

## Setup & Execution
```powershell
# Install backend dependencies with uv
uv pip install -r requirements.txt

# Run dataset ingestion script
python ml/data/download_or_generate.py

# Run MLflow training pipeline
python ml/train.py

# Launch FastAPI development server
uvicorn app.main:app --reload --port 8000
```
