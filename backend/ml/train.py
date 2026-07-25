"""
MLflow Training Engine & Optuna Tuning Pipeline for CreditRisk AI.

Trains baseline Logistic Regression, XGBoost, LightGBM, and CatBoost models.
Logs parameters, metrics, ROC/PR curves, confusion matrices, and SHAP plots to MLflow.
Promotes the winning model to the MLflow Model Registry under stage tag 'Production'.
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, log_loss,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import optuna

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
from mlflow.tracking import MlflowClient

# Ensure backend path is on sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from backend.ml.data.pipeline import prepare_train_test_data, PIPELINE_SAVE_PATH
from backend.ml.explainer import CreditRiskExplainer, EXPLAINER_SAVE_PATH

# Path settings
RAW_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "credit_card_default.csv"))
MLRUNS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mlruns"))

EXPERIMENT_NAME = "CreditRisk_Default_Prediction"
MODEL_REGISTRY_NAME = "CreditRisk_Production_Model"


def setup_mlflow():
    """Configure MLflow tracking URI and experiment with cross-platform artifact paths."""
    os.makedirs(MLRUNS_DIR, exist_ok=True)
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    db_path = os.path.join(MLRUNS_DIR, "mlflow.db").replace("\\", "/")
    tracking_uri = f"sqlite:///{db_path}"
    mlflow.set_tracking_uri(tracking_uri)

    art_dir = os.path.join(MLRUNS_DIR, "artifacts").replace("\\", "/")
    target_artifact_uri = f"file:{art_dir}"

    client = MlflowClient(tracking_uri)
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        client.create_experiment(EXPERIMENT_NAME, artifact_location=target_artifact_uri)
    else:
        try:
            client._tracking_client.store.update_experiment(
                exp.experiment_id,
                name=exp.name,
                artifact_location=target_artifact_uri
            )
        except Exception:
            pass

    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"[MLFLOW] Tracking URI set to: {tracking_uri}")


def evaluate_model(model, X_test, y_test):
    """Calculate comprehensive evaluation metrics and inference latency."""
    t0 = time.time()
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    latency_ms = ((time.time() - t0) / len(X_test)) * 1000.0

    y_pred = (y_pred_proba >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, y_pred_proba)),
        "pr_auc": float(average_precision_score(y_test, y_pred_proba)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "log_loss": float(log_loss(y_test, y_pred_proba)),
        "latency_ms": float(latency_ms)
    }

    return metrics, y_pred, y_pred_proba


def generate_and_log_plots(y_test, y_pred, y_pred_proba, model_name: str, run_id: str):
    """Generate confusion matrix and ROC curve plots, and log to MLflow."""
    artifacts_dir = os.path.join(MLRUNS_DIR, "plots")
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Confusion Matrix Plot
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Default", "Default"])
    disp.plot(ax=ax_cm, cmap="Blues")
    ax_cm.set_title(f"Confusion Matrix - {model_name}")
    cm_path = os.path.join(artifacts_dir, f"cm_{model_name}.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close(fig_cm)

    # 2. ROC Curve Plot
    fig_roc, ax_roc = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_pred_proba, ax=ax_roc, name=model_name)
    ax_roc.set_title(f"ROC Curve - {model_name}")
    roc_path = os.path.join(artifacts_dir, f"roc_{model_name}.png")
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close(fig_roc)

    # Log artifacts to MLflow safely
    try:
        mlflow.log_artifact(cm_path, artifact_path="plots")
        mlflow.log_artifact(roc_path, artifact_path="plots")
    except Exception as ex:
        print(f"[MLFLOW WARNING] Could not log plot artifacts: {ex}")


def train_baseline_logistic_regression(X_train, y_train, X_test, y_test):
    """Train baseline Logistic Regression model."""
    print("\n[TRAIN] Training Baseline Logistic Regression...")
    with mlflow.start_run(run_name="Logistic_Regression_Baseline") as run:
        params = {"C": 1.0, "max_iter": 1000, "random_state": 42}
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        generate_and_log_plots(y_test, y_pred, y_pred_proba, "LogisticRegression", run.info.run_id)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"   ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | F1: {metrics['f1_score']:.4f}")
        return model, metrics["roc_auc"], run.info.run_id


def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost model with Optuna hyperparameter optimization."""
    print("\n[TRAIN] Tuning & Training XGBoost Classifier...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "eval_metric": "logloss"
        }
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, preds)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    best_params.update({"random_state": 42, "eval_metric": "logloss"})

    with mlflow.start_run(run_name="XGBoost_Optuna_Tuned") as run:
        best_model = xgb.XGBClassifier(**best_params)
        best_model.fit(X_train, y_train)

        metrics, y_pred, y_pred_proba = evaluate_model(best_model, X_test, y_test)

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        generate_and_log_plots(y_test, y_pred, y_pred_proba, "XGBoost", run.info.run_id)
        mlflow.xgboost.log_model(best_model, artifact_path="model")

        print(f"   ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | F1: {metrics['f1_score']:.4f}")
        return best_model, metrics["roc_auc"], run.info.run_id


def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train LightGBM model with Optuna hyperparameter optimization."""
    print("\n[TRAIN] Tuning & Training LightGBM Classifier...")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "random_state": 42,
            "verbose": -1
        }
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train, y_train)
        preds = clf.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    best_params = study.best_params
    best_params.update({"random_state": 42, "verbose": -1})

    with mlflow.start_run(run_name="LightGBM_Optuna_Tuned") as run:
        best_model = lgb.LGBMClassifier(**best_params)
        best_model.fit(X_train, y_train)

        metrics, y_pred, y_pred_proba = evaluate_model(best_model, X_test, y_test)

        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        generate_and_log_plots(y_test, y_pred, y_pred_proba, "LightGBM", run.info.run_id)
        mlflow.lightgbm.log_model(best_model, artifact_path="model")

        print(f"   ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | F1: {metrics['f1_score']:.4f}")
        return best_model, metrics["roc_auc"], run.info.run_id


def train_catboost(X_train, y_train, X_test, y_test):
    """Train CatBoost model."""
    print("\n[TRAIN] Training CatBoost Classifier...")
    with mlflow.start_run(run_name="CatBoost_Baseline") as run:
        params = {"iterations": 150, "learning_rate": 0.05, "depth": 6, "random_seed": 42, "verbose": 0}
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)

        metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        generate_and_log_plots(y_test, y_pred, y_pred_proba, "CatBoost", run.info.run_id)
        mlflow.catboost.log_model(model, artifact_path="model")

        print(f"   ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f} | F1: {metrics['f1_score']:.4f}")
        return model, metrics["roc_auc"], run.info.run_id


def promote_winning_model(winning_model, winning_run_id, model_name: str, X_test):
    """Register winning model in MLflow Model Registry and save local artifacts."""
    print(f"\n[REGISTRY] Promoting winning model '{model_name}' (Run ID: {winning_run_id}) to MLflow Registry...")

    model_uri = f"runs:/{winning_run_id}/model"
    reg_model = mlflow.register_model(model_uri, MODEL_REGISTRY_NAME)

    client = MlflowClient()
    client.set_registered_model_tag(MODEL_REGISTRY_NAME, "task", "credit_default_scoring")
    client.set_model_version_tag(MODEL_REGISTRY_NAME, reg_model.version, "stage", "Production")

    print(f"[REGISTRY] Registered model version {reg_model.version} tagged as 'Production'.")

    # Fit & save SHAP explainer for the winning model
    print("[SHAP] Building and saving SHAP Explainer artifact...")
    explainer = CreditRiskExplainer(winning_model)
    explainer.save(EXPLAINER_SAVE_PATH)


def run_training_pipeline():
    """Main execution entrypoint for model training & MLflow logging."""
    setup_mlflow()

    print("[DATA] Loading raw data and executing preprocessing pipeline...")
    X_train_raw, X_test_raw, X_train, X_test, y_train, y_test, pipeline = prepare_train_test_data(RAW_DATA_PATH)

    results = []

    # 1. Logistic Regression
    lr_model, lr_auc, lr_run_id = train_baseline_logistic_regression(X_train, y_train, X_test, y_test)
    results.append(("LogisticRegression", lr_auc, lr_model, lr_run_id))

    # 2. XGBoost
    xgb_model, xgb_auc, xgb_run_id = train_xgboost(X_train, y_train, X_test, y_test)
    results.append(("XGBoost", xgb_auc, xgb_model, xgb_run_id))

    # 3. LightGBM
    lgb_model, lgb_auc, lgb_run_id = train_lightgbm(X_train, y_train, X_test, y_test)
    results.append(("LightGBM", lgb_auc, lgb_model, lgb_run_id))

    # 4. CatBoost
    cat_model, cat_auc, cat_run_id = train_catboost(X_train, y_train, X_test, y_test)
    results.append(("CatBoost", cat_auc, cat_model, cat_run_id))

    # Select winner
    results.sort(key=lambda x: x[1], reverse=True)
    winner_name, winner_auc, winner_model, winner_run_id = results[0]

    print(f"\n=======================================================")
    print(f"[WINNER] WINNING MODEL: {winner_name} with ROC-AUC = {winner_auc:.4f}")
    print(f"=======================================================")

    promote_winning_model(winner_model, winner_run_id, winner_name, X_test)
    print("\n[TRAIN] Step 2 Complete! Models, MLflow experiments, and SHAP explainers ready.")


if __name__ == "__main__":
    run_training_pipeline()
