"""
MLflow Fraud Detection Pipeline
Replaces Kubeflow pipeline with MLflow experiment tracking
Tasks: Data Loading, Preprocessing, Training, Evaluation, Conditional Deployment
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.models import infer_signature
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ============================================================================
# Configuration
# ============================================================================

MLFLOW_TRACKING_URI = "mlruns"
EXPERIMENT_NAME = "fraud-detection-pipeline"
DATA_DIR = "data"
MODEL_DIR = "model"
AUC_THRESHOLD = 0.85

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)  

# ============================================================================
# Step 1: Load and Validate Data
# ============================================================================

def load_and_validate_data():
    print("\n" + "="*50)
    print("STEP 1: Load and Validate Data")
    print("="*50)

    with mlflow.start_run(run_name="1_load_validate", nested=True):
        # Load sample data
        X_train = pd.read_csv(f"{DATA_DIR}/X_train_sample.csv")
        X_test = pd.read_csv(f"{DATA_DIR}/X_test_sample.csv")
        y_train = pd.read_csv(f"{DATA_DIR}/y_train_sample.csv").values.ravel()
        y_test = pd.read_csv(f"{DATA_DIR}/y_test_sample.csv").values.ravel()

        # Validation checks
        assert X_train.shape[0] == len(y_train), "Train size mismatch"
        assert X_test.shape[0] == len(y_test), "Test size mismatch"
        assert X_train.shape[1] == X_test.shape[1], "Feature count mismatch"

        # Log data stats
        mlflow.log_params({
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "n_features": X_train.shape[1],
            "fraud_rate_train": float(y_train.mean()),
            "fraud_rate_test": float(y_test.mean())
        })

        print(f"✓ X_train: {X_train.shape}")
        print(f"✓ X_test: {X_test.shape}")
        print(f"✓ Fraud rate train: {y_train.mean():.3f}")
        print(f"✓ Data validation: PASSED")

        mlflow.log_metric("validation_passed", 1)

    return X_train, X_test, y_train, y_test

# ============================================================================
# Step 2: Preprocessing
# ============================================================================

def preprocess_data(X_train, X_test, y_train):
    print("\n" + "="*50)
    print("STEP 2: Preprocessing")
    print("="*50)

    with mlflow.start_run(run_name="2_preprocess", nested=True):
        # Handle missing values
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Log preprocessing stats
        missing_before = X_train.isnull().sum().sum()
        mlflow.log_params({
            "missing_values_handled": int(missing_before),
            "scaling": "StandardScaler",
            "imputation": "median"
        })

        # Save scaler
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(scaler, f"{MODEL_DIR}/scaler_mlflow.pkl")
        mlflow.log_artifact(f"{MODEL_DIR}/scaler_mlflow.pkl")

        print(f"✓ Missing values handled: {missing_before}")
        print(f"✓ Scaling applied: StandardScaler")
        print(f"✓ Preprocessing: COMPLETE")

    return X_train_scaled, X_test_scaled

# ============================================================================
# Step 3: Model Training
# ============================================================================

def train_models(X_train, y_train):
    print("\n" + "="*50)
    print("STEP 3: Model Training")
    print("="*50)

    models = {}

    # --- XGBoost ---
    with mlflow.start_run(run_name="3a_train_xgboost", nested=True):
        print("Training XGBoost...")
        xgb_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.05,
            "scale_pos_weight": int((y_train == 0).sum() / (y_train == 1).sum()),
            "tree_method": "hist",
            "random_state": 42
        }
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_train, y_train)
        models["xgboost"] = xgb

        mlflow.log_params(xgb_params)
        mlflow.xgboost.log_model(xgb, "xgboost_model")
        joblib.dump(xgb, f"{MODEL_DIR}/xgboost_mlflow.pkl")
        print(f"✓ XGBoost trained")

    # --- LightGBM ---
    with mlflow.start_run(run_name="3b_train_lightgbm", nested=True):
        print("Training LightGBM...")
        lgbm_params = {
            "n_estimators": 100,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "is_unbalance": True,
            "random_state": 42,
            "verbose": -1
        }
        lgbm = LGBMClassifier(**lgbm_params)
        lgbm.fit(X_train, y_train)
        models["lightgbm"] = lgbm

        mlflow.log_params(lgbm_params)
        mlflow.sklearn.log_model(lgbm, "lightgbm_model")
        joblib.dump(lgbm, f"{MODEL_DIR}/lightgbm_mlflow.pkl")
        print(f"✓ LightGBM trained")

    # --- Hybrid (RF + feature selection) ---
    with mlflow.start_run(run_name="3c_train_hybrid", nested=True):
        print("Training Hybrid RF...")
        rf_params = {
            "n_estimators": 100,
            "max_depth": 8,
            "class_weight": "balanced",
            "random_state": 42
        }
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)
        models["hybrid_rf"] = rf

        mlflow.log_params(rf_params)
        mlflow.sklearn.log_model(rf, "hybrid_rf_model")
        joblib.dump(rf, f"{MODEL_DIR}/hybrid_rf_mlflow.pkl")
        print(f"✓ Hybrid RF trained")

    return models

# ============================================================================
# Step 4: Evaluation
# ============================================================================

def evaluate_models(models, X_test, y_test):
    print("\n" + "="*50)
    print("STEP 4: Model Evaluation")
    print("="*50)

    results = {}

    for name, model in models.items():
        with mlflow.start_run(run_name=f"4_evaluate_{name}", nested=True):
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba)
            cm = confusion_matrix(y_test, y_pred)

            metrics = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "auc_roc": auc,
                "true_negatives": int(cm[0][0]),
                "false_positives": int(cm[0][1]),
                "false_negatives": int(cm[1][0]),
                "true_positives": int(cm[1][1])
            }

            mlflow.log_metrics(metrics)
            results[name] = metrics

            print(f"\n{name.upper()}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1:        {f1:.4f}")
            print(f"  AUC-ROC:   {auc:.4f}")

    # Log comparison table
    results_df = pd.DataFrame(results).T
    results_df.to_csv("data/model_comparison_mlflow.csv")
    mlflow.log_artifact("data/model_comparison_mlflow.csv")

    return results

# ============================================================================
# Step 5: Cost-Sensitive Comparison
# ============================================================================

def cost_sensitive_comparison(X_train, X_test, y_train, y_test):
    print("\n" + "="*50)
    print("STEP 5: Cost-Sensitive vs Standard Training")
    print("="*50)

    with mlflow.start_run(run_name="5_cost_sensitive", nested=True):
        # Standard training
        std_model = XGBClassifier(n_estimators=100, random_state=42)
        std_model.fit(X_train, y_train)
        std_pred = std_model.predict(X_test)
        std_recall = recall_score(y_test, std_pred, zero_division=0)
        std_auc = roc_auc_score(y_test, std_model.predict_proba(X_test)[:, 1])

        # Cost-sensitive training (higher penalty for false negatives)
        ratio = int((y_train == 0).sum() / (y_train == 1).sum())
        cs_model = XGBClassifier(
            n_estimators=100,
            scale_pos_weight=ratio * 5,  # 5x penalty for false negatives
            random_state=42
        )
        cs_model.fit(X_train, y_train)
        cs_pred = cs_model.predict(X_test)
        cs_recall = recall_score(y_test, cs_pred, zero_division=0)
        cs_auc = roc_auc_score(y_test, cs_model.predict_proba(X_test)[:, 1])

        mlflow.log_metrics({
            "standard_recall": std_recall,
            "standard_auc": std_auc,
            "cost_sensitive_recall": cs_recall,
            "cost_sensitive_auc": cs_auc,
            "recall_improvement": cs_recall - std_recall
        })

        print(f"  Standard    → Recall: {std_recall:.4f}, AUC: {std_auc:.4f}")
        print(f"  Cost-Sensitive → Recall: {cs_recall:.4f}, AUC: {cs_auc:.4f}")
        print(f"  Recall improvement: {cs_recall - std_recall:+.4f}")

    return cs_model

# ============================================================================
# Step 6: Conditional Deployment
# ============================================================================

def conditional_deploy(models, results):
    print("\n" + "="*50)
    print("STEP 6: Conditional Deployment")
    print("="*50)

    with mlflow.start_run(run_name="6_deploy_decision", nested=True):
        # Find best model by AUC
        best_model_name = max(results, key=lambda x: results[x]["auc_roc"])
        best_auc = results[best_model_name]["auc_roc"]
        best_recall = results[best_model_name]["recall"]

        mlflow.log_params({
            "best_model": best_model_name,
            "deployment_threshold": AUC_THRESHOLD
        })
        mlflow.log_metrics({
            "best_auc": best_auc,
            "best_recall": best_recall
        })

        if best_auc >= AUC_THRESHOLD:
            decision = "DEPLOY"
            print(f"✓ Model APPROVED: {best_model_name}")
            print(f"  AUC: {best_auc:.4f} >= threshold {AUC_THRESHOLD}")

            # Register model in MLflow Model Registry
            best_model = models[best_model_name]
            mlflow.sklearn.log_model(
                best_model,
                "best_model",
                registered_model_name="fraud-detection-model"
            )

            # Save as production model
            joblib.dump(best_model, f"{MODEL_DIR}/xgboost.pkl")
            print(f"✓ Model saved to {MODEL_DIR}/xgboost.pkl")
        else:
            decision = "REJECT"
            print(f"✗ Model REJECTED: AUC {best_auc:.4f} < threshold {AUC_THRESHOLD}")

        mlflow.log_param("deployment_decision", decision)
        print(f"\n{'='*50}")
        print(f"DEPLOYMENT DECISION: {decision}")
        print(f"{'='*50}")

    return decision

# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline():
    print("\n" + "="*60)
    print("FRAUD DETECTION MLFLOW PIPELINE")
    print("IEEE CIS Fraud Detection Dataset")
    print("="*60)

    # Set experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="fraud_detection_pipeline"):
        # Step 1: Load data
        X_train, X_test, y_train, y_test = load_and_validate_data()

        # Step 2: Preprocess
        X_train_scaled, X_test_scaled = preprocess_data(X_train, X_test, y_train)

        # Step 3: Train models
        models = train_models(X_train_scaled, y_train)

        # Step 4: Evaluate
        results = evaluate_models(models, X_test_scaled, y_test)

        # Step 5: Cost-sensitive comparison
        cs_model = cost_sensitive_comparison(X_train_scaled, X_test_scaled, y_train, y_test)

        # Step 6: Deploy decision
        decision = conditional_deploy(models, results)

        # Log final summary
        mlflow.log_param("pipeline_status", "COMPLETE")
        mlflow.log_param("final_decision", decision)

        print("\n✓ Pipeline complete!")
        print(f"✓ View results at: http://localhost:5000")
        print(f"✓ Experiment: {EXPERIMENT_NAME}")

if __name__ == "__main__":
    run_pipeline()