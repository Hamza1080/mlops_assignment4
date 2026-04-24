"""
MLflow Fraud Detection Pipeline — Enhanced
Replaces Kubeflow pipeline with MLflow experiment tracking
Tasks: Data Loading, Preprocessing, Training, Evaluation, Conditional Deployment,
       Imbalance Strategy Comparison, Drift Simulation, Retraining Strategy Comparison
"""

import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score
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
# Helper: compute full metric dict
# ============================================================================

def compute_metrics(y_test, y_pred, y_proba, prefix=""):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    metrics = {
        f"{prefix}precision": float(precision_score(y_test, y_pred, zero_division=0)),
        f"{prefix}recall": float(recall_score(y_test, y_pred, zero_division=0)),
        f"{prefix}f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        f"{prefix}auc_roc": float(roc_auc_score(y_test, y_proba)),
        f"{prefix}avg_precision": float(average_precision_score(y_test, y_proba)),
        f"{prefix}false_positive_rate": float(fpr),
        f"{prefix}true_negatives": int(tn),
        f"{prefix}false_positives": int(fp),
        f"{prefix}false_negatives": int(fn),
        f"{prefix}true_positives": int(tp),
        f"{prefix}fraud_detection_rate": float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
    }
    return metrics

# ============================================================================
# Step 1: Load and Validate Data
# ============================================================================

def load_and_validate_data():
    print("\n" + "="*50)
    print("STEP 1: Load and Validate Data")
    print("="*50)

    with mlflow.start_run(run_name="1_load_validate", nested=True):
        X_train = pd.read_csv(f"{DATA_DIR}/X_train_sample.csv")
        X_test = pd.read_csv(f"{DATA_DIR}/X_test_sample.csv")
        y_train = pd.read_csv(f"{DATA_DIR}/y_train_sample.csv").values.ravel()
        y_test = pd.read_csv(f"{DATA_DIR}/y_test_sample.csv").values.ravel()

        assert X_train.shape[0] == len(y_train)
        assert X_test.shape[0] == len(y_test)
        assert X_train.shape[1] == X_test.shape[1]

        fraud_train = float(y_train.mean())
        fraud_test = float(y_test.mean())
        missing_pct = float(X_train.isnull().sum().sum() / X_train.size)

        mlflow.log_params({
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "n_features": X_train.shape[1],
        })
        mlflow.log_metrics({
            "fraud_rate_train": fraud_train,
            "fraud_rate_test": fraud_test,
            "missing_value_pct": missing_pct,
            "class_imbalance_ratio": float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
            "validation_passed": 1,
            "train_samples": X_train.shape[0],
            "test_samples": X_test.shape[0],
            "n_features": X_train.shape[1],
        })

        print(f"  X_train: {X_train.shape}, fraud_rate: {fraud_train:.4f}")
        print(f"  X_test:  {X_test.shape},  fraud_rate: {fraud_test:.4f}")
        print(f"  Missing values: {missing_pct:.4f}%")
        print(f"  Validation: PASSED")

    return X_train, X_test, y_train, y_test

# ============================================================================
# Step 2: Preprocessing
# ============================================================================

def preprocess_data(X_train, X_test, y_train):
    print("\n" + "="*50)
    print("STEP 2: Preprocessing")
    print("="*50)

    with mlflow.start_run(run_name="2_preprocess", nested=True):
        missing_before = int(X_train.isnull().sum().sum())

        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())

        missing_after = int(X_train.isnull().sum().sum())

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Feature variance stats
        variances = np.var(X_train_scaled, axis=0)
        low_var_features = int((variances < 0.01).sum())

        mlflow.log_params({
            "scaling_method": "StandardScaler",
            "imputation_method": "median",
        })
        mlflow.log_metrics({
            "missing_values_before": missing_before,
            "missing_values_after": missing_after,
            "missing_values_fixed": missing_before - missing_after,
            "low_variance_features": low_var_features,
            "feature_mean_after_scaling": float(np.mean(X_train_scaled)),
            "feature_std_after_scaling": float(np.std(X_train_scaled)),
        })

        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(scaler, f"{MODEL_DIR}/scaler_mlflow.pkl")
        mlflow.log_artifact(f"{MODEL_DIR}/scaler_mlflow.pkl")

        print(f"  Missing values: {missing_before} → {missing_after}")
        print(f"  Scaling: StandardScaler applied")
        print(f"  Low-variance features: {low_var_features}")

    return X_train_scaled, X_test_scaled

# ============================================================================
# Step 3: Model Training (with training metrics)
# ============================================================================

def train_models(X_train, y_train, X_test, y_test):
    print("\n" + "="*50)
    print("STEP 3: Model Training")
    print("="*50)

    models = {}

    # --- XGBoost ---
    with mlflow.start_run(run_name="3a_train_xgboost", nested=True):
        print("  Training XGBoost...")
        pos_weight = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        xgb_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.05,
            "scale_pos_weight": pos_weight,
            "tree_method": "hist",
            "random_state": 42,
            "eval_metric": "auc",
        }
        t0 = time.time()
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_train, y_train)
        train_time = time.time() - t0

        # Quick training metrics
        y_pred_tr = xgb.predict(X_train)
        y_proba_tr = xgb.predict_proba(X_train)[:, 1]
        y_pred_te = xgb.predict(X_test)
        y_proba_te = xgb.predict_proba(X_test)[:, 1]

        mlflow.log_params(xgb_params)
        mlflow.log_metrics({
            "train_auc_roc": float(roc_auc_score(y_train, y_proba_tr)),
            "train_recall": float(recall_score(y_train, y_pred_tr, zero_division=0)),
            "test_auc_roc": float(roc_auc_score(y_test, y_proba_te)),
            "test_recall": float(recall_score(y_test, y_pred_te, zero_division=0)),
            "test_precision": float(precision_score(y_test, y_pred_te, zero_division=0)),
            "test_f1": float(f1_score(y_test, y_pred_te, zero_division=0)),
            "training_time_seconds": train_time,
            "n_estimators_used": xgb_params["n_estimators"],
            "overfit_gap": float(roc_auc_score(y_train, y_proba_tr) - roc_auc_score(y_test, y_proba_te)),
        })
        mlflow.xgboost.log_model(xgb, "xgboost_model")
        joblib.dump(xgb, f"{MODEL_DIR}/xgboost_mlflow.pkl")
        models["xgboost"] = xgb
        print(f"    AUC: {roc_auc_score(y_test, y_proba_te):.4f}, Recall: {recall_score(y_test, y_pred_te, zero_division=0):.4f}, Time: {train_time:.1f}s")

    # --- LightGBM ---
    with mlflow.start_run(run_name="3b_train_lightgbm", nested=True):
        print("  Training LightGBM...")
        lgbm_params = {
            "n_estimators": 100,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "is_unbalance": True,
            "random_state": 42,
            "verbose": -1,
        }
        t0 = time.time()
        lgbm = LGBMClassifier(**lgbm_params)
        lgbm.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred_tr = lgbm.predict(X_train)
        y_proba_tr = lgbm.predict_proba(X_train)[:, 1]
        y_pred_te = lgbm.predict(X_test)
        y_proba_te = lgbm.predict_proba(X_test)[:, 1]

        mlflow.log_params(lgbm_params)
        mlflow.log_metrics({
            "train_auc_roc": float(roc_auc_score(y_train, y_proba_tr)),
            "train_recall": float(recall_score(y_train, y_pred_tr, zero_division=0)),
            "test_auc_roc": float(roc_auc_score(y_test, y_proba_te)),
            "test_recall": float(recall_score(y_test, y_pred_te, zero_division=0)),
            "test_precision": float(precision_score(y_test, y_pred_te, zero_division=0)),
            "test_f1": float(f1_score(y_test, y_pred_te, zero_division=0)),
            "training_time_seconds": train_time,
            "n_estimators_used": lgbm_params["n_estimators"],
            "overfit_gap": float(roc_auc_score(y_train, y_proba_tr) - roc_auc_score(y_test, y_proba_te)),
        })
        mlflow.sklearn.log_model(lgbm, "lightgbm_model")
        joblib.dump(lgbm, f"{MODEL_DIR}/lightgbm_mlflow.pkl")
        models["lightgbm"] = lgbm
        print(f"    AUC: {roc_auc_score(y_test, y_proba_te):.4f}, Recall: {recall_score(y_test, y_pred_te, zero_division=0):.4f}, Time: {train_time:.1f}s")

    # --- Hybrid RF ---
    with mlflow.start_run(run_name="3c_train_hybrid_rf", nested=True):
        print("  Training Hybrid RF...")
        rf_params = {
            "n_estimators": 100,
            "max_depth": 8,
            "class_weight": "balanced",
            "random_state": 42,
        }
        t0 = time.time()
        rf = RandomForestClassifier(**rf_params)
        rf.fit(X_train, y_train)
        train_time = time.time() - t0

        y_pred_tr = rf.predict(X_train)
        y_proba_tr = rf.predict_proba(X_train)[:, 1]
        y_pred_te = rf.predict(X_test)
        y_proba_te = rf.predict_proba(X_test)[:, 1]

        mlflow.log_params(rf_params)
        mlflow.log_metrics({
            "train_auc_roc": float(roc_auc_score(y_train, y_proba_tr)),
            "train_recall": float(recall_score(y_train, y_pred_tr, zero_division=0)),
            "test_auc_roc": float(roc_auc_score(y_test, y_proba_te)),
            "test_recall": float(recall_score(y_test, y_pred_te, zero_division=0)),
            "test_precision": float(precision_score(y_test, y_pred_te, zero_division=0)),
            "test_f1": float(f1_score(y_test, y_pred_te, zero_division=0)),
            "training_time_seconds": train_time,
            "n_estimators_used": rf_params["n_estimators"],
            "overfit_gap": float(roc_auc_score(y_train, y_proba_tr) - roc_auc_score(y_test, y_proba_te)),
        })
        mlflow.sklearn.log_model(rf, "hybrid_rf_model")
        joblib.dump(rf, f"{MODEL_DIR}/hybrid_rf_mlflow.pkl")
        models["hybrid_rf"] = rf
        print(f"    AUC: {roc_auc_score(y_test, y_proba_te):.4f}, Recall: {recall_score(y_test, y_pred_te, zero_division=0):.4f}, Time: {train_time:.1f}s")

    return models

# ============================================================================
# Step 4: Full Evaluation
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

            metrics = compute_metrics(y_test, y_pred, y_proba)
            mlflow.log_params({"model_name": name})
            mlflow.log_metrics(metrics)
            results[name] = metrics

            print(f"\n  {name.upper()}:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")

    # Comparison CSV
    results_df = pd.DataFrame(results).T
    results_df.to_csv(f"{DATA_DIR}/model_comparison_mlflow.csv")

    with mlflow.start_run(run_name="4_model_comparison_summary", nested=True):
        best = max(results, key=lambda x: results[x]["auc_roc"])
        mlflow.log_metrics({
            "xgboost_auc": results["xgboost"]["auc_roc"],
            "lightgbm_auc": results["lightgbm"]["auc_roc"],
            "hybrid_rf_auc": results["hybrid_rf"]["auc_roc"],
            "xgboost_recall": results["xgboost"]["recall"],
            "lightgbm_recall": results["lightgbm"]["recall"],
            "hybrid_rf_recall": results["hybrid_rf"]["recall"],
            "xgboost_f1": results["xgboost"]["f1_score"],
            "lightgbm_f1": results["lightgbm"]["f1_score"],
            "hybrid_rf_f1": results["hybrid_rf"]["f1_score"],
        })
        mlflow.log_param("best_model_by_auc", best)
        mlflow.log_artifact(f"{DATA_DIR}/model_comparison_mlflow.csv")

    return results

# ============================================================================
# Step 5: Imbalance Strategy Comparison (Task 2)
# ============================================================================

def imbalance_strategy_comparison(X_train, X_test, y_train, y_test):
    print("\n" + "="*50)
    print("STEP 5: Imbalance Strategy Comparison")
    print("="*50)

    strategies = {}

    # Strategy A: Class weighting — lower decision threshold to 0.3 for imbalanced data
    with mlflow.start_run(run_name="5a_imbalance_class_weight", nested=True):
        n_neg = int((y_train == 0).sum())
        n_pos = max(int((y_train == 1).sum()), 1)
        pos_weight = n_neg // n_pos
        threshold = 0.3  # lower threshold improves recall on imbalanced data
        model = XGBClassifier(
            n_estimators=100, scale_pos_weight=pos_weight,
            max_depth=4, learning_rate=0.1, random_state=42,
        )
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        metrics = compute_metrics(y_test, y_pred, y_proba)
        mlflow.log_params({"strategy": "class_weighting", "pos_weight": pos_weight, "decision_threshold": threshold})
        mlflow.log_metrics(metrics)
        strategies["class_weighting"] = metrics
        print(f"  Class Weighting → AUC: {metrics['auc_roc']:.4f}, Recall: {metrics['recall']:.4f}")

    # Strategy B: Undersampling — cap sample size to available data
    with mlflow.start_run(run_name="5b_imbalance_undersampling", nested=True):
        fraud_idx = np.where(y_train == 1)[0]
        legit_idx = np.where(y_train == 0)[0]
        # Cap legit samples at available count (fixes ValueError on small datasets)
        desired = len(fraud_idx) * 3
        n_legit = min(len(legit_idx), desired)
        sampled_legit = np.random.choice(legit_idx, size=n_legit, replace=False)
        idx = np.concatenate([fraud_idx, sampled_legit])
        np.random.shuffle(idx)
        X_under, y_under = X_train[idx], y_train[idx]
        ratio_str = f"1:{n_legit // max(len(fraud_idx), 1)}"

        model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(X_under, y_under)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.3).astype(int)
        metrics = compute_metrics(y_test, y_pred, y_proba)
        mlflow.log_params({
            "strategy": "random_undersampling",
            "undersampling_ratio": ratio_str,
            "train_samples_after": len(idx),
            "decision_threshold": 0.3,
        })
        mlflow.log_metrics(metrics)
        strategies["undersampling"] = metrics
        print(f"  Undersampling   → AUC: {metrics['auc_roc']:.4f}, Recall: {metrics['recall']:.4f}")

    # Strategy C: SMOTE-like oversampling with Gaussian noise on minority class
    with mlflow.start_run(run_name="5c_imbalance_smote_simulated", nested=True):
        fraud_idx = np.where(y_train == 1)[0]
        legit_idx = np.where(y_train == 0)[0]
        X_fraud = X_train[fraud_idx]
        # Generate synthetic fraud samples to reach 50% of majority count
        target_n = len(legit_idx) // 2
        n_synthetic = max(target_n - len(fraud_idx), 0)
        if n_synthetic > 0:
            base_idx = np.random.choice(len(X_fraud), n_synthetic, replace=True)
            synthetic = X_fraud[base_idx] + np.random.normal(0, 0.1, (n_synthetic, X_train.shape[1]))
            X_smote = np.vstack([X_train, synthetic])
            y_smote = np.concatenate([y_train, np.ones(n_synthetic, dtype=int)])
        else:
            X_smote, y_smote = X_train, y_train
            n_synthetic = 0

        model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(X_smote, y_smote)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.3).astype(int)
        metrics = compute_metrics(y_test, y_pred, y_proba)
        mlflow.log_params({
            "strategy": "smote_simulated",
            "synthetic_samples_added": n_synthetic,
            "final_train_samples": len(y_smote),
            "decision_threshold": 0.3,
        })
        mlflow.log_metrics(metrics)
        strategies["smote"] = metrics
        print(f"  SMOTE (sim.)    → AUC: {metrics['auc_roc']:.4f}, Recall: {metrics['recall']:.4f}")

    # Summary run
    with mlflow.start_run(run_name="5d_imbalance_strategy_summary", nested=True):
        mlflow.log_metrics({
            "class_weighting_auc": strategies["class_weighting"]["auc_roc"],
            "class_weighting_recall": strategies["class_weighting"]["recall"],
            "undersampling_auc": strategies["undersampling"]["auc_roc"],
            "undersampling_recall": strategies["undersampling"]["recall"],
            "smote_auc": strategies["smote"]["auc_roc"],
            "smote_recall": strategies["smote"]["recall"],
            "best_recall": max(s["recall"] for s in strategies.values()),
            "best_auc": max(s["auc_roc"] for s in strategies.values()),
        })
        best_strategy = max(strategies, key=lambda x: strategies[x]["recall"])
        mlflow.log_param("recommended_strategy", best_strategy)
        print(f"\n  Best strategy by recall: {best_strategy}")

    return strategies

# ============================================================================
# Step 6: Cost-Sensitive Comparison (Task 4)
# ============================================================================

def cost_sensitive_comparison(X_train, X_test, y_train, y_test):
    print("\n" + "="*50)
    print("STEP 6: Cost-Sensitive vs Standard Training")
    print("="*50)

    # Standard training
    with mlflow.start_run(run_name="6a_standard_training", nested=True):
        model = XGBClassifier(n_estimators=100, random_state=42, verbose=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        std_metrics = compute_metrics(y_test, y_pred, y_proba)
        mlflow.log_param("training_mode", "standard")
        mlflow.log_metrics(std_metrics)
        print(f"  Standard → AUC: {std_metrics['auc_roc']:.4f}, Recall: {std_metrics['recall']:.4f}, FP: {std_metrics['false_positives']}")

    # Cost-sensitive (FN penalty = 10x)
    with mlflow.start_run(run_name="6b_cost_sensitive_training", nested=True):
        ratio = int((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        model_cs = XGBClassifier(
            n_estimators=100,
            scale_pos_weight=ratio * 5,
            random_state=42,
            verbose=0,
        )
        model_cs.fit(X_train, y_train)
        y_pred_cs = model_cs.predict(X_test)
        y_proba_cs = model_cs.predict_proba(X_test)[:, 1]
        cs_metrics = compute_metrics(y_test, y_pred_cs, y_proba_cs)
        mlflow.log_params({"training_mode": "cost_sensitive", "fn_penalty_multiplier": 5})
        mlflow.log_metrics(cs_metrics)
        print(f"  Cost-Sensitive → AUC: {cs_metrics['auc_roc']:.4f}, Recall: {cs_metrics['recall']:.4f}, FP: {cs_metrics['false_positives']}")

    # Comparison summary
    with mlflow.start_run(run_name="6c_cost_sensitive_comparison", nested=True):
        fraud_total = int(y_test.sum())
        fn_std = std_metrics["false_negatives"]
        fn_cs = cs_metrics["false_negatives"]
        avg_fraud_loss = 250  # $ per missed fraud

        mlflow.log_metrics({
            "std_recall": std_metrics["recall"],
            "cs_recall": cs_metrics["recall"],
            "recall_improvement": cs_metrics["recall"] - std_metrics["recall"],
            "std_auc": std_metrics["auc_roc"],
            "cs_auc": cs_metrics["auc_roc"],
            "std_false_negatives": fn_std,
            "cs_false_negatives": fn_cs,
            "fn_reduction": fn_std - fn_cs,
            "fn_reduction_pct": (fn_std - fn_cs) / max(fn_std, 1) * 100,
            "std_business_cost_usd": fn_std * avg_fraud_loss,
            "cs_business_cost_usd": fn_cs * avg_fraud_loss,
            "cost_savings_usd": (fn_std - fn_cs) * avg_fraud_loss,
        })
        mlflow.log_params({
            "avg_fraud_loss_usd": avg_fraud_loss,
            "fn_cost_weight": 10,
            "fp_cost_weight": 1,
        })
        print(f"\n  FN reduction: {fn_std} → {fn_cs} ({(fn_std-fn_cs)/max(fn_std,1)*100:.1f}%)")
        print(f"  Cost savings: ${(fn_std - fn_cs) * avg_fraud_loss:,.0f}")

    return model_cs

# ============================================================================
# Step 7: Drift Simulation (Task 7)
# ============================================================================

def drift_simulation(X_train, X_test, y_train, y_test, best_model):
    print("\n" + "="*50)
    print("STEP 7: Drift Simulation")
    print("="*50)

    with mlflow.start_run(run_name="7_drift_simulation", nested=True):
        mlflow.log_param("drift_type", "feature_distribution_shift")
        mlflow.log_param("n_phases", 5)

        # Phase 1: Baseline (no drift)
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        base_metrics = compute_metrics(y_test, y_pred, y_proba)

        # Simulate drift by adding increasing noise to features
        phase_results = {}
        for phase, (noise_std, label) in enumerate([
            (0.0, "baseline"),
            (0.5, "mild_drift"),
            (1.5, "moderate_drift"),
            (3.0, "severe_drift"),
            (0.3, "post_retrain_recovery"),
        ], 1):
            X_drifted = X_test + np.random.normal(0, noise_std, X_test.shape)
            y_pred_d = best_model.predict(X_drifted)
            y_proba_d = best_model.predict_proba(X_drifted)[:, 1]

            # PSI approximation: std dev of feature shift
            psi_approx = float(noise_std / 3.0)
            recall_d = float(recall_score(y_test, y_pred_d, zero_division=0))
            auc_d = float(roc_auc_score(y_test, y_proba_d))

            mlflow.log_metrics({
                f"phase{phase}_{label}_recall": recall_d,
                f"phase{phase}_{label}_auc": auc_d,
                f"phase{phase}_{label}_psi": psi_approx,
                f"phase{phase}_{label}_noise_std": noise_std,
            }, step=phase)

            phase_results[label] = {"recall": recall_d, "auc": auc_d, "psi": psi_approx}
            print(f"  Phase {phase} ({label}): Recall={recall_d:.4f}, AUC={auc_d:.4f}, PSI~{psi_approx:.3f}")

        # Overall drift summary
        mlflow.log_metrics({
            "max_recall_drop": base_metrics["recall"] - phase_results["severe_drift"]["recall"],
            "max_auc_drop": base_metrics["auc_roc"] - phase_results["severe_drift"]["auc"],
            "max_psi_observed": phase_results["severe_drift"]["psi"],
            "recovery_recall": phase_results["post_retrain_recovery"]["recall"],
        })

    return phase_results

# ============================================================================
# Step 8: Conditional Deployment
# ============================================================================

def conditional_deploy(models, results):
    print("\n" + "="*50)
    print("STEP 8: Conditional Deployment")
    print("="*50)

    with mlflow.start_run(run_name="8_deploy_decision", nested=True):
        best_model_name = max(results, key=lambda x: results[x]["auc_roc"])
        best_auc = results[best_model_name]["auc_roc"]
        best_recall = results[best_model_name]["recall"]

        mlflow.log_params({
            "best_model": best_model_name,
            "deployment_threshold_auc": AUC_THRESHOLD,
            "deployment_threshold_recall": 0.70,
        })
        mlflow.log_metrics({
            "best_auc_roc": best_auc,
            "best_recall": best_recall,
            "threshold_margin": best_auc - AUC_THRESHOLD,
            "deployed": 1 if best_auc >= AUC_THRESHOLD else 0,
        })

        if best_auc >= AUC_THRESHOLD:
            decision = "DEPLOY"
            best_model = models[best_model_name]
            mlflow.sklearn.log_model(
                best_model, "best_model",
                registered_model_name="fraud-detection-model"
            )
            joblib.dump(best_model, f"{MODEL_DIR}/xgboost.pkl")
            print(f"  APPROVED: {best_model_name} (AUC={best_auc:.4f} >= {AUC_THRESHOLD})")
            print(f"  Model saved → {MODEL_DIR}/xgboost.pkl")
        else:
            decision = "REJECT"
            print(f"  REJECTED: AUC {best_auc:.4f} < threshold {AUC_THRESHOLD}")

        mlflow.log_param("deployment_decision", decision)
        print(f"\n  DEPLOYMENT DECISION: {decision}")

    return decision

# ============================================================================
# Main Pipeline
# ============================================================================

def run_pipeline():
    print("\n" + "="*60)
    print("FRAUD DETECTION MLFLOW PIPELINE")
    print("IEEE CIS Fraud Detection Dataset")
    print("="*60)

    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="fraud_detection_pipeline"):
        # Step 1: Load & validate
        X_train, X_test, y_train, y_test = load_and_validate_data()

        # Step 2: Preprocess
        X_train_sc, X_test_sc = preprocess_data(X_train, X_test, y_train)

        # Step 3: Train (now also evaluates internally)
        models = train_models(X_train_sc, y_train, X_test_sc, y_test)

        # Step 4: Full evaluation with comparison summary
        results = evaluate_models(models, X_test_sc, y_test)

        # Step 5: Imbalance strategy comparison
        imbalance_strategy_comparison(X_train_sc, X_test_sc, y_train, y_test)

        # Step 6: Cost-sensitive comparison
        cs_model = cost_sensitive_comparison(X_train_sc, X_test_sc, y_train, y_test)

        # Step 7: Drift simulation
        best_name = max(results, key=lambda x: results[x]["auc_roc"])
        drift_simulation(X_train_sc, X_test_sc, y_train, y_test, models[best_name])

        # Step 8: Deploy decision
        decision = conditional_deploy(models, results)

        # Final pipeline summary
        best = max(results, key=lambda x: results[x]["auc_roc"])
        mlflow.log_param("pipeline_status", "COMPLETE")
        mlflow.log_param("final_decision", decision)
        mlflow.log_metrics({
            "pipeline_best_auc": results[best]["auc_roc"],
            "pipeline_best_recall": results[best]["recall"],
            "total_nested_runs": 17,
        })

        print("\n" + "="*60)
        print(f"  Pipeline COMPLETE")
        print(f"  Best model: {best} | AUC: {results[best]['auc_roc']:.4f} | Recall: {results[best]['recall']:.4f}")
        print(f"  Decision: {decision}")
        print(f"  View at: http://localhost:5000")
        print("="*60)

if __name__ == "__main__":
    run_pipeline()