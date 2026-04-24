# IEEE CIS Fraud Detection — MLOps System

> **Assignment #04 | MLOps (BS Data Science) | Deadline: April 25, 2026**  
> Full end-to-end MLOps pipeline for real-time fraud detection with automated drift detection and retraining.

[![CI/CD](https://github.com/Hamza1080/mlops_assignment4/actions/workflows/ci.yml/badge.svg)](https://github.com/Hamza1080/mlops_assignment4/actions)
[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://python.org)
[![MLflow](https://img.shields.io/badge/mlflow-3.11-orange.svg)](https://mlflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100-green.svg)](https://fastapi.tiangolo.com)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Repository                         │
│                     code push / pull request                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ triggers
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GitHub Actions CI/CD                        │
│   Stage 1: Lint + Tests → Stage 2: Docker Build →               │
│   Stage 3: MLflow Pipeline → Stage 4: Intelligent Trigger       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ runs
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MLflow Pipeline                             │
│  Ingest → Validate → Preprocess → Train → Evaluate → Deploy     │
│  (XGBoost | LightGBM | Hybrid RF | Cost-Sensitive | SHAP)       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ saves model.pkl
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI Inference API                          │
│         /predict   /batch_predict   /metrics   /health           │
│              12 custom Prometheus metrics exposed                │
└──────────────────────────┬──────────────────────────────────────┘
                           │ scrape every 10s
                           ▼
┌───────────────────────────────────────┐
│            Prometheus                  │
│   16 alert rules (model + data + API) │
└──────────┬──────────────┬─────────────┘
           │              │
           ▼              ▼
┌──────────────┐  ┌──────────────────────┐
│   Grafana    │  │    Alertmanager       │
│ 3 dashboards │  │ webhook → GitHub API  │
└──────────────┘  └──────────┬───────────┘
                             │ repository_dispatch
                             ▼
                  ┌──────────────────────┐
                  │   GitHub Actions      │
                  │  Intelligent Trigger  │
                  │  → mlflow_pipeline.py │
                  └──────────────────────┘
                  (feedback loop: drift detected → retrain automatically)
```

---

## Results

| Model | Precision | Recall | F1 | AUC-ROC |
|-------|-----------|--------|----|---------|
| **XGBoost (Best)** | 0.874 | 0.882 | 0.878 | **0.908** |
| LightGBM | 0.861 | 0.871 | 0.866 | 0.905 |
| Hybrid RF | 0.843 | 0.856 | 0.849 | 0.891 |
| Logistic Regression | 0.781 | 0.792 | 0.786 | 0.880 |

- **Class imbalance:** SMOTE (590k → 33k/33k balanced)
- **Cost-sensitive:** scale_pos_weight=100 (FN=$100, FP=$1)
- **Memory optimization:** 50% reduction via float64→float32 downcast
- **Drift threshold:** PSI > 0.20 triggers retraining

---

## Project Structure

```
mlops_assignment4/
├── .github/
│   └── workflows/
│       └── ci.yml                  # 4-stage CI/CD pipeline
├── src/
│   └── app.py                      # FastAPI inference API + Prometheus metrics
├── monitoring/
│   ├── prometheus.yml              # Scrape config
│   ├── alert_rules.yaml            # 16 alert rules
│   ├── alertmanager.yml            # Webhook routing to GitHub
│   └── docker-compose.yml          # Full monitoring stack
├── docker/
│   ├── Dockerfile.train            # Training image
│   └── Dockerfile.api              # API image
├── tests/
│   ├── test_preprocessing.py       # 15 unit tests
│   └── test_model.py               # 18 unit tests
├── scripts/
│   ├── analyze_drift.py            # Drift analysis
│   └── determine_strategy.py       # Retraining strategy logic
├── data/
│   ├── X_train_sample.csv          # Training sample
│   └── X_test_sample.csv           # Test sample
├── model/
│   ├── xgboost.pkl                 # Best model (AUC 0.908)
│   ├── scaler.pkl                  # StandardScaler
│   └── best_threshold.json         # Optimal threshold (0.42)
├── notebooks/
│   └── mlops_assignement_4_latest_new.ipynb
├── mlflow_pipeline.py              # Main MLflow pipeline
└── requirements.txt
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/Hamza1080/mlops_assignment4.git
cd mlops_assignment4
pip install -r requirements.txt
```

### 2. Run MLflow pipeline

```bash
# Terminal 1 — start MLflow server
mlflow server --host 0.0.0.0 --port 5000

# Terminal 2 — run pipeline
python mlflow_pipeline.py
```

View results at `http://localhost:5000`

### 3. Start FastAPI

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

Test endpoints:
- `http://localhost:8000/health`
- `http://localhost:8000/docs`
- `http://localhost:8000/metrics`

### 4. Start monitoring stack

```bash
cd monitoring
docker-compose up -d
```

| Service | URL |
|---------|-----|
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (admin/changeme) |
| Alertmanager | http://localhost:9093 |

### 5. Run tests

```bash
pytest tests/ -v --cov=src
```

---

## Tasks Implemented

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | MLflow Pipeline (7 stages, conditional deployment) | ✅ |
| Task 2 | Data challenges (SMOTE, missing values, encoding) | ✅ |
| Task 3 | Model complexity (XGBoost, LightGBM, Hybrid RF) | ✅ |
| Task 4 | Cost-sensitive learning (scale_pos_weight=100) | ✅ |
| Task 5 | CI/CD with intelligent triggers (GitHub Actions) | ✅ |
| Task 6 | Monitoring (Prometheus + Grafana, 16 alerts) | ✅ |
| Task 7 | Drift simulation (PSI, temporal split) | ✅ |
| Task 8 | Retraining strategy (hybrid recommended) | ✅ |
| Task 9 | Explainability (SHAP TreeExplainer) | ✅ |

---

## CI/CD Pipeline

The GitHub Actions workflow has 4 stages:

```
Push to main
    │
    ├── Stage 1: CI
    │   ├── flake8 linting
    │   ├── pytest (33 tests)
    │   └── coverage report
    │
    ├── Stage 2: Build
    │   ├── Docker image (training)
    │   └── Docker image (API)
    │
    ├── Stage 3: Deploy
    │   └── Run mlflow_pipeline.py
    │
    └── Stage 4: Intelligent Trigger (on drift alert)
        ├── Analyze drift severity
        ├── Determine retraining strategy
        └── Execute mlflow_pipeline.py if needed
```

**Drift feedback loop:**
```
Prometheus detects recall < 0.70
    → Alertmanager fires webhook
    → GitHub repository_dispatch event
    → Stage 4 runs automatically
    → Model retrained with latest data
```

---

## Monitoring & Alerts

### Alert Rules (16 total)

| Alert | Condition | Severity |
|-------|-----------|----------|
| LowModelRecall | recall < 0.70 | Critical |
| LowModelAUC | AUC < 0.85 | Critical |
| DataDriftDetected | PSI > 0.20 | Warning |
| HighFalsePositiveRate | FPR > 0.15 | Warning |
| APIHighLatency | P95 > 1.0s | Warning |
| APIErrorRateHigh | error rate > 5% | Critical |
| PredictionConfidenceLow | confidence < 0.60 | Warning |
| HighMissingDataRate | missing > 5% | Warning |

### Test alerts manually

```powershell
# Trigger bad metrics (fires alerts after 5 min)
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/test/set_metrics" `
  -ContentType "application/json" `
  -Body '{"recall": 0.50, "auc": 0.75, "psi": 0.30}'

# Restore healthy metrics
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/test/set_metrics" `
  -ContentType "application/json" `
  -Body '{"recall": 0.88, "auc": 0.908, "psi": 0.05}'
```

---

## Retraining Strategies

| Strategy | Response Time | Compute Cost | Stability | Recovery |
|----------|--------------|--------------|-----------|----------|
| Threshold-based | ~5 min | High | Reactive | Best |
| Periodic (7-day) | Up to 7 days | Low | Predictable | Delayed |
| **Hybrid (recommended)** | ~5 min | Medium | Balanced | Near-best |

**Implemented:** Threshold-based via Prometheus → Alertmanager → GitHub Actions  
**Recommended for production:** Hybrid strategy

---

## Dataset

[IEEE CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)

- 590,000+ transactions
- 433 features (transaction + identity tables)
- 3.5% fraud rate (severe class imbalance)
- Handled with SMOTE balancing

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Pipeline orchestration | MLflow 3.11 |
| ML models | XGBoost, LightGBM, Scikit-learn |
| Inference API | FastAPI + Uvicorn |
| Monitoring | Prometheus + Grafana |
| Alerting | Alertmanager |
| CI/CD | GitHub Actions |
| Containerization | Docker + Docker Compose |
| Explainability | SHAP TreeExplainer |

---

## Author

**Hamza Zahid** — BS Data Science  
GitHub: [@Hamza1080](https://github.com/Hamza1080)
