# MLOps Assignment 4: Fraud Detection System

## What We're Building

A fraud detection system that:
1. Takes transaction data as input
2. Trains ML models to detect fraud
3. Automatically runs in a pipeline (Kubeflow)
4. Monitors itself and alerts if something breaks
5. Automatically retrains when performance drops

## Folder Structure Explained

```
mlops_assignment_4/
├── data/              → Put CSV files here (you'll download from Kaggle)
├── src/               → Our Python code (data loading, models, etc)
├── notebooks/         → Jupyter notebooks for exploration and analysis
├── kubeflow/          → Pipeline definitions (how to orchestrate runs)
├── monitoring/        → Prometheus + Grafana configs (watch the system)
├── cicd/              → CI/CD configs (automate testing/deployment)
├── docker/            → Docker files (package code for production)
├── tests/             → Unit tests (verify code works)
└── README.md          → This file
```

## Why Each Folder?

- **data/** - Raw data stays separate from code
- **src/** - Reusable Python modules we'll import everywhere
- **notebooks/** - Interactive exploration before we write production code
- **kubeflow/** - Defines the ML pipeline (what runs in what order)
- **monitoring/** - Prometheus scrapes metrics, Grafana shows dashboards
- **cicd/** - GitHub Actions / Jenkins to automate everything
- **docker/** - Package code + dependencies for consistent execution
- **tests/** - Automated tests so we know our code works

## Next Step

Download the dataset and put it in `data/` folder. Then we'll write ONE simple Python script to load it and look at what we're working with.
