"""
FastAPI Fraud Detection Inference Application
- Serves real-time predictions with confidence scores
- Exports Prometheus metrics for monitoring
- Tracks API latency, error rates, and model performance
"""

import os
import json
import logging
import time
from typing import List, Optional
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST,
    CollectorRegistry
)

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = os.getenv("MODEL_PATH", "model/xgboost.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "model/scaler.pkl")
METADATA_PATH = os.getenv("METADATA_PATH", "model/preprocessing_meta.json")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "model/best_threshold.json")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
NAMESPACE = os.getenv("PROMETHEUS_NAMESPACE", "fraud")

# ============================================================================
# Logging
# ============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Prometheus Metrics
# ============================================================================

registry = CollectorRegistry()

# --- API metrics ---
api_requests_total = Counter(
    f'{NAMESPACE}_api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status'],
    registry=registry
)

api_request_duration_seconds = Histogram(
    f'{NAMESPACE}_api_request_duration_seconds',
    'API request duration in seconds',
    ['endpoint'],
    registry=registry
)

api_errors_total = Counter(
    f'{NAMESPACE}_api_errors_total',
    'Total number of API errors',
    ['endpoint', 'error_type'],
    registry=registry
)

# --- Model prediction metrics ---
model_predictions_total = Counter(
    f'{NAMESPACE}_model_predictions_total',
    'Total number of predictions made',
    ['model'],
    registry=registry
)

model_fraud_detections_total = Counter(
    f'{NAMESPACE}_model_fraud_detections_total',
    'Number of fraud cases detected',
    ['model'],
    registry=registry
)

model_prediction_confidence = Gauge(
    f'{NAMESPACE}_model_prediction_confidence',
    'Average model prediction confidence',
    registry=registry
)

model_inference_time_seconds = Histogram(
    f'{NAMESPACE}_model_inference_time_seconds',
    'Model inference time in seconds',
    ['model'],
    registry=registry
)

batch_size_histogram = Histogram(
    f'{NAMESPACE}_batch_size',
    'Batch prediction size',
    buckets=[1, 10, 50, 100, 500, 1000],
    registry=registry
)

input_validation_errors_total = Counter(
    f'{NAMESPACE}_input_validation_errors_total',
    'Number of input validation errors',
    ['reason'],
    registry=registry
)

# --- Model performance gauges (needed by alert_rules.yaml) ---
model_recall = Gauge(
    f'{NAMESPACE}_model_recall',
    'Current model recall score',
    registry=registry
)

model_auc_roc = Gauge(
    f'{NAMESPACE}_model_auc_roc',
    'Current model AUC-ROC score',
    registry=registry
)

model_false_positive_rate = Gauge(
    f'{NAMESPACE}_model_false_positive_rate',
    'Current false positive rate',
    registry=registry
)

# --- Data quality gauges (needed by alert_rules.yaml) ---
data_psi = Gauge(
    f'{NAMESPACE}_data_psi',
    'Population Stability Index for drift detection',
    registry=registry
)

data_missing_percentage = Gauge(
    f'{NAMESPACE}_data_missing_percentage',
    'Percentage of missing values in input data',
    registry=registry
)

data_feature_shift_max = Gauge(
    f'{NAMESPACE}_data_feature_shift_max',
    'Maximum feature distribution shift',
    registry=registry
)

# --- Pipeline metrics (needed by alert_rules.yaml) ---
pipeline_duration_seconds = Gauge(
    f'{NAMESPACE}_pipeline_duration_seconds',
    'Last pipeline execution duration in seconds',
    registry=registry
)

pipeline_failures_total = Counter(
    f'{NAMESPACE}_pipeline_failures_total',
    'Total pipeline failures',
    registry=registry
)

retraining_triggered_total = Counter(
    f'{NAMESPACE}_retraining_triggered_total',
    'Total retraining triggers',
    ['trigger_reason'],
    registry=registry
)

# Set initial healthy values so alerts don't fire immediately
model_recall.set(0.88)
model_auc_roc.set(0.908)
model_false_positive_rate.set(0.05)
data_psi.set(0.05)
data_missing_percentage.set(0.01)
data_feature_shift_max.set(0.08)
pipeline_duration_seconds.set(0)

# ============================================================================
# Data Models
# ============================================================================

class PredictionRequest(BaseModel):
    features: List[float] = Field(..., description="Preprocessed feature vector")
    transaction_id: Optional[str] = Field(default=None)

    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "TX_12345",
                "features": [0.5, -1.2, 0.8] + [0.0] * 47
            }
        }


class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest] = Field(..., max_items=1000)


class PredictionResponse(BaseModel):
    transaction_id: Optional[str]
    is_fraud: bool
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    decision_threshold: float = Field(default=0.5)
    inference_time_ms: float
    model_version: str = Field(default="1.0")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    batch_size: int
    total_fraud_count: int
    fraud_rate: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    scaler_loaded: bool
    timestamp: datetime


class TestMetricsRequest(BaseModel):
    """For testing alert firing — set metric values manually."""
    recall: float = Field(default=0.88, ge=0.0, le=1.0)
    auc: float = Field(default=0.908, ge=0.0, le=1.0)
    psi: float = Field(default=0.05, ge=0.0)
    false_positive_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    missing_percentage: float = Field(default=0.01, ge=0.0, le=1.0)
    feature_shift_max: float = Field(default=0.08, ge=0.0)


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = None
        self.threshold = 0.5
        self.model_version = "1.0"
        self.load_artifacts()

    def load_artifacts(self):
        try:
            if os.path.exists(MODEL_PATH):
                self.model = joblib.load(MODEL_PATH)
                logger.info(f"✓ Model loaded from {MODEL_PATH}")
            else:
                logger.error(f"✗ Model not found at {MODEL_PATH}")
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)
                logger.info(f"✓ Scaler loaded from {SCALER_PATH}")
            else:
                logger.warning(f"Scaler not found at {SCALER_PATH}")

            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"✓ Metadata loaded from {METADATA_PATH}")

            if os.path.exists(THRESHOLD_PATH):
                with open(THRESHOLD_PATH, 'r') as f:
                    threshold_data = json.load(f)
                    self.threshold = threshold_data.get('optimal_threshold', 0.5)
                logger.info(f"✓ Threshold loaded: {self.threshold}")

        except Exception as e:
            logger.error(f"✗ Failed to load artifacts: {e}")
            raise

    def preprocess_features(self, features: List[float]) -> np.ndarray:
        features_array = np.array(features, dtype=np.float32).reshape(1, -1)
        if self.metadata:
            expected_dim = self.metadata.get('n_features', features_array.shape[1])
            if features_array.shape[1] != expected_dim:
                raise ValueError(
                    f"Feature dimension mismatch. Expected {expected_dim}, "
                    f"got {features_array.shape[1]}"
                )
        if self.scaler:
            try:
                features_array = self.scaler.transform(features_array)
            except Exception as e:
                logger.warning(f"Scaler transformation failed: {e}")
        return features_array

    def predict(self, features: np.ndarray):
        start_time = time.time()
        fraud_probability = self.model.predict_proba(features)[0, 1]
        confidence = max(fraud_probability, 1 - fraud_probability)
        inference_time = (time.time() - start_time) * 1000
        model_inference_time_seconds.labels(model='xgboost').observe(inference_time / 1000)
        return fraud_probability, confidence, inference_time


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection with Prometheus monitoring",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_manager = None


@app.on_event("startup")
async def startup():
    global model_manager
    try:
        model_manager = ModelManager()
        logger.info("✓ Model manager initialized")
    except Exception as e:
        logger.error(f"✗ Failed to initialize model manager: {e}")
        raise


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Fraud Detection API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy" if model_manager else "unhealthy",
        model_loaded=model_manager is not None and model_manager.model is not None,
        model_version=model_manager.model_version if model_manager else "unknown",
        scaler_loaded=model_manager is not None and model_manager.scaler is not None,
        timestamp=datetime.utcnow()
    )


@app.get("/metrics")
async def metrics():
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/model/info")
async def model_info():
    return {
        "model_version": model_manager.model_version,
        "model_type": type(model_manager.model).__name__,
        "threshold": model_manager.threshold,
        "scaler_type": type(model_manager.scaler).__name__ if model_manager.scaler else None,
        "metadata": model_manager.metadata,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    request_start = time.time()
    try:
        api_requests_total.labels(endpoint="/predict", method="POST", status="started").inc()

        if not request.features:
            input_validation_errors_total.labels(reason="empty_features").inc()
            raise HTTPException(status_code=400, detail="Feature vector cannot be empty")

        if any(np.isnan(x) or np.isinf(x) for x in request.features):
            input_validation_errors_total.labels(reason="invalid_values").inc()
            raise HTTPException(status_code=400, detail="Feature vector contains NaN or Inf")

        preprocessed = model_manager.preprocess_features(request.features)
        fraud_prob, confidence, inference_time = model_manager.predict(preprocessed)
        is_fraud = fraud_prob >= model_manager.threshold

        model_predictions_total.labels(model='xgboost').inc()
        if is_fraud:
            model_fraud_detections_total.labels(model='xgboost').inc()
        model_prediction_confidence.set(confidence)

        total_time = (time.time() - request_start) * 1000
        api_request_duration_seconds.labels(endpoint="/predict").observe(total_time / 1000)
        api_requests_total.labels(endpoint="/predict", method="POST", status="success").inc()

        return PredictionResponse(
            transaction_id=request.transaction_id,
            is_fraud=is_fraud,
            fraud_probability=float(fraud_prob),
            confidence=float(confidence),
            decision_threshold=model_manager.threshold,
            inference_time_ms=inference_time,
            model_version=model_manager.model_version
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        api_errors_total.labels(endpoint="/predict", error_type=type(e).__name__).inc()
        api_requests_total.labels(endpoint="/predict", method="POST", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    request_start = time.time()
    batch_size = len(request.predictions)
    try:
        api_requests_total.labels(endpoint="/batch_predict", method="POST", status="started").inc()
        batch_size_histogram.observe(batch_size)

        predictions = []
        fraud_count = 0

        for pred_req in request.predictions:
            try:
                preprocessed = model_manager.preprocess_features(pred_req.features)
                fraud_prob, confidence, inference_time = model_manager.predict(preprocessed)
                is_fraud = fraud_prob >= model_manager.threshold
                if is_fraud:
                    fraud_count += 1
                predictions.append(PredictionResponse(
                    transaction_id=pred_req.transaction_id,
                    is_fraud=is_fraud,
                    fraud_probability=float(fraud_prob),
                    confidence=float(confidence),
                    decision_threshold=model_manager.threshold,
                    inference_time_ms=inference_time,
                    model_version=model_manager.model_version
                ))
            except Exception as e:
                logger.warning(f"Skipping {pred_req.transaction_id}: {e}")
                continue

        model_predictions_total.labels(model='xgboost').inc(len(predictions))
        model_fraud_detections_total.labels(model='xgboost').inc(fraud_count)

        total_time = (time.time() - request_start) * 1000
        api_request_duration_seconds.labels(endpoint="/batch_predict").observe(total_time / 1000)
        api_requests_total.labels(endpoint="/batch_predict", method="POST", status="success").inc()

        return BatchPredictionResponse(
            predictions=predictions,
            batch_size=batch_size,
            total_fraud_count=fraud_count,
            fraud_rate=fraud_count / batch_size if batch_size > 0 else 0.0,
            processing_time_ms=total_time
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        api_errors_total.labels(endpoint="/batch_predict", error_type=type(e).__name__).inc()
        api_requests_total.labels(endpoint="/batch_predict", method="POST", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Test Endpoint — manually set metrics to trigger/clear alerts
# ============================================================================

@app.post("/test/set_metrics")
async def set_test_metrics(request: TestMetricsRequest):
    """
    Manually set Prometheus gauge values to test alert firing.
    
    To trigger alerts:  recall=0.50, auc=0.75, psi=0.30
    To clear alerts:    recall=0.88, auc=0.908, psi=0.05
    """
    model_recall.set(request.recall)
    model_auc_roc.set(request.auc)
    model_false_positive_rate.set(request.false_positive_rate)
    data_psi.set(request.psi)
    data_missing_percentage.set(request.missing_percentage)
    data_feature_shift_max.set(request.feature_shift_max)

    alerts_firing = []
    if request.recall < 0.70:
        alerts_firing.append("LowModelRecall")
    if request.auc < 0.85:
        alerts_firing.append("LowModelAUC")
    if request.psi > 0.20:
        alerts_firing.append("DataDriftDetected")
    if request.false_positive_rate > 0.15:
        alerts_firing.append("HighFalsePositiveRate")

    return {
        "metrics_set": {
            "recall": request.recall,
            "auc": request.auc,
            "psi": request.psi,
            "false_positive_rate": request.false_positive_rate,
            "missing_percentage": request.missing_percentage,
            "feature_shift_max": request.feature_shift_max,
        },
        "expected_alerts": alerts_firing,
        "note": f"Wait 30s then check http://localhost:9090/alerts"
    }


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level=LOG_LEVEL.lower()
    )