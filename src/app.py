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
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CollectorRegistry

# ============================================================================
# Configuration
# ============================================================================

MODEL_PATH = os.getenv("MODEL_PATH", "models/xgboost.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
METADATA_PATH = os.getenv("METADATA_PATH", "models/preprocessing_meta.json")
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", "models/best_threshold.json")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
NAMESPACE = os.getenv("PROMETHEUS_NAMESPACE", "fraud")

# ============================================================================
# Logging Setup
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

# API metrics
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

# Model metrics
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

# Batch prediction metrics
batch_size_histogram = Histogram(
    f'{NAMESPACE}_batch_size',
    'Batch prediction size',
    buckets=[1, 10, 50, 100, 500, 1000],
    registry=registry
)

# Data quality metrics
input_validation_errors_total = Counter(
    f'{NAMESPACE}_input_validation_errors_total',
    'Number of input validation errors',
    ['reason'],
    registry=registry
)

# ============================================================================
# Data Models
# ============================================================================

class PredictionRequest(BaseModel):
    """Single transaction for fraud detection."""
    features: List[float] = Field(
        ...,
        description="Preprocessed feature vector (530 dimensions)"
    )
    transaction_id: Optional[str] = Field(
        default=None,
        description="Unique transaction identifier for logging"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "transaction_id": "TX_12345",
                "features": [0.5, -1.2, 0.8] + [0.0] * 527  # 530 dims
            }
        }


class BatchPredictionRequest(BaseModel):
    """Batch of transactions for fraud detection."""
    predictions: List[PredictionRequest] = Field(
        ...,
        max_items=1000,
        description="Up to 1000 transactions per batch"
    )


class PredictionResponse(BaseModel):
    """Response for a single prediction."""
    transaction_id: Optional[str]
    is_fraud: bool
    fraud_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted fraud probability (0-1)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in prediction"
    )
    decision_threshold: float = Field(
        default=0.5,
        description="Threshold used for fraud classification"
    )
    inference_time_ms: float = Field(
        ...,
        description="Inference time in milliseconds"
    )
    model_version: str = Field(
        default="1.0",
        description="Model version that generated prediction"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of prediction"
    )


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: List[PredictionResponse]
    batch_size: int
    total_fraud_count: int
    fraud_rate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Percentage of transactions flagged as fraud"
    )
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_version: str
    scaler_loaded: bool
    timestamp: datetime


# ============================================================================
# Model Loading
# ============================================================================

class ModelManager:
    """Manages model and preprocessing artifacts."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = None
        self.threshold = 0.5
        self.model_version = "1.0"
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load model, scaler, and metadata."""
        try:
            # Load model
            if os.path.exists(MODEL_PATH):
                self.model = joblib.load(MODEL_PATH)
                logger.info(f"✓ Model loaded from {MODEL_PATH}")
            else:
                logger.error(f"✗ Model not found at {MODEL_PATH}")
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
            
            # Load scaler
            if os.path.exists(SCALER_PATH):
                self.scaler = joblib.load(SCALER_PATH)
                logger.info(f"✓ Scaler loaded from {SCALER_PATH}")
            else:
                logger.warning(f"Scaler not found at {SCALER_PATH}")
            
            # Load metadata
            if os.path.exists(METADATA_PATH):
                with open(METADATA_PATH, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"✓ Metadata loaded from {METADATA_PATH}")
            else:
                logger.warning(f"Metadata not found at {METADATA_PATH}")
            
            # Load optimal threshold
            if os.path.exists(THRESHOLD_PATH):
                with open(THRESHOLD_PATH, 'r') as f:
                    threshold_data = json.load(f)
                    self.threshold = threshold_data.get('optimal_threshold', 0.5)
                logger.info(f"✓ Optimal threshold loaded: {self.threshold}")
            else:
                logger.warning(f"Threshold not found. Using default: {self.threshold}")
        
        except Exception as e:
            logger.error(f"✗ Failed to load artifacts: {e}")
            raise
    
    def preprocess_features(self, features: List[float]) -> np.ndarray:
        """Validate and preprocess features."""
        features_array = np.array(features, dtype=np.float32).reshape(1, -1)
        
        # Validate dimensions
        if self.metadata:
            expected_dim = self.metadata.get('n_features', 530)
            if features_array.shape[1] != expected_dim:
                raise ValueError(
                    f"Feature dimension mismatch. Expected {expected_dim}, "
                    f"got {features_array.shape[1]}"
                )
        
        # Apply scaler if available
        if self.scaler:
            try:
                features_array = self.scaler.transform(features_array)
            except Exception as e:
                logger.warning(f"Scaler transformation failed: {e}")
        
        return features_array
    
    def predict(self, features: np.ndarray) -> tuple[float, float]:
        """
        Make prediction and return fraud probability and confidence.
        
        Returns:
            (fraud_probability, confidence)
        """
        start_time = time.time()
        
        try:
            # Get probability prediction
            fraud_probability = self.model.predict_proba(features)[0, 1]
            
            # Calculate confidence as max(prob, 1-prob)
            confidence = max(fraud_probability, 1 - fraud_probability)
            
            inference_time = (time.time() - start_time) * 1000
            model_inference_time_seconds.labels(model='xgboost').observe(inference_time / 1000)
            
            return fraud_probability, confidence, inference_time
        
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection with explainability",
    version="1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
model_manager = None

@app.on_event("startup")
async def startup():
    """Initialize model manager on startup."""
    global model_manager
    try:
        model_manager = ModelManager()
        logger.info("✓ Model manager initialized successfully")
    except Exception as e:
        logger.error(f"✗ Failed to initialize model manager: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with service status
    """
    return HealthResponse(
        status="healthy" if model_manager else "unhealthy",
        model_loaded=model_manager is not None and model_manager.model is not None,
        model_version=model_manager.model_version if model_manager else "unknown",
        scaler_loaded=model_manager is not None and model_manager.scaler is not None,
        timestamp=datetime.utcnow()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Single transaction fraud prediction.
    
    Args:
        request: PredictionRequest with transaction features
    
    Returns:
        PredictionResponse with fraud prediction and confidence
    
    Raises:
        HTTPException: For validation or inference errors
    """
    request_start = time.time()
    
    try:
        api_requests_total.labels(
            endpoint="/predict",
            method="POST",
            status="started"
        ).inc()
        
        # Validate input
        if not request.features or len(request.features) == 0:
            input_validation_errors_total.labels(reason="empty_features").inc()
            raise HTTPException(
                status_code=400,
                detail="Feature vector cannot be empty"
            )
        
        if any(np.isnan(x) or np.isinf(x) for x in request.features):
            input_validation_errors_total.labels(reason="invalid_values").inc()
            raise HTTPException(
                status_code=400,
                detail="Feature vector contains NaN or Inf values"
            )
        
        # Preprocess features
        preprocessed = model_manager.preprocess_features(request.features)
        
        # Generate prediction
        fraud_prob, confidence, inference_time = model_manager.predict(preprocessed)
        is_fraud = fraud_prob >= model_manager.threshold
        
        # Update metrics
        model_predictions_total.labels(model='xgboost').inc()
        if is_fraud:
            model_fraud_detections_total.labels(model='xgboost').inc()
        model_prediction_confidence.set(confidence)
        
        # Total API time
        total_time = (time.time() - request_start) * 1000
        api_request_duration_seconds.labels(endpoint="/predict").observe(total_time / 1000)
        api_requests_total.labels(
            endpoint="/predict",
            method="POST",
            status="success"
        ).inc()
        
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
        api_requests_total.labels(
            endpoint="/predict",
            method="POST",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Batch transaction fraud prediction.
    
    Args:
        request: BatchPredictionRequest with up to 1000 transactions
    
    Returns:
        BatchPredictionResponse with fraud predictions for all transactions
    """
    request_start = time.time()
    batch_size = len(request.predictions)
    
    try:
        api_requests_total.labels(
            endpoint="/batch_predict",
            method="POST",
            status="started"
        ).inc()
        
        batch_size_histogram.observe(batch_size)
        
        predictions = []
        fraud_count = 0
        
        for pred_req in request.predictions:
            try:
                # Preprocess
                preprocessed = model_manager.preprocess_features(pred_req.features)
                
                # Predict
                fraud_prob, confidence, inference_time = model_manager.predict(preprocessed)
                is_fraud = fraud_prob >= model_manager.threshold
                
                if is_fraud:
                    fraud_count += 1
                
                predictions.append(
                    PredictionResponse(
                        transaction_id=pred_req.transaction_id,
                        is_fraud=is_fraud,
                        fraud_probability=float(fraud_prob),
                        confidence=float(confidence),
                        decision_threshold=model_manager.threshold,
                        inference_time_ms=inference_time,
                        model_version=model_manager.model_version
                    )
                )
            
            except Exception as e:
                logger.warning(f"Skipping prediction for {pred_req.transaction_id}: {e}")
                continue
        
        # Update metrics
        model_predictions_total.labels(model='xgboost').inc(len(predictions))
        model_fraud_detections_total.labels(model='xgboost').inc(fraud_count)
        
        total_time = (time.time() - request_start) * 1000
        api_request_duration_seconds.labels(endpoint="/batch_predict").observe(total_time / 1000)
        api_requests_total.labels(
            endpoint="/batch_predict",
            method="POST",
            status="success"
        ).inc()
        
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
        api_requests_total.labels(
            endpoint="/batch_predict",
            method="POST",
            status="error"
        ).inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns:
        Prometheus-formatted metrics for all tracked counters/gauges/histograms
    """
    return generate_latest(registry).decode('utf-8')


@app.get("/model/info")
async def model_info():
    """Get model and configuration information."""
    return {
        "model_version": model_manager.model_version,
        "model_type": type(model_manager.model).__name__,
        "threshold": model_manager.threshold,
        "scaler_type": type(model_manager.scaler).__name__ if model_manager.scaler else None,
        "metadata": model_manager.metadata,
        "model_path": MODEL_PATH,
        "scaler_path": SCALER_PATH
    }


@app.get("/")
async def root():
    """API documentation redirect."""
    return {
        "message": "Fraud Detection API",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
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
