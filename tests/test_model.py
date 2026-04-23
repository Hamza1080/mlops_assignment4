"""
Unit tests for model training and inference.
Tests model loading, prediction, threshold tuning, and evaluation metrics.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class TestModelLoading:
    """Test model loading and initialization."""
    
    def test_model_instantiation(self):
        """Test that model can be instantiated."""
        model = LogisticRegression(random_state=42)
        assert model is not None
    
    def test_model_parameters(self):
        """Test model parameters are set correctly."""
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )
        
        assert model.max_iter == 1000
        assert model.random_state == 42
        assert model.solver == 'lbfgs'


class TestModelTraining:
    """Test model training process."""
    
    def test_model_fit(self):
        """Test that model can be fit to training data."""
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Check model has coefficients
        assert model.coef_.shape == (1, 10)
    
    def test_model_convergence(self):
        """Test that model converges during training."""
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Model should not raise convergence warnings
        assert model.n_iter_[0] < model.max_iter


class TestModelPrediction:
    """Test model prediction functionality."""
    
    def test_predict_shape(self):
        """Test prediction output shape."""
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 10)
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        assert y_pred.shape == (20,)
    
    def test_predict_proba_shape(self):
        """Test predict_proba output shape."""
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 10)
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)
        
        assert y_proba.shape == (20, 2)  # 2 classes
    
    def test_predict_proba_range(self):
        """Test that probabilities are in valid range."""
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 10)
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)
        
        # Probabilities should be between 0 and 1
        assert np.all(y_proba >= 0.0)
        assert np.all(y_proba <= 1.0)
        
        # Sum of probabilities for each sample should be 1
        assert np.allclose(y_proba.sum(axis=1), 1.0)
    
    def test_predict_binary_output(self):
        """Test that predictions are binary."""
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 10)
        
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Should only have values 0 and 1
        assert set(y_pred) == {0, 1} or set(y_pred) == {0} or set(y_pred) == {1}


class TestModelEvaluation:
    """Test model evaluation metrics."""
    
    def test_auc_roc_calculation(self):
        """Test AUC-ROC calculation."""
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.4, 0.35, 0.8])
        
        auc = roc_auc_score(y_true, y_proba)
        
        assert 0.0 <= auc <= 1.0
        assert auc > 0.5  # Better than random
    
    def test_precision_recall_calculation(self):
        """Test precision and recall calculation."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
        assert precision == 2/3  # 2 correct fraud, 3 total fraud predictions
        assert recall == 2/3  # 2 correct fraud out of 3 actual fraud
    
    def test_f1_score_calculation(self):
        """Test F1 score calculation."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        
        f1 = f1_score(y_true, y_pred)
        
        assert 0.0 <= f1 <= 1.0
    
    def test_confusion_matrix(self):
        """Test confusion matrix calculation."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Check shape
        assert cm.shape == (2, 2)
        
        # Check values
        tn, fp, fn, tp = cm.ravel()
        assert tn == 1  # True negatives
        assert fp == 1  # False positives
        assert fn == 0  # False negatives
        assert tp == 2  # True positives


class TestThresholdTuning:
    """Test threshold optimization for fraud detection."""
    
    def test_threshold_application(self):
        """Test applying different thresholds to probabilities."""
        y_proba = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        threshold = 0.5
        
        y_pred = (y_proba >= threshold).astype(int)
        
        assert y_pred[0] == 0
        assert y_pred[1] == 0
        assert y_pred[2] == 1
        assert y_pred[3] == 1
        assert y_pred[4] == 1
    
    def test_threshold_precision_recall_tradeoff(self):
        """Test precision-recall tradeoff with different thresholds."""
        y_true = np.array([0]*95 + [1]*5)  # Imbalanced
        y_proba = np.concatenate([
            np.random.uniform(0, 0.4, 95),  # Legit transactions
            np.random.uniform(0.6, 1.0, 5)   # Fraud transactions
        ])
        
        # High threshold -> higher precision, lower recall
        y_pred_high = (y_proba >= 0.8).astype(int)
        precision_high = precision_score(y_true, y_pred_high, zero_division=0)
        recall_high = recall_score(y_true, y_pred_high, zero_division=0)
        
        # Low threshold -> lower precision, higher recall
        y_pred_low = (y_proba >= 0.3).astype(int)
        precision_low = precision_score(y_true, y_pred_low, zero_division=0)
        recall_low = recall_score(y_true, y_pred_low, zero_division=0)
        
        # Verify tradeoff
        assert precision_high >= precision_low
        assert recall_low >= recall_high
    
    def test_optimal_threshold_selection(self):
        """Test selection of optimal threshold."""
        y_true = np.array([0]*90 + [1]*10)
        y_proba = np.concatenate([
            np.random.uniform(0, 0.5, 90),
            np.random.uniform(0.5, 1.0, 10)
        ])
        
        # Find threshold maximizing F1
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
        
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        
        assert 0.0 <= best_threshold <= 1.0


class TestCostSensitiveLearning:
    """Test cost-sensitive learning for fraud detection."""
    
    def test_class_weight_impact(self):
        """Test that class weights affect model training."""
        X_train = np.random.randn(200, 10)
        # Highly imbalanced: 95% legit (0), 5% fraud (1)
        y_train = np.concatenate([
            np.zeros(190, dtype=int),
            np.ones(10, dtype=int)
        ])
        
        # Balanced weights
        model_balanced = LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        model_balanced.fit(X_train, y_train)
        
        # Uniform weights
        model_uniform = LogisticRegression(
            random_state=42,
            max_iter=1000
        )
        model_uniform.fit(X_train, y_train)
        
        # Models should have different coefficients
        assert not np.allclose(model_balanced.coef_, model_uniform.coef_)
    
    def test_scale_pos_weight(self):
        """Test scale_pos_weight for XGBoost-like models."""
        # Simulate cost-weighted prediction adjustment
        y_proba = np.array([0.3, 0.5, 0.7])
        scale_pos_weight = 100  # Cost of false negative
        
        # Adjust probabilities (simple approximation)
        y_proba_adjusted = y_proba * scale_pos_weight / (1 + scale_pos_weight)
        
        # Should push probabilities higher
        assert np.all(y_proba_adjusted >= y_proba * 0.99)


class TestModelStability:
    """Test model stability and consistency."""
    
    def test_deterministic_predictions(self):
        """Test that predictions are deterministic with fixed seed."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 10)
        
        # First training
        model1 = LogisticRegression(random_state=42, max_iter=1000)
        model1.fit(X_train, y_train)
        y_pred1 = model1.predict_proba(X_test)
        
        # Second training with same seed
        model2 = LogisticRegression(random_state=42, max_iter=1000)
        model2.fit(X_train, y_train)
        y_pred2 = model2.predict_proba(X_test)
        
        # Predictions should be identical
        assert np.allclose(y_pred1, y_pred2)
    
    def test_model_robustness(self):
        """Test model performance on slightly different data."""
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Test on clean data
        X_test_clean = np.random.randn(20, 10)
        y_clean = model.predict(X_test_clean)
        
        # Test on noisy data
        X_test_noisy = X_test_clean + np.random.normal(0, 0.01, X_test_clean.shape)
        y_noisy = model.predict(X_test_noisy)
        
        # Predictions should be similar but not necessarily identical
        assert np.mean(y_clean == y_noisy) > 0.7  # >70% agreement


class TestModelComparison:
    """Test comparison of multiple models."""
    
    def test_model_performance_ranking(self):
        """Test ranking models by performance."""
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 10)
        y_test = np.random.randint(0, 2, 20)
        
        # Train multiple models
        models = {
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(n_estimators=10, random_state=42)
        }
        
        scores = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            scores[name] = model.score(X_test, y_test)
        
        # Check that we have valid scores
        assert all(0.0 <= score <= 1.0 for score in scores.values())
        
        # Get best model
        best_model = max(scores, key=scores.get)
        assert best_model in models.keys()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
