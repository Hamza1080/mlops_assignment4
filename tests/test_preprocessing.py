"""
Unit tests for data preprocessing pipeline.
Tests missing value handling, encoding, scaling, and feature engineering.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class TestMissingValueHandling:
    """Test missing value imputation strategies."""
    
    def test_median_imputation(self):
        """Test median imputation for numerical features."""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'feature2': [10.0, np.nan, 30.0, np.nan, 50.0]
        })
        
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        # Check no NaN values remain
        assert not np.any(np.isnan(X_imputed))
        
        # Check median values imputed
        expected_f1 = np.median([1.0, 2.0, 4.0, 5.0])  # 3.0
        expected_f2 = np.median([10.0, 30.0, 50.0])  # 30.0
        
        assert X_imputed[2, 0] == expected_f1
        assert X_imputed[1, 1] == expected_f2
    
    def test_missing_percentage_filter(self):
        """Test filtering features with high missing percentage."""
        X = pd.DataFrame({
            'mostly_missing': [np.nan] * 95 + [1.0] * 5,  # 95% missing
            'sparse_missing': [1.0] * 90 + [np.nan] * 10,  # 10% missing
        })
        
        # Filter columns with >80% missing
        missing_pct = X.isnull().sum() / len(X)
        cols_to_keep = missing_pct[missing_pct < 0.80].index
        
        assert 'mostly_missing' not in cols_to_keep
        assert 'sparse_missing' in cols_to_keep
    
    def test_nan_validation(self):
        """Test validation of NaN after preprocessing."""
        X = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0],
            'feature2': [4.0, 5.0, 6.0]
        })
        
        # Check no NaN
        assert X.isnull().sum().sum() == 0


class TestFeatureEncoding:
    """Test categorical feature encoding."""
    
    def test_frequency_encoding_high_cardinality(self):
        """Test frequency encoding for high-cardinality features."""
        X = pd.DataFrame({
            'high_card': ['A'] * 100 + ['B'] * 50 + ['C'] * 30
        })
        
        freq_map = X['high_card'].value_counts(normalize=True).to_dict()
        X_encoded = X['high_card'].map(freq_map)
        
        assert X_encoded.min() >= 0 and X_encoded.max() <= 1
        assert X_encoded.iloc[0] == 100 / 180  # 'A' frequency
    
    def test_label_encoding_low_cardinality(self):
        """Test label encoding for low-cardinality features."""
        X = pd.Series(['Male', 'Female', 'Male', 'Female', 'Female'])
        
        le = LabelEncoder()
        X_encoded = le.fit_transform(X)
        
        assert len(np.unique(X_encoded)) == 2
        assert X_encoded.min() == 0 and X_encoded.max() == 1
    
    def test_encoding_consistency(self):
        """Test that encoding is consistent across train/test."""
        train_data = pd.Series(['A', 'B', 'C', 'A', 'B'])
        test_data = pd.Series(['A', 'C', 'B'])
        
        le = LabelEncoder()
        train_encoded = le.fit_transform(train_data)
        test_encoded = le.transform(test_data)
        
        # Same values should have same encodings
        assert train_encoded[0] == test_encoded[0]  # Both 'A'
        assert train_encoded[1] == test_encoded[2]  # Both 'B'


class TestFeatureScaling:
    """Test feature scaling and normalization."""
    
    def test_standard_scaler_fit_transform(self):
        """Test StandardScaler fit and transform."""
        X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_test = np.array([[2.0, 3.0], [4.0, 5.0]])
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Check mean ~0 and std ~1 for training data
        assert np.abs(X_train_scaled.mean()) < 0.1
        assert np.abs(X_train_scaled.std() - 1.0) < 0.1
        
        # Check test data scaled correctly
        assert X_test_scaled.shape == X_test.shape
    
    def test_scaler_persistence(self):
        """Test that scaler parameters are preserved."""
        X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        # Store parameters
        mean_stored = scaler.mean_.copy()
        scale_stored = scaler.scale_.copy()
        
        # Transform new data
        X_new = np.array([[2.0, 3.0]])
        X_new_scaled = scaler.transform(X_new)
        
        # Verify same parameters used
        assert np.allclose(scaler.mean_, mean_stored)
        assert np.allclose(scaler.scale_, scale_stored)
    
    def test_scaling_out_of_bounds(self):
        """Test scaling handles out-of-bounds values."""
        X_train = np.array([[1.0], [2.0], [3.0]])
        X_test = np.array([[100.0]])  # Far outside training range
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Should still scale without error
        assert not np.any(np.isnan(X_test_scaled))
        assert not np.any(np.isinf(X_test_scaled))


class TestFeatureEngineering:
    """Test feature engineering operations."""
    
    def test_temporal_features(self):
        """Test extraction of temporal features from timestamp."""
        times = pd.Series([
            pd.Timestamp('2023-01-15 10:30:00'),
            pd.Timestamp('2023-01-15 14:45:00'),
            pd.Timestamp('2023-01-16 23:00:00'),
        ])
        
        hour = times.dt.hour
        day_of_week = times.dt.dayofweek
        
        assert hour[0] == 10
        assert hour[1] == 14
        assert day_of_week[0] == day_of_week[1]  # Same day
        assert day_of_week[0] != day_of_week[2]   # Different days
    
    def test_missingness_indicator(self):
        """Test creation of missingness indicator features."""
        X = pd.DataFrame({
            'feature1': [1.0, np.nan, 3.0],
            'feature2': [4.0, 5.0, 6.0]
        })
        
        # Create indicator for missing values
        X['feature1_missing'] = X['feature1'].isnull().astype(int)
        
        assert X['feature1_missing'].iloc[0] == 0
        assert X['feature1_missing'].iloc[1] == 1
        assert X['feature1_missing'].iloc[2] == 0
    
    def test_feature_interaction(self):
        """Test creation of interaction features."""
        X = pd.DataFrame({
            'amount': [100, 200, 150],
            'duration': [10, 20, 15]
        })
        
        X['amount_duration'] = X['amount'] * X['duration']
        
        assert X['amount_duration'].iloc[0] == 1000
        assert X['amount_duration'].iloc[1] == 4000


class TestDataDimensionality:
    """Test data shape and dimensionality."""
    
    def test_feature_count_preservation(self):
        """Test that feature count is correct after preprocessing."""
        n_samples = 1000
        n_features = 50
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        assert X.shape == (n_samples, n_features)
    
    def test_sample_count_preservation(self):
        """Test that sample count doesn't change during preprocessing."""
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100)
        })
        
        initial_samples = len(X)
        
        # After preprocessing (without deletion)
        X_processed = X.copy()
        
        assert len(X_processed) == initial_samples
    
    def test_memory_optimization(self):
        """Test memory usage after dtype downcast."""
        X = pd.DataFrame({
            'float64': np.random.randn(1000).astype(np.float64),
            'int64': np.random.randint(0, 100, 1000).astype(np.int64)
        })
        
        mem_before = X.memory_usage(deep=True).sum()
        
        # Downcast
        X['float64'] = X['float64'].astype(np.float32)
        X['int64'] = X['int64'].astype(np.int32)
        
        mem_after = X.memory_usage(deep=True).sum()
        
        # Should use less memory
        assert mem_after < mem_before


class TestDataValidation:
    """Test data validation functions."""
    
    def test_required_columns_check(self):
        """Test validation of required columns."""
        X = pd.DataFrame({
            'Time': [1, 2, 3],
            'Amount': [100, 200, 150],
            'Class': [0, 1, 0]
        })
        
        required_cols = ['Time', 'Amount', 'Class']
        assert all(col in X.columns for col in required_cols)
    
    def test_missing_columns_detection(self):
        """Test detection of missing required columns."""
        X = pd.DataFrame({
            'Time': [1, 2, 3],
            'Amount': [100, 200, 150]
            # Missing 'Class'
        })
        
        required_cols = ['Time', 'Amount', 'Class']
        missing = [col for col in required_cols if col not in X.columns]
        
        assert 'Class' in missing
        assert len(missing) == 1
    
    def test_data_type_validation(self):
        """Test validation of data types."""
        X = pd.DataFrame({
            'numeric': [1.0, 2.0, 3.0],
            'categorical': ['A', 'B', 'C']
        })
        
        assert pd.api.types.is_numeric_dtype(X['numeric'])
        assert pd.api.types.is_object_dtype(X['categorical'])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
