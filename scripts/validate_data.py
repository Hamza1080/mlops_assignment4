#!/usr/bin/env python3
"""
Data validation script for fraud detection pipeline.
Validates schema, missing values, and data quality.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np


def validate_schema(df: pd.DataFrame, required_cols: List[str]) -> Dict[str, bool]:
    """
    Validate that all required columns exist.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
    
    Returns:
        Dict with validation results
    """
    results = {}
    missing = set(required_cols) - set(df.columns)
    
    if missing:
        print(f"✗ Missing required columns: {missing}")
        results['schema_valid'] = False
        results['missing_columns'] = list(missing)
    else:
        print(f"✓ All {len(required_cols)} required columns present")
        results['schema_valid'] = True
        results['missing_columns'] = []
    
    return results


def validate_missing_values(
    df: pd.DataFrame,
    max_missing_pct: float = 0.95
) -> Dict[str, any]:
    """
    Validate missing value patterns.
    
    Args:
        df: DataFrame to validate
        max_missing_pct: Maximum allowed missing percentage
    
    Returns:
        Dict with validation results
    """
    results = {}
    missing_pct = df.isnull().sum() / len(df)
    
    # Check overall missing rate
    overall_missing = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    print(f"Overall missing rate: {overall_missing:.2%}")
    
    # Check features with high missing rates
    high_missing = missing_pct[missing_pct > max_missing_pct]
    if len(high_missing) > 0:
        print(f"⚠ {len(high_missing)} features exceed {max_missing_pct:.0%} missing threshold:")
        for col, pct in high_missing.items():
            print(f"  - {col}: {pct:.2%}")
        results['high_missing_features'] = high_missing.to_dict()
    else:
        print(f"✓ No features exceed {max_missing_pct:.0%} missing threshold")
    
    results['missing_valid'] = len(high_missing) == 0
    results['overall_missing_pct'] = overall_missing
    
    return results


def validate_data_types(df: pd.DataFrame) -> Dict[str, any]:
    """
    Validate data types are appropriate.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dict with validation results
    """
    results = {
        'numeric_columns': [],
        'categorical_columns': [],
        'datetime_columns': []
    }
    
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            results['numeric_columns'].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            results['datetime_columns'].append(col)
        else:
            results['categorical_columns'].append(col)
    
    print(f"✓ Data types validated:")
    print(f"  - {len(results['numeric_columns'])} numeric")
    print(f"  - {len(results['categorical_columns'])} categorical")
    print(f"  - {len(results['datetime_columns'])} datetime")
    
    return results


def validate_duplicates(df: pd.DataFrame) -> Dict[str, any]:
    """
    Check for duplicate rows.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dict with validation results
    """
    results = {}
    
    # Check complete duplicates
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        print(f"⚠ Found {n_duplicates} duplicate rows ({n_duplicates/len(df):.2%})")
        results['has_duplicates'] = True
        results['n_duplicates'] = int(n_duplicates)
    else:
        print(f"✓ No duplicate rows found")
        results['has_duplicates'] = False
        results['n_duplicates'] = 0
    
    return results


def validate_value_ranges(df: pd.DataFrame) -> Dict[str, any]:
    """
    Check for outliers and invalid value ranges.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dict with validation results
    """
    results = {'outlier_features': {}}
    
    # Check numeric columns for outliers (IQR method)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
        
        if outliers > 0:
            results['outlier_features'][col] = int(outliers)
    
    if results['outlier_features']:
        print(f"⚠ Outliers detected in {len(results['outlier_features'])} features")
        for col, count in results['outlier_features'].items():
            print(f"  - {col}: {count} outliers")
    else:
        print(f"✓ No significant outliers detected")
    
    return results


def main():
    """Main validation routine."""
    parser = argparse.ArgumentParser(description='Validate data quality')
    parser.add_argument('--data-path', type=str, required=True, help='Path to data file')
    parser.add_argument('--check-schema', action='store_true', help='Check schema')
    parser.add_argument('--check-missing', action='store_true', help='Check missing values')
    parser.add_argument('--check-duplicates', action='store_true', help='Check duplicates')
    parser.add_argument('--output', type=str, default='validation_report.json', help='Output report path')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_path}...")
    try:
        if args.data_path.endswith('.csv'):
            df = pd.read_csv(args.data_path)
        elif args.data_path.endswith('.parquet'):
            df = pd.read_parquet(args.data_path)
        else:
            raise ValueError(f"Unsupported file format: {args.data_path}")
    except Exception as e:
        print(f"✗ Failed to load data: {e}")
        sys.exit(1)
    
    print(f"✓ Loaded {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    # Run validations
    report = {
        'data_shape': {'rows': df.shape[0], 'columns': df.shape[1]},
        'validations': {}
    }
    
    # Required columns for fraud detection
    required_cols = ['Time', 'Amount', 'Class']
    report['validations']['schema'] = validate_schema(df, required_cols)
    print()
    
    # Missing values
    report['validations']['missing_values'] = validate_missing_values(df)
    print()
    
    # Data types
    report['validations']['data_types'] = validate_data_types(df)
    print()
    
    # Duplicates
    if args.check_duplicates or True:
        report['validations']['duplicates'] = validate_duplicates(df)
        print()
    
    # Value ranges
    report['validations']['value_ranges'] = validate_value_ranges(df)
    print()
    
    # Summary
    all_valid = all(
        v.get('schema_valid', True) and v.get('missing_valid', True)
        for v in report['validations'].values()
    )
    
    if all_valid:
        print("✓ DATA VALIDATION PASSED")
        exit_code = 0
    else:
        print("✗ DATA VALIDATION FAILED")
        exit_code = 1
    
    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nValidation report saved to {args.output}")
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
