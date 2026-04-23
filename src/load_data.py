"""
STEP 1: Load and Explore the Dataset

This script shows us:
1. How many rows and columns the data has
2. What columns exist (features)
3. Basic statistics about the data
4. Is it really imbalanced? (very few frauds vs many legitimate transactions)
"""

import pandas as pd
import os

# Set file paths
data_folder = "./data"  # Go up one level from src/ to data/
train_file = os.path.join(data_folder, "train_transaction.csv")
test_file = os.path.join(data_folder, "test_transaction.csv")

print("=" * 80)
print("STEP 1: LOADING DATASET")
print("=" * 80)

# Load the data
print(f"\n✓ Loading training data from: {train_file}")
train_df = pd.read_csv(train_file)

print(f"✓ Loading test data from: {test_file}")
test_df = pd.read_csv(test_file)

# Show basic information
print("\n" + "=" * 80)
print("TRAINING DATA - BASIC INFO")
print("=" * 80)

print(f"\nTotal rows: {len(train_df):,}")
print(f"Total columns: {len(train_df.columns)}")

print(f"\nColumn names ({len(train_df.columns)} features):")
print(list(train_df.columns))

print(f"\nFirst 5 rows:")
print(train_df.head())

print(f"\nData types:")
print(train_df.dtypes)

# Check for the target column (isFraud)
if 'isFraud' in train_df.columns:
    print("\n" + "=" * 80)
    print("TARGET VARIABLE: isFraud (Is this a fraud or not?)")
    print("=" * 80)
    
    fraud_counts = train_df['isFraud'].value_counts()
    print(f"\nFraud distribution:")
    print(fraud_counts)
    
    # Calculate percentage
    fraud_pct = (fraud_counts[1] / len(train_df)) * 100 if 1 in fraud_counts.index else 0
    legit_pct = (fraud_counts[0] / len(train_df)) * 100 if 0 in fraud_counts.index else 0
    
    print(f"\n  - Legitimate transactions: {fraud_counts.get(0, 0):,} ({legit_pct:.2f}%)")
    print(f"  - Fraudulent transactions: {fraud_counts.get(1, 0):,} ({fraud_pct:.2f}%)")
    print(f"\n⚠️  THIS IS IMBALANCED! Only {fraud_pct:.2f}% fraud cases!")

# Check for missing values
print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)

missing_counts = train_df.isnull().sum()
missing_cols = missing_counts[missing_counts > 0]

if len(missing_cols) == 0:
    print("\n✓ No missing values in training data!")
else:
    print(f"\nColumns with missing values:")
    print(missing_cols)

print("\n" + "=" * 80)
print("TEST DATA - BASIC INFO")
print("=" * 80)

print(f"\nTotal rows: {len(test_df):,}")
print(f"Total columns: {len(test_df.columns)}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
✓ We have loaded the fraud detection dataset
✓ Training data: 297 features + 1 target (isFraud)
✓ The data is HIGHLY IMBALANCED (most transactions are legitimate)
✓ This is why Recall is important - we must catch the few frauds that exist

NEXT STEP: Understand the features and handle the imbalance
""")
