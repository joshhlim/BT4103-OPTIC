"""
BT4103 General ML Preprocessing Pipeline
Prepares cleaned data for ANY ML model (CatBoost, XGBoost, Random Forest, etc.)

Key Steps:
1. Remove data leakage columns
2. Handle datetime features
3. Identify categorical vs numerical features
4. Create train/test/validation splits
5. Handle outliers (optional)
6. Save preprocessed data in model-agnostic format

NOTE: Model-specific transformations (encoding, scaling, etc.) should be done
in separate model-specific scripts.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Target variable - choose one:
TARGET = "ACTUAL_SURGERY_DURATION"  # Main target: knife-to-skin → closure (minutes)
# TARGET = "ACTUAL_USAGE_DURATION"  # Alternative: total OR time (minutes)

# Categorical features (will need encoding for most models except tree-based)
CATEGORICAL_FEATURES = [
    'LOCATION', 'ROOM', 'CASE_STATUS', 'OPERATION_TYPE', 
    'EMERGENCY_PRIORITY', 'SURGICAL_CODE', 'DISCIPLINE', 
    'SURGEON', 'ANAESTHETIST_TEAM', 'ANAESTHETIST_MCR_NO',
    'ANESTHESIA', 'EQUIPMENT', 'ADMISSION_STATUS', 
    'ADMISSION_CLASS_TYPE', 'ADMISSION_TYPE', 'ADMISSION_WARD',
    'AOH', 'BLOOD', 'IMPLANT', 'CANCER_INDICATOR', 
    'TRAUMA_INDICATOR', 'planned_valid', 'Delay_Category'
]

# Data leakage columns - these reveal the target!
LEAKAGE_COLUMNS = [
    # All ACTUAL columns (except target)
    'ACTUAL_RECEPTION_IN_TIME', 'ACTUAL_ENTER_OR_TIME',
    'ACTUAL_ANAESTHESIA_INDUCTION', 'ACTUAL_SURGERY_PREP_TIME',
    'ACTUAL_KNIFE_TO_SKIN_TIME', 'ACTUAL_SKIN_CLOSURE',
    'ACTUAL_PATIENT_REVERSAL_TIME', 'ACTUAL_EXIT_OR_TIME',
    'ACTUAL_EXIT_RECOVERY_TIME', 'ACTUAL_OR_CLEANUP_TIME',
    
    # DIFF columns (calculated from actuals)
    'DIFF_SURGERY_DURATION', 'DIFF_USAGE_DURATION',
    
    # Delay columns (derived from actuals)
    '_Delay_norm', 'ENTER_START_DELAY', 'KNIFE_START_DELAY',
    'EXIT_OR_DELAY', 'Is_Late', 'Reason_Is_Late',
    
    # Metadata
    'actual_valid',
    
    # Alternative targets (if not using them)
    'ACTUAL_USAGE_DURATION',  # Will be removed if TARGET is ACTUAL_SURGERY_DURATION
    'ACTUAL_SURGERY_DURATION',  # Will be removed if TARGET is ACTUAL_USAGE_DURATION
]

# ============================================================================
# 1. LOAD AND INSPECT DATA
# ============================================================================

def load_cleaned_data(filepath):
    """Load the cleaned dataset"""
    print("=" * 70)
    print("GENERAL ML PREPROCESSING PIPELINE")
    print("=" * 70)
    print(f"\nLoading data from: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    
    return df

# ============================================================================
# 2. REMOVE DATA LEAKAGE
# ============================================================================

def remove_leakage_columns(df, target, leakage_cols):
    """Remove columns that would leak information about the target"""
    print("\n=== REMOVING DATA LEAKAGE ===")
    
    # Check which leakage columns actually exist
    existing_leakage = [col for col in leakage_cols if col in df.columns]
    
    # Ensure target is not removed
    if target in existing_leakage:
        existing_leakage.remove(target)
    
    # Remove alternative target if not the main target
    if target == 'ACTUAL_SURGERY_DURATION' and 'ACTUAL_USAGE_DURATION' in existing_leakage:
        pass  # Already in leakage list
    elif target == 'ACTUAL_USAGE_DURATION' and 'ACTUAL_SURGERY_DURATION' in existing_leakage:
        pass  # Already in leakage list
    
    print(f"Removing {len(existing_leakage)} leakage columns:")
    print(f"  {', '.join(existing_leakage[:5])}...")
    
    df = df.drop(columns=existing_leakage)
    print(f"✓ Remaining columns: {len(df.columns)}")
    
    return df

# ============================================================================
# 3. FEATURE ENGINEERING FROM DATETIME
# ============================================================================

def engineer_datetime_features(df):
    """Extract useful features from datetime columns"""
    print("\n=== ENGINEERING DATETIME FEATURES ===")
    
    datetime_cols = [
        'PLANNED_PATIENT_CALL_TIME', 'PLANNED_PATIENT_FETCH_TIME',
        'PLANNED_RECEPTION_IN_TIME', 'PLANNED_ENTER_OR_TIME',
        'PLANNED_ANAESTHESIA_INDUCTION', 'PLANNED_SURGERY_PREP_TIME',
        'PLANNED_KNIFE_TO_SKIN_TIME', 'PLANNED_SKIN_CLOSURE',
        'PLANNED_PATIENT_REVERSAL_TIME', 'PLANNED_EXIT_OR_TIME',
        'PLANNED_OR_CLEANUP_TIME', 'PLANNED_EXIT_RECOVERY_TIME',
        'PATIENT_CALL_TIME', 'PATIENT_FETCH_TIME'
    ]
    
    # Convert to datetime if not already
    for col in datetime_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Extract features from planned start time (knife-to-skin)
    if 'PLANNED_KNIFE_TO_SKIN_TIME' in df.columns:
        df['planned_hour'] = df['PLANNED_KNIFE_TO_SKIN_TIME'].dt.hour
        df['planned_day_of_week'] = df['PLANNED_KNIFE_TO_SKIN_TIME'].dt.dayofweek
        df['planned_month'] = df['PLANNED_KNIFE_TO_SKIN_TIME'].dt.month
        df['planned_quarter'] = df['PLANNED_KNIFE_TO_SKIN_TIME'].dt.quarter
        df['planned_year'] = df['PLANNED_KNIFE_TO_SKIN_TIME'].dt.year
        
        # Boolean time indicators (as integers for compatibility)
        df['is_morning'] = ((df['planned_hour'] >= 6) & (df['planned_hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['planned_hour'] >= 12) & (df['planned_hour'] < 18)).astype(int)
        df['is_evening'] = ((df['planned_hour'] >= 18) & (df['planned_hour'] < 22)).astype(int)
        df['is_night'] = ((df['planned_hour'] >= 22) | (df['planned_hour'] < 6)).astype(int)
        df['is_weekend'] = (df['planned_day_of_week'] >= 5).astype(int)
        
        print("✓ Created temporal features:")
        print("  - Numerical: hour, day_of_week, month, quarter, year")
        print("  - Binary: is_morning, is_afternoon, is_evening, is_night, is_weekend")
    
    # Drop original datetime columns (most models can't use them)
    datetime_to_drop = [col for col in datetime_cols if col in df.columns]
    df = df.drop(columns=datetime_to_drop)
    print(f"✓ Dropped {len(datetime_to_drop)} datetime columns")
    
    return df

# ============================================================================
# 4. HANDLE HIGH CARDINALITY CATEGORICALS
# ============================================================================

def handle_high_cardinality(df, threshold=100, min_freq=10):
    """Group rare categories in high-cardinality features"""
    print("\n=== HANDLING HIGH CARDINALITY ===")
    
    high_card_cols = ['PATIENT_CODE', 'ADMISSION_BED', 'SURGEON', 
                      'SURGICAL_CODE', 'ANAESTHETIST_MCR_NO', 'EQUIPMENT']
    
    for col in high_card_cols:
        if col not in df.columns:
            continue
        
        n_unique = df[col].nunique()
        
        if n_unique > threshold:
            # Group categories appearing less than min_freq times
            value_counts = df[col].value_counts()
            rare_categories = value_counts[value_counts < min_freq].index
            
            if len(rare_categories) > 0:
                df[col] = df[col].replace(rare_categories, 'RARE')
                print(f"  {col}: {n_unique} → {df[col].nunique()} categories (grouped {len(rare_categories)} rare)")
    
    return df

# ============================================================================
# 5. HANDLE TEXT COLUMNS
# ============================================================================

def handle_text_columns(df, categorical_features, max_categories=1000):
    """Handle or drop text columns like DIAGNOSIS and Remarks"""
    print("\n=== HANDLING TEXT COLUMNS ===")
    
    text_cols = ['DIAGNOSIS', 'Remarks']
    
    for col in text_cols:
        if col not in df.columns:
            continue
        
        n_unique = df[col].nunique()
        print(f"  {col}: {n_unique} unique values")
        
        if n_unique > max_categories:
            print(f"    → Dropping (too many unique values)")
            df = df.drop(columns=[col])
            if col in categorical_features:
                categorical_features.remove(col)
        else:
            print(f"    → Keeping as categorical feature")
            if col not in categorical_features:
                categorical_features.append(col)
    
    return df, categorical_features

# ============================================================================
# 6. IDENTIFY FEATURE TYPES
# ============================================================================

def identify_feature_types(df, target, categorical_features):
    """Identify which features are categorical vs numerical"""
    print("\n=== IDENTIFYING FEATURE TYPES ===")
    
    # Ensure target is not in features
    all_features = [col for col in df.columns if col != target]
    
    # Categorical features (present in dataframe)
    cat_features = [col for col in categorical_features if col in all_features]
    
    # Numerical features (everything else)
    num_features = [col for col in all_features if col not in cat_features]
    
    print(f"✓ Total features: {len(all_features)}")
    print(f"  - Categorical: {len(cat_features)}")
    print(f"    Examples: {cat_features[:3]}")
    print(f"  - Numerical: {len(num_features)}")
    print(f"    Examples: {num_features[:3]}")
    
    return all_features, cat_features, num_features

# ============================================================================
# 7. HANDLE OUTLIERS (OPTIONAL)
# ============================================================================

def handle_outliers(df, target, method='clip', iqr_multiplier=3.0):
    """
    Handle outliers in target variable
    
    Parameters:
    - method: 'clip' (cap values), 'remove' (drop rows), or 'keep' (no action)
    - iqr_multiplier: how many IQRs away to consider outlier
    """
    print("\n=== HANDLING OUTLIERS ===")
    
    if method == 'keep':
        print("Skipping outlier handling (keeping all data)")
        return df
    
    # Calculate outlier bounds using IQR method
    Q1 = df[target].quantile(0.25)
    Q3 = df[target].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - iqr_multiplier * IQR
    upper_bound = Q3 + iqr_multiplier * IQR
    
    # Don't allow negative lower bound for duration
    lower_bound = max(0, lower_bound)
    
    print(f"Target: {target}")
    print(f"  Q1: {Q1:.1f}, Q3: {Q3:.1f}, IQR: {IQR:.1f}")
    print(f"  Outlier bounds: [{lower_bound:.1f}, {upper_bound:.1f}]")
    
    # Count outliers
    outliers = (df[target] < lower_bound) | (df[target] > upper_bound)
    n_outliers = outliers.sum()
    print(f"  Found {n_outliers} outliers ({n_outliers/len(df)*100:.2f}%)")
    
    if method == 'clip':
        # Cap values at bounds
        df[target] = df[target].clip(lower=lower_bound, upper=upper_bound)
        print(f"✓ Clipped outliers to bounds")
    
    elif method == 'remove':
        # Remove outlier rows
        df = df[~outliers].copy()
        print(f"✓ Removed {n_outliers} outlier rows")
    
    return df

# ============================================================================
# 8. CREATE TRAIN/VAL/TEST SPLITS
# ============================================================================

def create_splits(df, target, test_size=0.2, val_size=0.1, random_state=42):
    """
    Create train/validation/test splits
    
    Returns: X_train, X_val, X_test, y_train, y_val, y_test
    """
    print("\n=== CREATING DATA SPLITS ===")
    
    # Separate features and target
    X = df.drop(columns=[target])
    y = df[target]
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)  # adjust for remaining data
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    print(f"✓ Data splits created:")
    print(f"  Train: {len(X_train)} rows ({len(X_train)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} rows ({len(X_val)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} rows ({len(X_test)/len(df)*100:.1f}%)")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# ============================================================================
# 9. DATA QUALITY CHECKS
# ============================================================================

def perform_quality_checks(df, target):
    """Perform final data quality checks"""
    print("\n=== DATA QUALITY CHECKS ===")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("⚠ Warning: Missing values found:")
        print(missing[missing > 0])
    else:
        print("✓ No missing values")
    
    # Check for infinite values in numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    inf_check = np.isinf(df[num_cols]).sum()
    if inf_check.sum() > 0:
        print("⚠ Warning: Infinite values found:")
        print(inf_check[inf_check > 0])
    else:
        print("✓ No infinite values")
    
    # Check target variable
    print(f"\nTarget variable: {target}")
    print(f"  Mean: {df[target].mean():.2f} minutes")
    print(f"  Median: {df[target].median():.2f} minutes")
    print(f"  Std: {df[target].std():.2f} minutes")
    print(f"  Min: {df[target].min():.2f} minutes")
    print(f"  Max: {df[target].max():.2f} minutes")
    
    # Check for negative durations
    if (df[target] < 0).any():
        print(f"⚠ Warning: {(df[target] < 0).sum()} negative values in target!")
    else:
        print("✓ No negative values in target")
    
    return True

# ============================================================================
# 10. SAVE PREPROCESSED DATA
# ============================================================================

def save_preprocessed_data(X_train, X_val, X_test, y_train, y_val, y_test, 
                           cat_features, num_features, target, output_dir="preprocessed_data"):
    """Save preprocessed data and metadata in model-agnostic format"""
    print("\n=== SAVING PREPROCESSED DATA ===")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits as CSV (compatible with all models)
    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_val.to_csv(f"{output_dir}/X_val.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False, header=['target'])
    y_val.to_csv(f"{output_dir}/y_val.csv", index=False, header=['target'])
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False, header=['target'])
    
    # Save metadata for model-specific scripts
    metadata = {
        'target': target,
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'n_features': len(X_train.columns),
        'n_categorical': len(cat_features),
        'n_numerical': len(num_features),
        'categorical_features': cat_features,
        'numerical_features': num_features,
        'all_features': list(X_train.columns),
        'target_stats': {
            'mean': float(y_train.mean()),
            'std': float(y_train.std()),
            'min': float(y_train.min()),
            'max': float(y_train.max())
        }
    }
    
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save feature types separately for easy loading
    feature_types = pd.DataFrame({
        'feature': list(X_train.columns),
        'type': ['categorical' if col in cat_features else 'numerical' for col in X_train.columns]
    })
    feature_types.to_csv(f"{output_dir}/feature_types.csv", index=False)
    
    print(f"✓ Saved to {output_dir}/")
    print(f"  Files created:")
    print(f"    - X_train.csv, X_val.csv, X_test.csv")
    print(f"    - y_train.csv, y_val.csv, y_test.csv")
    print(f"    - metadata.json (feature info, stats)")
    print(f"    - feature_types.csv (categorical vs numerical)")

# ============================================================================
# 11. MAIN PREPROCESSING PIPELINE
# ============================================================================

def preprocess_for_ml(
    input_filepath,
    target=TARGET,
    remove_outliers=True,
    outlier_method='clip',
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    save_data=True,
    output_dir="preprocessed_data"
):
    """
    Complete preprocessing pipeline for general ML models
    
    Parameters:
    - input_filepath: path to cleaned CSV
    - target: target variable name
    - remove_outliers: whether to handle outliers
    - outlier_method: 'clip', 'remove', or 'keep'
    - test_size: proportion for test set (default 0.2)
    - val_size: proportion for validation set (default 0.1)
    - random_state: random seed for reproducibility
    - save_data: whether to save preprocessed data
    - output_dir: directory to save preprocessed data
    
    Returns:
    - Dictionary with all data and metadata
    """
    
    # Step 1: Load data
    df = load_cleaned_data(input_filepath)
    
    # Step 2: Remove leakage
    df = remove_leakage_columns(df, target, LEAKAGE_COLUMNS)
    
    # Step 3: Engineer datetime features
    df = engineer_datetime_features(df)
    
    # Step 4: Handle high cardinality
    df = handle_high_cardinality(df)
    
    # Step 5: Handle text columns
    cat_features_copy = CATEGORICAL_FEATURES.copy()
    df, cat_features_copy = handle_text_columns(df, cat_features_copy)
    
    # Step 6: Identify feature types
    all_features, cat_features, num_features = identify_feature_types(
        df, target, cat_features_copy
    )
    
    # Step 7: Quality checks
    perform_quality_checks(df, target)
    
    # Step 8: Handle outliers
    if remove_outliers:
        df = handle_outliers(df, target, method=outlier_method)
    
    # Step 9: Create splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(
        df, target, test_size=test_size, val_size=val_size, random_state=random_state
    )
    
    # Step 10: Save data
    if save_data:
        save_preprocessed_data(
            X_train, X_val, X_test, y_train, y_val, y_test,
            cat_features, num_features, target, output_dir
        )
    
    # Summary
    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"✓ Ready for ML modeling!")
    print(f"✓ Target: {target}")
    print(f"✓ Features: {len(all_features)} ({len(cat_features)} categorical, {len(num_features)} numerical)")
    print(f"✓ Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    print(f"\nNext steps:")
    print(f"  1. Load data from '{output_dir}/' folder")
    print(f"  2. Apply model-specific preprocessing (encoding, scaling, etc.)")
    print(f"  3. Train your model!")
    
    # Return everything for programmatic use
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'categorical_features': cat_features,
        'numerical_features': num_features,
        'all_features': all_features,
        'target': target,
        'output_dir': output_dir
    }

# ============================================================================
# 12. HELPER FUNCTION TO LOAD PREPROCESSED DATA
# ============================================================================

def load_preprocessed_data(data_dir="preprocessed_data"):
    """
    Load preprocessed data for model training
    
    Returns: Dictionary with all data and metadata
    """
    print(f"Loading preprocessed data from {data_dir}/...")
    
    # Load splits
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_val = pd.read_csv(f"{data_dir}/X_val.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    
    y_train = pd.read_csv(f"{data_dir}/y_train.csv")['target']
    y_val = pd.read_csv(f"{data_dir}/y_val.csv")['target']
    y_test = pd.read_csv(f"{data_dir}/y_test.csv")['target']
    
    # Load metadata
    with open(f"{data_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Load feature types
    feature_types = pd.read_csv(f"{data_dir}/feature_types.csv")
    
    print(f"✓ Loaded {len(X_train)} train, {len(X_val)} val, {len(X_test)} test samples")
    print(f"✓ Target: {metadata['target']}")
    print(f"✓ Features: {metadata['n_features']}")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'metadata': metadata,
        'feature_types': feature_types,
        'categorical_features': metadata['categorical_features'],
        'numerical_features': metadata['numerical_features']
    }

# ============================================================================
# 13. USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Configuration
    INPUT_FILE = "/Users/nigelcmc/Downloads/Final_Cleaned_Dataset_OPTIC_7.csv"
    OUTPUT_DIR = "Proprocessed_Dataset.csv"
    
    # Run preprocessing pipeline
    results = preprocess_for_ml(
        input_filepath=INPUT_FILE,
        target="ACTUAL_SURGERY_DURATION",  # or "ACTUAL_USAGE_DURATION"
        remove_outliers=True,
        outlier_method='clip',  # 'clip', 'remove', or 'keep'
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        save_data=True,
        output_dir=OUTPUT_DIR
    )
    
    print("\n" + "=" * 70)
    print("✓ Preprocessing complete!")
    print("=" * 70)
    print("\nData saved to:", OUTPUT_DIR)
    print("\nTo load for model training:")
    print(f"  data = load_preprocessed_data('{OUTPUT_DIR}')")
    print(f"  X_train = data['X_train']")
    print(f"  y_train = data['y_train']")
    print("  # Apply model-specific preprocessing, then train!")