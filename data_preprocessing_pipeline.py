"""
BT4103 General ML Preprocessing Pipeline
Prepares cleaned data for ANY ML model (CatBoost, XGBoost, Random Forest, etc.)

Key Steps:
1. Remove data leakage columns
2. Handle datetime features
3. Identify categorical vs numerical features
4. Handle text columns and outliers
5. Save single preprocessed dataset

NOTE: Train/test split and random_state handling should be done in the respective ML model files.
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

TARGETS = ["ACTUAL_SURGERY_DURATION", "ACTUAL_USAGE_DURATION"]  # Both targets retained

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
    'actual_valid'
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

def remove_leakage_columns(df, leakage_cols):
    """Remove columns that would leak information about the targets"""
    print("\n=== REMOVING DATA LEAKAGE ===")
    
    # Check which leakage columns actually exist
    existing_leakage = [col for col in leakage_cols if col in df.columns]
    df = df.drop(columns=existing_leakage)
    print(f"✓ Removed {len(existing_leakage)} leakage columns")
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
# 5a. CUSTOM FUNCTION TO CATEGORIZE IMPLANT COLUMN
# ============================================================================

def categorize_implant(text):
    if pd.isna(text) or str(text).strip() == "" or str(text).lower() == "0":
        return "Unknown"
    text = str(text).lower()

    # Orthopedic & Spine
    if any(k in text for k in ["screw", "plate", "rod", "arthrex", "zimmer", "synthes", "bone"]):
        return "Orthopedic"
    if any(k in text for k in ["solera", "spine", "cage", "medtronic", "pedicle"]):
        return "Spine Surgery"
    
    # Neurosurgery
    if any(k in text for k in ["neuro", "cranial", "burr", "crani", "mesh"]):
        return "Neurosurgery"
    
    # ENT
    if any(k in text for k in ["eustachian", "sinus", "trache", "balloon", "ear", "nose", "throat"]):
        return "ENT"
    
    # Ophthalmology
    if any(k in text for k in ["lens", "cataract", "eye", "vitrectomy"]):
        return "Ophthalmology"
    
    # Dental / Maxillofacial
    if any(k in text for k in ["mandible", "maxilla", "dental", "tooth", "oral", "implant"]):
        return "Dental / Maxillofacial"
    
    # Cardiac / Vascular
    if any(k in text for k in ["stent", "valve", "graft", "aorta", "cardiac"]):
        return "Cardiac / Vascular"
    
    # Urology
    if any(k in text for k in ["catheter", "stent", "nephro", "ureter", "urine"]):
        return "Urology"
    
    # Gynecology / Obstetrics
    if any(k in text for k in ["hysterec", "uterus", "pregnan", "ovary"]):
        return "Gynecology / Obstetrics"
    
    # Laparoscopic / General Surgery
    if any(k in text for k in ["lapar", "trocar", "port", "camera"]):
        return "Laparoscopic / General Surgery"
    
    # Upper Limb
    if any(k in text for k in ["shoulder", "elbow", "hand", "radius", "ulna"]):
        return "Orthopedic – Upper Limb"
    
    # Lower Limb
    if any(k in text for k in ["knee", "hip", "femur", "tibia", "ankle"]):
        return "Orthopedic – Lower Limb"
    
    # Plastic / Reconstructive
    if any(k in text for k in ["flap", "graft", "reconstruct"]):
        return "Plastic / Reconstructive"
    
    # Endoscopy Equipment
    if any(k in text for k in ["scope", "endosc", "camera", "light source"]):
        return "Endoscopy Equipment"
    
    # Power Tools / Instruments
    if any(k in text for k in ["drill", "burr", "saw", "cutter"]):
        return "Power Tools / Instruments"
    
    # Vendor / Logistics
    if any(k in text for k in ["tray", "set", "vendor", "system", "standby"]):
        return "Vendor / Logistics"
    
    return "Other"

# ============================================================================
# 5b. CUSTOM FUNCTION TO CATEGORIZE DIAGNOSIS COLUMN
# ============================================================================

def categorize_diagnosis(text):
    if pd.isna(text):
        return 'Unknown'

    text = str(text).lower().strip()

    # --- Oncological ---
    if 'cancer' in text or 'carcinoma' in text or 'malignant' in text or 'tumor' in text or 'ca ' in text:
        return 'Cancer'

    # --- Urological / Renal ---
    if 'stone' in text or 'nephrolith' in text or 'urolith' in text:
        return 'Gallstone/Kidney Stone'
    if 'hydronephrosis' in text or 'renal' in text or 'kidney' in text:
        return 'Renal/Urological'

    # --- Trauma / Injury ---
    if 'fracture' in text or 'rupture' in text or 'tear' in text or 'injury' in text or 'trauma' in text:
        return 'Trauma'

    # --- Obstetric / Gynecologic ---
    if 'pregnancy' in text or 'gravid' in text or 'gestation' in text:
        return 'Pregnancy'
    if 'fibroid' in text or 'ovarian cyst' in text or 'endometriosis' in text or 'uterus' in text:
        return 'Gynecological'
    if 'subfert' in text or 'infertile' in text or 'ivf' in text:
        return 'Fertility'

    # --- Neurological ---
    if 'stroke' in text or 'infarct' in text or 'cva' in text or 'ischemia' in text:
        return 'Stroke/Ischemia'
    if 'hydrocephalus' in text or 'mening' in text or 'seizure' in text or 'brain' in text:
        return 'Neuro'

    # --- Cardiovascular ---
    if 'myocardial' in text or 'heart' in text or 'cardiac' in text or 'aortic' in text:
        return 'Cardiac'
    if 'aneurysm' in text or 'thrombosis' in text or 'embol' in text or 'vascular' in text:
        return 'Vascular'

    # --- Gastrointestinal / Hepatic ---
    if 'liver' in text or 'hepatic' in text or 'cirrhosis' in text or 'hepatitis' in text:
        return 'Liver/Hepatic'
    if 'appendicitis' in text or 'bowel' in text or 'colon' in text or 'intestin' in text:
        return 'Gastrointestinal'
    if 'gallbladder' in text or 'cholecyst' in text:
        return 'Gallbladder'

    # --- Respiratory ---
    if 'pneumonia' in text or 'lung' in text or 'pleural' in text or 'bronch' in text:
        return 'Respiratory'

    # --- Musculoskeletal ---
    if 'arthritis' in text or 'osteolysis' in text or 'joint' in text or 'muscle' in text:
        return 'Musculoskeletal'

    # --- Infection / Inflammation ---
    if 'abscess' in text or 'infection' in text or 'sepsis' in text or 'inflamm' in text:
        return 'Infectious/Inflammatory'

    # --- Pediatric / Congenital ---
    if 'congenital' in text or 'neonate' in text or 'infant' in text:
        return 'Pediatric/Congenital'

    # --- Endocrine / Metabolic ---
    if 'thyroid' in text or 'diabetes' in text or 'adrenal' in text:
        return 'Endocrine/Metabolic'

    # --- Others ---
    if 'lesion' in text or 'mass' in text or 'growth' in text:
        return 'Mass/Lesion'
    if 'post op' in text or 'postoperative' in text or 'follow up' in text:
        return 'Post-Operative/Follow-up'

    return 'Other'

# ============================================================================
# 5. HANDLE TEXT COLUMNS
# ============================================================================

def handle_text_columns(df, categorical_features, max_categories=1000):
    """
    Handle or drop text columns like IMPLANT, DIAGNOSIS, and IMPLANT.
    - Applies categorize_implant() to IMPLANT.
    - Applies categorize_diagnosis() to DIAGNOSIS.
    - Drops or keeps Remarks depending on cardinality.
    """
    print("\n=== HANDLING TEXT COLUMNS ===")

    text_cols = ['IMPLANT', 'DIAGNOSIS', 'Remarks']

    for col in text_cols:
        if col not in df.columns:
            continue

        n_unique = df[col].nunique()
        print(f"  {col}: {n_unique} unique values")

        # --- Special handling: IMPLANT -------------------------------------
        if col == 'IMPLANT':
            print("    → Applying categorize_implant() function")
            df[col] = df[col].apply(categorize_implant)
            if col not in categorical_features:
                categorical_features.append(col)
            continue

        # --- Special handling: DIAGNOSIS -----------------------------------
        if col == 'DIAGNOSIS':
            print("    → Applying categorize_diagnosis() function")
            df[col] = df[col].apply(categorize_diagnosis)
            if col not in categorical_features:
                categorical_features.append(col)
            continue

        # --- Default handling: Remarks or others ----------------------------
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
# 7. HANDLE OUTLIERS
# ============================================================================

def handle_outliers(df, target, method='clip', iqr_multiplier=1.5):
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
# 8. DATA QUALITY CHECKS
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
# 9. SAVE PREPROCESSED DATA
# ============================================================================

def save_preprocessed_data(df, cat_features, num_features, output_file="Proprocessed_Dataset.csv"):
    print("\n=== SAVING PREPROCESSED DATA ===")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✓ Saved full preprocessed dataset (with both targets) to {output_file}")
    metadata = {
        'targets': TARGETS,
        'n_rows': len(df),
        'n_features': len(df.columns),
        'n_categorical': len(cat_features),
        'n_numerical': len(num_features),
        'categorical_features': cat_features,
        'numerical_features': num_features
    }
    with open(output_file.replace(".csv", "_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("✓ Metadata saved")
    return metadata

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def preprocess_for_ml(
    input_filepath,
    remove_outliers=True,
    outlier_method='clip',
    save_data=True,
    output_file="Preprocessed_Dataset.csv"
):
    df = load_cleaned_data(input_filepath)
    
    # Step 2: Remove leakage
    df = remove_leakage_columns(df, LEAKAGE_COLUMNS)
    
    # Step 3: Engineer datetime features
    df = engineer_datetime_features(df)
    
    # Step 4: Handle high cardinality
    # df = handle_high_cardinality(df)
    
    # Step 5: Handle text columns (and keep both targets in df)
    cat_features_copy = CATEGORICAL_FEATURES.copy()
    df, cat_features_copy = handle_text_columns(df, cat_features_copy)
    all_features, cat_features, num_features = identify_feature_types(df, None, cat_features_copy)
    perform_quality_checks(df, TARGETS[0])

    if remove_outliers:
        for t in TARGETS:
            if t in df.columns:
                df = handle_outliers(df, t, method=outlier_method)

    if save_data:
        metadata = save_preprocessed_data(df, cat_features, num_features, output_file)
    else:
        metadata = None

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"✓ Dataset ready for ML model script.")
    print(f"✓ Output file: {output_file}")
    print(f"✓ Targets retained: {', '.join(TARGETS)}")
    print(f"✓ Features: {len(all_features)} ({len(cat_features)} categorical, {len(num_features)} numerical)")
    return df, metadata

# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    INPUT_FILE = "Final_Cleaned_Dataset_OPTIC_7.csv"
    OUTPUT_FILE = "Proprocessed_Dataset.csv"
    preprocess_for_ml(
        input_filepath=INPUT_FILE,
        remove_outliers=True,
        outlier_method='clip',
        save_data=True,
        output_file=OUTPUT_FILE
    )
