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
    'ROOM', 'CASE_STATUS', 'OPERATION_TYPE', 
    'EMERGENCY_PRIORITY',
    'DISCIPLINE', 
    'SURGEON', 'ANAESTHETIST_TEAM',
    'ANAESTHETIST_MCR_NO',
    'ANESTHESIA', 'EQUIPMENT', 'ADMISSION_STATUS', 
    'ADMISSION_CLASS_TYPE', 'ADMISSION_TYPE', 'ADMISSION_WARD',
    'AOH', 'BLOOD', 'IMPLANT', 'CANCER_INDICATOR', 
    'TRAUMA_INDICATOR', 'planned_valid', 'Delay_Category'
]

# Columns we wish to exclude
LEAKAGE_COLUMNS = [
    # All ACTUAL columns reveal the target (except target)
    'ACTUAL_RECEPTION_IN_TIME', 'ACTUAL_ENTER_OR_TIME',
    'ACTUAL_ANAESTHESIA_INDUCTION', 'ACTUAL_SURGERY_PREP_TIME',
    'ACTUAL_KNIFE_TO_SKIN_TIME', 'ACTUAL_SKIN_CLOSURE',
    'ACTUAL_PATIENT_REVERSAL_TIME', 'ACTUAL_EXIT_OR_TIME',
    'ACTUAL_EXIT_RECOVERY_TIME', 'ACTUAL_OR_CLEANUP_TIME',
    
    # Planned columns (since we are predicting duration, we assume planned times are unknown)
    'PLANNED_SURGERY_DURATION', 'PLANNED_USAGE_DURATION',

    # DIFF columns (calculated from actuals)
    'DIFF_SURGERY_DURATION', 'DIFF_USAGE_DURATION',
    
    # Delay columns (derived from actuals)
    '_Delay_norm', 'ENTER_START_DELAY', 'KNIFE_START_DELAY',
    'EXIT_OR_DELAY', 'Is_Late', 'Reason_Is_Late',
    
    # Metadata
    'actual_valid',

    # Multicollinearity issues (removed based on correlation analysis)
    'ANAESTHETIST_TEAM', # (Relationship note:
        # TEAM is finer-grained than MCR_NO — each Head Anaesthetist (MCR_NO) can lead multiple teams,
        # as their assisting staff may differ across surgeries. However, since the Head Anaesthetist’s
        # availability directly influences surgery timing and scheduling (affecting early/late durations
        # and slot assignments), MCR_NO is selected as the key feature for modelling, rather than TEAM.
        # )
    'LOCATION', # (our EDA informed us that ROOM is a finer-grained attribute of LOCATION. In other words, One-to-many (1 → N) relationship between ROOM and LOCATION.)
    
    # High cardinality / overfitting concerns
    'PATIENT_CODE',
    'Remarks'
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
    
    high_card_cols = ['ADMISSION_BED', 'SURGEON', 
                      'ANAESTHETIST_MCR_NO']
    
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
# 4b. DEFINE CUSTOM TRANSFORMERS
# ============================================================================

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

class EquipmentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, min_count=8):
        self.min_count = min_count

    def fit(self, X, y=None):
        # Step 1: Split and flatten
        all_equipment = X['EQUIPMENT'].fillna('').apply(self._split_equipment)
        flattened = [item for sublist in all_equipment for item in sublist if item != '0']

        # Step 2: Count occurrences
        counts = pd.Series(flattened).value_counts()

        # Step 3: Keep only equipment appearing >= min_count times
        self.common_equipment = set(counts[counts >= self.min_count].index)
        self.common_equipment.add("others")
        return self

    def transform(self, X):
        transformed = []
        for val in X['EQUIPMENT'].fillna(''):
            items = self._split_equipment(val)

            # If '0' or empty → no equipment at all
            if not items or items == ['0']:
                transformed.append([])  # No active equipment
                continue

            # Replace rare ones with "others"
            items = [
                item if item in self.common_equipment else "others"
                for item in items
            ]
            transformed.append(items)

        # One-hot encode only the retained categories
        mlb = MultiLabelBinarizer(sparse_output=True)
        mlb.fit([list(self.common_equipment)])
        self.mlb = mlb
        return mlb.transform(transformed)

    def _split_equipment(self, val):
        return [item.strip().lower() for item in str(val).split(';') if item.strip()]

# ============================================================================
# 5a. CUSTOM FUNCTION TO CATEGORIZE IMPLANT COLUMN
# ============================================================================

def categorize_implant(text):
    import pandas as pd

    if pd.isna(text):
        return 'Unknown'

    text = str(text).lower().strip()

    # === Orthopaedic Implants & Prostheses ===
    if any(k in text for k in [
        'implant', 'plate', 'screw', 'rod', 'nail', 'cage', 'spacer', 'stem', 
        'tibial', 'femur', 'hip', 'knee', 'ankle', 'shoulder', 'spine', 
        'clavicle', 'radius', 'tibia', 'humerus', 'fixator', 'locking', 'prosthesis'
    ]):
        return 'Orthopaedic Implants & Prostheses'

    # === Surgical Instruments & Tools ===
    elif any(k in text for k in [
        'drill', 'saw', 'reamer', 'driver', 'screwdriver', 'clamp', 'cutter',
        'scalpel', 'forcep', 'plier', 'rongeur', 'retractor', 'guide',
        'tray', 'instrument', 'handpiece', 'shaver', 'osteotome', 'probe', 'suction'
    ]):
        return 'Surgical Instruments & Tools'

    # === Sutures & Soft Tissue Fixation ===
    elif any(k in text for k in [
        'suture', 'anchor', 'fastfix', 'fibrewire', 'biosuture', 'juggerknot',
        'ethibond', 'prolene', 'biosure', 'labral', 'tendon', 'ligament'
    ]):
        return 'Sutures & Soft Tissue Fixation'

    # === Neurosurgical / Spine Devices ===
    elif any(k in text for k in [
        'neuro', 'neuromonitoring', 'spinal', 'spine', 'nuvasive', 'cortical',
        'cervical', 'lumbar', 'thoracic', 'bone graft', 'graft', 'fusion'
    ]):
        return 'Neurosurgical / Spine Devices'

    # === Cardiovascular / Valve / Vascular Devices ===
    elif any(k in text for k in [
        'stent', 'valve', 'pacemaker', 'catheter', 'shunt', 'balloon', 'wire',
        'vascular', 'arterial', 'venous'
    ]):
        return 'Cardiovascular / Valve / Vascular Devices'

    # === Visualization / Imaging Equipment ===
    elif any(k in text for k in [
        'camera', 'microscope', 'endoscope', 'arthroscope', 'laparoscope',
        'scope', 'navigation', 'monitor', 'intensifier', 'system', 'imaging', 'ultrasound'
    ]):
        return 'Visualization / Imaging Equipment'

    # === Power Systems & Machines ===
    elif any(k in text for k in [
        'power', 'machine', 'equipment', 'powertool', 'harmonic', 'ligasure', 'coblator',
        'ultrasonic', 'thunderbeat'
    ]):
        return 'Power Systems & Machines'

    # === Surgical Sets / Trays / Kits ===
    elif any(k in text for k in [
        'set', 'kit', 'modular', 'consignment', 'universal', 'tools', 'tray'
    ]):
        return 'Surgical Sets / Trays / Kits'

    # === Biomaterials / Bone Substitutes ===
    elif any(k in text for k in [
        'cement', 'floseal', 'collagen', 'bone', 'allograft', 'synvisc',
        'oxinium', 'syntellix', 'biodesign', 'palacos'
    ]):
        return 'Biomaterials / Bone Substitutes'

    # === Wound Care / Dressing / Drainage ===
    elif any(k in text for k in [
        'vac', 'dressing', 'drain', 'renasys', 'aquamantys', 'prp',
        'phenol', 'paraffin', 'wound'
    ]):
        return 'Wound Care / Dressing / Drainage'

    # === Dental / Maxillofacial / Craniofacial ===
    elif any(k in text for k in [
        'dental', 'orthognathic', 'cranioplasty', 'max', 'facial', 'jaw', 'orthofix'
    ]):
        return 'Dental / Maxillofacial / Craniofacial'

    # === Injection / Biologicals ===
    elif any(k in text for k in [
        'prp', 'triamcinolone', 'injection', 'stem', 'plasma', 'platelet'
    ]):
        return 'Injection / Biologicals'

    # === Brands & Manufacturers ===
    elif any(k in text for k in [
        'arthrex', 'stryker', 'zimmer', 'depuy', 'smith', 'nephew', 'medtronic',
        'synthes', 'biomet', 'nuvasive', 'medartis', 'globus', 'syntes', 
        'oxford', 'exeter', 'persona', 'tomofix', 'triathlon', 'vanguard', 'solera', 'colibri'
    ]):
        return 'Brands & Manufacturers'

    # === Consumables / Disposables ===
    elif any(k in text for k in [
        'needle', 'syringe', 'catheter', 'tube', 'mesh', 'bag', 'bottle', 'filter'
    ]):
        return 'Consumables / Disposables'

    # === Surgical Furniture & Accessories ===
    elif any(k in text for k in [
        'table', 'trolley', 'chair', 'stand', 'holder', 'arm', 'frame', 'traction'
    ]):
        return 'Surgical Furniture & Accessories'

    # === Patient / Anatomy Reference ===
    elif any(k in text for k in [
        'anterior', 'posterior', 'medial', 'lateral', 'proximal', 'distal', 
        'tibial', 'femur', 'hip', 'knee', 'hand', 'wrist', 'shoulder', 'elbow'
    ]):
        return 'Patient / Anatomy Reference'

    # === Navigation / Robotics / Computer-Assisted ===
    elif any(k in text for k in [
        'navigation', 'robotic', 'stealth', 'carto', 'navio', '3d', 'positioning'
    ]):
        return 'Navigation / Robotics / Computer-Assisted'

    # === Fixation / Support / External Devices ===
    elif any(k in text for k in [
        'brace', 'splint', 'external', 'traction', 'orthosis', 'cast', 'bandage'
    ]):
        return 'Fixation / Support / External Devices'

    else:
        return 'Other'

# ============================================================================
# 5b. CUSTOM FUNCTION TO CATEGORIZE DIAGNOSIS COLUMN
# ============================================================================

def categorize_diagnosis(text):
    if pd.isna(text):
        return 'Unknown'

    text = str(text).lower().strip()

    # === Category logic ===
    if any(k in text for k in [
        'cataract', 'glaucoma', 'retina', 'retinal', 'pterygium', 'vitrectomy',
        'macular', 'entropion', 'ptosis', 'pseudophakia', 'eye', 'cornea',
        'exotropia', 'esotropia', 'strabismus'
    ]):
        return 'Ophthalmology'

    elif any(k in text for k in [
        'fracture', 'dislocation', 'osteoporosis', 'arthritis', 'osteoarthritis',
        'scoliosis', 'ligament', 'tendon', 'meniscus', 'rotator', 'clavicle',
        'femur', 'tibia', 'humerus', 'joint', 'bone', 'spine', 'shoulder',
        'knee', 'hip', 'ankle', 'wrist', 'back', 'limb', 'elbow', 'hand', 'foot'
    ]):
        return 'Musculoskeletal / Orthopaedic'

    elif any(k in text for k in [
        'coronary', 'artery', 'atrial', 'mitral', 'ischemic', 'fibrillation',
        'tachycardia', 'bradycardia', 'aneurysm', 'angina', 'pacemaker',
        'aortic', 'valve', 'vascular', 'venous', 'varicose', 'regurgitation',
        'myocardial', 'cardiac', 'heart'
    ]):
        return 'Cardiac / Vascular'

    elif any(k in text for k in [
        'pneumonia', 'empyema', 'pleural', 'lung', 'pneumothorax', 'rhinitis',
        'sinusitis', 'asthma', 'bronchitis', 'apnea', 'snoring', 'stridor',
        'nasal', 'sinus', 'epistaxis'
    ]):
        return 'Respiratory / Thoracic'

    elif any(k in text for k in [
        'appendicitis', 'cholecystitis', 'cholelithiasis', 'cholangitis',
        'gallstones', 'gallbladder', 'biliary', 'liver', 'hepatitis',
        'esophageal', 'gastritis', 'ulcer', 'colitis', 'pancreatitis', 'hernia',
        'constipation', 'bowel', 'rectal', 'intestinal', 'abdominal', 'abdomen'
    ]):
        return 'Gastrointestinal / Hepatic'

    elif any(k in text for k in [
        'anemia', 'lymphoma', 'leukemia', 'myeloma', 'lymphadenopathy',
        'pancytopenia', 'bleed', 'hematemesis', 'coagulopathy', 'thrombosed'
    ]):
        return 'Haematological / Lymphatic'

    elif any(k in text for k in [
        'carcinoma', 'adenocarcinoma', 'sarcoma', 'tumour', 'cancer',
        'neoplasm', 'malignant', 'benign', 'metastasis', 'lesion', 'polyp',
        'mass', 'nodule', 'growth'
    ]):
        return 'Oncology (Tumour)'

    elif any(k in text for k in [
        'renal', 'kidney', 'ureter', 'ureteric', 'bladder', 'urethral',
        'hydronephrosis', 'nephritis', 'pyelonephritis', 'urinary', 'incontinence'
    ]):
        return 'Renal / Urinary'

    elif any(k in text for k in [
        'fibroid', 'fibroids', 'ovarian', 'ovary', 'uterine', 'endometrial',
        'endometriosis', 'subfertility', 'pregnancy', 'abortion', 'miscarriage',
        'breech', 'ectopic', 'cervical', 'uterus', 'menorrhagia', 'menopausal', 'pmb'
    ]):
        return 'Reproductive / Gynaecological'

    elif any(k in text for k in [
        'caesarean', 'lscs', 'twins', 'breech', 'delivery', 'labour', 'fetal',
        'placenta', 'postpartum', 'rpoc'
    ]):
        return 'Obstetric / Perinatal'

    elif any(k in text for k in [
        'abscess', 'ulcer', 'cellulitis', 'lipoma', 'sebaceous', 'carbuncle',
        'keloid', 'cyst', 'lesion', 'wound', 'scar', 'infection', 'dermoid',
        'melanoma', 'gangrene'
    ]):
        return 'Dermatological / Skin'

    elif any(k in text for k in [
        'infection', 'sepsis', 'osteomyelitis', 'cellulitis', 'hepatitis',
        'tb', 'pneumonia', 'cholangitis', 'tonsillitis', 'sinusitis'
    ]):
        return 'Infectious Disease'

    elif any(k in text for k in [
        'diabetes', 'thyroid', 'goiter', 'obesity', 'parathyroid',
        'hyperparathyroidism', 'adrenal', 'pituitary', 'metabolic'
    ]):
        return 'Endocrine / Metabolic'

    elif any(k in text for k in [
        'stroke', 'subdural', 'hydrocephalus', 'meningioma', 'hematoma',
        'glioblastoma', 'neuropathy', 'syncope', 'cerebral', 'nerve', 'myelopathy',
        'spinal', 'epilepsy'
    ]):
        return 'Neurological / Neurosurgical'

    elif any(k in text for k in [
        'depression', 'bipolar', 'schizophrenia', 'psychosis', 'schizoaffective',
        'anxiety', 'catatonia', 'psychotic', 'mdd'
    ]):
        return 'Psychiatric / Mental Health'

    elif any(k in text for k in [
        'otitis', 'tonsillitis', 'sinusitis', 'rhinitis', 'hearing', 'tonsil',
        'nasopharyngeal', 'adenoid', 'parotid', 'epistaxis', 'cholesteatoma',
        'ear', 'pinna', 'uvp', 'snoring'
    ]):
        return 'ENT (Ear, Nose, Throat)'

    elif any(k in text for k in [
        'dental', 'caries', 'tooth', 'teeth', 'oral', 'jaw', 'mandible',
        'maxillary', 'parotid', 'facial', 'tongue', 'buccal', 'lip', 'cleft'
    ]):
        return 'Dental / Maxillofacial'

    elif any(k in text for k in [
        'phimosis', 'circumcision', 'penile', 'hydrocele', 'prostate', 'testis',
        'scrotal', 'varicocele', 'epididymis', 'urethral', 'azoospermia'
    ]):
        return 'Urological / Male Reproductive'

    elif any(k in text for k in [
        'laceration', 'injury', 'trauma', 'crush', 'amputation', 'stab',
        'burn', 'scar', 'wound', 'ulcer', 'traumatic'
    ]):
        return 'Trauma / Injury'

    elif any(k in text for k in [
        'congenital', 'atresia', 'cleft', 'hypospadias', 'syndrome',
        'malformation', 'anomaly', 'hydrocephalus'
    ]):
        return 'Congenital / Developmental'

    elif any(k in text for k in [
        'arthritis', 'spondylitis', 'lupus', 'vasculitis', 'dermatitis',
        'autoimmune', 'rheumatic'
    ]):
        return 'Autoimmune / Inflammatory'

    elif any(k in text for k in [
        'screening', 'biopsy', 'repair', 'excision', 'amputation', 'debridement',
        'closure', 'intubation', 'transplant', 'investigation', 'revision', 'drainage'
    ]):
        return 'Procedural / Post-Surgical'

    else:
        return 'Other / Unclassified'

# ============================================================================
# 5. HANDLE TEXT COLUMNS
# ============================================================================

def handle_text_columns(df, categorical_features):
    """
    Handle or drop text columns like IMPLANT, DIAGNOSIS, and IMPLANT.
    - Applies categorize_implant() to IMPLANT.
    - Applies categorize_diagnosis() to DIAGNOSIS.
    """
    print("\n=== HANDLING TEXT COLUMNS ===")

    text_cols = ['IMPLANT', 'DIAGNOSIS']

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
    # print(f"    Examples: {cat_features[:3]}")
    result = ', '.join(cat_features)
    print(result)
    
    print(f"  - Numerical: {len(num_features)}")
    # print(f"    Examples: {num_features[:3]}")
    result = ', '.join(num_features)
    print(result)
    
    return all_features, cat_features, num_features

# ============================================================================
# 7. HANDLE OUTLIERS
# ============================================================================

def handle_outliers(df, target, method='remove'):
    """
    Handle outliers in target variable
    
    Parameters:
    - method: 'clip' (cap values), 'remove' (drop rows), or 'keep' (no action)
    """
    print("\n=== HANDLING OUTLIERS ===")
    
    if method == 'keep':
        print("Skipping outlier handling (keeping all data)")
        return df
    
    # Don't allow negative lower bound for duration + cap surgey/usage at 24 hours
    lower_bound = 5
    upper_bound = 1440
    
    print(f"Target: {target}")
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

    # Step 3b: Create "number of operations conducted" feature
    print("\n=== CALCULATING NUMBER OF OPERATIONS CONDUCTED ===")
    if 'SURGICAL_CODE' in df.columns:
        df['num_operations_conducted'] = (
            df['SURGICAL_CODE']
            .fillna('')
            .apply(lambda x: len([s for s in str(x).split(';') if s.strip()]))
        )
        print("✓ Added column: num_operations_conducted")
    else:
        print("⚠ SURGICAL_CODE column not found — skipping operation count feature.")

    # Step 3c: Keep only the first surgical code
    print("\n=== CLEANING SURGICAL_CODE COLUMN ===")
    if 'SURGICAL_CODE' in df.columns:
        df['SURGICAL_CODE'] = (
            df['SURGICAL_CODE']
            .astype(str)
            .str.split(';')
            .str[0]
            .str.strip()
        )
        print("✓ Retained only the first code in SURGICAL_CODE")
    else:
        print("⚠ SURGICAL_CODE column not found — skipping cleanup")

    # Step 4: Handle high cardinality
    df = handle_high_cardinality(df)
    
    # Step 5: Handle text columns (and keep both targets in df)
    cat_features_copy = CATEGORICAL_FEATURES.copy()
    df, cat_features_copy = handle_text_columns(df, cat_features_copy)
    
    # Step 5b: Apply custom transformers for multi-valued categorical fields
    print("\n=== APPLYING CUSTOM TRANSFORMERS ===")

    if 'EQUIPMENT' in df.columns:
        print("→ Expanding EQUIPMENT into one-hot encoded columns (≥8 threshold, no explicit 'No Additional Equipment')")
        equip_trans = EquipmentTransformer(min_count=8).fit(df)
        equip_matrix = equip_trans.transform(df)
        equip_df = pd.DataFrame.sparse.from_spmatrix(
            equip_matrix,
            index=df.index,
            columns=[f"EQUIPMENT_{c.title().replace(' ', '_')}" for c in equip_trans.mlb.classes_]
        )
        df = pd.concat([df.drop(columns=['EQUIPMENT']), equip_df], axis=1)

    all_features, cat_features, num_features = identify_feature_types(df, None, cat_features_copy)
    perform_quality_checks(df, TARGETS[0])
    perform_quality_checks(df, TARGETS[1])

    if remove_outliers:
        for t in TARGETS:
            if t in df.columns:
                df = handle_outliers(df, t, method=outlier_method)

    # ========================================================================
    # ADD SURGEON / OPERATION-TYPE MEDIAN STATISTICS
    # ========================================================================
    print("\n=== ADDING SURGEON AND OPERATION-TYPE MEDIAN STATISTICS ===")

    # --- Median surgery and usage durations per SURGEON ---
    if {'SURGEON', 'ACTUAL_SURGERY_DURATION', 'ACTUAL_USAGE_DURATION'}.issubset(df.columns):
        surgeon_stats = (
            df.groupby('SURGEON')
              .agg(
                  SURGEON_MEDIAN_SURGERY=('ACTUAL_SURGERY_DURATION', 'median'),
                  SURGEON_MEDIAN_USAGE=('ACTUAL_USAGE_DURATION', 'median'),
                  SURGEON_MEDIAN_CONFIDENCE=('SURGEON', 'count')
              )
              .reset_index()
        )
        df = df.merge(surgeon_stats, on='SURGEON', how='left')
        print(f"✓ Added surgeon-level median statistics for {len(surgeon_stats)} surgeons")

    # --- Median surgery and usage durations per OPERATION_TYPE ---
    if {'OPERATION_TYPE', 'ACTUAL_SURGERY_DURATION', 'ACTUAL_USAGE_DURATION'}.issubset(df.columns):
        procedure_stats = (
            df.groupby('OPERATION_TYPE')
              .agg(
                  OPERATION_TYPE_MEDIAN_SURGERY=('ACTUAL_SURGERY_DURATION', 'median'),
                  OPERATION_TYPE_MEDIAN_USAGE=('ACTUAL_USAGE_DURATION', 'median'),
                  OPERATION_TYPE_MEDIAN_CONFIDENCE=('OPERATION_TYPE', 'count')
              )
              .reset_index()
        )
        df = df.merge(procedure_stats, on='OPERATION_TYPE', how='left')
        print(f"✓ Added operation-type-level median statistics for {len(procedure_stats)} procedures")              

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
        outlier_method='remove',
        save_data=True,
        output_file=OUTPUT_FILE
    )
