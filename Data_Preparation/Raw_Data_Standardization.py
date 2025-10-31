"""
ONE-TIME USE ONLY: Corrects historical data (2016-2022) to match proper data entry protocols.
Future data exports should already conform to these standards, making this file obsolete.

Input:  Raw_Dataset.csv (raw messy dataset)
Output: Standardized_Raw_Dataset.csv (Simulation of a raw dataset with proper data entry protocols)
"""
import pandas as pd
import numpy as np
import re
import datetime
import warnings
import time
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
    tqdm.pandas()
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("TQDM not installed")

# CONFIGURATION
INPUT_FILE = "/Users/joshlim/Documents/GitHub/BT4103-OPTIC/Data/Raw_Dataset.csv"
VALIDATION_FILE = "/Users/joshlim/Documents/GitHub/BT4103-OPTIC/Data/Validation_Dataset.csv"
OUTPUT_FILE = "/Users/joshlim/Documents/GitHub/BT4103-OPTIC/Data/Standardized_Raw_Dataset.csv"
MIN_DATE = pd.Timestamp("2016-12-31")
MAX_DATE = pd.Timestamp("2022-02-25")
PLANNED_COLS = [
    "PLANNED_PATIENT_CALL_TIME",
    "PLANNED_PATIENT_FETCH_TIME",
    "PLANNED_RECEPTION_IN_TIME",
    "PLANNED_ENTER_OR_TIME",
    "PLANNED_ANAESTHESIA_INDUCTION",
    "PLANNED_SURGERY_PREP_TIME",
    "PLANNED_KNIFE_TO_SKIN_TIME",
    "PLANNED_SKIN_CLOSURE",
    "PLANNED_PATIENT_REVERSAL_TIME",
    "PLANNED_EXIT_OR_TIME",
    "PLANNED_OR_CLEANUP_TIME",
    "PLANNED_EXIT_RECOVERY_TIME",
]
ACTUAL_COLS = [
    "PATIENT_CALL_TIME",
    "PATIENT_FETCH_TIME",
    "ACTUAL_RECEPTION_IN_TIME",
    "ACTUAL_ENTER_OR_TIME",
    "ACTUAL_ANAESTHESIA_INDUCTION",
    "ACTUAL_SURGERY_PREP_TIME",
    "ACTUAL_KNIFE_TO_SKIN_TIME",
    "ACTUAL_SKIN_CLOSURE",
    "ACTUAL_PATIENT_REVERSAL_TIME",
    "ACTUAL_EXIT_OR_TIME",
    "ACTUAL_OR_CLEANUP_TIME",
    "ACTUAL_EXIT_RECOVERY_TIME",
]
# HELPER FUNCTIONS
# Regex patterns for detecting time-only and date-like strings
_time_only_re = re.compile(r'^\s*\d{1,2}:\d{2}(:\d{2})?\s*(?:[AaPp][Mm])?\s*$')
_date_like_re = re.compile(r'\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}')

def looks_like_date_string(s: str) -> bool:
    # Check if string contains an explicit date pattern.
    if not isinstance(s, str):
        return False
    return bool(_date_like_re.search(s.strip()))

def is_time_only_string(s: str) -> bool:
    # Check if string is time-only
    if not isinstance(s, str):
        return False
    return bool(_time_only_re.match(s.strip()))

def find_start_date_from_row(row, cols):
    # Scan columns in order and return the first valid date (normalized to 00:00:00).
    # A value is considered to contain a date if:
    # - It's a pandas Timestamp/datetime with year > 1900
    # - Or the string contains a date pattern and parses to a valid Timestamp
    for col in cols:
        if col not in row.index:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        if isinstance(val, pd.Timestamp):
            if val.year > 1900:
                return val.normalize()
            continue
        if isinstance(val, datetime.datetime):
            if val.year > 1900:
                return pd.Timestamp(val).normalize()
        try:
            s = str(val).strip()
        except Exception:
            continue
        if looks_like_date_string(s):
            parsed = pd.to_datetime(s, errors="coerce", dayfirst=False)
            if not pd.isna(parsed) and parsed.year > 1900:
                return parsed.normalize()
    return None

def combine_time_with_date(time_str: str, base_date: pd.Timestamp):
    # Parse time-only string and combine with base_date
    if base_date is None:
        return None
    parsed = pd.to_datetime(time_str, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp.combine(base_date, parsed.time())

def build_validation_lookup_dict(validation_df, cols):
    # Pre-build a lookup dictionary for faster validation lookups.
    lookup = {}
    if "OPERATION_ID" not in validation_df.columns:
        return lookup
    # Group by OPERATION_ID for O(1) lookups
    for opid, group in validation_df.groupby("OPERATION_ID"):
        if group.empty:
            continue
        # Find the first valid date for this OPERATION_ID
        start_date = find_start_date_from_row(group.iloc[0], cols)
        if start_date is not None:
            lookup[opid] = start_date
    
    return lookup

def attach_dates_to_time_only_strings(row, planned_cols, actual_cols, planned_lookup, actual_lookup):
    # For each row:
    # 1. Find planned_start (first planned col with valid date)
    # 2. Find actual_start (first actual col with valid date)
    # 3. Validate both are within MIN_DATE..MAX_DATE; if not, retrieve from lookup dict
    # 4. For time-only strings, attach the appropriate start date
    # 5. For full datetimes, validate they're in range or fix them
    # Find start dates
    planned_start = find_start_date_from_row(row, planned_cols)
    actual_start = find_start_date_from_row(row, actual_cols)
    def in_range(ts):
        return isinstance(ts, pd.Timestamp) and (MIN_DATE <= ts <= MAX_DATE)
    opid = row.get("OPERATION_ID", None)

    # Fix planned_start if out of range
    if not in_range(planned_start):
        alt_planned = planned_lookup.get(opid)
        if in_range(alt_planned):
            planned_start = alt_planned
        else:
            planned_start = None
    
    # Fix actual_start if out of range
    if not in_range(actual_start):
        alt_actual = actual_lookup.get(opid)
        if in_range(alt_actual):
            actual_start = alt_actual
        else:
            actual_start = None
    
    # if actual_start missing, use planned_start
    if actual_start is None and planned_start is not None:
        actual_start = planned_start
    
    # Process planned columns
    for col in planned_cols:
        if col not in row.index:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        
        # If already Timestamp/datetime
        if isinstance(val, (pd.Timestamp, datetime.datetime)):
            ts = pd.Timestamp(val)
            if MIN_DATE <= ts <= MAX_DATE:
                row[col] = ts
            elif planned_start is not None:
                row[col] = pd.Timestamp.combine(planned_start, ts.time())
            else:
                row[col] = pd.NaT
            continue
        
        # String handling
        s = str(val).strip()
        if is_time_only_string(s):
            if planned_start is not None:
                combined = combine_time_with_date(s, planned_start)
                if combined is not None:
                    row[col] = combined
        else:
            parsed = pd.to_datetime(s, errors="coerce", dayfirst=False)
            if not pd.isna(parsed):
                if MIN_DATE <= parsed <= MAX_DATE:
                    row[col] = parsed
                elif planned_start is not None:
                    row[col] = pd.Timestamp.combine(planned_start, parsed.time())
                else:
                    row[col] = pd.NaT
    
    # Process actual columns
    for col in actual_cols:
        if col not in row.index:
            continue
        val = row[col]
        if pd.isna(val):
            continue
        
        if isinstance(val, (pd.Timestamp, datetime.datetime)):
            ts = pd.Timestamp(val)
            if MIN_DATE <= ts <= MAX_DATE:
                row[col] = ts
            elif actual_start is not None:
                row[col] = pd.Timestamp.combine(actual_start, ts.time())
            else:
                row[col] = pd.NaT
            continue
        
        s = str(val).strip()
        if is_time_only_string(s):
            if actual_start is not None:
                combined = combine_time_with_date(s, actual_start)
                if combined is not None:
                    row[col] = combined
        else:
            parsed = pd.to_datetime(s, errors="coerce", dayfirst=False)
            if not pd.isna(parsed):
                if MIN_DATE <= parsed <= MAX_DATE:
                    row[col] = parsed
                elif actual_start is not None:
                    row[col] = pd.Timestamp.combine(actual_start, parsed.time())
                else:
                    row[col] = pd.NaT
    
    return row

def sync_columns_vectorized(data, col_a, col_b):
    # Vectorized column syncing
    a = data[col_a]
    b = data[col_b]
    
    # Where A is missing but B exists, copy B to A
    mask_a = a.isna() & b.notna()
    data.loc[mask_a, col_a] = data.loc[mask_a, col_b]
    
    # Where B is missing but A exists, copy A to B
    mask_b = b.isna() & a.notna()
    data.loc[mask_b, col_b] = data.loc[mask_b, col_a]
    
    return data

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def main():
    start_time = time.time()
    
    print("=" * 80)
    print("STANDARDIZATION PIPELINE".center(80))
    print("=" * 80)
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)
    step_start = time.time()
    
    # Load data
    data = pd.read_csv(INPUT_FILE)
    validation_data = pd.read_csv(VALIDATION_FILE)
    print(f"Loaded {len(data):,} rows from {INPUT_FILE}")
    print(f"Loaded {len(validation_data):,} validation rows")
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: STANDARDIZE TEXT FIELDS")
    print("=" * 80)
    step_start = time.time()
    
    # Emergency Priority standardization
    if "EMERGENCY_PRIORITY" in data.columns:
        data["EMERGENCY_PRIORITY"] = data["EMERGENCY_PRIORITY"].replace({
            'P3a': 'P3A',
            'P2': 'P2B',
            'P3': 'P3B',
            'P3b': 'P3B'
        })
        print("Standardized EMERGENCY_PRIORITY values")
    
    # AOH standardization
    if "AOH" in data.columns:
        data["AOH"] = data["AOH"].astype(str).str.lower().str.strip()
        data["AOH"] = data["AOH"].replace({
            "0": "false", "na": "false", "n/a": "false", 
            "-": "false", "null": "false", "nan": "false"
        })
        print("Standardized AOH to true/false")
    
    # ADMISSION_STATUS: Remove numeric entries
    if "ADMISSION_STATUS" in data.columns:
        before = len(data)
        mask = data["ADMISSION_STATUS"].astype(str).str.match(r'^\d+$', na=False)
        data = data[~mask]
        after = len(data)
        if before != after:
            print(f"Removed {before - after} rows with numeric ADMISSION_STATUS")
    
    # CANCER_INDICATOR standardization
    if "CANCER_INDICATOR" in data.columns:
        data["CANCER_INDICATOR"] = data["CANCER_INDICATOR"].astype(str).str.lower().str.strip()
        data["CANCER_INDICATOR"] = data["CANCER_INDICATOR"].replace({
            "0": "false", "na": "false", "n/a": "false", 
            "-": "false", "null": "false", "nan": "false"
        })
        before = len(data)
        data = data[data["CANCER_INDICATOR"].isin(["false", "true"])]
        after = len(data)
        if before != after:
            print(f"Removed {before - after} rows with invalid CANCER_INDICATOR")
        print("Standardized CANCER_INDICATOR to true/false")
    
    # TRAUMA_INDICATOR standardization
    if "TRAUMA_INDICATOR" in data.columns:
        data["TRAUMA_INDICATOR"] = data["TRAUMA_INDICATOR"].astype(str).str.lower().str.strip()
        data["TRAUMA_INDICATOR"] = data["TRAUMA_INDICATOR"].replace({
            "0": "false", "na": "false", "n/a": "false", 
            "-": "false", "null": "false", "nan": "false"
        })
        print("Standardized TRAUMA_INDICATOR to true/false")
    
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: HANDLE OPTIONAL COLUMNS")
    print("=" * 80)
    step_start = time.time()
    
    optional_cols = ["Delay_Reason", "Remarks", "IMPLANT", "EQUIPMENT", "EMERGENCY_PRIORITY"]
    for col in optional_cols:
        if col in data.columns:
            data[col].fillna("0", inplace=True)
            print(f"Filled missing {col} with '0'")
    
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: FIX DATE/TIME COLUMNS")
    print("=" * 80)
    step_start = time.time()
    
    # Build lookup dictionaries once
    planned_lookup = build_validation_lookup_dict(validation_data, PLANNED_COLS)
    actual_lookup = build_validation_lookup_dict(validation_data, ACTUAL_COLS)
    print(f"Built lookup dicts: {len(planned_lookup)} planned, {len(actual_lookup)} actual")
    
    if HAS_TQDM:
        tqdm.pandas(desc="Fixing dates")
        data = data.progress_apply(
            lambda r: attach_dates_to_time_only_strings(r, PLANNED_COLS, ACTUAL_COLS, planned_lookup, actual_lookup), 
            axis=1
        )
    else:
        data = data.apply(
            lambda r: attach_dates_to_time_only_strings(r, PLANNED_COLS, ACTUAL_COLS, planned_lookup, actual_lookup), 
            axis=1
        )
    
    print("Attached dates to time-only strings")
    print("Validated all dates are within 2016-2022 range")
    
    # Convert all datetime columns to proper datetime dtype
    for col in PLANNED_COLS + ACTUAL_COLS:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    print("Converted all datetime columns to datetime64[ns]")
    
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: SIMPLE COLUMN SYNCING")
    print("=" * 80)
    step_start = time.time()
    
    # Sync patient call/fetch times
    if "PLANNED_PATIENT_CALL_TIME" in data.columns and "PLANNED_PATIENT_FETCH_TIME" in data.columns:
        data = sync_columns_vectorized(data, "PLANNED_PATIENT_CALL_TIME", "PLANNED_PATIENT_FETCH_TIME")
        print("Synced PLANNED_PATIENT_CALL_TIME ↔ PLANNED_PATIENT_FETCH_TIME")
    
    # Sync cleanup/exit times
    if "PLANNED_OR_CLEANUP_TIME" in data.columns and "PLANNED_EXIT_OR_TIME" in data.columns:
        data = sync_columns_vectorized(data, "PLANNED_OR_CLEANUP_TIME", "PLANNED_EXIT_OR_TIME")
        print("Synced PLANNED_OR_CLEANUP_TIME ↔ PLANNED_EXIT_OR_TIME")
    
    # Copy ACTUAL_RECEPTION_IN_TIME to PATIENT_CALL_TIME if missing
    if "PATIENT_CALL_TIME" in data.columns and "ACTUAL_RECEPTION_IN_TIME" in data.columns:
        mask = data["PATIENT_CALL_TIME"].isna() & data["ACTUAL_RECEPTION_IN_TIME"].notna()
        data.loc[mask, "PATIENT_CALL_TIME"] = data.loc[mask, "ACTUAL_RECEPTION_IN_TIME"]
        print(f"Filled {mask.sum()} missing PATIENT_CALL_TIME from ACTUAL_RECEPTION_IN_TIME")
    
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: HANDLE ADMISSION-RELATED COLUMNS")
    print("=" * 80)
    step_start = time.time()
    
    admission_cols = [
        "ADMISSION_STATUS", "ADMISSION_CLASS_TYPE", 
        "ADMISSION_TYPE", "ADMISSION_WARD", "ADMISSION_BED"
    ]
    for col in admission_cols:
        if col in data.columns:
            data[col].fillna("Not Admitted", inplace=True)
    print(f"Filled missing admission columns with 'Not Admitted'")
    
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 7: HANDLE CLINICAL STAFF COLUMNS")
    print("=" * 80)
    step_start = time.time()
    
    clinician_cols = ["SURGEON", "ANAESTHETIST_TEAM", "ANAESTHETIST_MCR_NO"]
    for col in clinician_cols:
        if col in data.columns:
            data[col].fillna("Unknown", inplace=True)
    print(f"Filled missing clinician columns with 'Unknown'")
    
    if "DIAGNOSIS" in data.columns:
        data["DIAGNOSIS"].fillna("Not Recorded", inplace=True)
        print("Filled missing DIAGNOSIS with 'Not Recorded'")
    
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 8: SAVE STANDARDIZED DATASET")
    print("=" * 80)
    step_start = time.time()
    
    data.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved {len(data):,} rows to {OUTPUT_FILE}")
    
    print(f"Step completed in {time.time() - step_start:.1f}s")
    
    # ========================================================================
    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("STANDARDIZATION COMPLETE")
    print("=" * 80)
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Final row count: {len(data):,}")
    print(f"Final column count: {len(data.columns)}")
    print(f"\n⏱  TOTAL RUNTIME: {total_time/60:.1f} minutes ({total_time:.1f}s)")
    print("=" * 80)

if __name__ == "__main__":
    main()