"""
Data_Cleaning_Dashboard.py

Streamlit-compatible cleaner:
- Accepts a DataFrame as input (df_in)
- Returns cleaned DataFrame + optional legend DataFrame + logs (as string)
- Reuses logic from Data_Cleaning.py
"""

import importlib
import os, sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
from typing import Callable, Tuple, Optional
# ---- load Data_Preparation/Data_Cleaning.py by file path ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANER_FILE = os.path.join(PROJECT_ROOT, "Data_Preparation", "Data_Cleaning.py")

spec = importlib.util.spec_from_file_location("Data_Cleaning_module", CLEANER_FILE)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load Data_Cleaning.py at {CLEANER_FILE}")
DC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(DC)  # type: ignore[attr-defined]
try:
    DC.USE_PARALLEL = False
    DC.N_CORES = 1
except Exception:
    pass

# ---- alias the pieces we need from the loaded module ----
build_operation_legend = DC.build_operation_legend
clean_equipment_vectorized = DC.clean_equipment_vectorized
clean_implant_vectorized = DC.clean_implant_vectorized
clean_diagnosis_vectorized = DC.clean_diagnosis_vectorized
enforce_ordering_vectorized = DC.enforce_ordering_vectorized
impute_planned_from_knife_vectorized = DC.impute_planned_from_knife_vectorized
impute_patient_fetch_time_vectorized = DC.impute_patient_fetch_time_vectorized
compute_statistical_marks = DC.compute_statistical_marks
impute_induction_prep_reversal_cleanup_vectorized = DC.impute_induction_prep_reversal_cleanup_vectorized
validate_datetime_order_vectorized = DC.validate_datetime_order_vectorized
validate_duration_vectorized = DC.validate_duration_vectorized
process_delay_reasons_vectorized = DC.process_delay_reasons_vectorized
convert_to_correct_dtypes = DC.convert_to_correct_dtypes

PLANNED_COLS = DC.PLANNED_COLS
ACTUAL_COLS = DC.ACTUAL_COLS
DROP_COLS = DC.DROP_COLS


def _drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove index column and anything listed in DROP_COLS, if present."""
    out = df.copy()
    if "Unnamed: 0" in out.columns:
        out = out.drop(columns=["Unnamed: 0"])
    drop_existing = [c for c in DROP_COLS if c in out.columns]
    if drop_existing:
        out = out.drop(columns=drop_existing)
    return out


def _parse_known_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce all known time columns to datetimes (errors → NaT)."""
    for col in list(PLANNED_COLS) + list(ACTUAL_COLS):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _create_feature_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create duration/delay features and convert timedeltas to minutes."""

    # Surgery duration (knife → closure)
    if "ACTUAL_SKIN_CLOSURE" in df.columns and "ACTUAL_KNIFE_TO_SKIN_TIME" in df.columns:
        df["ACTUAL_SURGERY_DURATION"] = df["ACTUAL_SKIN_CLOSURE"] - df["ACTUAL_KNIFE_TO_SKIN_TIME"]
    if "PLANNED_SKIN_CLOSURE" in df.columns and "PLANNED_KNIFE_TO_SKIN_TIME" in df.columns:
        df["PLANNED_SURGERY_DURATION"] = df["PLANNED_SKIN_CLOSURE"] - df["PLANNED_KNIFE_TO_SKIN_TIME"]
    if "ACTUAL_SURGERY_DURATION" in df.columns and "PLANNED_SURGERY_DURATION" in df.columns:
        df["DIFF_SURGERY_DURATION"] = df["ACTUAL_SURGERY_DURATION"] - df["PLANNED_SURGERY_DURATION"]

    # OR usage duration (enter OR → exit OR)
    if "ACTUAL_EXIT_OR_TIME" in df.columns and "ACTUAL_ENTER_OR_TIME" in df.columns:
        df["ACTUAL_USAGE_DURATION"] = df["ACTUAL_EXIT_OR_TIME"] - df["ACTUAL_ENTER_OR_TIME"]
    if "PLANNED_EXIT_OR_TIME" in df.columns and "PLANNED_ENTER_OR_TIME" in df.columns:
        df["PLANNED_USAGE_DURATION"] = df["PLANNED_EXIT_OR_TIME"] - df["PLANNED_ENTER_OR_TIME"]
    if "ACTUAL_USAGE_DURATION" in df.columns and "PLANNED_USAGE_DURATION" in df.columns:
        df["DIFF_USAGE_DURATION"] = df["ACTUAL_USAGE_DURATION"] - df["PLANNED_USAGE_DURATION"]

    # Start delays
    if "ACTUAL_ENTER_OR_TIME" in df.columns and "PLANNED_ENTER_OR_TIME" in df.columns:
        df["ENTER_START_DELAY"] = df["ACTUAL_ENTER_OR_TIME"] - df["PLANNED_ENTER_OR_TIME"]
    if "ACTUAL_KNIFE_TO_SKIN_TIME" in df.columns and "PLANNED_KNIFE_TO_SKIN_TIME" in df.columns:
        df["KNIFE_START_DELAY"] = df["ACTUAL_KNIFE_TO_SKIN_TIME"] - df["PLANNED_KNIFE_TO_SKIN_TIME"]
    if "ACTUAL_EXIT_OR_TIME" in df.columns and "PLANNED_EXIT_OR_TIME" in df.columns:
        df["EXIT_OR_DELAY"] = df["ACTUAL_EXIT_OR_TIME"] - df["PLANNED_EXIT_OR_TIME"]

    # Convert timedelta features to minutes
    td_cols = [
        "ACTUAL_SURGERY_DURATION", "PLANNED_SURGERY_DURATION", "DIFF_SURGERY_DURATION",
        "ACTUAL_USAGE_DURATION", "PLANNED_USAGE_DURATION", "DIFF_USAGE_DURATION",
        "ENTER_START_DELAY", "KNIFE_START_DELAY", "EXIT_OR_DELAY",
    ]
    for c in td_cols:
        if c in df.columns:
            df[c] = df[c].dt.total_seconds() / 60.0

    return df


def _remove_out_of_scope_and_dupes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "LOCATION" in out.columns:
        out = out[out["LOCATION"] != "OUT OF OT ROOMS"].copy()
    out = out.drop_duplicates()
    return out


def _drop_missing_critical(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing critical fields (only those that exist)."""
    out = df.copy()
    critical_cols = [
        "OPERATION_ID", "LOCATION", "ROOM", "CASE_STATUS", "OPERATION_TYPE",
        "SURGICAL_CODE", "DISCIPLINE", "ANESTHESIA", "AOH", "BLOOD",
        "CANCER_INDICATOR", "TRAUMA_INDICATOR",
    ]
    existing = [c for c in critical_cols if c in out.columns]
    if not existing:
        return out.dropna()
    out.dropna(subset=existing)
    return out.dropna()


def clean_in_memory(
    df_in: pd.DataFrame,
    *,
    save_legend_path: Optional[str] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Run the full cleaning pipeline in memory with progress tracking.
    
    Args:
        df_in: Input DataFrame
        save_legend_path: Optional path to save operation legend
        progress_callback: Optional callback function(step_num, step_name) for progress updates
        
    Returns:
        (cleaned_df, legend_df_or_None)
    """
    TOTAL_STEPS = 13
    
    def update_progress(step: int, message: str):
        """Helper to update progress if callback provided"""
        if progress_callback:
            progress_callback(step, message)
    
    data = df_in.copy()

    # Step 1: Parse datetimes
    update_progress(1, "Parsing datetime columns...")
    data = _parse_known_datetimes(data)

    # Step 2: Drop unnecessary columns
    update_progress(2, "Removing unnecessary columns...")
    data = _drop_unnecessary_columns(data)

    # Step 3: Build operation legend
    update_progress(3, "Building operation legend...")
    legend_df: Optional[pd.DataFrame] = None
    if "NATURE" in data.columns and "SURGICAL_CODE" in data.columns:
        data, legend_df = build_operation_legend(data, code_col="SURGICAL_CODE", nature_col="NATURE")
        if save_legend_path:
            os.makedirs(os.path.dirname(save_legend_path) or ".", exist_ok=True)
            legend_df.to_csv(save_legend_path, index=False)

    # Step 4: Clean text columns
    update_progress(4, "Cleaning text columns (equipment, implant, diagnosis)...")
    if "EQUIPMENT" in data.columns:
        data = clean_equipment_vectorized(data, col="EQUIPMENT")
    if "IMPLANT" in data.columns:
        data = clean_implant_vectorized(data, col="IMPLANT")
    if "DIAGNOSIS" in data.columns:
        data = clean_diagnosis_vectorized(data, col="DIAGNOSIS")

    # Step 5: Planned timeline imputation
    update_progress(5, "Imputing planned timeline...")
    col_pairs = [
        ("PLANNED_KNIFE_TO_SKIN_TIME", "PLANNED_SKIN_CLOSURE"),
        ("PLANNED_SKIN_CLOSURE", "PLANNED_PATIENT_REVERSAL_TIME"),
        ("PLANNED_PATIENT_REVERSAL_TIME", "PLANNED_EXIT_OR_TIME"),
        ("PLANNED_EXIT_OR_TIME", "PLANNED_EXIT_RECOVERY_TIME"),
        ("PLANNED_EXIT_RECOVERY_TIME", "PLANNED_OR_CLEANUP_TIME"),
    ]
    data = enforce_ordering_vectorized(data, col_pairs)
    data = impute_planned_from_knife_vectorized(data)

    # Step 6: Actual timeline imputation
    update_progress(6, "Imputing actual timeline...")
    data = impute_patient_fetch_time_vectorized(data)
    marks = compute_statistical_marks(data)
    data = impute_induction_prep_reversal_cleanup_vectorized(data, marks)

    # Step 7: Validate timeline ordering
    update_progress(7, "Validating timeline ordering...")
    planned_valid_order = validate_datetime_order_vectorized(data, PLANNED_COLS)
    actual_valid_order = validate_datetime_order_vectorized(data, ACTUAL_COLS)
    order_valid_mask = planned_valid_order & actual_valid_order
    num_invalid_order = (~order_valid_mask).sum()
    data = data[order_valid_mask].copy()

    # Step 8: Validate duration limits
    update_progress(8, "Validating duration limits (72h max)...")
    planned_valid_duration = validate_duration_vectorized(data, PLANNED_COLS, max_hours=72)
    actual_valid_duration = validate_duration_vectorized(data, ACTUAL_COLS, max_hours=72)
    duration_valid_mask = planned_valid_duration & actual_valid_duration
    num_invalid_duration = (~duration_valid_mask).sum()
    data = data[duration_valid_mask].copy()

    # Step 9: Remove out-of-scope and duplicates
    update_progress(9, "Removing out-of-scope records and duplicates...")
    data = _remove_out_of_scope_and_dupes(data)

    # Step 10: Process delay reasons
    update_progress(10, "Processing delay reasons and classifications...")
    data = process_delay_reasons_vectorized(data)

    # Step 11: Create feature variables
    update_progress(11, "Creating feature variables (durations, delays)...")
    data = _create_feature_variables(data)

    # Step 12: Drop rows with missing critical data
    update_progress(12, "Removing rows with missing critical data...")
    data = _drop_missing_critical(data)

    # Step 13: Convert to correct data types
    update_progress(13, "Converting to correct data types...")
    data = convert_to_correct_dtypes(data)

    return data, legend_df