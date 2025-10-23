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
from typing import Tuple, Optional
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
    out = df.copy()
    for col in list(PLANNED_COLS) + list(ACTUAL_COLS):
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    return out


def _create_feature_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Create duration/delay features and convert timedeltas to minutes."""
    out = df.copy()

    # Surgery duration (knife → closure)
    if "ACTUAL_SKIN_CLOSURE" in out.columns and "ACTUAL_KNIFE_TO_SKIN_TIME" in out.columns:
        out["ACTUAL_SURGERY_DURATION"] = out["ACTUAL_SKIN_CLOSURE"] - out["ACTUAL_KNIFE_TO_SKIN_TIME"]
    if "PLANNED_SKIN_CLOSURE" in out.columns and "PLANNED_KNIFE_TO_SKIN_TIME" in out.columns:
        out["PLANNED_SURGERY_DURATION"] = out["PLANNED_SKIN_CLOSURE"] - out["PLANNED_KNIFE_TO_SKIN_TIME"]
    if "ACTUAL_SURGERY_DURATION" in out.columns and "PLANNED_SURGERY_DURATION" in out.columns:
        out["DIFF_SURGERY_DURATION"] = out["ACTUAL_SURGERY_DURATION"] - out["PLANNED_SURGERY_DURATION"]

    # OR usage duration (enter OR → exit OR)
    if "ACTUAL_EXIT_OR_TIME" in out.columns and "ACTUAL_ENTER_OR_TIME" in out.columns:
        out["ACTUAL_USAGE_DURATION"] = out["ACTUAL_EXIT_OR_TIME"] - out["ACTUAL_ENTER_OR_TIME"]
    if "PLANNED_EXIT_OR_TIME" in out.columns and "PLANNED_ENTER_OR_TIME" in out.columns:
        out["PLANNED_USAGE_DURATION"] = out["PLANNED_EXIT_OR_TIME"] - out["PLANNED_ENTER_OR_TIME"]
    if "ACTUAL_USAGE_DURATION" in out.columns and "PLANNED_USAGE_DURATION" in out.columns:
        out["DIFF_USAGE_DURATION"] = out["ACTUAL_USAGE_DURATION"] - out["PLANNED_USAGE_DURATION"]

    # Start delays
    if "ACTUAL_ENTER_OR_TIME" in out.columns and "PLANNED_ENTER_OR_TIME" in out.columns:
        out["ENTER_START_DELAY"] = out["ACTUAL_ENTER_OR_TIME"] - out["PLANNED_ENTER_OR_TIME"]
    if "ACTUAL_KNIFE_TO_SKIN_TIME" in out.columns and "PLANNED_KNIFE_TO_SKIN_TIME" in out.columns:
        out["KNIFE_START_DELAY"] = out["ACTUAL_KNIFE_TO_SKIN_TIME"] - out["PLANNED_KNIFE_TO_SKIN_TIME"]
    if "ACTUAL_EXIT_OR_TIME" in out.columns and "PLANNED_EXIT_OR_TIME" in out.columns:
        out["EXIT_OR_DELAY"] = out["ACTUAL_EXIT_OR_TIME"] - out["PLANNED_EXIT_OR_TIME"]

    # Convert timedelta features to minutes
    td_cols = [
        "ACTUAL_SURGERY_DURATION", "PLANNED_SURGERY_DURATION", "DIFF_SURGERY_DURATION",
        "ACTUAL_USAGE_DURATION", "PLANNED_USAGE_DURATION", "DIFF_USAGE_DURATION",
        "ENTER_START_DELAY", "KNIFE_START_DELAY", "EXIT_OR_DELAY",
    ]
    for c in td_cols:
        if c in out.columns:
            out[c] = out[c].dt.total_seconds() / 60.0

    return out


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
    save_legend_path: Optional[str] = None
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Run the full cleaning pipeline in memory (no printing, no I/O).
    Returns (cleaned_df, legend_df_or_None).
    If save_legend_path is provided, the legend CSV is also written to that path.
    """
    data = df_in.copy()

    # 1) Parse datetimes early (only known time columns)
    data = _parse_known_datetimes(data)

    # 2) Drop unnecessary cols
    data = _drop_unnecessary_columns(data)

    # 3) Build operation legend; drop 'NATURE'
    legend_df: Optional[pd.DataFrame] = None
    if "NATURE" in data.columns and "SURGICAL_CODE" in data.columns:
        data, legend_df = build_operation_legend(data, code_col="SURGICAL_CODE", nature_col="NATURE")
        if save_legend_path:
            os.makedirs(os.path.dirname(save_legend_path) or ".", exist_ok=True)
            legend_df.to_csv(save_legend_path, index=False)

    # 4) Clean text columns where available
    if "EQUIPMENT" in data.columns:
        data = clean_equipment_vectorized(data, col="EQUIPMENT")
    if "IMPLANT" in data.columns:
        data = clean_implant_vectorized(data, col="IMPLANT")
    if "DIAGNOSIS" in data.columns:
        data = clean_diagnosis_vectorized(data, col="DIAGNOSIS")

    # 5) Planned timeline: enforce ordering + impute from knife/closure
    col_pairs = [
        ("PLANNED_KNIFE_TO_SKIN_TIME", "PLANNED_SKIN_CLOSURE"),
        ("PLANNED_SKIN_CLOSURE", "PLANNED_PATIENT_REVERSAL_TIME"),
        ("PLANNED_PATIENT_REVERSAL_TIME", "PLANNED_EXIT_OR_TIME"),
        ("PLANNED_EXIT_OR_TIME", "PLANNED_EXIT_RECOVERY_TIME"),
        ("PLANNED_EXIT_RECOVERY_TIME", "PLANNED_OR_CLEANUP_TIME"),
    ]
    data = enforce_ordering_vectorized(data, col_pairs)
    data = impute_planned_from_knife_vectorized(data)

    # 6) Actual timeline: impute fetch + impute using statistical marks
    data = impute_patient_fetch_time_vectorized(data)
    marks = compute_statistical_marks(data)
    data = impute_induction_prep_reversal_cleanup_vectorized(data, marks)

    # 7) Validate timelines and filter invalid rows
    # Validate non-decreasing order (vectorized)
    planned_valid_order = validate_datetime_order_vectorized(data, PLANNED_COLS)
    actual_valid_order = validate_datetime_order_vectorized(data, ACTUAL_COLS)
    
    order_valid_mask = planned_valid_order & actual_valid_order
    num_invalid_order = (~order_valid_mask).sum()
    data = data[order_valid_mask].copy()
    print(f"✓ Removed {num_invalid_order:,} rows with non-decreasing validation failures")

    # Validate duration (72 hours max) (vectorized)
    planned_valid_duration = validate_duration_vectorized(data, PLANNED_COLS, max_hours=72)
    actual_valid_duration = validate_duration_vectorized(data, ACTUAL_COLS, max_hours=72)
    
    duration_valid_mask = planned_valid_duration & actual_valid_duration
    num_invalid_duration = (~duration_valid_mask).sum()
    data = data[duration_valid_mask].copy()
    print(f"✓ Removed {num_invalid_duration:,} rows exceeding 72-hour duration")

    # 8) Scope & duplicates
    data = _remove_out_of_scope_and_dupes(data)

    # 9) Delay reason taxonomy + late flags
    data = process_delay_reasons_vectorized(data)

    # 10) Features (minutes)
    data = _create_feature_variables(data)

    # 11) Drop rows with missing critical fields (only those present)
    data = _drop_missing_critical(data)

    # 12) Final dtype normalization
    data = convert_to_correct_dtypes(data)

    return data, legend_df