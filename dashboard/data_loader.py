"""Data loading and filtering"""
import contextlib
import importlib
import io
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from datetime import timedelta
import runpy
import sys
import traceback
from typing import Optional, Tuple, Dict
import pandas as pd
import numpy as np
import streamlit as st
from config import COLS, TIME_COLS
import tempfile
import hashlib
from data_cleaning_dashboard import clean_in_memory


MAX_BYTES = 2 * 1024 * 1024 * 1024
SUPPORTED_TYPES = {"csv", "gz", "parquet", "xlsx", "xls"}

# Typed errors for the UI layer to catch & message nicely
class FileTooLargeError(Exception): ...
class UnsupportedFileTypeError(Exception): ...
class CleaningFailedError(Exception): ...
class SchemaMismatchError(Exception): ...

def _file_ext(name: str) -> str:
    return (name.rsplit(".", 1)[-1] if "." in name else "").lower()

def validate_upload(uploaded_file) -> None:
    ext = _file_ext(uploaded_file.name)
    if ext not in SUPPORTED_TYPES:
        raise UnsupportedFileTypeError(f"{ext}")
    if uploaded_file.size > MAX_BYTES:
        raise FileTooLargeError(f"~{uploaded_file.size / (1024**3):.2f} GB")

def validate_cleaned_schema(cleaned_df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [c for c in required_cols if c not in cleaned_df.columns]
    if missing:
        raise SchemaMismatchError(f"Missing required columns after cleaning: {missing}")

    
def normalize_upload_to_df(uploaded_file) -> pd.DataFrame:
    """
    Unified reader for csv/gz/parquet/xlsx/xls -> DataFrame.
    IMPORTANT: read EVERYTHING as strings (dtype=str) and avoid date parsing.
               Cleaning will handle types/casting later.
    """
    name = uploaded_file.name.lower()
    buf = io.BytesIO(uploaded_file.getbuffer())

    if name.endswith(".csv"):
        buf.seek(0)
        return pd.read_csv(buf, dtype=str, low_memory=False)  # <-- strings only
    if name.endswith(".gz"):
        buf.seek(0)
        return pd.read_csv(buf, dtype=str, low_memory=False, compression="gzip")
    if name.endswith(".parquet"):
        buf.seek(0)
        # Parquet has a schema already; to be consistent, cast to string (optional)
        df = pd.read_parquet(buf)
        return df.astype(str)
    if name.endswith((".xlsx", ".xls")):
        buf.seek(0)
        # dtype=str works on pandas >= 1.5 for read_excel
        return pd.read_excel(buf, dtype=str)
    raise UnsupportedFileTypeError(_file_ext(name))



def _df_to_csv_buf(df: pd.DataFrame) -> io.BytesIO:
    b = io.BytesIO()
    df.to_csv(b, index=False)
    b.seek(0)
    return b

def _df_to_parquet_buf(df: pd.DataFrame) -> Optional[io.BytesIO]:
    try:
        b = io.BytesIO()
        df.to_parquet(b, index=False)  # requires pyarrow
        b.seek(0)
        return b
    except Exception:
        return None

def _hash_bytes(name: str, size: int, raw: memoryview) -> str:
    m = hashlib.md5()
    m.update(name.encode("utf-8"))
    m.update(str(size).encode("utf-8"))
    m.update(raw)
    return m.hexdigest()


# Cleaning bridge to Data_Cleaning.py
CLEANER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "Data_Preparation",
    "Data_Cleaning.py",
)

def run_cleaning(df_in: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Directly run ../Data_Preparation/Data_Cleaning.py.
    Injects INPUT_FILE and OUTPUT_FILE globals and captures stdout/stderr.
    Returns (cleaned_df, logs_text).
    """
    log_buf = io.StringIO()
    cleaner_path = CLEANER_PATH
    cleaner_dir = os.path.dirname(cleaner_path)

    if not os.path.exists(cleaner_path):
        raise CleaningFailedError(
            f"Expected cleaner at {cleaner_path}, but file was not found.\n"
            "Ensure Data_Cleaning.py exists under ../Data_Preparation/ relative to dashboard/."
        )

    with tempfile.TemporaryDirectory() as tmpdir, \
         contextlib.redirect_stdout(log_buf), \
         contextlib.redirect_stderr(log_buf):

        in_path  = os.path.join(tmpdir, "input.csv")
        out_path = os.path.join(tmpdir, "cleaned.csv")
        df_in.to_csv(in_path, index=False)

        prev_cwd = os.getcwd()
        sys_path_added = False

        try:
            # Switch to cleaner dir so relative imports/files in the cleaner work
            os.chdir(cleaner_dir)
            if cleaner_dir not in sys.path:
                sys.path.insert(0, cleaner_dir)
                sys_path_added = True

            print(f"[data_loader] Running cleaner: {cleaner_path}")
            runpy.run_path(
                cleaner_path,
                run_name="__main__",  # emulate: python Data_Cleaning.py
                init_globals={
                    "INPUT_FILE": in_path,
                    "OUTPUT_FILE": out_path,
                },
            )

            if not os.path.exists(out_path):
                raise CleaningFailedError(
                    f"Cleaner ran but did not produce an output file.\nExpected at: {out_path}"
                )

            cleaned = pd.read_csv(out_path)

        except Exception as e:
            tb = traceback.format_exc()
            raise CleaningFailedError(f"{type(e).__name__}: {e}\n{tb}")

        finally:
            os.chdir(prev_cwd)
            if sys_path_added:
                try:
                    sys.path.remove(cleaner_dir)
                except ValueError:
                    pass

    logs = log_buf.getvalue()
    if cleaned.empty:
        raise CleaningFailedError("Cleaning produced an empty dataset.")
    return cleaned, logs

# To be called by app.py
def preprocess_upload(uploaded_file):
    validate_upload(uploaded_file)
    raw_df = normalize_upload_to_df(uploaded_file)  # you already read as strings

    # Clean entirely in memory (no temp files, no runpy)
    cleaned_df, legend_df = clean_in_memory(raw_df)

    # (Optional) schema check against COLS
    required = [v for v in COLS.values() if isinstance(v, str)]
    if required:
        validate_cleaned_schema(cleaned_df, required)

    downloads = {
        "csv": _df_to_csv_buf(cleaned_df),
        "parquet": _df_to_parquet_buf(cleaned_df),
    }
    # You can also expose legend_df for download if you want.
    return cleaned_df, downloads, ""  # logs omitted intentionally

# Preprocess_upload with progress tracking
def preprocess_upload_with_progress(uploaded_file, progress_bar=None, status_text=None):
    """
    Enhanced version with progress tracking for Streamlit UI.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        progress_bar: Streamlit progress bar widget (optional)
        status_text: Streamlit text widget for status messages (optional)
        
    Returns:
        (cleaned_df, downloads, logs)
    """
    TOTAL_STEPS = 13
    
    # Update status helper
    def update_status(step: int, message: str):
        if progress_bar:
            progress = step / TOTAL_STEPS
            progress_bar.progress(progress)
        if status_text:
            status_text.text(f"Step {step}/{TOTAL_STEPS}: {message}")
    
    # Step 0: Validation
    update_status(0, "Validating uploaded file...")
    validate_upload(uploaded_file)
    
    # Step 0.5: Reading file
    update_status(0, "Reading uploaded file...")
    raw_df = normalize_upload_to_df(uploaded_file)
    
    # Create progress callback for cleaning
    def progress_callback(step: int, message: str):
        update_status(step, message)
    
    # Run cleaning with progress updates
    cleaned_df, legend_df = clean_in_memory(
        raw_df,
        progress_callback=progress_callback
    )
    
    # Final validation
    update_status(TOTAL_STEPS, "Finalizing and validating schema...")
    required = [v for v in COLS.values() if isinstance(v, str)]
    if required:
        validate_cleaned_schema(cleaned_df, required)

    # Prepare downloads
    update_status(TOTAL_STEPS, "Preparing download files...")
    downloads = {
        "csv": _df_to_csv_buf(cleaned_df),
        "parquet": _df_to_parquet_buf(cleaned_df),
    }
    
    return cleaned_df, downloads, ""

@st.cache_data(show_spinner=False)
def load_data_sample(file_bytes: bytes, name: str, n_rows: int = 50000):
    """Load sample for filter building"""
    with io.BytesIO(file_bytes) as tmp:
        if name.lower().endswith(('.csv', '.csv.gz', '.gz')):
            compression = 'gzip' if name.lower().endswith('.gz') else None
            return pd.read_csv(tmp, nrows=n_rows, compression=compression)
        elif name.lower().endswith('.parquet'):
            return pd.read_parquet(tmp)
        else:
            return pd.read_excel(tmp, nrows=n_rows)

@st.cache_data(show_spinner=True)
def load_data_full(file_bytes: bytes, name: str):
    """Load full dataset"""
    with io.BytesIO(file_bytes) as tmp:
        if name.lower().endswith(('.csv', '.csv.gz', '.gz')):
            compression = 'gzip' if name.lower().endswith('.gz') else None
            return pd.read_csv(tmp, compression=compression)
        elif name.lower().endswith('.parquet'):
            return pd.read_parquet(tmp)
        else:
            return pd.read_excel(tmp)

def parse_datetime_columns(df, cols):
    """Parse datetime columns in place"""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

def apply_filters(df, filters: Dict):
    """Apply global filters"""
    result = df.copy()
    
    if 'date_range' in filters and filters['date_col'] in result.columns:
        date_series = pd.to_datetime(result[filters['date_col']], errors='coerce')
        start, end = filters['date_range']
        result = result[(date_series >= start) & (date_series < end)]
    
    for col, key in [
        ('ROOM', 'rooms'),
        ('DISCIPLINE', 'disciplines'),
        ('SURGEON', 'surgeons'),
        ('CASE_STATUS', 'statuses')
    ]:
        if key in filters and filters[key] and col in result.columns:
            result = result[result[col].astype(str).isin(filters[key])]
    
    if 'emergency_mode' in filters and filters['emergency_mode'] != 'All':
        if 'EMERGENCY_PRIORITY' in result.columns:
            if filters['emergency_mode'] == 'Emergency only':
                mask = result['EMERGENCY_PRIORITY'].astype(str).str.contains(
                    'emerg', case=False, na=False
                )
            else:
                mask = ~result['EMERGENCY_PRIORITY'].astype(str).str.contains(
                    'emerg', case=False, na=False
                )
            result = result[mask]
    
    return result.reset_index(drop=True)


# Legacy data loader which loads raw file into Dashboard
def load_and_filter_data(uploaded_file) -> Tuple[pd.DataFrame, Dict, int, int]:
    """Main data loading function"""
    file_name = uploaded_file.name
    
    # Load sample for filters
    with st.spinner("Loading data..."):
        sample_df = load_data_sample(uploaded_file.getvalue(), file_name)
        parse_datetime_columns(sample_df, TIME_COLS)
    
    # Build global filters
    st.sidebar.header("Global Filters")
    st.sidebar.caption("Applied across all tabs")
    
    date_basis = st.sidebar.radio(
        "Date basis for filtering",
        ["Surgery Start Time", "Booking Date"]
    )
    
    date_col = COLS['surg_start'] if date_basis == "Surgery Start Time" else 'PLANNED_PATIENT_CALL_TIME'
    if date_col not in sample_df.columns:
        date_col = [c for c in TIME_COLS if c in sample_df.columns][0]
    
    global_filters = {'date_col': date_col}
    
    # Date range
    if date_col in sample_df.columns:
        date_series = pd.to_datetime(sample_df[date_col], errors='coerce')
        min_date = date_series.min()
        max_date = date_series.max()
        
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.sidebar.date_input(
                "Date Range",
                value=(min_date.date(), max_date.date())
            )
            if len(date_range) == 2:
                start = pd.to_datetime(date_range[0])
                end = pd.to_datetime(date_range[1]) + timedelta(days=1)
                global_filters['date_range'] = (start, end)
    
    # Status filter
    if COLS['status'] in sample_df.columns:
        status_options = sorted(sample_df[COLS['status']].dropna().unique())
        selected_status = st.sidebar.multiselect(
            "Case Status",
            status_options,
            default=status_options
        )
        if selected_status and len(selected_status) < len(status_options):
            global_filters['statuses'] = selected_status
    
    # Emergency filter
    if 'EMERGENCY_PRIORITY' in sample_df.columns:
        emerg_mode = st.sidebar.selectbox(
            "Emergency Classification",
            ["All Cases", "Emergency Only", "Elective Only"]
        )
        if emerg_mode != "All Cases":
            global_filters['emergency_mode'] = emerg_mode.replace(
                " Cases", ""
            ).replace(" Only", " only")
    
    st.sidebar.markdown("---")
    st.sidebar.header("Analysis Parameters")
    
    late_threshold = st.sidebar.slider("Late Threshold (minutes)", 0, 30, 10, 1)
    on_time_band = st.sidebar.slider("On-Time Tolerance (±minutes)", 0, 30, 5, 1)
    
    # Load full data
    with st.spinner("Loading and filtering data..."):
        data = load_data_full(uploaded_file.getvalue(), file_name)
        parse_datetime_columns(data, TIME_COLS)
        data = apply_filters(data, global_filters)
    
    # Compute derived columns
    if COLS['knife_delay'] in data.columns:
        data['_knife_delay_num'] = pd.to_numeric(data[COLS['knife_delay']], errors='coerce')
        data['_is_late'] = (data['_knife_delay_num'] > late_threshold).astype(int)
    else:
        data['_knife_delay_num'] = np.nan
        data['_is_late'] = 0
    
    if COLS['actual_dur'] in data.columns and COLS['planned_dur'] in data.columns:
        if 'DIFF_SURGERY_DURATION' not in data.columns:
            data['DIFF_SURGERY_DURATION'] = (
                pd.to_numeric(data[COLS['actual_dur']], errors='coerce') -
                pd.to_numeric(data[COLS['planned_dur']], errors='coerce')
            )
    
    if COLS['usage_dur'] in data.columns and COLS['planned_usage'] in data.columns:
        if 'DIFF_USAGE_DURATION' not in data.columns:
            data['DIFF_USAGE_DURATION'] = (
                pd.to_numeric(data[COLS['usage_dur']], errors='coerce') -
                pd.to_numeric(data[COLS['planned_usage']], errors='coerce')
            )
    
    return data, global_filters, late_threshold, on_time_band


# New Helper to load cleaned df
def load_and_filter_data_from_df(cleaned_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict, int, int]:
    """
    Same output as load_and_filter_data(...), but starts from an already-cleaned DataFrame.
    Avoids any re-reading of the raw file.
    """
    # Sample for sidebar controls
    sample_df = cleaned_df.copy()
    parse_datetime_columns(sample_df, TIME_COLS)

    # ----- Build global filters (same UI as original) -----
    st.sidebar.header("Global Filters")
    st.sidebar.caption("Applied across all tabs")

    date_basis = st.sidebar.radio(
        "Date basis for filtering",
        ["Surgery Start Time", "Booking Date"]
    )

    date_col = COLS['surg_start'] if date_basis == "Surgery Start Time" else 'PLANNED_PATIENT_CALL_TIME'
    if date_col not in sample_df.columns:
        avail_time_cols = [c for c in TIME_COLS if c in sample_df.columns]
        date_col = avail_time_cols[0] if avail_time_cols else None

    global_filters: Dict = {'date_col': date_col} if date_col else {}

    # Date range widget
    if date_col and date_col in sample_df.columns:
        date_series = pd.to_datetime(sample_df[date_col], errors='coerce')
        min_date = date_series.min()
        max_date = date_series.max()
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.sidebar.date_input(
                "Date Range",
                value=(min_date.date(), max_date.date())
            )
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                start = pd.to_datetime(date_range[0])
                end = pd.to_datetime(date_range[1]) + timedelta(days=1)
                global_filters['date_range'] = (start, end)

    # Status filter
    if COLS['status'] in sample_df.columns:
        status_options = sorted(sample_df[COLS['status']].dropna().unique())
        selected_status = st.sidebar.multiselect(
            "Case Status",
            status_options,
            default=status_options
        )
        if selected_status and len(selected_status) < len(status_options):
            global_filters['statuses'] = selected_status

    # Emergency filter
    if 'EMERGENCY_PRIORITY' in sample_df.columns:
        emerg_mode = st.sidebar.selectbox(
            "Emergency Classification",
            ["All Cases", "Emergency Only", "Elective Only"]
        )
        if emerg_mode != "All Cases":
            global_filters['emergency_mode'] = emerg_mode.replace(" Cases", "").replace(" Only", " only")

    st.sidebar.markdown("---")
    st.sidebar.header("Analysis Parameters")
    late_threshold = st.sidebar.slider("Late Threshold (minutes)", 0, 30, 10, 1)
    on_time_band  = st.sidebar.slider("On-Time Tolerance (±minutes)", 0, 30, 5, 1)

    # Use the cleaned data only 
    data = cleaned_df.copy()
    parse_datetime_columns(data, TIME_COLS)
    data = apply_filters(data, global_filters)

    # Derived columns 
    if COLS['knife_delay'] in data.columns:
        data['_knife_delay_num'] = pd.to_numeric(data[COLS['knife_delay']], errors='coerce')
        data['_is_late'] = (data['_knife_delay_num'] > late_threshold).astype(int)
    else:
        data['_knife_delay_num'] = np.nan
        data['_is_late'] = 0

    if COLS['actual_dur'] in data.columns and COLS['planned_dur'] in data.columns:
        if 'DIFF_SURGERY_DURATION' not in data.columns:
            data['DIFF_SURGERY_DURATION'] = (
                pd.to_numeric(data[COLS['actual_dur']], errors='coerce')
                - pd.to_numeric(data[COLS['planned_dur']], errors='coerce')
            )

    if COLS['usage_dur'] in data.columns and COLS['planned_usage'] in data.columns:
        if 'DIFF_USAGE_DURATION' not in data.columns:
            data['DIFF_USAGE_DURATION'] = (
                pd.to_numeric(data[COLS['usage_dur']], errors='coerce')
                - pd.to_numeric(data[COLS['planned_usage']], errors='coerce')
            )

    return data, global_filters, late_threshold, on_time_band