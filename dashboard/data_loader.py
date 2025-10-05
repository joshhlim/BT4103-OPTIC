"""Data loading and filtering"""
import io
from datetime import timedelta
from typing import Tuple, Dict
import pandas as pd
import numpy as np
import streamlit as st
from config import COLS, TIME_COLS

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