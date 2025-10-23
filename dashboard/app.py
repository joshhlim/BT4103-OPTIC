"""Main dashboard entry point"""

import hashlib
import os, sys
import uuid
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from datetime import timedelta
from config import DARK_THEME_CSS, COLS
from data_loader import (
    preprocess_upload,
    load_and_filter_data_from_df,
    FileTooLargeError,
    UnsupportedFileTypeError,
    CleaningFailedError,
    SchemaMismatchError,
)
from components.styling import metric_card
from tabs import overview, lateness, planning, duration, utilization

# ---------- helpers ----------
def _file_sig(uploaded_file) -> str:
    """Stable signature so we only clean when the file really changes."""
    raw = uploaded_file.getvalue()  # bytes
    h = hashlib.md5()
    h.update(uploaded_file.name.encode())
    h.update(str(len(raw)).encode())
    # avoid hashing huge file fully: hash first + last 1MB (still stable for changes)
    if len(raw) > 2_000_000:
        h.update(raw[:1_000_000])
        h.update(raw[-1_000_000:])
    else:
        h.update(raw)
    return f"{uploaded_file.name}|{len(raw)}|{h.hexdigest()}"


st.set_page_config(page_title="OT Efficiency", layout="wide", initial_sidebar_state="expanded")
st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Operating Theatre Efficiency")
st.sidebar.markdown("---")
st.sidebar.header("Data Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload dataset",
    type=['csv', 'gz', 'parquet', 'xlsx', 'xls'],
)

if not uploaded_file:
    st.title("Operating Theatre Efficiency Dashboard")
    st.markdown("""
    ### Welcome
    
    Upload your dataset to analyze:
    - **Lateness** - Identify delays and causes
    - **Planning** - Compare planned vs actual
    - **Duration** - Analyze surgery patterns
    - **Utilization** - Optimize resources
    """)
    st.stop()

# ---------- clean once per unique file ----------
sig = _file_sig(uploaded_file)
needs_clean = st.session_state.get("clean_sig") != sig

if needs_clean:
    with st.spinner("Cleaning your data..."):
        try:
            cleaned_df, downloads, clean_logs = preprocess_upload(uploaded_file)
        except FileTooLargeError as e:
            st.error("Cleaning failed"); st.markdown(f"- File is too large ({e}). Max allowed is 2 GB."); st.stop()
        except UnsupportedFileTypeError as e:
            st.error("Cleaning failed"); st.markdown(f"- Unsupported file type `{e}`. Allowed: csv, gz, parquet, xlsx, xls."); st.stop()
        except (SchemaMismatchError, CleaningFailedError) as e:
            st.error("Cleaning failed"); st.markdown(
                "- Columns may not match the cleaning script’s expected schema\n"
                "- Or an internal error occurred in the cleaning step\n\n"
                f"**Detail:** {type(e).__name__}: {e}"
            ); st.stop()
        except Exception as e:
            st.error("Cleaning failed"); st.markdown(f"- Unexpected error during cleaning\n\n**Detail:** {type(e).__name__}: {e}"); st.stop()

        # persist results so future reruns / widget interactions DON’T re-clean
        st.session_state["clean_sig"] = sig
        st.session_state["cleaned_df"] = cleaned_df
        st.session_state["downloads_csv"] = downloads["csv"]
        st.session_state["downloads_parquet"] = downloads["parquet"]

        st.success(f"Cleaning successful — rows: {len(cleaned_df):,} • columns: {len(cleaned_df.columns):,}")

# ---------- downloads never trigger cleaning ----------
c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "Download cleaned CSV",
        data=st.session_state["downloads_csv"],
        file_name="cleaned_dataset.csv",
        mime="text/csv",
        use_container_width=True
    )
with c2:
    pq = st.session_state["downloads_parquet"]
    if pq is not None:
        st.download_button(
            "Download cleaned Parquet",
            data=pq,
            file_name="cleaned_dataset.parquet",
            mime="application/octet-stream",
            use_container_width=True
        )
    else:
        st.caption("Parquet download unavailable (install `pyarrow`).")

# --- always materialize cleaned_df from session_state on every rerun ---
cleaned_df = st.session_state.get("cleaned_df")

if cleaned_df is None:
    # We haven't cleaned anything yet this session (or state was cleared).
    # Keep the page empty instead of crashing.
    st.info("Upload a file to begin. The dashboard will populate after cleaning.")
    st.stop()

# Load data
data, global_filters, late_threshold, on_time_band = load_and_filter_data_from_df(cleaned_df)

if data.empty:
    st.warning("No data matches filters.")
    st.stop()

# Global KPIs
st.title("Operating Theatre Efficiency Dashboard")

date_info = ""
if 'date_range' in global_filters:
    start_date = global_filters['date_range'][0].strftime('%d %b %Y')
    end_date = (global_filters['date_range'][1] - timedelta(days=1)).strftime('%d %b %Y')
    date_info = f"Analysis Period: {start_date} to {end_date}"
else:
    date_info = "Analysis Period: All Data"

st.caption(date_info)
st.markdown("")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    metric_card("Total Cases", f"{len(data):,}")

with col2:
    if COLS['actual_dur'] in data.columns:
        import pandas as pd
        avg = pd.to_numeric(data[COLS['actual_dur']], errors='coerce').mean()
        metric_card("Avg Duration", f"{avg:.0f} min")
    else:
        metric_card("Avg Duration", "N/A")

with col3:
    late_count = data['_is_late'].sum() if '_is_late' in data.columns else 0
    late_rate = (late_count / len(data) * 100) if len(data) > 0 else 0
    metric_card("Late Cases", f"{late_count:,}", f"Rate: {late_rate:.1f}%")

with col4:
    if COLS['status'] in data.columns:
        cancelled = (
            data[COLS['status']].astype(str).str.contains('cancel', case=False, na=False)
        ).sum()
        cancel_rate = (cancelled / len(data) * 100) if len(data) > 0 else 0
        metric_card("Cancellations", f"{cancelled:,}", f"Rate: {cancel_rate:.1f}%")
    else:
        metric_card("Cancellations", "N/A")

with col5:
    if COLS['room'] in data.columns:
        n_rooms = data[COLS['room']].nunique()
        metric_card("Active Rooms", f"{n_rooms}")
    else:
        metric_card("Active Rooms", "N/A")

st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview", "Lateness", "Planning", "Duration", "Utilization"
])

with tab1:
    overview.render(data, global_filters, late_threshold, on_time_band)

with tab2:
    lateness.render(data, late_threshold, on_time_band)

with tab3:
    planning.render(data, on_time_band)

with tab4:
    duration.render(data, on_time_band)

with tab5:
    utilization.render(data)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #b8b8b8; padding: 20px;'>
    <p><strong>Operating Theatre Efficiency Dashboard</strong></p>
    <p style='font-size: 0.85rem;'>Optimize scheduling • Reduce delays • Maximize utilization</p>
</div>
""", unsafe_allow_html=True)