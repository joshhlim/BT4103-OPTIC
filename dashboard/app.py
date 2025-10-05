"""Main dashboard entry point"""
import streamlit as st
from datetime import timedelta
from config import DARK_THEME_CSS, COLS
from data_loader import load_and_filter_data
from components.styling import metric_card
from tabs import overview, lateness, planning, duration, utilization

st.set_page_config(page_title="OT Efficiency", layout="wide", initial_sidebar_state="expanded")
st.markdown(DARK_THEME_CSS, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Operating Theatre Efficiency")
st.sidebar.markdown("---")
st.sidebar.header("Data Upload")

uploaded_file = st.sidebar.file_uploader(
    "Upload dataset",
    type=['csv', 'gz', 'parquet', 'xlsx', 'xls']
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

# Load data
data, global_filters, late_threshold, on_time_band = load_and_filter_data(uploaded_file)

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