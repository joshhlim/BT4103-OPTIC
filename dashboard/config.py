"""Configuration and constants"""
import plotly.express as px

COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17becf',
    'ontime': '#66c2a5',
    'late': '#fc8d62',
    'early': '#8da0cb',
}

QUALITATIVE_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

px.defaults.color_discrete_sequence = QUALITATIVE_COLORS

FONT = dict(family="Inter, Arial, Helvetica, sans-serif", size=12, color='#e8e8e8')

COLS = {
    'room': 'ROOM',
    'status': 'CASE_STATUS',
    'discipline': 'DISCIPLINE',
    'surgeon': 'SURGEON',
    'emergency': 'EMERGENCY_PRIORITY',
    'room_start': 'ACTUAL_ENTER_OR_TIME',
    'room_end': 'ACTUAL_EXIT_OR_TIME',
    'surg_start': 'ACTUAL_KNIFE_TO_SKIN_TIME',
    'surg_end': 'ACTUAL_SKIN_CLOSURE',
    'actual_dur': 'ACTUAL_SURGERY_DURATION',
    'planned_dur': 'PLANNED_SURGERY_DURATION',
    'usage_dur': 'ACTUAL_USAGE_DURATION',
    'planned_usage': 'PLANNED_USAGE_DURATION',
    'knife_delay': 'KNIFE_START_DELAY',
}

TIME_COLS = [
    'ACTUAL_ENTER_OR_TIME', 'ACTUAL_EXIT_OR_TIME',
    'ACTUAL_KNIFE_TO_SKIN_TIME', 'ACTUAL_SKIN_CLOSURE',
    'PLANNED_ENTER_OR_TIME', 'PLANNED_EXIT_OR_TIME',
    'PLANNED_KNIFE_TO_SKIN_TIME', 'PLANNED_SKIN_CLOSURE',
    'PLANNED_PATIENT_CALL_TIME',
]

DARK_THEME_CSS = """
<style>
    .main {background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #e8e8e8;}
    .main p, .main span, .main div, .main label, .main h1, .main h2, .main h3 {color: #e8e8e8 !important;}
    .stTabs [data-baseweb="tab-list"] {gap: 4px; background-color: transparent;}
    .stTabs [data-baseweb="tab"] {background-color: rgba(40, 40, 60, 0.6); border-radius: 4px 4px 0 0;
        padding: 12px 24px; font-weight: 600; color: #b8b8b8; border: 1px solid rgba(100, 100, 100, 0.3);}
    .stTabs [aria-selected="true"] {background-color: #1f77b4; color: #ffffff; border-color: #1f77b4;}
    .stExpander {background-color: rgba(30, 30, 45, 0.6); border-radius: 8px; border: 1px solid rgba(100, 100, 100, 0.3);}
    .stExpander summary {font-weight: 600; color: #e8e8e8 !important;}
    h3 {color: #ffffff !important; border-bottom: 2px solid #1f77b4; padding-bottom: 8px; margin-top: 2rem;}
    [data-testid="stSidebar"] {background: linear-gradient(180deg, #1e1e30 0%, #252538 100%);}
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span {color: #e8e8e8 !important;}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {color: #ffffff !important;}
</style>
"""