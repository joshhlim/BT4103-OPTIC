"""Styling components for charts and metrics"""
import streamlit as st
from config import FONT

def apply_chart_style(fig, title=None, legend=True, height=None):
    """Apply consistent dark theme to plotly figures"""
    layout_updates = {
        'template': 'plotly_dark',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': FONT,
        'margin': dict(l=60, r=30, t=90, b=60),
        'hoverlabel': dict(font=FONT, bgcolor='rgba(30,30,30,0.95)', bordercolor='#555'),
    }
    
    if title:
        layout_updates['title'] = {
            'text': f'<b>{title}</b>',
            'font': {'size': 18, 'color': '#e8e8e8'},
            'x': 0.02,
            'xanchor': 'left',
            'y': 0.98,
            'yanchor': 'top'
        }
    
    if legend:
        layout_updates['legend'] = dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(30,30,30,0.8)',
            bordercolor='#555',
            borderwidth=1,
            font=dict(color='#e8e8e8')
        )
    else:
        layout_updates['showlegend'] = False
    
    if height:
        layout_updates['height'] = height
    
    fig.update_layout(**layout_updates)
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(100,100,100,0.3)',
        linecolor='rgba(100,100,100,0.5)',
        color='#e8e8e8'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(100,100,100,0.3)',
        linecolor='rgba(100,100,100,0.5)',
        color='#e8e8e8'
    )
    return fig

def metric_card(label: str, value: str, help_text: str = None):
    """Create a styled metric card with professional solid background"""
    # Use a single professional color for all cards
    background_color = '#2c5282'  # Professional blue
    
    # Always include help_text div, but make it invisible if no text provided
    # This ensures consistent height across all cards
    if help_text:
        help_html = f"<p style='color: rgba(255,255,255,0.75); font-size: 0.8rem; margin: 8px 0 0 0; line-height: 1.3; min-height: 32px;'>{help_text}</p>"
    else:
        help_html = "<p style='min-height: 32px; margin: 8px 0 0 0;'>&nbsp;</p>"
    
    st.markdown(
        f"""
        <div style='background: {background_color}; padding: 20px; border-radius: 12px; 
                    box-shadow: 0 4px 12px rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1);
                    height: 180px; display: flex; flex-direction: column; justify-content: space-between;'>
            <div>
                <p style='color: rgba(255,255,255,0.85); font-size: 0.9rem; margin: 0 0 8px 0; 
                          font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;'>{label}</p>
                <p style='color: white; font-size: 2.2rem; margin: 0; font-weight: 700;
                          text-shadow: 0 2px 4px rgba(0,0,0,0.2);'>{value}</p>
            </div>
            {help_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def metric_row_with_info(info_text: str):
    """Create an info icon for a row of metrics"""
    with st.popover("ⓘ About These Metrics", use_container_width=False):
        st.markdown(info_text)

def chart_header_with_info(title: str, info_text: str):
    """Create a chart title with an expandable info section"""
    col1, col2 = st.columns([10, 1])
    with col1:
        st.markdown(f"#### {title}")
    with col2:
        with st.popover("ⓘ", use_container_width=False):
            st.caption(info_text)

def section_header(text: str, help_text: str = None):
    """Create a styled section header"""
    st.markdown(f"### {text}")
    if help_text:
        st.caption(help_text)
    st.markdown("")

def section_header_with_info(text: str, info_text: str):
    """Create a section header with an info popover"""
    col1, col2 = st.columns([10, 1])
    with col1:
        st.markdown(f"### {text}")
    with col2:
        with st.popover("ⓘ", use_container_width=False):
            st.markdown(info_text)
    st.markdown("")

def tab_intro(intro_text: str):
    """Create an introductory info box at the top of a tab"""
    st.info(intro_text)
    st.markdown("")