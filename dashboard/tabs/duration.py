"""Duration analysis tab - COMPLETE"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import COLS, COLORS
from components.styling import (apply_chart_style, metric_card, section_header, 
                                chart_header_with_info, tab_intro, metric_row_with_info,
                                section_header_with_info)
from components.filters import add_tab_filters

def render(data, on_time_band):
    section_header(
        "Duration Analysis",
        "Analyze surgery duration patterns and identify factors affecting timing"
    )
    
    # Tab introduction
    tab_intro("""
    **Understanding Duration Analysis** - This tab helps you understand how long procedures actually take versus how long 
    they were planned to take. You can analyze two types of duration: (1) **Surgery Duration** measures from knife-to-skin 
    to skin-closure (the actual procedure time), and (2) **OR Usage Duration** measures from patient entering the OR to 
    exiting (includes setup and cleanup). Understanding duration patterns helps improve scheduling accuracy and identify 
    procedures that consistently take longer or shorter than expected.
    """)
    
    tab_data = add_tab_filters(data, 'duration')
    
    # Duration type selector with help text
    col1, col2 = st.columns([1, 3])
    with col1:
        duration_type = st.radio(
            "Duration Type",
            ["Surgery Duration", "OR Usage Duration"],
            help="Surgery Duration: knife-to-skin → skin-closure (procedure only). OR Usage Duration: enter OR → exit OR (includes all OR time)"
        )
    with col2:
        st.info("""
        **Which duration type should I choose?**
        - **Surgery Duration**: Best for analyzing surgeon performance and procedure complexity. Excludes setup/cleanup time.
        - **OR Usage Duration**: Best for capacity planning and scheduling. Includes all time the room is occupied.
        """)
    
    # Determine which duration column to use
    if duration_type == "Surgery Duration":
        dur_col = COLS['actual_dur']
        plan_col = COLS['planned_dur']
        diff_col = 'DIFF_SURGERY_DURATION'
    else:
        dur_col = COLS['usage_dur']
        plan_col = COLS['planned_usage']
        diff_col = 'DIFF_USAGE_DURATION'
    
    if dur_col in tab_data.columns:
        dur_data = tab_data.copy()
        dur_data['duration'] = pd.to_numeric(dur_data[dur_col], errors='coerce')
        dur_data = dur_data[dur_data['duration'] > 0].dropna(subset=['duration'])
        
        # Calculate diff if not exists
        if diff_col not in dur_data.columns and plan_col in dur_data.columns:
            dur_data[diff_col] = (
                pd.to_numeric(dur_data[dur_col], errors='coerce') -
                pd.to_numeric(dur_data[plan_col], errors='coerce')
            )
        
        if not dur_data.empty:
            # MOVED TO TOP: Overruns and underruns analysis
            if diff_col in dur_data.columns:
                section_header_with_info(
                    "Overruns and Underruns Analysis",
                    f"""
                    This section compares actual durations to planned durations to assess planning accuracy:
                    
                    - **Overrun**: Procedure took longer than planned (by more than {on_time_band} min tolerance)
                    - **Underrun**: Procedure finished earlier than planned (by more than {on_time_band} min tolerance)
                    - **Tolerance Band**: ±{on_time_band} min allows for minor variations without counting as over/underruns
                    
                    **Why this matters**: Excessive overruns cause downstream delays and reduce daily case throughput. 
                    Underruns represent wasted capacity and may indicate overly conservative planning. The ideal is to 
                    minimize both by improving planning accuracy.
                    """
                )
                
                diff_data = dur_data[diff_col].dropna()
                overruns = diff_data[diff_data > on_time_band]
                underruns = diff_data[diff_data < -on_time_band]
                
                metric_row_with_info("""
                **Understanding These Metrics:**
                
                - **Overrun Cases/Total Time**: Shows volume and cumulative impact of procedures taking longer than planned
                - **Underrun Cases/Total Time**: Shows volume and potential wasted capacity from finishing early
                - **Overrun/Underrun Rate**: Percentage of cases outside the tolerance band
                - **Idle Capacity**: Total hours lost due to procedures finishing earlier than expected - this could be used for additional cases
                """)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Overruns**")
                    metric_card("Cases", f"{len(overruns):,}")
                    metric_card("Total Time", f"{overruns.sum()/60:.0f} hrs")
                    metric_card("Median", f"{overruns.median():.0f} min" if len(overruns) > 0 else "N/A")
                
                with col2:
                    st.markdown("**Underruns**")
                    metric_card("Cases", f"{len(underruns):,}")
                    metric_card("Total Time", f"{abs(underruns.sum())/60:.0f} hrs")
                    metric_card("Median", f"{abs(underruns.median()):.0f} min" if len(underruns) > 0 else "N/A")
                
                with col3:
                    st.markdown("**Efficiency**")
                    overrun_pct = len(overruns) / len(diff_data) * 100 if len(diff_data) > 0 else 0
                    metric_card("Overrun Rate", f"{overrun_pct:.1f}%")
                    underrun_pct = len(underruns) / len(diff_data) * 100 if len(diff_data) > 0 else 0
                    metric_card("Underrun Rate", f"{underrun_pct:.1f}%")
                    
                    idle_time = abs(underruns.sum()) / 60
                    metric_card("Idle Capacity", f"{idle_time:.0f} hrs", "From underruns")
                
                st.markdown("---")
            
            # Summary statistics
            section_header("Duration Summary Statistics")
            
            metric_row_with_info("""
            **Statistical Measures Explained:**
            
            - **Mean**: Simple average of all durations. Affected by extreme values
            - **Median**: Middle value - half of procedures are shorter, half are longer. More resistant to outliers
            - **Std Dev** (Standard Deviation): Measures variability. Higher = more unpredictable durations. Lower = more consistent
            - **P90** (90th Percentile): 90% of procedures are shorter than this. Use for conservative planning
            - **P95** (95th Percentile): Only 5% of procedures exceed this duration. Helps identify worst-case scenarios
            """)
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                metric_card("Mean", f"{dur_data['duration'].mean():.0f} min")
            with col2:
                metric_card("Median", f"{dur_data['duration'].median():.0f} min")
            with col3:
                metric_card("Std Dev", f"{dur_data['duration'].std():.0f} min")
            with col4:
                metric_card("P90", f"{dur_data['duration'].quantile(0.90):.0f} min")
            with col5:
                metric_card("P95", f"{dur_data['duration'].quantile(0.95):.0f} min")
            
            st.markdown("---")
            
            # Overall distribution
            section_header("Duration Distribution")
            
            # Use 3:1 ratio for better balance
            col1, col2 = st.columns([3, 1])
            
            with col1:
                chart_header_with_info(
                    f"{duration_type} Distribution",
                    "This histogram shows the frequency distribution of procedure durations. The x-axis represents duration in minutes, while the y-axis shows the number of cases. The dashed line marks the median (50th percentile). A normal bell-shaped curve suggests predictable durations, while multiple peaks may indicate different procedure types. Use the slider to adjust the range and focus on specific duration windows."
                )
                
                max_dur = st.slider(
                    "Maximum duration to display (minutes)", 
                    100, 600, 400, 50, 
                    key='dur_max',
                    help="Adjust this slider to zoom into different duration ranges. Lower values focus on shorter procedures, higher values show the full range."
                )
                plot_data = dur_data[dur_data['duration'] <= max_dur]
                
                fig = px.histogram(
                    plot_data,
                    x='duration',
                    nbins=60,
                    color_discrete_sequence=[COLORS['primary']]
                )
                median_val = plot_data['duration'].median()
                fig.add_vline(
                    x=median_val,
                    line_dash="dash",
                    line_color="#00f2fe",
                    annotation_text=f"Median: {median_val:.0f} min",
                    annotation_position="top right"
                )
                apply_chart_style(fig, "", legend=False, height=450)
                fig.update_xaxes(title='Duration (minutes)')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                chart_header_with_info(
                    "Duration Statistics",
                    "This box plot provides a statistical summary. The box contains the middle 50% of durations (25th-75th percentiles), with the line showing the median. The diamond represents the mean. Whiskers extend to show the typical range, while dots indicate outliers (unusually long cases). A long upper whisker or many outliers suggest occasional extended procedures."
                )
                
                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=dur_data['duration'].clip(0, 600),
                    marker_color=COLORS['primary'],
                    boxmean='sd',
                    name='Duration',
                    width=0.7
                ))
                apply_chart_style(fig, "", legend=False, height=420)
                fig.update_yaxes(title='Duration (minutes)')
                fig.update_xaxes(showticklabels=False)
                fig.update_layout(margin=dict(l=50, r=20, t=20, b=40))
                st.plotly_chart(fig, use_container_width=True)
            
            # Duration by discipline
            if COLS['discipline'] in dur_data.columns:
                section_header("Duration Patterns by Discipline")
                
                chart_header_with_info(
                    f"{duration_type} by Discipline",
                    "This box plot compares duration distributions across different medical disciplines. Each box shows the duration range for that specialty, with the line indicating the median. Taller boxes indicate more variable durations, while higher boxes show longer average times. Use this to identify which disciplines require more operating time and have greater scheduling uncertainty."
                )
                
                top_n = st.slider(
                    "Number of disciplines to analyze", 
                    5, 20, 12, 
                    key='dur_topn',
                    help="Select how many top disciplines to display. Fewer shows top performers clearly, more provides comprehensive comparison."
                )
                top_disciplines = dur_data[COLS['discipline']].value_counts().head(top_n).index
                disc_dur = dur_data[dur_data[COLS['discipline']].isin(top_disciplines)]
                
                # Box plot
                fig = px.box(
                    disc_dur,
                    x=COLS['discipline'],
                    y='duration',
                    color=COLS['discipline'],
                    points=False
                )
                apply_chart_style(fig, "", legend=False, height=500)
                fig.update_xaxes(tickangle=-45)
                fig.update_yaxes(title='Duration (minutes)', range=[0, 500])
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics table
                disc_stats = disc_dur.groupby(COLS['discipline'])['duration'].agg([
                    ('Cases', 'count'),
                    ('Median', 'median'),
                    ('Mean', 'mean'),
                    ('Std Dev', 'std'),
                    ('P90', lambda x: x.quantile(0.90))
                ]).round(0).sort_values('Median', ascending=False).reset_index()
                
                with st.expander("View detailed duration statistics"):
                    st.dataframe(disc_stats, use_container_width=True, hide_index=True)
            
            # Duration by emergency priority
            if 'EMERGENCY_PRIORITY' in dur_data.columns:
                section_header_with_info(
                    "Duration by Emergency Priority",
                    """
                    Emergency cases may have different duration patterns than elective cases due to complexity, urgency, 
                    or incomplete pre-operative assessment. Understanding these differences helps with capacity planning 
                    and resource allocation for different priority levels.
                    """
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    chart_header_with_info(
                        f"{duration_type} by Priority",
                        "This box plot compares procedure durations across emergency priority levels (P0 = most urgent, P3 = least urgent). Each box represents the duration distribution for that priority level. Emergency cases (P0, P1) may show different duration patterns than elective cases due to urgency or complexity. The boxes are ordered by priority level for easy comparison."
                    )
                    
                    priority_order = ['P0', 'P1', 'P2A', 'P2B', 'P3A', 'P3B']
                    fig = px.box(
                        dur_data,
                        x='EMERGENCY_PRIORITY',
                        y='duration',
                        category_orders={'EMERGENCY_PRIORITY': priority_order},
                        color='EMERGENCY_PRIORITY',
                        points=False
                    )
                    apply_chart_style(fig, "", legend=False)
                    fig.update_yaxes(title='Duration (minutes)', range=[0, 400])
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    chart_header_with_info(
                        f"Median {duration_type} by Priority",
                        "This bar chart shows the median (typical) duration for each priority level, sorted in priority order (P0 first). Bar height indicates the typical duration. The bars are color-coded by intensity, with darker shades indicating longer durations. Compare across priorities to understand if emergency cases typically take more or less time than elective procedures."
                    )
                    
                    priority_stats = dur_data.groupby('EMERGENCY_PRIORITY')['duration'].agg(['median', 'count']).reset_index()
                    priority_stats = priority_stats[priority_stats['count'] >= 5]
                    
                    # Sort by priority order
                    priority_order = ['P0', 'P1', 'P2A', 'P2B', 'P3A', 'P3B']
                    priority_stats['sort_key'] = priority_stats['EMERGENCY_PRIORITY'].map(
                        {p: i for i, p in enumerate(priority_order)}
                    )
                    priority_stats = priority_stats.sort_values('sort_key')
                    
                    fig = px.bar(
                        priority_stats,
                        x='EMERGENCY_PRIORITY',
                        y='median',
                        color='median',
                        color_continuous_scale='Blues',
                        text='median',
                        category_orders={'EMERGENCY_PRIORITY': priority_order}
                    )
                    fig.update_traces(texttemplate='%{text:.0f} min', textposition='outside')
                    apply_chart_style(fig, "", legend=False)
                    fig.update_yaxes(title='Median Duration (minutes)')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid duration data available after filtering.")
    else:
        st.info(f"{duration_type} analysis requires {dur_col} column.")