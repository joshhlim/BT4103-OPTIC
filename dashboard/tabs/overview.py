"""Overview tab"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import COLS, COLORS, QUALITATIVE_COLORS
from components.styling import (apply_chart_style, section_header, metric_card, 
                                chart_header_with_info, tab_intro, metric_row_with_info)
from components.filters import add_tab_filters

def render(data, global_filters, late_threshold, on_time_band):
    section_header(
        "Executive Summary",
        "High-level performance indicators across all operational dimensions"
    )
    
    # Tab introduction
    tab_intro("""
    **Welcome to the Overview Dashboard** - This tab provides a high-level summary of your operating theatre 
    performance. Use this as your starting point to identify areas that need attention, then navigate to 
    specific tabs for detailed analysis. The metrics below show your current performance across key dimensions: 
    timeliness, planning accuracy, efficiency, and capacity utilization.
    """)
    
    tab_data = add_tab_filters(data, 'overview')
    
    # Performance scorecard with metric cards
    metric_row_with_info("""
    **Understanding These Metrics:**
    
    - **On-Time Rate**: Percentage of cases that started within your defined tolerance band (±{} min). Higher is better. Target: >80%
    
    - **Planning Accuracy**: Mean Absolute Error (MAE) - the average difference between planned and actual surgery durations in minutes. Lower is better. Target: <15 min
    
    - **Median Duration**: The typical (middle) surgery duration. Half of procedures are shorter, half are longer. Use this to understand your baseline procedure times.
    
    - **Rooms Utilized**: Number of different operating rooms used in the selected period. More rooms indicate higher facility usage but may also suggest capacity constraints.
    """.format(on_time_band))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if '_knife_delay_num' in tab_data.columns:
            delays = tab_data['_knife_delay_num'].dropna()
            on_time = ((delays >= -on_time_band) & (delays <= on_time_band)).sum()
            on_time_rate = (on_time / len(delays) * 100) if len(delays) > 0 else 0
            metric_card("On-Time Rate", f"{on_time_rate:.1f}%", "Cases within tolerance band")
        else:
            metric_card("On-Time Rate", "N/A")
    
    with col2:
        if 'DIFF_SURGERY_DURATION' in tab_data.columns:
            diff = tab_data['DIFF_SURGERY_DURATION'].dropna()
            mae = diff.abs().mean()
            metric_card("Planning Accuracy", f"{mae:.0f} min", "Mean Absolute Error")
        else:
            metric_card("Planning Accuracy", "N/A")
    
    with col3:
        if COLS['actual_dur'] in tab_data.columns:
            durations = pd.to_numeric(tab_data[COLS['actual_dur']], errors='coerce').dropna()
            metric_card("Median Duration", f"{durations.median():.0f} min", "Surgery duration")
        else:
            metric_card("Median Duration", "N/A")
    
    with col4:
        if COLS['room'] in tab_data.columns:
            rooms_used = tab_data[COLS['room']].nunique()
            metric_card("Rooms Utilized", f"{rooms_used}", "Active operating rooms")
        else:
            metric_card("Rooms Utilized", "N/A")
    
    st.markdown("---")
    
    # Monthly trend
    section_header("Performance Trends")
    
    if global_filters['date_col'] in tab_data.columns:
        trend_data = tab_data.copy()
        trend_data['month'] = pd.to_datetime(
            trend_data[global_filters['date_col']], errors='coerce'
        ).dt.to_period('M').astype(str)
        
        # Create stacked bar chart data
        monthly = trend_data.groupby(['month', '_is_late']).size().reset_index(name='count')
        monthly['status'] = monthly['_is_late'].map({0: 'On-Time', 1: 'Late'})
        
        # Calculate total and late rate for hover
        monthly_totals = trend_data.groupby('month').agg({
            '_is_late': ['count', 'sum', 'mean']
        }).reset_index()
        monthly_totals.columns = ['month', 'total_cases', 'late_cases', 'late_rate']
        monthly_totals = monthly_totals[monthly_totals['total_cases'] >= 5]
        
        if not monthly.empty and len(monthly_totals) > 1:
            # Merge late rate into the stacked data
            monthly = monthly.merge(
                monthly_totals[['month', 'late_rate']], 
                on='month', 
                how='left'
            )
            
            chart_header_with_info(
                "Monthly Case Volume by Timeliness Status",
                "This stacked bar chart shows the monthly distribution of cases by timeliness status. Each bar represents the total cases for that month, split into on-time (bottom, green) and late (top, red) cases. Hover over the bars to see exact counts and the late rate percentage. Increasing red portions indicate growing lateness issues, while consistent green suggests good time management."
            )
            
            fig = px.bar(
                monthly,
                x='month',
                y='count',
                color='status',
                color_discrete_map={'On-Time': COLORS['success'], 'Late': COLORS['danger']},
                custom_data=['late_rate']
            )
            
            fig.update_traces(
                hovertemplate='<b>%{x}</b><br>Count: %{y}<br>Late Rate: %{customdata[0]:.1%}<extra></extra>'
            )
            
            fig.update_layout(
                yaxis=dict(title='Number of Cases'),
                xaxis=dict(title='Month'),
                hovermode='x unified',
                height=450,
                barmode='stack'
            )
            
            apply_chart_style(fig, "")
            st.plotly_chart(fig, use_container_width=True)
    
    # Distributions
    col1, col2 = st.columns(2)
    
    with col1:
        if COLS['discipline'] in tab_data.columns:
            disc_data = tab_data[COLS['discipline']].value_counts().head(12).reset_index()
            disc_data.columns = ['Discipline', 'Cases']
            
            chart_header_with_info(
                "Top 12 Disciplines by Case Volume",
                "This horizontal bar chart ranks the top 12 medical disciplines by case volume. Longer bars indicate higher activity levels. Use this to identify which specialties drive the most operating room demand and may require additional resources or scheduling attention."
            )
            
            fig = px.bar(
                disc_data,
                y='Discipline',
                x='Cases',
                orientation='h',
                color='Cases',
                color_continuous_scale='Blues',
                text='Cases'
            )
            fig.update_traces(textposition='outside')
            apply_chart_style(fig, "", legend=False, height=450)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'EMERGENCY_PRIORITY' in tab_data.columns:
            priority_data = tab_data[tab_data['EMERGENCY_PRIORITY'].astype(str) != '0'].copy()
            priority_counts = priority_data['EMERGENCY_PRIORITY'].value_counts().reset_index()
            priority_counts.columns = ['Priority', 'Cases']
            
            if not priority_counts.empty:
                chart_header_with_info(
                    "Emergency Priority Distribution",
                    "This donut chart shows the distribution of emergency priority classifications. Each segment represents the proportion of cases at that priority level. Larger segments indicate more common priority types. P0/P1 represent the most urgent cases (true emergencies), P2 represents urgent cases, and P3 indicates lower urgency (semi-elective). Understanding your priority mix helps with resource planning and capacity allocation."
                )
                
                fig = px.pie(
                    priority_counts,
                    names='Priority',
                    values='Cases',
                    hole=0.4,
                    color_discrete_sequence=QUALITATIVE_COLORS
                )
                apply_chart_style(fig, "", height=450)
                st.plotly_chart(fig, use_container_width=True)