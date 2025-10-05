"""Planning accuracy tab - COMPLETE"""
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
        "Planning Accuracy Analysis",
        "Compare planned versus actual timings to improve scheduling reliability"
    )
    
    tab_intro(f"""
    **Understanding Planning Accuracy** - This tab evaluates how well your planned surgery durations match actual durations. 
    Good planning accuracy means fewer cascading delays, better resource utilization, and more predictable schedules. A case is 
    considered "on-time" when the actual duration is within **±{on_time_band} minutes** of the planned duration (adjustable in sidebar). 
    Use this analysis to identify disciplines or surgeons with systematic planning biases and improve future scheduling estimates.
    """)
    
    tab_data = add_tab_filters(data, 'planning')
    
    if COLS['actual_dur'] in tab_data.columns and COLS['planned_dur'] in tab_data.columns:
        plan_data = tab_data.copy()
        plan_data['actual'] = pd.to_numeric(plan_data[COLS['actual_dur']], errors='coerce')
        plan_data['planned'] = pd.to_numeric(plan_data[COLS['planned_dur']], errors='coerce')
        plan_data['diff'] = plan_data['actual'] - plan_data['planned']
        plan_data = plan_data.dropna(subset=['actual', 'planned'])
        
        if not plan_data.empty:
            # Summary metrics
            metric_row_with_info("""
            **Planning Accuracy Metrics Explained:**
            
            - **MAE** (Mean Absolute Error): Average prediction error regardless of direction. Lower is better. Target: <15 min
            
            - **Mean Bias**: Average over/under estimation. Positive = systematic underestimation (taking longer than planned). 
              Negative = systematic overestimation (finishing earlier). Target: close to 0
            
            - **Within Tolerance**: Percentage of cases where actual duration was within ±{} min of planned. Higher is better. Target: >70%
            
            - **Overruns**: Cases that took significantly longer than planned (beyond tolerance band)
            
            - **Underruns**: Cases that finished significantly earlier than planned (beyond tolerance band)
            """.format(on_time_band))
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                mae = plan_data['diff'].abs().mean()
                metric_card("MAE", f"{mae:.1f} min", "Mean Absolute Error")
            
            with col2:
                bias = plan_data['diff'].mean()
                metric_card("Mean Bias", f"{bias:+.1f} min", "Avg over/under estimation")
            
            with col3:
                within_band = ((plan_data['diff'].abs() <= on_time_band)).sum()
                accuracy = (within_band / len(plan_data) * 100)
                metric_card("Within Tolerance", f"{accuracy:.1f}%", f"±{on_time_band} min")
            
            with col4:
                overrun = (plan_data['diff'] > on_time_band).sum()
                overrun_pct = (overrun / len(plan_data) * 100)
                metric_card("Overruns", f"{overrun:,}", f"{overrun_pct:.1f}%")
            
            with col5:
                underrun = (plan_data['diff'] < -on_time_band).sum()
                underrun_pct = (underrun / len(plan_data) * 100)
                metric_card("Underruns", f"{underrun:,}", f"{underrun_pct:.1f}%")
            
            st.markdown("---")
            
            # Scatter plot
            section_header("Planned vs Actual Duration Comparison")
            
            sample_size = min(5000, len(plan_data))
            plot_data = plan_data.sample(sample_size, random_state=42) if len(plan_data) > sample_size else plan_data
            
            def categorize(diff):
                if diff < -on_time_band:
                    return 'Underrun'
                elif diff > on_time_band:
                    return 'Overrun'
                else:
                    return f'On-Time (±{on_time_band}min)'
            
            plot_data['category'] = plot_data['diff'].apply(categorize)
            
            chart_header_with_info(
                f"Planned vs Actual Duration (sample: {sample_size:,} cases)",
                "This scatter plot compares planned durations (x-axis) against actual durations (y-axis). Each dot represents one case, colored by accuracy: green for on-time (within tolerance), blue for underruns (finished early), and red for overruns (took longer). The dashed diagonal line represents perfect predictions. Points above the line indicate underestimation, while points below show overestimation. Clustering along the diagonal suggests good planning accuracy."
            )
            
            fig = px.scatter(
                plot_data,
                x='planned',
                y='actual',
                color='category',
                color_discrete_map={
                    'Underrun': COLORS['early'],
                    f'On-Time (±{on_time_band}min)': COLORS['success'],
                    'Overrun': COLORS['late']
                },
                opacity=0.5,
                render_mode='webgl'
            )
            
            max_val = max(plot_data['planned'].max(), plot_data['actual'].max())
            fig.add_shape(
                type='line',
                x0=0, y0=0, x1=max_val, y1=max_val,
                line=dict(color='white', dash='dash', width=2),
                opacity=0.5
            )
            
            apply_chart_style(fig, "")
            fig.update_xaxes(title='Planned Duration (minutes)')
            fig.update_yaxes(title='Actual Duration (minutes)', scaleanchor='x', scaleratio=1)
            st.plotly_chart(fig, use_container_width=True, key="planning_scatter_chart")
            
            # Distribution comparison
            section_header("Duration Distribution Comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                chart_header_with_info(
                    "Duration Distribution: Planned vs Actual (0-500 min)",
                    "This overlapping histogram compares the frequency distributions of planned (orange) and actual (blue) durations. The x-axis shows duration in minutes, while the y-axis shows case count. When distributions align closely, planning is accurate. If actual durations (blue) are consistently shifted right of planned (orange), procedures are systematically taking longer than expected."
                )
                
                hist_data = pd.DataFrame({
                    'Duration': pd.concat([plan_data['planned'], plan_data['actual']]),
                    'Type': ['Planned'] * len(plan_data) + ['Actual'] * len(plan_data)
                })
                hist_data = hist_data[hist_data['Duration'].between(0, 500)]
                
                fig = px.histogram(
                    hist_data,
                    x='Duration',
                    color='Type',
                    nbins=50,
                    barmode='overlay',
                    opacity=0.7,
                    color_discrete_map={'Planned': '#ff7f0e', 'Actual': '#1f77b4'}
                )
                apply_chart_style(fig, "")
                fig.update_xaxes(title='Duration (minutes)')
                st.plotly_chart(fig, use_container_width=True, key="planning_duration_histogram")
            
            with col2:
                chart_header_with_info(
                    "Prediction Error Distribution (-120 to +120 min)",
                    "This histogram shows the distribution of prediction errors (actual minus planned duration). The x-axis represents the error in minutes, with negative values indicating overestimation (finished early) and positive values showing underestimation (took longer). The white dashed line marks zero (perfect prediction), while the red dotted line shows the mean bias. A centered distribution around zero indicates unbiased predictions."
                )
                
                diff_clipped = plan_data['diff'].clip(-120, 120)
                fig = px.histogram(
                    diff_clipped,
                    nbins=60,
                    color_discrete_sequence=[COLORS['warning']]
                )
                fig.add_vline(
                    x=0,
                    line_dash="dash",
                    line_color="white",
                    annotation_text="Perfect Match",
                    annotation_position="top"
                )
                fig.add_vline(
                    x=bias,
                    line_dash="dot",
                    line_color=COLORS['danger'],
                    annotation_text=f"Mean: {bias:+.0f}",
                    annotation_position="bottom"
                )
                apply_chart_style(fig, "", legend=False)
                fig.update_xaxes(title='Difference: Actual - Planned (minutes)')
                st.plotly_chart(fig, use_container_width=True, key="planning_error_histogram")
            
            # Accuracy by discipline
            if COLS['discipline'] in plan_data.columns:
                section_header_with_info(
                    "Planning Accuracy by Discipline",
                    """
                    Different disciplines may have different planning accuracy due to procedure complexity, variability, or estimation 
                    practices. Identifying disciplines with poor planning accuracy allows targeted training or systematic time adjustments.
                    """
                )
                
                disc_accuracy = plan_data.groupby(COLS['discipline']).agg({
                    'diff': ['mean', lambda x: x.abs().mean(), 'count']
                }).reset_index()
                disc_accuracy.columns = ['Discipline', 'Mean_Bias', 'MAE', 'Case_Count']
                disc_accuracy = disc_accuracy[disc_accuracy['Case_Count'] >= 20].sort_values('MAE', ascending=False).head(15)
                
                chart_header_with_info(
                    "Top 15 Disciplines by Planning Error (min 20 cases)",
                    "This dual-axis chart shows planning accuracy by discipline. Orange bars represent Mean Absolute Error (MAE)—the average magnitude of prediction errors regardless of direction, with taller bars indicating less accurate planning. The red line shows Mean Bias—whether the discipline tends to overrun (positive) or underrun (negative). High MAE with positive bias suggests consistent underestimation requiring longer time allocations."
                )
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=disc_accuracy['Discipline'],
                    y=disc_accuracy['MAE'],
                    name='MAE',
                    marker_color=COLORS['warning'],
                    text=disc_accuracy['MAE'].round(1),
                    textposition='outside'
                ))
                
                fig.add_trace(go.Scatter(
                    x=disc_accuracy['Discipline'],
                    y=disc_accuracy['Mean_Bias'],
                    name='Mean Bias',
                    yaxis='y2',
                    mode='lines+markers',
                    line=dict(width=3, color=COLORS['danger']),
                    marker=dict(size=10)
                ))
                
                fig.update_layout(
                    yaxis=dict(title='Mean Absolute Error (minutes)'),
                    yaxis2=dict(title='Mean Bias (minutes)', overlaying='y', side='right'),
                    hovermode='x unified',
                    height=500
                )
                fig.update_xaxes(tickangle=-45)
                
                apply_chart_style(fig, "")
                st.plotly_chart(fig, use_container_width=True, key="planning_discipline_chart")
        else:
            st.info("No valid planning data available after filtering.")
    else:
        st.info("Planning accuracy analysis requires ACTUAL_SURGERY_DURATION and PLANNED_SURGERY_DURATION columns.")