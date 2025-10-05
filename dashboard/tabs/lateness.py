"""Lateness analysis tab - COMPLETE"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import COLS, COLORS
from components.styling import (apply_chart_style, metric_card, section_header, 
                                chart_header_with_info, tab_intro, metric_row_with_info,
                                section_header_with_info)
from components.filters import add_tab_filters

def render(data, late_threshold, on_time_band):
    section_header("Lateness Analysis", "Identify delays, quantify impact, and understand root causes")
    
    # Tab introduction
    tab_intro(f"""
    **Understanding Lateness** - A case is considered "late" when the knife-to-skin time (surgery start) occurs more than **{late_threshold} minutes** 
    after the planned start time. This threshold is adjustable in the sidebar under "Analysis Parameters". Lateness cascades through the 
    day, causing subsequent surgeries to be delayed, reducing overall theatre efficiency, and potentially impacting patient satisfaction. 
    Use this tab to identify patterns, root causes, and opportunities to improve on-time performance.
    """)
    
    tab_data = add_tab_filters(data, 'lateness')
    late_data = tab_data[tab_data['_is_late'] == 1].copy()
    
    if late_data.empty:
        st.success("No late cases identified in the selected period.")
        st.info(f"Current late threshold: >{late_threshold} minutes delay")
        return
    
    # Summary metrics
    metric_row_with_info("""
    **Understanding Delay Metrics:**
    
    - **Late Cases**: Total number of surgeries that started more than {} minutes after the planned time
    
    - **Late Rate**: Percentage of all cases that were late. Lower is better. Target: <20%
    
    - **Median Delay**: The typical delay duration for late cases (50th percentile). Half of late cases had shorter delays, half had longer
    
    - **Mean Delay**: The average delay duration. If this is much higher than the median, you have some extremely late cases skewing the average
    
    - **P90 Delay**: 90th percentile - only 10% of late cases had delays longer than this. This shows your worst-case scenarios (excluding extreme outliers)
    """.format(late_threshold))
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        metric_card("Late Cases", f"{len(late_data):,}")
    with col2:
        late_rate = (len(late_data) / len(tab_data) * 100) if len(tab_data) > 0 else 0
        metric_card("Late Rate", f"{late_rate:.1f}%")
    with col3:
        median_delay = late_data['_knife_delay_num'].median()
        metric_card("Median Delay", f"{median_delay:.0f} min")
    with col4:
        mean_delay = late_data['_knife_delay_num'].mean()
        metric_card("Mean Delay", f"{mean_delay:.0f} min")
    with col5:
        p90_delay = late_data['_knife_delay_num'].quantile(0.90)
        metric_card("P90 Delay", f"{p90_delay:.0f} min")
    
    st.markdown("---")
    
    # Delay distribution
    section_header_with_info(
        "Delay Distribution",
        """
        This section visualizes how delays are distributed across your late cases. The histogram shows the frequency 
        distribution of delay durations, while the box plot provides statistical summary. Understanding the distribution 
        helps you identify whether delays are clustered around specific durations or widely varied, which informs 
        different intervention strategies.
        """
    )
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        chart_header_with_info(
            "Distribution of Knife Start Delays (0-180 min)",
            "This histogram shows the frequency distribution of knife start delays for late cases. The x-axis represents delay duration in minutes (0-180), while the y-axis shows the number of cases. The dashed line marks the median delay. Taller bars indicate more common delay durations. If delays cluster near the threshold, minor improvements could significantly reduce late cases."
        )
        
        delays = late_data['_knife_delay_num'].clip(0, 180)
        fig = px.histogram(
            delays,
            nbins=50,
            labels={'value': 'Delay (minutes)', 'count': 'Number of Cases'},
            color_discrete_sequence=[COLORS['danger']]
        )
        fig.add_vline(
            x=median_delay,
            line_dash="dash",
            line_color="#00f2fe",
            annotation_text=f"Median: {median_delay:.0f} min",
            annotation_position="top right"
        )
        apply_chart_style(fig, "", legend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        chart_header_with_info(
            "Delay Statistics",
            "This box-and-whisker plot summarizes delay statistics. The box spans the 25th-75th percentiles (middle 50% of delays), with the line inside showing the median. The diamond represents the mean (average) delay - if it's much higher than the median, extreme delays are pulling the average up. Whiskers extend to show the range, with dots indicating outliers (extreme delays beyond typical patterns). Use this to quickly assess the spread and identify if you have a few very problematic cases."
        )
        
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=late_data['_knife_delay_num'].clip(0, 300),
            name='Delay',
            marker_color=COLORS['danger'],
            boxmean='sd',
            width=0.5
        ))
        apply_chart_style(fig, "", legend=False, height=400)
        fig.update_yaxes(title='Delay (minutes)')
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Impact analysis
    section_header_with_info(
        "Lateness Impact by Category",
        """
        Understanding which disciplines have the most late cases helps prioritize improvement efforts. High-volume 
        disciplines with high late rates cause the most disruption. Focus on the top 3-5 disciplines for maximum impact.
        """
    )
    
    if COLS['discipline'] in late_data.columns:
        # Prepare stacked bar chart data
        all_disc = tab_data.groupby([COLS['discipline'], '_is_late']).size().reset_index(name='count')
        all_disc['status'] = all_disc['_is_late'].map({0: 'On-Time', 1: 'Late'})
        
        # Calculate totals and late rates
        disc_totals = tab_data.groupby(COLS['discipline']).agg({
            '_is_late': ['sum', 'count', 'mean']
        }).reset_index()
        disc_totals.columns = ['Discipline', 'Late_Count', 'Total_Count', 'Late_Rate']
        disc_totals = disc_totals[disc_totals['Total_Count'] >= 10].sort_values('Late_Count', ascending=False).head(15)
        
        # Filter to top 15 disciplines
        top_disciplines = disc_totals['Discipline'].tolist()
        all_disc = all_disc[all_disc[COLS['discipline']].isin(top_disciplines)]
        
        # Add late rate for hover
        all_disc = all_disc.merge(
            disc_totals[['Discipline', 'Late_Rate', 'Total_Count']], 
            left_on=COLS['discipline'], 
            right_on='Discipline',
            how='left'
        )
        
        # Sort by late count for display
        all_disc['sort_order'] = all_disc['Discipline'].map({d: i for i, d in enumerate(top_disciplines)})
        all_disc = all_disc.sort_values('sort_order')
        
        chart_header_with_info(
            "Top 15 Disciplines by Late Case Volume (min 10 cases)",
            "This stacked bar chart displays the top 15 disciplines with the most late cases. Each bar shows total case volume, split between on-time (green) and late (red) cases. Longer bars indicate higher activity. Hover to see the late rate percentage. Disciplines with large red portions or high late rates may need workflow improvements or additional scheduling time."
        )
        
        fig = px.bar(
            all_disc,
            x=all_disc['count'],
            y='Discipline',
            color='status',
            orientation='h',
            color_discrete_map={'On-Time': COLORS['success'], 'Late': COLORS['danger']},
            custom_data=['Late_Rate', 'Total_Count'],
            category_orders={'Discipline': top_disciplines}
        )
        
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>Count: %{x}<br>Total Cases: %{customdata[1]}<br>Late Rate: %{customdata[0]:.1%}<extra></extra>'
        )
        
        fig.update_layout(
            xaxis=dict(title='Number of Cases'),
            yaxis=dict(title=''),
            height=500,
            barmode='stack'
        )
        
        apply_chart_style(fig, "")
        st.plotly_chart(fig, use_container_width=True)
    
    # Root cause analysis
    if 'Delay_Category' in late_data.columns:
        section_header_with_info(
            "Root Cause Analysis",
            """
            Identifying the reasons for delays is critical for implementing effective solutions. Focus improvement efforts 
            on the top 3-5 most frequent causes to achieve the greatest impact on reducing lateness.
            """
        )
        
        reason_data = late_data['Delay_Category'].value_counts().head(12).reset_index()
        reason_data.columns = ['Reason', 'Count']
        reason_data['Percentage'] = (reason_data['Count'] / len(late_data) * 100).round(1)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            chart_header_with_info(
                "Top Delay Reasons",
                "This horizontal bar chart ranks the top reasons for delays by frequency. Bar length represents the number of late cases attributed to each cause, while the percentage labels show what portion of all late cases each reason represents. Focus improvement efforts on the top 3-5 causes to achieve the greatest impact."
            )
            
            fig = px.bar(
                reason_data,
                y='Reason',
                x='Count',
                orientation='h',
                color='Count',
                color_continuous_scale='Reds',
                text='Percentage'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            apply_chart_style(fig, "", legend=False, height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Delay Reasons Summary**")
            st.dataframe(
                reason_data[['Reason', 'Count', 'Percentage']],
                use_container_width=True,
                hide_index=True
            )
    
    # Temporal patterns
    section_header_with_info(
        "Temporal Patterns",
        """
        Understanding when delays occur helps identify systematic issues. Consistent patterns by day or time suggest 
        operational problems (e.g., Monday mornings, afternoon cascading delays) that can be addressed through scheduling 
        or workflow changes.
        """
    )
    
    if COLS['surg_start'] in late_data.columns:
        late_data['weekday'] = pd.to_datetime(late_data[COLS['surg_start']], errors='coerce').dt.day_name()
        late_data['hour'] = pd.to_datetime(late_data[COLS['surg_start']], errors='coerce').dt.hour
        
        col1, col2 = st.columns(2)
        
        with col1:
            chart_header_with_info(
                "Late Cases by Day of Week",
                "This bar chart shows which days of the week experience the most late cases. Taller bars indicate more lateness problems on those days. Consistent patterns (e.g., Mondays always worse) suggest systematic issues like weekend backlogs or staffing challenges on specific days."
            )
            
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            weekday_counts = late_data['weekday'].value_counts().reindex(weekday_order, fill_value=0).reset_index()
            weekday_counts.columns = ['Weekday', 'Late_Cases']
            
            fig = px.bar(
                weekday_counts,
                x='Weekday',
                y='Late_Cases',
                color='Late_Cases',
                color_continuous_scale='Reds'
            )
            apply_chart_style(fig, "", legend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            chart_header_with_info(
                "Late Cases by Hour of Day",
                "This line chart displays late case frequency by hour of the day (24-hour format). Peaks indicate times when delays are most common. Morning peaks may suggest overnight preparation issues, while afternoon peaks could indicate cumulative delays from earlier cases."
            )
            
            hourly = late_data['hour'].value_counts().sort_index().reset_index()
            hourly.columns = ['Hour', 'Late_Cases']
            
            fig = px.line(
                hourly,
                x='Hour',
                y='Late_Cases',
                markers=True,
                color_discrete_sequence=[COLORS['danger']]
            )
            apply_chart_style(fig, "", legend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    # Room-level analysis
    if COLS['room'] in late_data.columns:
        section_header_with_info(
            "Lateness by Operating Room",
            """
            Some operating rooms may consistently have higher late rates due to equipment issues, location, staffing patterns, 
            or the types of cases scheduled there. Identifying problem rooms helps target facility-specific interventions.
            """
        )
        
        room_late = tab_data.groupby(COLS['room']).agg({
            '_is_late': ['sum', 'count', 'mean'],
            '_knife_delay_num': lambda x: x[x > 0].median()
        }).reset_index()
        room_late.columns = ['Room', 'Late_Count', 'Total_Count', 'Late_Rate', 'Median_Delay']
        room_late = room_late[room_late['Total_Count'] >= 15].sort_values('Late_Rate', ascending=False).head(15)
        
        chart_header_with_info(
            "Rooms with Highest Late Rates (min 15 cases)",
            "This bar chart ranks operating rooms by their late rate (percentage of cases that start late). Bar height represents the late rate, while color intensity indicates the median delay duration for that room. Darker reds show both high late rates and longer delays. Rooms consistently at the top may have equipment, staffing, or workflow issues requiring investigation."
        )
        
        fig = px.bar(
            room_late,
            x='Room',
            y='Late_Rate',
            color='Median_Delay',
            color_continuous_scale='Reds',
            text=room_late['Late_Rate'].apply(lambda x: f'{x:.1%}'),
            hover_data={'Late_Count': True, 'Total_Count': True}
        )
        fig.update_traces(textposition='outside')
        apply_chart_style(fig, "")
        fig.update_yaxes(tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)