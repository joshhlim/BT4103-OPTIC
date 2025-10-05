"""Utilization analysis tab - COMPLETE"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from config import COLS, COLORS
from components.styling import (apply_chart_style, metric_card, section_header, 
                                chart_header_with_info, tab_intro, metric_row_with_info,
                                section_header_with_info)
from components.filters import add_tab_filters

def render(data):
    section_header(
        "Utilization Analysis",
        "Optimize resource allocation and maximize theatre capacity"
    )
    
    tab_intro("""
    **Understanding Utilization** - This tab measures how effectively you're using your operating room capacity. Utilization rate 
    is calculated as: (Actual hours used / Available capacity) × 100%. You define the available capacity by setting a standard 
    session length below. Optimal utilization is typically **70-85%** - high enough to maximize resource use, but with buffer 
    for emergencies and variability. Rates consistently above 90% suggest capacity constraints, while rates below 60% indicate 
    underutilization and potential inefficiency.
    """)
    
    tab_data = add_tab_filters(data, 'utilization')
    
    # Session length parameter with explanation
    col1, col2 = st.columns([1, 2])
    with col1:
        session_hours = st.number_input(
            "Standard session length (hours/day)",
            min_value=4,
            max_value=16,
            value=8,
            step=1,
            help="Set the standard daily operating hours for your facility. This is used as the baseline (100%) for calculating utilization rates."
        )
    with col2:
        st.info("""
        **Why adjust this parameter?** Your standard session length defines 100% capacity. If your facility typically operates 
        8-hour days, use 8. For 24-hour facilities or extended-hours programs, adjust accordingly. Changing this value will 
        recalculate all utilization percentages.
        """)
    
    st.markdown("")
    
    if (COLS['room'] in tab_data.columns and 
        COLS['room_start'] in tab_data.columns and 
        COLS['room_end'] in tab_data.columns):
        
        util_data = tab_data.copy()
        
        # Calculate durations
        util_data['start'] = pd.to_datetime(util_data[COLS['room_start']], errors='coerce')
        util_data['end'] = pd.to_datetime(util_data[COLS['room_end']], errors='coerce')
        util_data = util_data.dropna(subset=['start', 'end'])
        
        util_data['duration_hours'] = (util_data['end'] - util_data['start']).dt.total_seconds() / 3600
        util_data['date'] = util_data['start'].dt.date
        
        # Room utilization summary
        section_header("Room Utilization Overview")
        
        room_util = util_data.groupby(COLS['room']).agg({
            'duration_hours': 'sum',
            'date': 'nunique'
        }).reset_index()
        room_util.columns = ['Room', 'Total_Hours_Used', 'Active_Days']
        room_util['Available_Hours'] = room_util['Active_Days'] * session_hours
        room_util['Utilization_Pct'] = (room_util['Total_Hours_Used'] / room_util['Available_Hours'] * 100).round(1)
        room_util = room_util.sort_values('Utilization_Pct', ascending=False)
        
        # Explanation box BEFORE metrics
        st.info("""
        **Understanding Utilization Rates Over 100%:**
        
        Utilization can exceed 100% when actual operating hours surpass the standard session length. This occurs when:
        - Rooms run beyond scheduled hours due to emergency cases or overruns
        - Multiple overlapping procedures are performed (parallel usage)
        - Extended shifts or overtime operations take place
        
        High utilization (>90%) may indicate capacity constraints, while rates consistently over 100% suggest the need to 
        reassess standard session lengths or consider additional resources.
        """)
        
        # Summary metrics
        metric_row_with_info("""
        **Utilization Metrics Explained:**
        
        - **Average Utilization**: Mean utilization rate across all rooms. Target: 70-85%
        
        - **Total Capacity**: Sum of all available operating hours (rooms × active days × session length)
        
        - **Hours Used**: Actual total hours rooms were occupied with cases
        
        - **Unused Capacity**: Potential additional hours available. Some buffer is healthy for flexibility and emergencies
        """)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_util = room_util['Utilization_Pct'].mean()
            metric_card("Average Utilization", f"{avg_util:.1f}%")
        
        with col2:
            total_capacity = room_util['Available_Hours'].sum()
            metric_card("Total Capacity", f"{total_capacity:.0f} hrs")
        
        with col3:
            total_used = room_util['Total_Hours_Used'].sum()
            metric_card("Hours Used", f"{total_used:.0f} hrs")
        
        with col4:
            unused = total_capacity - total_used
            metric_card("Unused Capacity", f"{unused:.0f} hrs")
        
        st.markdown("---")
        
        # Room utilization chart
        section_header("Utilization Rate by Room")
        
        top_rooms = st.slider(
            "Number of rooms to display", 
            5, 30, 15, 
            key='util_rooms',
            help="Select how many rooms to display. Fewer shows top/bottom performers clearly, more provides comprehensive view."
        )
        plot_rooms = room_util.head(top_rooms)
        
        chart_header_with_info(
            f"Room Utilization Rates (Top {top_rooms} rooms)",
            "This bar chart shows the utilization rate for each operating room as a percentage of standard capacity. Bar height represents the utilization rate, with color intensity indicating efficiency: green for optimal (around 80%), yellow for moderate, and red for underutilized or overutilized. The dashed green line marks the 80% target, while the dotted red line shows 100% capacity. Hover over bars to see total hours used and active days."
        )
        
        fig = px.bar(
            plot_rooms,
            x='Room',
            y='Utilization_Pct',
            color='Utilization_Pct',
            color_continuous_scale='RdYlGn',
            range_color=[0, 100],
            text=plot_rooms['Utilization_Pct'].apply(lambda x: f'{x:.1f}%'),
            hover_data={'Total_Hours_Used': ':.0f', 'Active_Days': True}
        )
        fig.add_hline(
            y=80,
            line_dash="dash",
            line_color="green",
            annotation_text="Target: 80%",
            annotation_position="right"
        )
        fig.add_hline(
            y=100,
            line_dash="dot",
            line_color="red",
            annotation_text="Full Capacity",
            annotation_position="right"
        )
        fig.update_traces(textposition='outside')
        apply_chart_style(fig, "", height=500)
        fig.update_yaxes(title='Utilization (%)', range=[0, max(110, plot_rooms['Utilization_Pct'].max() + 10)])
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed utilization table
        with st.expander("View detailed room utilization data"):
            display_util = room_util.copy()
            display_util['Total_Hours_Used'] = display_util['Total_Hours_Used'].round(1)
            display_util['Available_Hours'] = display_util['Available_Hours'].round(0)
            st.dataframe(display_util, use_container_width=True, hide_index=True)
        
        # Time series utilization
        section_header("Utilization Trends Over Time")
        
        daily_util = util_data.groupby(['date', COLS['room']])['duration_hours'].sum().reset_index()
        daily_util['utilization'] = (daily_util['duration_hours'] / session_hours * 100).clip(0, 150)
        
        # Calculate aggregate metrics
        daily_avg = daily_util.groupby('date')['utilization'].mean().reset_index()
        daily_avg.columns = ['date', 'avg_utilization']
        
        col1, col2 = st.columns([1, 3])
        with col1:
            viz_option = st.radio(
                "Visualization Type",
                ["Average Across All Rooms", "Individual Rooms"],
                help="Average view shows overall facility trends. Individual rooms view allows comparison of specific rooms over time."
            )
        
        if viz_option == "Average Across All Rooms":
            chart_header_with_info(
                "Average Daily Utilization Across All Rooms",
                "This line chart displays the average daily utilization rate across all operating rooms over time. The y-axis shows utilization percentage, while the x-axis shows dates. The blue line with markers tracks daily average utilization. The green dashed line marks the 80% target, and the red dotted line shows 100% capacity. Consistent trends above or below target may indicate systemic capacity issues or scheduling inefficiencies."
            )
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=daily_avg['date'],
                y=daily_avg['avg_utilization'],
                mode='lines+markers',
                name='Average Utilization',
                line=dict(width=3, color='#1f77b4'),
                marker=dict(size=6)
            ))
            
            fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="80% Target", opacity=0.7)
            fig.add_hline(y=100, line_dash="dot", line_color="red", annotation_text="Full Capacity", opacity=0.7)
            
            apply_chart_style(fig, "")
            fig.update_yaxes(title='Utilization (%)', range=[0, 120])
            fig.update_xaxes(title='Date')
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            all_rooms = sorted(util_data[COLS['room']].unique())
            default_rooms = list(room_util.head(3)['Room']) if len(room_util) >= 3 else all_rooms[:3]
            
            selected_rooms = st.multiselect(
                "Select up to 5 rooms to compare",
                options=all_rooms,
                default=default_rooms,
                max_selections=5,
                key='util_ts_rooms',
                help="Compare utilization trends for specific rooms. Select rooms with different utilization rates to identify patterns."
            )
            
            if selected_rooms:
                chart_header_with_info(
                    "Daily Utilization Rate by Selected Rooms",
                    "This multi-line chart compares daily utilization trends for individual rooms over time. Each colored line represents one room's daily utilization. The gray dashed line shows the 80% target, while the red dotted line marks 100% capacity. Use this to identify rooms with consistently high or low utilization, or to spot patterns and anomalies. Volatile lines suggest unpredictable scheduling, while steady lines indicate consistent utilization."
                )
                
                ts_data = daily_util[daily_util[COLS['room']].isin(selected_rooms)]
                
                fig = px.line(
                    ts_data,
                    x='date',
                    y='utilization',
                    color=COLS['room'],
                    markers=True,
                    line_shape='spline'
                )
                fig.add_hline(y=80, line_dash="dash", line_color="gray", annotation_text="80% Target", opacity=0.5)
                fig.add_hline(y=100, line_dash="dot", line_color="red", annotation_text="Full Capacity", opacity=0.5)
                apply_chart_style(fig, "")
                fig.update_yaxes(title='Utilization (%)', range=[0, 120])
                fig.update_xaxes(title='Date')
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Limit selection to 5 rooms maximum for clarity. Target utilization is 80%.")
            else:
                st.info("Please select at least one room to visualize.")
        
        # Activity heatmap
        section_header_with_info(
            "Activity Patterns",
            """
            Understanding when operating rooms are busiest helps with staffing decisions, capacity planning, and identifying 
            opportunities to shift cases to underutilized time slots.
            """
        )
        
        if len(util_data) > 50:
            heatmap_data = util_data.copy()
            heatmap_data['weekday'] = heatmap_data['start'].dt.day_name()
            heatmap_data['hour'] = heatmap_data['start'].dt.hour
            
            weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_pivot = heatmap_data.groupby(['weekday', 'hour']).size().reset_index(name='cases')
            
            # Create a proper pivot table for heatmap
            heatmap_matrix = heatmap_pivot.pivot(index='weekday', columns='hour', values='cases').fillna(0)
            # Reorder rows by weekday
            heatmap_matrix = heatmap_matrix.reindex(weekday_order)
            
            chart_header_with_info(
                "Case Volume Heatmap: Day of Week x Hour of Day",
                "This heatmap shows case volume patterns by day of week (rows) and hour of day (columns). Color intensity indicates volume: red for high activity, yellow for moderate, and green for low activity. Use this to identify peak operating times, underutilized time slots, and staffing needs. Consistent patterns reveal scheduling habits, while gaps may represent opportunities for better capacity utilization."
            )
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_matrix.values,
                x=heatmap_matrix.columns,
                y=heatmap_matrix.index,
                colorscale='RdYlGn_r',
                hoverongaps=False
            ))
            apply_chart_style(fig, "", legend=False, height=400)
            fig.update_xaxes(title='Hour of Day', dtick=1, side='bottom')
            fig.update_yaxes(title='')
            st.plotly_chart(fig, use_container_width=True)
        
        # Capacity recommendations
        section_header("Capacity Optimization Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Underutilized Rooms (< 60%)**")
            underutil = room_util[room_util['Utilization_Pct'] < 60].sort_values('Utilization_Pct')
            if not underutil.empty:
                st.dataframe(
                    underutil[['Room', 'Utilization_Pct', 'Active_Days']].head(10),
                    use_container_width=True,
                    hide_index=True
                )
                total_wasted = (underutil['Available_Hours'] - underutil['Total_Hours_Used']).sum()
                st.metric("Potential Capacity Gain", f"{total_wasted:.0f} hrs")
                st.caption("These rooms have significant unused capacity that could be reallocated or consolidated.")
            else:
                st.info("No underutilized rooms identified.")
        
        with col2:
            st.markdown("**Overutilized Rooms (> 90%)**")
            overutil = room_util[room_util['Utilization_Pct'] > 90].sort_values('Utilization_Pct', ascending=False)
            if not overutil.empty:
                st.dataframe(
                    overutil[['Room', 'Utilization_Pct', 'Active_Days']].head(10),
                    use_container_width=True,
                    hide_index=True
                )
                st.caption("These rooms may benefit from extended hours or case redistribution to prevent bottlenecks and delays.")
            else:
                st.info("No overutilized rooms identified.")
    
    else:
        st.info("Utilization analysis requires ROOM, ACTUAL_ENTER_OR_TIME, and ACTUAL_EXIT_OR_TIME columns.")