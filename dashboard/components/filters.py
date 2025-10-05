"""Filter components"""
import streamlit as st
from config import COLS

def add_tab_filters(df, prefix: str):
    """Add per-tab filters in an expander"""
    with st.expander("Additional Filters for This Tab"):
        filtered = df.copy()
        cols = st.columns(3)
        
        with cols[0]:
            if COLS['room'] in filtered.columns:
                rooms = st.multiselect(
                    "Filter Rooms",
                    sorted(filtered[COLS['room']].dropna().unique()),
                    key=f'{prefix}_rooms'
                )
                if rooms:
                    filtered = filtered[filtered[COLS['room']].isin(rooms)]
        
        with cols[1]:
            if COLS['discipline'] in filtered.columns:
                disciplines = st.multiselect(
                    "Filter Disciplines",
                    sorted(filtered[COLS['discipline']].dropna().unique()),
                    key=f'{prefix}_disciplines'
                )
                if disciplines:
                    filtered = filtered[filtered[COLS['discipline']].isin(disciplines)]
        
        with cols[2]:
            if COLS['surgeon'] in filtered.columns:
                surgeons = st.multiselect(
                    "Filter Surgeons",
                    sorted(filtered[COLS['surgeon']].dropna().unique()),
                    key=f'{prefix}_surgeons'
                )
                if surgeons:
                    filtered = filtered[filtered[COLS['surgeon']].isin(surgeons)]
        
        st.caption(f"Showing {len(filtered):,} of {len(df):,} cases after tab filters")
    
    return filtered