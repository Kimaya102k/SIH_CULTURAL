"""
data_loader.py
---------------
Central place for loading all CSV datasets used across the app.
Streamlit's caching decorator is used so files are read from disk once
per session, keeping the app fast even as pages are switched.
"""

import os
import pandas as pd
import streamlit as st

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


@st.cache_data
def load_states():
    df = pd.read_csv(os.path.join(DATA_DIR, "states.csv"))
    return df


@st.cache_data
def load_festivals():
    df = pd.read_csv(os.path.join(DATA_DIR, "festivals.csv"))
    return df


@st.cache_data
def load_art_forms():
    df = pd.read_csv(os.path.join(DATA_DIR, "art_forms.csv"))
    return df


@st.cache_data
def load_cuisines():
    df = pd.read_csv(os.path.join(DATA_DIR, "cuisines.csv"))
    return df


@st.cache_data
def load_languages():
    df = pd.read_csv(os.path.join(DATA_DIR, "languages.csv"))
    return df


@st.cache_data
def load_monuments():
    df = pd.read_csv(os.path.join(DATA_DIR, "monuments.csv"))
    return df


@st.cache_data
def load_all():
    """Convenience loader returning every dataset as a dict of DataFrames."""
    return {
        "states": load_states(),
        "festivals": load_festivals(),
        "art_forms": load_art_forms(),
        "cuisines": load_cuisines(),
        "languages": load_languages(),
        "monuments": load_monuments(),
    }
