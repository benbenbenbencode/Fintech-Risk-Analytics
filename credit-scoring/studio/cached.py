"""Streamlit-cached wrappers around the heavier policy computations."""
import json

import streamlit as st

from . import policy


@st.cache_data(show_spinner=False)
def collections_plan():
    return policy.collections_plan()


@st.cache_data(show_spinner=False)
def cli_proposals(params_json: str, products: tuple):
    return policy.cli_proposals(json.loads(params_json), products)


@st.cache_data(show_spinner=False)
def strategy_curve(product: str):
    return policy.strategy_curve(product)
