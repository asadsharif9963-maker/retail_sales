import streamlit as st


def init():
    for k, v in {
        "raw_df": None, "schema": None, "schema_meta": None,
        "panel": None, "freq": None, "bundle": None, "bundle_meta": None,
        "forecast": None, "history": [], "backtest": None,
    }.items():
        st.session_state.setdefault(k, v)


def has_data():
    return st.session_state.get("panel") is not None


def has_forecast():
    return st.session_state.get("forecast") is not None


def require_data():
    if not has_data():
        st.markdown("### No dataset loaded")
        st.info("Upload a dataset from the **Dataset** page to begin.")
        st.stop()
