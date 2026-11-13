import streamlit as st
import pandas as pd
from src.data_loader import load_dataset, list_sheets
from src.column_mapper import detect_schema
from src.data_validator import quality_report, validate_for_forecasting
from src.preprocessing import build_panel

st.markdown('<div class="eyebrow">Data</div>', unsafe_allow_html=True)
st.title("Dataset")

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded is None:
    st.markdown('<div class="empty">Drag and drop a CSV or Excel file to begin.</div>', unsafe_allow_html=True)
    st.stop()

try:
    sheets = list_sheets(uploaded)
    sheet = st.selectbox("Sheet", sheets) if sheets else 0
    df = load_dataset(uploaded, sheet_name=sheet)
except Exception as e:
    st.error(f"Could not read file: {e}")
    st.stop()

st.caption(f"{df.shape[0]:,} rows · {df.shape[1]} columns")
st.dataframe(df.head(50), use_container_width=True)

auto = detect_schema(df)
schema = dict(auto["schema"])

st.subheader("Column mapping")
cols = ["—"] + list(df.columns)

with st.expander("Auto-detected roles — click to review or override", expanded=True):
    c1, c2 = st.columns(2)
    schema["date"] = c1.selectbox("Date column", cols,
                                    index=cols.index(schema["date"]) if schema.get("date") in cols else 0,
                                    key="map_date")
    schema["target"] = c2.selectbox("Target column", cols,
                                      index=cols.index(schema["target"]) if schema.get("target") in cols else 0,
                                      key="map_target")
    schema["product"] = c1.selectbox("Product column (optional)", cols,
                                       index=cols.index(schema["product"]) if schema.get("product") in cols else 0,
                                       key="map_prod")
    schema["store"] = c2.selectbox("Store column (optional)", cols,
                                     index=cols.index(schema["store"]) if schema.get("store") in cols else 0,
                                     key="map_store")
    schema["category"] = c1.selectbox("Category column (optional)", cols,
                                        index=cols.index(schema["category"]) if schema.get("category") in cols else 0,
                                        key="map_cat")
    schema["quantity"] = c2.selectbox("Quantity column (optional)", cols,
                                        index=cols.index(schema["quantity"]) if schema.get("quantity") in cols else 0,
                                        key="map_qty")

schema = {k: (None if v == "—" else v) for k, v in schema.items()}
schema["series_keys"] = [c for c in [schema.get("product"), schema.get("store"), schema.get("category")] if c]

st.subheader("Validation")
val = validate_for_forecasting(df, schema)
if val["ok"]:
    st.success("Dataset is compatible with the forecasting pipeline.")
else:
    st.error("Dataset cannot be used for forecasting:")
    for i in val["issues"]:
        st.write("• " + i)

if val["issues"] and val["ok"]:
    with st.expander("Warnings"):
        for i in val["issues"]:
            st.write("• " + i)

with st.expander("Data quality report"):
    rep = quality_report(df)
    st.write(f"Duplicates: **{rep['duplicates']}**")
    if rep["missing_pct"]:
        st.write("Missing values:")
        st.dataframe(pd.Series(rep["missing_pct"], name="%").to_frame(), use_container_width=True)

if st.button("Load dataset into session", type="primary", disabled=not val["ok"]):
    try:
        with st.spinner("Building panel…"):
            panel, freq = build_panel(df, schema)
        st.session_state.update({
            "raw_df": df, "schema": schema, "schema_meta": auto,
            "panel": panel, "freq": freq,
            "bundle": None, "forecast": None, "backtest": None,
        })
        st.success(f"Loaded {panel['series_id'].nunique()} series · frequency **{freq}**")
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
