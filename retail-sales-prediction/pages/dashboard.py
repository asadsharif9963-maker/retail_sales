import streamlit as st
import pandas as pd
from src.session import has_data, has_forecast

st.markdown('<div class="eyebrow">Overview</div>', unsafe_allow_html=True)
st.title("Dashboard")

if not has_data():
    st.markdown('<div class="empty">Upload a retail dataset to see your dashboard.</div>', unsafe_allow_html=True)
    st.stop()

panel = st.session_state["panel"]
forecast = st.session_state.get("forecast")
schema = st.session_state["schema"]
freq = st.session_state["freq"]

total_sales = panel["y"].sum()
avg_sales = panel["y"].mean()
total_units = panel["quantity"].sum() if "quantity" in panel.columns else None

panel_sorted = panel.sort_values("date")
n = len(panel_sorted)
split = max(1, n // 10)
recent, prior = panel_sorted.tail(split)["y"].sum(), panel_sorted.iloc[-2*split:-split]["y"].sum()
growth = ((recent - prior) / prior * 100) if prior else None

forecast_total = forecast["y_pred"].sum() if has_forecast() else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total historical sales", f"{total_sales:,.0f}")
c2.metric("Average per step", f"{avg_sales:,.1f}")
c3.metric("Total units", f"{total_units:,.0f}" if total_units is not None else "—")
c4.metric("Forecasted revenue", f"{forecast_total:,.0f}" if forecast_total else "—")

if growth is not None:
    arrow = "▲" if growth >= 0 else "▼"
    color = "#2A6F62" if growth >= 0 else "#C4501E"
    st.markdown(
        f'<div class="card" style="padding:12px 18px">'
        f'<span class="card-title">Period-over-period growth</span> '
        f'<span style="color:{color};font-weight:600">{arrow} {abs(growth):.1f}%</span>'
        f'</div>', unsafe_allow_html=True)

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Best performing product")
    if schema.get("product") and schema["product"] in panel.columns:
        best = panel.groupby(schema["product"])["y"].sum().sort_values(ascending=False)
        st.markdown(f"**{best.index[0]}** — {best.iloc[0]:,.0f} total sales")
        st.dataframe(best.head(5).to_frame("sales"), use_container_width=True)
    else:
        st.caption("No product column in this dataset.")

with col_b:
    st.subheader("Top category")
    if schema.get("category") and schema["category"] in panel.columns:
        best = panel.groupby(schema["category"])["y"].sum().sort_values(ascending=False)
        st.markdown(f"**{best.index[0]}** — {best.iloc[0]:,.0f} total sales")
        st.dataframe(best.head(5).to_frame("sales"), use_container_width=True)
    else:
        st.caption("No category column in this dataset.")

st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
st.subheader("Recent predictions")
hist = st.session_state.get("history", [])
if hist:
    st.dataframe(pd.DataFrame(hist[-10:]), use_container_width=True)
else:
    st.caption("No predictions yet. Run a forecast from the Prediction page.")
