import streamlit as st
import plotly.graph_objects as go
from src.session import require_data
ACCENT = "#2A6F62"

st.markdown('<div class="eyebrow">Analytics</div>', unsafe_allow_html=True)
st.title("Sales analytics")
require_data()

panel = st.session_state["panel"]
forecast = st.session_state.get("forecast")

hist = panel.groupby("date")["y"].sum().reset_index()
fig = go.Figure()
fig.add_trace(go.Scatter(x=hist["date"], y=hist["y"], name="Historical", line=dict(color=ACCENT, width=2)))
if forecast is not None:
    fg = forecast.groupby("date")["y_pred"].sum().reset_index()
    fig.add_trace(go.Scatter(x=fg["date"], y=fg["y_pred"], name="Forecast", line=dict(color="#C4501E", width=2, dash="dash")))
fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

c1, c2 = st.columns(2)
p = panel.copy()
p["month"] = p["date"].dt.to_period("M").astype(str)
c1.subheader("Monthly totals")
c1.dataframe(p.groupby("month")["y"].sum().to_frame("sales"), use_container_width=True)
p["dow"] = p["date"].dt.day_name()
c2.subheader("Day-of-week totals")
c2.dataframe(p.groupby("dow")["y"].sum().to_frame("sales"), use_container_width=True)
