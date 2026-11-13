import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.session import require_data

st.markdown('<div class="eyebrow">Model</div>', unsafe_allow_html=True)
st.title("Model performance")
require_data()

bundle = st.session_state.get("bundle")
if bundle is None:
    st.info("Train a model from the Prediction page to see performance.")
    st.stop()

m = bundle["metrics"]
st.markdown(f'<span class="pill">{bundle["model_name"]}</span> '
            f'<span style="color:#8A8A82;font-size:0.85rem;margin-left:8px">'
            f'trained {bundle["trained_at"]}</span>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{m['MAE']:.2f}")
c2.metric("RMSE", f"{m['RMSE']:.2f}")
c3.metric("R²", f"{m['R2']:.3f}")
c4.metric("WAPE", f"{m['WAPE']:.2f}%")

st.caption("Metrics come from the chronological hold-out test window, never from the training data.")

bt = st.session_state.get("backtest")
if bt is not None:
    st.subheader("Actual vs predicted — test window")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt["date"], y=bt["y"], mode="markers", name="Actual", marker=dict(color="#2A6F62", size=5)))
    fig.add_trace(go.Scatter(x=bt["date"], y=bt["y_pred"], mode="markers", name="Predicted", marker=dict(color="#C4501E", size=5, symbol="diamond")))
    fig.update_layout(template="plotly_white", height=400, margin=dict(l=10, r=10, t=30, b=10))
st.plotly_chart(fig, use_container_width=True)

err = (bt["y"] - bt["y_pred"])
st.subheader("Error distribution")
st.dataframe(err.describe().to_frame("error"), use_container_width=True)
