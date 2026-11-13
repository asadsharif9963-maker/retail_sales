import streamlit as st
import plotly.express as px
from src.session import require_data

st.markdown('<div class="eyebrow">Analytics</div>', unsafe_allow_html=True)
st.title("Store analytics")
require_data()

panel = st.session_state["panel"]
schema = st.session_state["schema"]
store = schema.get("store")

if not store or store not in panel.columns:
    st.info("This dataset has no store column.")
    st.stop()

agg = panel.groupby(store)["y"].sum().sort_values(ascending=False)
st.subheader("Store performance")
st.dataframe(agg.to_frame("sales"), use_container_width=True)

fig = px.bar(x=agg.index, y=agg.values, template="plotly_white", labels={"x": store, "y": "sales"}, title="Sales by store")
fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

fc = st.session_state.get("forecast")
if fc is not None and store in fc.columns:
    st.subheader("Store-level forecast")
    fs = fc.groupby(store)["y_pred"].sum().sort_values(ascending=False)
    st.dataframe(fs.to_frame("forecasted_sales"), use_container_width=True)
