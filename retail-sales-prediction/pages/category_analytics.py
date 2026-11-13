import streamlit as st
import plotly.express as px
from src.session import require_data

st.markdown('<div class="eyebrow">Analytics</div>', unsafe_allow_html=True)
st.title("Category analytics")
require_data()

panel = st.session_state["panel"]
schema = st.session_state["schema"]
cat = schema.get("category")

if not cat or cat not in panel.columns:
    st.info("This dataset has no category column.")
    st.stop()

agg = panel.groupby(cat)["y"].sum().sort_values(ascending=False)
st.subheader("Category performance")
st.dataframe(agg.to_frame("sales"), use_container_width=True)

fig = px.bar(x=agg.index, y=agg.values, template="plotly_white", labels={"x": cat, "y": "sales"}, title="Total sales by category")
fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

fig2 = px.line(panel, x="date", y="y", color=cat, template="plotly_white", title="Category trends")
fig2.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig2, use_container_width=True)
