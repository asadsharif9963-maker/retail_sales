import streamlit as st
import plotly.express as px
from src.session import require_data

st.markdown('<div class="eyebrow">Analytics</div>', unsafe_allow_html=True)
st.title("Product analytics")
require_data()

panel = st.session_state["panel"]
schema = st.session_state["schema"]
prod = schema.get("product")

if not prod or prod not in panel.columns:
    st.info("This dataset has no product column.")
    st.stop()

agg = panel.groupby(prod)["y"].sum().sort_values(ascending=False)
c1, c2 = st.columns(2)
c1.subheader("Top products")
c1.dataframe(agg.head(10).to_frame("sales"), use_container_width=True)
c2.subheader("Lowest performers")
c2.dataframe(agg.tail(10).to_frame("sales"), use_container_width=True)

top = agg.head(8).index
sub = panel[panel[prod].isin(top)]
fig = px.line(sub, x="date", y="y", color=prod, template="plotly_white", title="Sales trend — top products")
fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, use_container_width=True)

fc = st.session_state.get("forecast")
if fc is not None and prod in fc.columns:
    st.subheader("Forecasted product sales")
    fp = fc.groupby(prod)["y_pred"].sum().sort_values(ascending=False)
    st.dataframe(fp.to_frame("forecasted_sales"), use_container_width=True)
