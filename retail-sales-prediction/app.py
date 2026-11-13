import streamlit as st

st.set_page_config(
    page_title="Meridian — Retail Forecasting",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.session import init
init()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', -apple-system, system-ui, sans-serif; }
.stApp { background: #FAFAF7; color: #1A1A16; }
section[data-testid="stSidebar"] { background: #F4F3EF; border-right: 1px solid #E8E6E1; }
section[data-testid="stSidebar"] * { color: #1A1A16; }
h1 { font-size: 1.55rem !important; font-weight: 600 !important; letter-spacing: -0.01em; margin-bottom: 0.2rem !important; }
h2 { font-size: 1.08rem !important; font-weight: 600 !important; margin-top: 1.2rem !important; }
h3 { font-size: 0.95rem !important; font-weight: 600 !important; }
p, label, span, div { font-size: 0.90rem; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1300px; }
div[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid #E8E6E1; border-radius: 10px;
    padding: 16px 18px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);
}
div[data-testid="stMetricLabel"] { font-size: 0.72rem !important; text-transform: uppercase;
    letter-spacing: 0.06em; color: #6B6B63 !important; font-weight: 500 !important; }
div[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 600 !important; color: #1A1A16 !important; }
.stButton > button {
    background: #2A6F62; color: #FFFFFF; border: none; border-radius: 8px;
    padding: 0.5rem 1.2rem; font-weight: 500; font-size: 0.88rem;
    transition: background 0.15s ease;
}
.stButton > button:hover { background: #235A50; color: #FFFFFF; }
.stDownloadButton > button {
    background: #FFFFFF; color: #1A1A16; border: 1px solid #D9D6CE;
    border-radius: 8px; font-weight: 500; font-size: 0.86rem;
}
.stDownloadButton > button:hover { border-color: #2A6F62; color: #2A6F62; }
[data-testid="stFileUploader"] {
    background: #FFFFFF; border: 1px dashed #C9C5BB; border-radius: 10px; padding: 12px;
}
[data-testid="stDataFrame"] { border: 1px solid #E8E6E1; border-radius: 8px; }
.card { background:#FFFFFF; border:1px solid #E8E6E1; border-radius:10px; padding:20px 22px; margin-bottom:14px; }
.card-title { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.06em;
              color:#6B6B63; font-weight:500; margin-bottom:6px; }
.card-value { font-size:1.7rem; font-weight:600; color:#1A1A16; letter-spacing:-0.02em; }
.card-note { font-size:0.78rem; color:#8A8A82; margin-top:4px; }
.eyebrow { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em;
           color:#8A8A82; font-weight:500; margin-bottom:4px; }
.section-rule { border-top:1px solid #E8E6E1; margin: 18px 0 12px 0; }
.empty { background:#FFFFFF; border:1px solid #E8E6E1; border-radius:10px;
         padding:42px 24px; text-align:center; color:#6B6B63; }
.pill { display:inline-block; padding:2px 10px; border-radius:20px; font-size:0.74rem;
        font-weight:500; background:#E6F1EF; color:#2A6F62; }
.pill-warn { background:#FBEEE6; color:#C4501E; }
</style>
""", unsafe_allow_html=True)

P = "pages/"
nav = {
    "Overview": [st.Page(P + "dashboard.py", title="Dashboard", default=True)],
    "Data": [st.Page(P + "upload.py", title="Dataset"),
             st.Page(P + "prediction.py", title="Prediction")],
    "Analytics": [st.Page(P + "sales_analytics.py", title="Sales"),
                  st.Page(P + "product_analytics.py", title="Products"),
                  st.Page(P + "category_analytics.py", title="Categories"),
                  st.Page(P + "store_analytics.py", title="Stores")],
    "Model": [st.Page(P + "model_performance.py", title="Performance")],
    "Export": [st.Page(P + "reports.py", title="Reports")],
}

st.sidebar.markdown("### ◆ Meridian")
st.sidebar.caption("Retail forecasting console")
st.sidebar.markdown("---")
pg = st.navigation(nav)
pg.run()
st.sidebar.markdown("---")
st.sidebar.caption("Session state persists across pages.")
