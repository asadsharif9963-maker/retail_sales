import streamlit as st
import pandas as pd
from src.session import require_data
from src.reporting import to_csv_bytes, to_excel_bytes, to_pdf_bytes

st.markdown('<div class="eyebrow">Export</div>', unsafe_allow_html=True)
st.title("Reports")
require_data()

fc = st.session_state.get("forecast")
panel = st.session_state["panel"]
bundle = st.session_state.get("bundle")

if fc is not None:
    st.subheader("Forecast export")
    c1, c2, c3 = st.columns(3)
    c1.download_button("Download CSV", to_csv_bytes(fc), file_name="forecast.csv", mime="text/csv")
    sheets = {"Forecast": fc, "Historical": panel}
    if bundle:
        sheets["Metrics"] = pd.DataFrame([bundle["metrics"]])
    c2.download_button("Download Excel", to_excel_bytes(sheets), file_name="forecast_report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    summary = [
        f"Model: {bundle['model_name']}" if bundle else "Model: baseline",
        f"Horizon: {len(fc['date'].unique())} steps",
        f"Total forecasted sales: {fc['y_pred'].sum():,.2f}",
        f"Series covered: {fc['series_id'].nunique()}",
    ]
    c3.download_button("Download PDF", to_pdf_bytes("Retail Forecast Report", summary, fc.head(50)), file_name="forecast_report.pdf", mime="application/pdf")
else:
    st.info("No forecast available yet.")

st.subheader("Prediction history")
hist = st.session_state.get("history", [])
if hist:
    hdf = pd.DataFrame(hist)
    st.dataframe(hdf, use_container_width=True)
    st.download_button("Download history", to_csv_bytes(hdf), file_name="prediction_history.csv", mime="text/csv")
else:
    st.caption("Nothing yet — run a forecast first.")
