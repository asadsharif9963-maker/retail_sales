import streamlit as st
import pandas as pd
from datetime import datetime, timezone
from src.session import require_data
from src.feature_engineering import build_feature_frame
from src.model_training import (
    chronological_split, holdout_steps,
    run_model_suite, train_one, fit_cat_maps,
    encode_cats
)
from src.evaluation import metrics
from src.forecasting import recursive_forecast

st.markdown('<div class="eyebrow">Data</div>', unsafe_allow_html=True)
st.title("Sales prediction")
require_data()

panel = st.session_state["panel"]
freq = st.session_state["freq"]
schema = st.session_state["schema"]

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    horizon = c1.number_input("Horizon (steps)", 1, 180, 30)
    freq_choice = c2.selectbox("Frequency", ["D", "W", "MS"], index=["D", "W", "MS"].index(freq))
    all_series = sorted(panel["series_id"].unique())
    selected = c3.multiselect("Series (optional)", all_series, default=all_series[:min(5, len(all_series))])
    st.caption("Assumption: price, discount and promotion are held at their last observed values across the forecast horizon.")
    run = st.button("Run forecast", type="primary")

if run:
    with st.status("Training model on this dataset…", expanded=True) as status:
        CAT_COLS = [c for c in [schema.get("product"), schema.get("store"), schema.get("category")] if c]
        feat_df, FEAT_COLS = build_feature_frame(panel, CAT_COLS)
        n_test, n_val = holdout_steps(freq_choice)
        train_df, val_df, test_df = chronological_split(feat_df, n_test, n_val)

        small = len(train_df) < 400
        suite, cat_maps = run_model_suite(train_df, val_df, FEAT_COLS, CAT_COLS, freq_choice, small)
        best = suite[0]
        st.write(f"Best model on validation: **{best['model']}** (WAPE {best['WAPE']:.2f}%)")

        tv = pd.concat([train_df, val_df], ignore_index=True)
        cat_maps = fit_cat_maps(tv, CAT_COLS)
        if best["model"] in ("CatBoost", "RandomForest"):
            enc = encode_cats(tv, CAT_COLS, cat_maps)
            model = train_one(best["model"], enc[FEAT_COLS].values, tv["y"].values, FEAT_COLS, CAT_COLS)
        else:
            model = None

        enc_t = encode_cats(test_df, CAT_COLS, cat_maps)
        if model is not None:
            preds = model.predict(enc_t[FEAT_COLS].values)
        else:
            from src.model_training import baseline_predictions
            preds = baseline_predictions(test_df, freq_choice, best["model"])
        test_metrics = metrics(test_df["y"].values, preds)

        bundle = {
            "model": model,
            "feature_names": FEAT_COLS,
            "cat_cols": CAT_COLS,
            "cat_maps": cat_maps,
            "lags": feat_df.attrs["lags"],
            "windows": feat_df.attrs["windows"],
            "freq": freq_choice,
            "series_keys": schema["series_keys"],
            "date_col": schema["date"],
            "target_col": schema["target"],
            "model_name": best["model"],
            "metrics": test_metrics,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        fc = recursive_forecast(panel, bundle, int(horizon), series_filter=selected or None)
        status.update(label="Forecast complete", state="complete")

    st.session_state["bundle"] = bundle
    st.session_state["forecast"] = fc
    st.session_state["backtest"] = test_df.assign(y_pred=preds)[["series_id", "date", "y", "y_pred"]]
    st.session_state["history"] = st.session_state.get("history", []) + [{
        "timestamp": bundle["trained_at"],
        "model": bundle["model_name"],
        "horizon": int(horizon),
        "series": len(fc["series_id"].unique()),
        "total_forecast": round(float(fc["y_pred"].sum()), 2),
        "wape": round(float(test_metrics["WAPE"]), 2),
    }]
    st.success("Model trained and forecast generated.")

if st.session_state.get("forecast") is not None:
    fc = st.session_state["forecast"]
    st.subheader("Forecast")
    agg = fc.groupby("date")["y_pred"].sum().reset_index()
    st.metric("Total forecasted sales", f"{agg['y_pred'].sum():,.0f}")
    st.dataframe(agg.tail(30), use_container_width=True)
