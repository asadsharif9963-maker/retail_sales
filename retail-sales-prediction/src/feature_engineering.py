import numpy as np
import pandas as pd

TIME_FEATURES = ["year","month","day","dow","doy","week","quarter",
                 "is_weekend","is_month_start","is_month_end"]


def add_time_features(df):
    d = df["date"]
    df = df.copy()
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["dow"] = d.dt.dayofweek
    df["doy"] = d.dt.dayofyear
    df["week"] = d.dt.isocalendar().week.astype(int)
    df["quarter"] = d.dt.quarter
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["is_month_start"] = d.dt.is_month_start.astype(int)
    df["is_month_end"] = d.dt.is_month_end.astype(int)
    return df


def select_lag_windows(panel):
    min_len = int(panel.groupby("series_id").size().min())
    lags = [l for l in (1, 7, 14, 28, 30, 90) if l < min_len]
    windows = [w for w in (7, 14, 30, 90) if w < min_len]
    return lags or [1], windows


def add_history_features(panel, lags, windows):
    df = panel.sort_values(["series_id", "date"]).copy()
    g = df.groupby("series_id", sort=False)

    for l in lags:
        df[f"lag_{l}"] = g["y"].shift(l)

    for w in windows:
        minp = max(2, w // 2)
        df[f"rmean_{w}"] = g["y"].transform(
            lambda x, w=w, minp=minp: x.shift(1).rolling(w, min_periods=minp).mean())
        df[f"rstd_{w}"] = g["y"].transform(
            lambda x, w=w, minp=minp: x.shift(1).rolling(w, min_periods=minp).std())
        df[f"rmed_{w}"] = g["y"].transform(
            lambda x, w=w, minp=minp: x.shift(1).rolling(w, min_periods=minp).median())
    return df


def build_feature_frame(panel, cat_cols):
    lags, windows = select_lag_windows(panel)
    df = add_time_features(panel)
    df = add_history_features(df, lags, windows)

    hist_cols = [f"lag_{l}" for l in lags] + \
                [f"{p}_{w}" for w in windows for p in ("rmean","rstd","rmed")]
    feat_cols = TIME_FEATURES + hist_cols + [c for c in cat_cols if c in df.columns]

    before = len(df)
    df = df.dropna(subset=hist_cols).reset_index(drop=True)
    df.attrs["lags"] = lags
    df.attrs["windows"] = windows
    df.attrs["feature_cols"] = feat_cols
    print(f"Feature frame: {len(df)} rows (dropped {before - len(df)} warm-up rows), "
          f"{len(feat_cols)} features, lags={lags}, windows={windows}")
    return df, feat_cols
