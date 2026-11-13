import numpy as np
import pandas as pd

ROLES = ["quantity", "price", "discount", "promo"]
AGG  = {"y": "sum", "quantity": "sum", "price": "mean", "discount": "mean", "promo": "max"}


def detect_frequency(dates):
    s = pd.Series(pd.to_datetime(pd.Series(dates).dropna().unique())).sort_values()
    gaps = s.diff().dt.days.dropna()
    if gaps.empty: return "D"
    med = gaps.median()
    if med <= 1.5:  return "D"
    if med <= 8:    return "W"
    return "MS"


def build_panel(df, schema, freq=None):
    """Convert raw data → regular panel: one row per (series_id, date)."""
    date_col, target = schema["date"], schema["target"]
    work = df.copy()
    work["date"] = pd.to_datetime(work[date_col], errors="coerce")
    if getattr(work["date"].dt, "tz", None) is not None:
        work["date"] = work["date"].dt.tz_localize(None)
    work["y"] = pd.to_numeric(work[target], errors="coerce")
    work = work[work["date"].notna() & work["y"].notna()].copy()
    if work.empty:
        raise ValueError("No valid rows after date/target parsing.")

    key_cols = [c for c in schema.get("series_keys", []) if c in work.columns]
    for c in key_cols:
        work[c] = work[c].fillna("UNK").astype(str)
    work["series_id"] = work[key_cols].astype(str).agg("|".join, axis=1) if key_cols else "ALL"

    for role in ROLES:
        col = schema.get(role)
        if col and col in work.columns:
            work[role] = pd.to_numeric(work[col], errors="coerce")

    if not freq:
        freq = detect_frequency(work["date"])

    existing = [r for r in ROLES if r in work.columns and work[r].notna().any()]
    agg = {k: v for k, v in AGG.items() if k == "y" or k in existing}

    frames = []
    for sid, g in work.groupby("series_id", sort=False):
        g = g.set_index("date").sort_index()
        agg_out = g.resample(freq).agg(agg)
        agg_out["y"] = agg_out["y"].fillna(0.0)
        for r in existing:
            agg_out[r] = agg_out[r].ffill().bfill()
        agg_out["series_id"] = sid
        for c in key_cols:
            agg_out[c] = g[c].iloc[0]
        frames.append(agg_out.reset_index())

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["series_id", "date"]).reset_index(drop=True)

    # Drop series with almost no history
    counts = panel.groupby("series_id")["date"].count()
    valid = counts[counts >= 8].index
    dropped = len(counts) - len(valid)
    panel = panel[panel["series_id"].isin(valid)].reset_index(drop=True)
    if dropped:
        print(f"Dropped {dropped} series with <8 observations.")

    panel.attrs["frequency"] = freq
    panel.attrs["series_keys"] = key_cols
    return panel, freq


def seasonal_period(freq):
    return {"D": 7, "W": 4, "MS": 12}.get(freq, 7)
