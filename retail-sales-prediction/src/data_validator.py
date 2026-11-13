import pandas as pd
import numpy as np


def quality_report(df):
    rep = {"shape": df.shape, "columns": list(df.columns)}
    miss = df.isna().sum()
    rep["missing"] = {c: int(v) for c, v in miss.items() if v > 0}
    rep["missing_pct"] = {c: round(100 * v / max(len(df), 1), 2) for c, v in miss.items() if v > 0}
    rep["duplicates"] = int(df.duplicated().sum())
    rep["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}

    num_summary, cat_summary, outliers = {}, {}, {}
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            s = df[c].dropna()
            if s.empty: continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = int(((s < lo) | (s > hi)).sum())
            num_summary[c] = {"min": float(s.min()), "max": float(s.max()),
                              "mean": float(s.mean()), "std": float(s.std() or 0)}
            if n_out: outliers[c] = n_out
        else:
            vc = df[c].astype(str).value_counts()
            cat_summary[c] = {"unique": int(df[c].nunique()),
                              "top": vc.index[0] if len(vc) else None,
                              "top_freq": int(vc.iloc[0]) if len(vc) else 0}

    rep["numeric_summary"] = num_summary
    rep["categorical_summary"] = cat_summary
    rep["outliers"] = outliers
    return rep


def validate_for_forecasting(df, schema):
    issues, ok = [], True
    if not schema.get("date"):   issues.append("Missing date column."); ok = False
    if not schema.get("target"): issues.append("Missing target column."); ok = False
    if ok:
        d = pd.to_datetime(df[schema["date"]], errors="coerce")
        bad = int(d.isna().sum())
        if bad == len(df):
            issues.append("Date column could not be parsed at all."); ok = False
        elif bad:
            issues.append(f"{bad} rows have invalid dates (will be dropped).")
        valid = d.dropna()
        if len(valid.unique()) < 5:
            issues.append("Fewer than 5 distinct dates — not enough history."); ok = False
        t = pd.to_numeric(df[schema["target"]], errors="coerce")
        if t.notna().sum() < 10:
            issues.append("Fewer than 10 numeric target values."); ok = False
    return {"ok": ok, "issues": issues}
