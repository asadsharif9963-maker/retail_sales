import numpy as np
import pandas as pd

def _future_dates(last, freq, n):
    if freq == "D":  return [last + pd.Timedelta(days=i) for i in range(1, n + 1)]
    if freq == "W":  return [last + pd.Timedelta(weeks=i) for i in range(1, n + 1)]
    return list(pd.date_range(start=last, periods=n + 1, freq="MS")[1:])

def _one_row(hist_y, date, static, lags, windows):
    r = {"date": date}
    r["year"], r["month"], r["day"] = date.year, date.month, date.day
    r["dow"], r["doy"], r["quarter"] = date.dayofweek, date.dayofyear, date.quarter
    r["week"] = date.isocalendar()[1]
    r["is_weekend"] = int(date.dayofweek >= 5)
    r["is_month_start"] = int(date.is_month_start)
    r["is_month_end"] = int(date.is_month_end)
    for l in lags:
        r[f"lag_{l}"] = hist_y[-l] if len(hist_y) >= l else np.nan
    for w in windows:
        arr = np.array(hist_y[-w:], dtype=float)
        if len(arr) >= max(2, w // 2):
            r[f"rmean_{w}"], r[f"rstd_{w}"], r[f"rmed_{w}"] = arr.mean(), arr.std(ddof=0), np.median(arr)
        else:
            r[f"rmean_{w}"] = r[f"rstd_{w}"] = r[f"rmed_{w}"] = np.nan
    for k, v in static.items():
        r[k] = v
    return r

def recursive_forecast(panel, bundle, horizon, series_filter=None):
    model     = bundle["model"]
    feat_cols = bundle["feature_names"]
    cat_cols  = bundle["cat_cols"]
    cat_maps  = bundle["cat_maps"]
    lags      = bundle["lags"]
    windows   = bundle["windows"]
    freq      = bundle["freq"]
    keys      = list(dict.fromkeys(bundle["series_keys"]))

    from .model_training import encode_cats

    if series_filter:
        panel = panel[panel["series_id"].isin(series_filter)]

    # Clean panel columns
    panel = panel.loc[:, ~panel.columns.duplicated()]

    rows = []
    for sid, g in panel.groupby("series_id", sort=False):
        g = g.sort_values("date")
        if len(g) < max(lags + [1]) + 1:
            continue
        hist = g["y"].astype(float).tolist()
        static = {c: g[c].iloc[-1] for c in keys if c in g.columns}
        for c in ("price", "discount", "promo", "quantity"):
            if c in g.columns:
                static[c] = g[c].iloc[-1]

        # Resolve date column dynamically regardless of case
        date_col = "date" if "date" in g.columns else g.columns[g.columns.str.lower() == "date"][0]
        dates = _future_dates(g[date_col].iloc[-1], freq, horizon)

        for d in dates:
            row = _one_row(hist, d, static, lags, windows)
            row["y_pred"] = 0.0
            rows.append({"series_id": sid, **row})
            df_row = pd.DataFrame([row])
            enc = encode_cats(df_row, cat_cols, cat_maps)
            X = enc.reindex(columns=feat_cols).values
            if model is None:
                yhat = float(hist[-1])
            else:
                try:
                    yhat = float(model.predict(X)[0])
                except Exception:
                    yhat = float(np.mean(hist[-7:])) if len(hist) >= 7 else float(hist[-1])
            yhat = max(0.0, yhat)
            rows[-1]["y_pred"] = yhat
            hist.append(yhat)

    if not rows:
        return pd.DataFrame(columns=["series_id", "date", "y_pred"] + keys)

    out = pd.DataFrame(rows)

    # Securely copy static keys to forecast output without groupby merge bugs
    for key in keys:
        if key in panel.columns:
            mapping = panel.groupby("series_id")[key].first().to_dict()
            out[key] = out["series_id"].map(mapping)

    return out[["series_id", *keys, "date", "y_pred"]].sort_values(["series_id", "date"]).reset_index(drop=True)
