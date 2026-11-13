import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .evaluation import metrics
from .preprocessing import seasonal_period

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- encoding
def fit_cat_maps(df, cat_cols):
    maps = {}
    for c in cat_cols:
        vals = sorted(df[c].astype(str).unique())
        maps[c] = {v: i for i, v in enumerate(vals)}
    return maps


def encode_cats(df, cat_cols, maps):
    out = df.copy()
    for c in cat_cols:
        out[c] = out[c].astype(str).map(maps[c]).fillna(-1).astype(int)
    return out


# ---------------------------------------------------------------- splitting
def chronological_split(panel, n_test, n_val):
    dates = np.sort(panel["date"].unique())
    if len(dates) < n_test + n_val + 5:
        raise ValueError(
            f"Not enough time steps ({len(dates)}) for test={n_test} + val={n_val}.")
    test_start = dates[-n_test]
    val_start  = dates[-(n_test + n_val)]
    train = panel[panel["date"] <  val_start]
    val   = panel[(panel["date"] >= val_start) & (panel["date"] < test_start)]
    test  = panel[panel["date"] >= test_start]
    return train, val, test


def holdout_steps(freq):
    return {"D": (14, 14), "W": (4, 4), "MS": (3, 3)}.get(freq, (14, 14))


# ---------------------------------------------------------------- models
def _cat_indices(feature_cols, cat_cols):
    return [feature_cols.index(c) for c in cat_cols if c in feature_cols]


def _fit_catboost(X, y, cat_idx):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.05,
                          loss_function="RMSE", verbose=False,
                          random_seed=42, cat_features=cat_idx,
                          allow_writing_files=False)
    m.fit(X, y)
    return m


def _fit_rf(X, y):
    m = RandomForestRegressor(n_estimators=300, max_depth=14, n_jobs=-1,
                              random_state=42)
    m.fit(X, y)
    return m


def train_one(model_name, X_tr, y_tr, feat_cols, cat_cols):
    cat_idx = _cat_indices(feat_cols, cat_cols)
    if model_name == "CatBoost":
        return _fit_catboost(X_tr, y_tr, cat_idx)
    if model_name == "RandomForest":
        return _fit_rf(X_tr, y_tr)
    raise ValueError(f"Unknown model {model_name}")


# ---------------------------------------------------------------- baselines
def baseline_predictions(df, freq, model_name):
    sp = seasonal_period(freq)
    col = f"lag_{sp}" if f"lag_{sp}" in df.columns else "lag_1"
    if model_name == "Naive":
        return df["lag_1"].values
    return df[col].values


def evaluate_model(model, X, y, feat_cols, cat_cols):
    if hasattr(model, "predict") and not isinstance(model, str):
        pred = model.predict(X)
    else:
        pred = model
    return metrics(y, pred)


def run_model_suite(train_df, val_df, feat_cols, cat_cols, freq, small_data=False):
    """Train/evaluate all candidates. Returns list of dicts sorted by WAPE."""
    results = []
    cat_maps = fit_cat_maps(train_df, cat_cols)

    val_enc = encode_cats(val_df, cat_cols, cat_maps)
    X_val = val_enc[feat_cols].values
    y_val = val_df["y"].values

    # Baselines (no training)
    for name in ("Naive", "SeasonalNaive"):
        preds = baseline_predictions(val_df, freq, name)
        m = metrics(y_val, preds)
        results.append({"model": name, "object": None, **m})

    # ML models
    train_enc = encode_cats(train_df, cat_cols, cat_maps)
    X_tr = train_enc[feat_cols].values
    y_tr = train_df["y"].values

    names = ["CatBoost", "RandomForest"]
    if small_data: names = ["RandomForest"]

    for name in names:
        try:
            mdl = train_one(name, X_tr, y_tr, feat_cols, cat_cols)
            preds = mdl.predict(X_val)
            m = metrics(y_val, preds)
            results.append({"model": name, "object": mdl, "cat_maps": cat_maps, **m})
        except Exception as e:
            print(f"  {name} failed: {e}")
            results.append({"model": name, "object": None, "error": str(e),
                            "WAPE": np.inf})

    results.sort(key=lambda r: (r.get("WAPE") if r.get("WAPE") is not None else np.inf))
    return results, cat_maps
