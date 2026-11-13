import numpy as np


def metrics(y_true, y_pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    mask = ~(np.isnan(y) | np.isnan(p))
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan,
                "MAPE": np.nan, "sMAPE": np.nan, "WAPE": np.nan, "n": 0}

    err = y - p
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    denom_var = np.sum((y - y.mean()) ** 2)
    r2 = float(1 - np.sum(err ** 2) / denom_var) if denom_var > 0 else np.nan

    nz = y != 0
    mape = float(np.mean(np.abs(err[nz] / y[nz])) * 100) if nz.any() else np.nan

    denom = (np.abs(y) + np.abs(p)) / 2
    safe = denom != 0
    smape = float(np.mean(np.abs(err[safe] / denom[safe])) * 100) if safe.any() else np.nan

    total = np.sum(np.abs(y))
    wape = float(np.sum(np.abs(err)) / total * 100) if total > 0 else np.nan

    return {"MAE": mae, "RMSE": rmse, "R2": r2,
            "MAPE": mape, "sMAPE": smape, "WAPE": wape, "n": int(len(y))}
