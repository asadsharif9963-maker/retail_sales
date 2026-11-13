import json
from pathlib import Path
import joblib
import pandas as pd


def save_bundle(bundle, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    cfg = {k: v for k, v in bundle.items() if k not in ("model", "cat_maps", "last_panel")}
    # convert non-serializable
    for k, v in list(cfg.items()):
        if isinstance(v, (pd.Timestamp,)):
            cfg[k] = str(v)
    joblib.dump(
        {"model": bundle.get("model"),
         "cat_maps": bundle.get("cat_maps"),
         "last_panel": bundle.get("last_panel")},
        out / "model.joblib"
    )
    (out / "model_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    return out


def load_bundle(in_dir):
    p = Path(in_dir)
    core = joblib.load(p / "model.joblib")
    cfg  = json.loads((p / "model_config.json").read_text())
    bundle = {**cfg, **core}
    return bundle


def compatibility_report(bundle, df, schema):
    issues, ok = [], True
    required_keys = bundle.get("series_keys", [])
    current_keys = schema.get("series_keys", [])
    missing = [k for k in required_keys if k not in current_keys]
    if missing:
        issues.append(f"Dataset is missing panel keys used by the model: {missing}")
        ok = False
    if schema.get("date") != bundle.get("date_col"):
        issues.append(f"Date column differs ('{schema.get('date')}' vs '{bundle.get('date_col')}').")
    if schema.get("target") != bundle.get("target_col"):
        issues.append(f"Target column differs ('{schema.get('target')}' vs '{bundle.get('target_col')}').")
    return {"ok": ok, "issues": issues, "retrain_recommended": not ok}
