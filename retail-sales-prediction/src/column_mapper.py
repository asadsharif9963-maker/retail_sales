import re
import pandas as pd

DATE_HINTS   = ["date","order date","invoice date","transaction date","purchase date","datetime","timestamp"]
TARGET_HINTS = ["sales","revenue","total amount","amount","total","gmv","turnover","net sales","income"]
QTY_HINTS    = ["quantity","qty","units sold","units","volume"]
PRICE_HINTS  = ["unit price","unitprice","price","rate"]
DISC_HINTS   = ["discount","markdown","disc"]
PROMO_HINTS  = ["promotion","promo","campaign","on promo"]
PROD_HINTS   = ["product id","product","item id","item","sku","product code","stock code","product name"]
STORE_HINTS  = ["store id","store","shop","branch","outlet","city"]
CAT_HINTS    = ["sub category","subcategory","product category","category","department","class","type"]

_EXCLUDE = set(QTY_HINTS + PRICE_HINTS + DISC_HINTS)


def _score(col, hints):
    c = str(col).lower().strip()
    for i, h in enumerate(hints):
        if c == h: return 100 - i
    for i, h in enumerate(hints):
        if h in c: return 60 - i
    return 0


def _best(df, hints, numeric=None, exclude=()):
    cands = []
    for c in df.columns:
        if c in exclude: continue
        if numeric is True and not pd.api.types.is_numeric_dtype(df[c]): continue
        if numeric is False and pd.api.types.is_numeric_dtype(df[c]): continue
        s = _score(c, hints)
        if s > 0: cands.append((s, c))
    return sorted(cands, reverse=True)


def _date_confidence(series):
    s = series.dropna().head(500)
    if s.empty: return 0.0
    if pd.api.types.is_datetime64_any_dtype(series): return 1.0
    parsed = pd.to_datetime(s, errors="coerce")
    return float(parsed.notna().mean())


def detect_schema(df):
    """Auto-detect column roles. Returns dict + candidate lists + warnings."""
    warnings = []
    candidates = {k: [c for _, c in _best(df, h)] for k, h in {
        "date": DATE_HINTS, "target": TARGET_HINTS, "quantity": QTY_HINTS,
        "price": PRICE_HINTS, "discount": DISC_HINTS, "promo": PROMO_HINTS,
        "product": PROD_HINTS, "store": STORE_HINTS, "category": CAT_HINTS,
    }.items()}

    # Date: score by parse success
    date_col, best = None, 0.0
    for c in df.columns:
        conf = _date_confidence(df[c])
        bonus = 0.15 if _score(c, DATE_HINTS) > 0 else 0.0
        if conf + bonus > best and conf > 0.5:
            best, date_col = conf + bonus, c
    if date_col is None:
        warnings.append("No date-like column detected. Manual mapping required.")

    # Target: numeric, prefers revenue/sales names, excludes qty/price
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    target_col, tbest = None, 0
    for c in numeric_cols:
        if c in _EXCLUDE or c == date_col: continue
        s = _score(c, TARGET_HINTS) + 10   # any numeric candidate is plausible
        if s > tbest: tbest, target_col = s, c
    if target_col is None:
        warnings.append("No numeric target candidate found. Select manually.")

    def pick(role, numeric):
        lst = _best(df, {"quantity":QTY_HINTS,"price":PRICE_HINTS,
                         "discount":DISC_HINTS}.get(role, []),
                    numeric=numeric, exclude={target_col, date_col})
        return lst[0][1] if lst else None

    product  = candidates["product"][0]  if candidates["product"]  else None
    store    = candidates["store"][0]    if candidates["store"]    else None
    category = candidates["category"][0] if candidates["category"] else None
    promo    = candidates["promo"][0]    if candidates["promo"]    else None

    schema = {
        "date": date_col, "target": target_col,
        "quantity": pick("quantity", True), "price": pick("price", True),
        "discount": pick("discount", True), "promo": promo,
        "product": product, "store": store, "category": category,
    }
    schema["series_keys"] = [c for c in [product, store, category] if c]
    if not schema["series_keys"]:
        warnings.append("No product/store/category detected — treating dataset as a single aggregate series.")
    if len(numeric_cols) == 0:
        warnings.append("Dataset has no numeric columns.")
    return {"schema": schema, "candidates": candidates, "warnings": warnings}
