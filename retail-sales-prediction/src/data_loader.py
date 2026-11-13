import pandas as pd


def list_sheets(source):
    try:
        xls = pd.ExcelFile(source)
        return xls.sheet_names
    except Exception:
        return []


def load_dataset(source, sheet_name=0):
    """Load a CSV/Excel file. `source` may be a path or a file-like object."""
    name = str(getattr(source, "name", source)).lower()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        return pd.read_excel(source, sheet_name=sheet_name or 0)
    if hasattr(source, "seek"):
        source.seek(0)
    try:
        df = pd.read_csv(source, sep=None, engine="python")
    except Exception:
        if hasattr(source, "seek"):
            source.seek(0)
        df = pd.read_csv(source)
    df.columns = [str(c).strip() for c in df.columns]
    return df
