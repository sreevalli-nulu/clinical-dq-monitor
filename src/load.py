from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def load_domain(name: str) -> pd.DataFrame:
    """Load one SDTM domain (e.g. 'dm', 'vs') from data/raw as a DataFrame."""
    path = RAW_DIR / f"{name.lower()}.xpt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/download_data.py first.")
    df = pd.read_sas(path, format="xport", encoding="utf-8")
    # SAS stores missing text as "" - convert to real NaN so .isna() catches it
    return df.replace(r"^\s*$", np.nan, regex=True)