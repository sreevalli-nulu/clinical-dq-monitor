import numpy as np
import pandas as pd


def add_study_day(df: pd.DataFrame, date_col: str, ref_col: str = "RFSTDTC") -> pd.DataFrame:
    """Add a study-day column (e.g. SVSTDTC -> SVSTDY).
    SDTM rule: first dose = day 1, day before = day -1, there is no day 0."""
    date = pd.to_datetime(df[date_col], errors="coerce")
    ref = pd.to_datetime(df[ref_col], errors="coerce")
    diff = (date - ref).dt.days
    df[date_col.replace("DTC", "DY")] = np.where(diff >= 0, diff + 1, diff)
    return df
