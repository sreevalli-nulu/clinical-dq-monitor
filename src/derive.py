import numpy as np
import pandas as pd


def parse_dtc(s: pd.Series) -> pd.Series:
    """Turn SDTM ISO-8601 text dates into real dates (date only, time dropped).

    - Complete dates ('2013-04-04' or '2013-04-04T08:30') are parsed from their first 10 characters.
      This avoids a pandas trap: a column that mixes both formats can silently lose one of them.
    - Partial dates ('2012-02', '1977') become NaT ON PURPOSE. Letting pandas guess a day
      would hide the incomplete-date problem we found in 2C.
    """
    s = s.astype("string")
    complete = s.where(s.str.len() >= 10).str[:10]
    return pd.to_datetime(complete, format="%Y-%m-%d", errors="coerce")


def add_study_day(df: pd.DataFrame, date_col: str, ref_col: str = "RFSTDTC",
                  out_col: str | None = None) -> pd.DataFrame:
    """Add a study-day column. SDTM rule: first dose = day 1, day before = day -1, no day 0.
    date_col and ref_col may hold text dates or real dates."""
    date = df[date_col] if pd.api.types.is_datetime64_any_dtype(df[date_col]) else parse_dtc(df[date_col])
    ref = df[ref_col] if pd.api.types.is_datetime64_any_dtype(df[ref_col]) else parse_dtc(df[ref_col])
    diff = (date - ref).dt.days
    df[out_col or date_col.replace("DTC", "DY")] = np.where(diff >= 0, diff + 1, diff)
    return df


def visit_type(visit: pd.Series, visitnum: pd.Series) -> pd.Series:
    """Classify each visit: SCREENING, UNSCHEDULED, FOLLOWUP (numbers 100+) or PLANNED.
    PLANNED includes baseline, the weekly visits, and the ECG and telephone '(T)' visits."""
    v = visit.astype("string")
    conditions = [
        v.str.startswith("SCREENING").fillna(False).to_numpy(dtype=bool),
        v.str.startswith("UNSCHEDULED").fillna(False).to_numpy(dtype=bool),
        (visitnum >= 100).to_numpy(dtype=bool),
    ]
    out = np.select(conditions, ["SCREENING", "UNSCHEDULED", "FOLLOWUP"], default="PLANNED")
    return pd.Series(out, index=visit.index)