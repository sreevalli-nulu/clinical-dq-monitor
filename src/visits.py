import numpy as np
import pandas as pd

from src.config import VISIT_WINDOW_DAYS
from src.derive import add_study_day, parse_dtc, visit_type
from src.load import load_domain
from src.subjects import build_subjects

SUBJECT_COLS = ["USUBJID", "SITEID", "ARM", "ACTARM", "DOSED", "RFSTDT"]
VISIT_COLS = ["VISITNUM", "VISIT", "VISIT_TYPE", "VISITDY", "DATE", "STUDY_DAY"]


def _prepare(domain: str, date_col: str) -> pd.DataFrame:
    """Load a domain, attach site/arm/population, parse dates, add study day and visit type."""
    df = load_domain(domain)
    subjects = build_subjects()[SUBJECT_COLS]
    df = df.merge(subjects, on="USUBJID", how="left", validate="many_to_one")
    if df["SITEID"].isna().any():
        raise ValueError(f"{domain.upper()} has subjects that are missing from DM")
    df["DATE"] = parse_dtc(df[date_col])
    df = add_study_day(df, "DATE", ref_col="RFSTDT", out_col="STUDY_DAY")
    df["VISIT_TYPE"] = visit_type(df["VISIT"], df["VISITNUM"])
    return df


def build_sv() -> pd.DataFrame:
    """One row per visit, with timing against the planned day."""
    sv = _prepare("sv", "SVSTDTC")
    # The window rule applies to planned visits after baseline that have a planned day.
    sv["WINDOW_APPLIES"] = (
        (sv["VISIT_TYPE"] == "PLANNED") & (sv["VISITNUM"] > 3)
        & sv["VISITDY"].notna() & sv["STUDY_DAY"].notna()
    )
    sv["DEVIATION"] = np.where(sv["WINDOW_APPLIES"], sv["STUDY_DAY"] - sv["VISITDY"], np.nan)
    sv["OUT_OF_WINDOW"] = sv["WINDOW_APPLIES"] & (sv["DEVIATION"].abs() > VISIT_WINDOW_DAYS)
    return sv[SUBJECT_COLS[:-1] + VISIT_COLS + ["WINDOW_APPLIES", "DEVIATION", "OUT_OF_WINDOW"]]


def build_vs() -> pd.DataFrame:
    """One row per vital-sign reading, standard units, with explicit NOT_DONE flag."""
    vs = _prepare("vs", "VSDTC")
    vs = vs.rename(columns={
        "VSTESTCD": "TESTCD", "VSTEST": "TEST", "VSPOS": "POS", "VSTPT": "TPT",
        "VSSTRESN": "RESULT", "VSSTRESU": "UNIT", "VSDY": "STUDY_DAY_SRC", "VSBLFL": "BASELINE_FLAG",
    })
    vs["NOT_DONE"] = vs["VSSTAT"].eq("NOT DONE")
    cols = (SUBJECT_COLS[:-1] + VISIT_COLS + ["STUDY_DAY_SRC", "TESTCD", "TEST", "POS", "TPT",
            "RESULT", "UNIT", "NOT_DONE", "BASELINE_FLAG"])
    return vs[cols]


def build_lb() -> pd.DataFrame:
    """One row per lab result, with reference ranges and flags for text-only and censored values."""
    lb = _prepare("lb", "LBDTC")
    lb = lb.rename(columns={
        "LBTESTCD": "TESTCD", "LBTEST": "TEST", "LBCAT": "CATEGORY",
        "LBSTRESN": "RESULT", "LBSTRESC": "RESULT_TXT", "LBSTRESU": "UNIT",
        "LBSTNRLO": "LO", "LBSTNRHI": "HI", "LBNRIND": "FLAG",
        "LBDY": "STUDY_DAY_SRC", "LBBLFL": "BASELINE_FLAG",
    })
    # Original (as-reported) value and range, kept for the unit-conversion check in Step 4
    lb["ORIG_RESULT"] = pd.to_numeric(lb["LBORRES"], errors="coerce")
    lb["ORIG_LO"] = pd.to_numeric(lb["LBORNRLO"], errors="coerce")
    lb["ORIG_HI"] = pd.to_numeric(lb["LBORNRHI"], errors="coerce")
    # No number: either censored ('<3.42' = below detection limit) or text by design (urine colour)
    text = lb["RESULT_TXT"].astype("string")
    lb["CENSORED"] = lb["RESULT"].isna() & text.str.match(r"^[<>]").fillna(False).to_numpy(dtype=bool)
    lb["TEXT_ONLY"] = lb["RESULT"].isna() & ~lb["CENSORED"]
    cols = (SUBJECT_COLS[:-1] + VISIT_COLS + ["STUDY_DAY_SRC", "CATEGORY", "TESTCD", "TEST",
            "RESULT", "RESULT_TXT", "UNIT", "LO", "HI", "FLAG", "ORIG_RESULT", "ORIG_LO", "ORIG_HI",
            "CENSORED", "TEXT_ONLY", "BASELINE_FLAG"])
    return lb[cols]