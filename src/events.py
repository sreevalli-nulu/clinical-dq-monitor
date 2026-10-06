import numpy as np
import pandas as pd

from src.derive import add_study_day, parse_dtc
from src.load import load_domain
from src.subjects import build_subjects

SEVERITY_ORDER = {"MILD": 1, "MODERATE": 2, "SEVERE": 3}
SEVERITY_NAME = {v: k for k, v in SEVERITY_ORDER.items()}


def _start_vs_dose(start_text: pd.Series, dose_date: pd.Series):
    """Compare an AE start date with the first-dose date AT THE PRECISION AVAILABLE.

    A partial date is a range: '2012-02' = 1 Feb to 29 Feb, '2013' = the whole year.
    - range entirely before the dose date  -> BEFORE_DOSE
    - range entirely on/after the dose date -> AFTER_DOSE
    - range contains the dose date          -> AMBIGUOUS (only the site can tell us)
    Returns (precision, relation).
    """
    s = start_text.astype("string")
    length = s.str.len().fillna(0).to_numpy()
    precision = np.select([length >= 10, length == 7, length == 4], ["DAY", "MONTH", "YEAR"], "MISSING")

    day = parse_dtc(s)
    month_start = pd.to_datetime(s.str[:7] + "-01", format="%Y-%m-%d", errors="coerce").where(precision == "MONTH")
    year_start = pd.to_datetime(s.str[:4] + "-01-01", format="%Y-%m-%d", errors="coerce").where(precision == "YEAR")
    low = day.fillna(month_start).fillna(year_start)
    high = day.fillna(month_start + pd.offsets.MonthEnd(0)).fillna(year_start + pd.offsets.YearEnd(0))

    relation = np.select(
        [dose_date.isna(), low.isna(), high < dose_date, low >= dose_date],
        ["NO_DOSE_DATE", "UNKNOWN", "BEFORE_DOSE", "AFTER_DOSE"],
        default="AMBIGUOUS",
    )
    return precision, relation


def build_ae() -> pd.DataFrame:
    """One row per adverse EVENT (not per record).

    The raw AE file has several records for one event (follow-up updates with the same sponsor
    id AESPID). We keep the latest record, the worst severity, and how many records there were.
    """
    ae = load_domain("ae")
    subjects = build_subjects()[["USUBJID", "SITEID", "ARM", "ACTARM", "DOSED", "RFSTDT"]]
    ae = ae.merge(subjects, on="USUBJID", how="left", validate="many_to_one")
    if ae["SITEID"].isna().any():
        raise ValueError("AE has subjects that are missing from DM")

    ae["SEV_RANK"] = ae["AESEV"].map(SEVERITY_ORDER)
    ae["UPDATED"] = parse_dtc(ae["AEDTC"])
    ae = ae.sort_values(["USUBJID", "AESPID", "UPDATED", "AESEQ"])

    keys = ["USUBJID", "AESPID"]
    grouped = ae.groupby(keys, sort=False)
    summary = grouped.agg(N_RECORDS=("AESEQ", "size"), MAX_SEV=("SEV_RANK", "max"), MIN_SEV=("SEV_RANK", "min"))
    ev = ae.groupby(keys, sort=False).tail(1).merge(summary, on=keys, validate="one_to_one")

    ev["SEVERITY"] = ev["MAX_SEV"].map(SEVERITY_NAME)
    ev["SEV_CHANGED"] = ev["MAX_SEV"] != ev["MIN_SEV"]
    ev["START_TEXT"] = ev["AESTDTC"]
    ev["START_DATE"] = parse_dtc(ev["AESTDTC"])
    ev["END_DATE"] = parse_dtc(ev["AEENDTC"])
    ev["LAST_UPDATE"] = ev["UPDATED"]
    ev = add_study_day(ev, "START_DATE", ref_col="RFSTDT", out_col="START_DAY")
    ev["START_PRECISION"], ev["START_VS_DOSE"] = _start_vs_dose(ev["AESTDTC"], ev["RFSTDT"])

    ev = ev.rename(columns={"AESER": "SERIOUS", "AEREL": "RELATED", "AEOUT": "OUTCOME",
                            "AEACN": "ACTION", "AESDTH": "DIED"})
    cols = ["USUBJID", "SITEID", "ARM", "ACTARM", "DOSED", "AESPID", "AETERM", "AEDECOD",
            "START_TEXT", "START_DATE", "START_PRECISION", "START_DAY", "START_VS_DOSE", "END_DATE",
            "SEVERITY", "SEV_CHANGED", "SERIOUS", "RELATED", "OUTCOME", "ACTION", "DIED",
            "N_RECORDS", "LAST_UPDATE"]
    return ev[cols].sort_values(["USUBJID", "AESPID"]).reset_index(drop=True)