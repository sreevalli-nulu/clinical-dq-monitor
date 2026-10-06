"""Step 5: site-level statistical checks (key risk indicators, z-scores, digit preference).

Idea: for each site, compute a rate (e.g. dropout rate), compare it with the rate in ALL OTHER
sites, and express the gap in standard errors (a z-score). Small sites are not scored.
"""
import numpy as np
import pandas as pd

from src.config import MIN_SITE_SUBJECTS

WATCH_Z = 2.0
ALERT_Z = 3.0
KRI_COLUMNS = ["KRI", "SITEID", "N", "EVENTS", "RATE", "REF_RATE", "Z", "STATUS", "DIRECTION"]


def rate_zscores(counts: pd.DataFrame, min_n: int = MIN_SITE_SUBJECTS) -> pd.DataFrame:
    """counts has one row per site with columns SITEID, N (units at risk) and EVENTS.

    Each site is compared with all OTHER sites combined (leave-one-out), so a bad site cannot
    hide by pulling the average toward itself. z = (site rate - rest rate) / standard error,
    where the standard error comes from the binomial formula sqrt(p(1-p)/n).
    """
    c = counts[["SITEID", "N", "EVENTS"]].copy()
    n_all, e_all = c["N"].sum(), c["EVENTS"].sum()
    c["RATE"] = c["EVENTS"] / c["N"]
    c["REF_RATE"] = (e_all - c["EVENTS"]) / (n_all - c["N"])
    se = np.sqrt(c["REF_RATE"] * (1 - c["REF_RATE"]) / c["N"])
    c["Z"] = ((c["RATE"] - c["REF_RATE"]) / se).where(se > 0)
    small = c["N"] < min_n
    c.loc[small, "Z"] = np.nan
    c["STATUS"] = np.select([small, c["Z"].abs() >= ALERT_Z, c["Z"].abs() >= WATCH_Z, c["Z"].isna()],
                            ["TOO_SMALL", "ALERT", "WATCH", "NO_SPREAD"], default="OK")
    c["DIRECTION"] = np.select([c["Z"] > 0, c["Z"] < 0], ["HIGH", "LOW"], default="")
    return c


def mean_zscores(per_patient: pd.DataFrame, min_n: int = MIN_SITE_SUBJECTS, high_only: bool = False) -> pd.DataFrame:
    """per_patient has one row per patient: SITEID, VALUE (a number between 0 and 1).

    Same leave-one-out idea, but the unit is the PATIENT, not the individual reading. Readings
    from one patient (and one clinic's staff and devices) are not independent, so counting every
    reading as separate evidence makes z-scores far too large. z = (site mean - rest mean) /
    (spread of the rest's patients / sqrt(patients at this site)).
    """
    rows = []
    for site, g in per_patient.groupby("SITEID"):
        rest = per_patient.loc[per_patient["SITEID"] != site, "VALUE"]
        n = len(g)
        se = rest.std(ddof=1) / np.sqrt(n)
        z = (g["VALUE"].mean() - rest.mean()) / se if se > 0 else np.nan
        rows.append({"SITEID": site, "N": n, "EVENTS": pd.NA, "RATE": g["VALUE"].mean(),
                     "REF_RATE": rest.mean(), "Z": z})
    c = pd.DataFrame(rows)
    small = c["N"] < min_n
    c.loc[small, "Z"] = np.nan
    big = c["Z"].abs() >= ALERT_Z
    watch = c["Z"].abs() >= WATCH_Z
    if high_only:  # for digit preference only a HIGH share is a concern; a low share is the healthy baseline
        big, watch = big & (c["Z"] > 0), watch & (c["Z"] > 0)
    c["STATUS"] = np.select([small, big, watch, c["Z"].isna()], ["TOO_SMALL", "ALERT", "WATCH", "NO_SPREAD"], default="OK")
    c["DIRECTION"] = np.select([c["Z"] > 0, c["Z"] < 0], ["HIGH", "LOW"], default="")
    return c


def subject_indicators(view: dict, findings: pd.DataFrame) -> pd.DataFrame:
    """One row per DOSED subject with 0/1 flags used by the site KRIs."""
    s = view["subjects"]
    s = s[s["DOSED"]][["USUBJID", "SITEID", "DISCONTINUED"]].copy()
    ae, sv = view["ae"], view["sv"]
    s["ANY_AE"] = s["USUBJID"].isin(ae.loc[ae["START_VS_DOSE"] != "BEFORE_DOSE", "USUBJID"])
    s["OUT_OF_WINDOW_ANY"] = s["USUBJID"].isin(sv.loc[sv["OUT_OF_WINDOW"].fillna(False).astype(bool), "USUBJID"])
    serious = findings.loc[findings["SEVERITY"].isin(["HIGH", "MEDIUM"]), "USUBJID"]
    s["DQ_FLAGGED"] = s["USUBJID"].isin(serious)
    return s.reset_index(drop=True)


def digit_patient_table(view: dict, min_readings: int = 10) -> pd.DataFrame:
    """Per patient: share of their BP/pulse readings whose last digit is 0 or 5.

    If digits were random this would be about 20%. These devices mostly give even numbers, which
    also gives about 20% ending in 0 (5 never occurs), so a share well above 20% means rounding.
    """
    vs = view["vs"]
    x = vs[vs["TESTCD"].isin(["SYSBP", "DIABP", "PULSE"]) & vs["RESULT"].notna()][["SITEID", "USUBJID", "RESULT"]].copy()
    x["EVENT"] = (x["RESULT"].round().astype(int) % 5 == 0)
    p = x.groupby(["SITEID", "USUBJID"])["EVENT"].agg(VALUE="mean", READINGS="size").reset_index()
    return p[p["READINGS"] >= min_readings].reset_index(drop=True)


KRI_LABELS = {
    "DROPOUT_RATE": "Dosed subjects who discontinued",
    "ANY_AE_RATE": "Dosed subjects with at least one adverse event (low = possible under-reporting)",
    "OUT_OF_WINDOW_PATIENT_RATE": "Dosed subjects with at least one out-of-window visit",
    "DQ_FLAG_RATE": "Dosed subjects with a HIGH/MEDIUM rule finding",
    "DIGIT_0_5_SHARE": "Average share of a patient's BP/pulse readings ending in 0 or 5 (digit preference; only HIGH is a concern)",
}


def build_site_kris(view: dict, findings: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (KRI, site) with rate, reference rate, z-score and status."""
    ind = subject_indicators(view, findings)
    specs = {"DROPOUT_RATE": "DISCONTINUED", "ANY_AE_RATE": "ANY_AE",
             "OUT_OF_WINDOW_PATIENT_RATE": "OUT_OF_WINDOW_ANY", "DQ_FLAG_RATE": "DQ_FLAGGED"}
    parts = []
    for kri, col in specs.items():
        g = ind.groupby("SITEID")[col].agg(N="size", EVENTS="sum").reset_index()
        parts.append(rate_zscores(g).assign(KRI=kri))
    parts.append(mean_zscores(digit_patient_table(view), high_only=True).assign(KRI="DIGIT_0_5_SHARE"))
    out = pd.concat(parts, ignore_index=True)
    out["EVENTS"] = out["EVENTS"].astype("Int64")
    return out[KRI_COLUMNS]


def kri_matrix(kris: pd.DataFrame) -> pd.DataFrame:
    """Wide view: rows = sites, columns = KRIs, cells = z-score."""
    return kris.pivot(index="SITEID", columns="KRI", values="Z")