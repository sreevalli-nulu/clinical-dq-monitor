"""Step 4: rule-based edit checks.

Every rule is a function that takes the clean trial view (dict of tables) and returns a
findings table with the same columns. A finding is a LEAD for a reviewer, not a verdict.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from src.config import AE_PRE_DOSE_DAYS, WEIGHT_JUMP_PCT
from src.derive import parse_dtc
from src.load import load_domain

FINDING_COLUMNS = ["RULE_ID", "RULE_NAME", "SEVERITY", "USUBJID", "SITEID",
                   "WHERE", "DETAIL", "VALUE"]
SEVERITY_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def _findings(df: pd.DataFrame, rule_id: str, name: str, severity: str,
              where: pd.Series, detail: pd.Series, value=None) -> pd.DataFrame:
    """Build a findings table in the standard shape from a table of flagged rows."""
    if df.empty:
        return pd.DataFrame(columns=FINDING_COLUMNS)
    out = pd.DataFrame({
        "RULE_ID": rule_id, "RULE_NAME": name, "SEVERITY": severity,
        "USUBJID": df["USUBJID"].to_numpy(), "SITEID": df["SITEID"].to_numpy(),
        "WHERE": where.to_numpy(), "DETAIL": detail.to_numpy(),
        "VALUE": (value if value is not None else pd.Series(np.nan, index=df.index)).to_numpy(),
    })
    return out[FINDING_COLUMNS]


# ---- R01 ------------------------------------------------------------------------------
def r01_arm_swap(view):
    """Planned arm differs from the arm actually received."""
    s = view["subjects"]
    d = s[s["DOSED"] & (s["ARM"] != s["ACTARM"])]
    return _findings(d, "R01", "ARM_SWAP", "LOW",
                     where="subject " + d["USUBJID"],
                     detail="planned " + d["ARM"] + " -> received " + d["ACTARM"]
                            + "; ended: " + d["DISPOSITION"])


# ---- R02 ------------------------------------------------------------------------------
def r02_vs_incomplete(view):
    """Blood pressure / pulse should have 3 readings per test per visit."""
    vs = view["vs"]
    bp = vs[vs["TESTCD"].isin(["SYSBP", "DIABP", "PULSE"])]
    n = (bp.groupby(["USUBJID", "SITEID", "VISITNUM", "VISIT", "TESTCD"]).size()
           .unstack(fill_value=0).reindex(columns=["SYSBP", "DIABP", "PULSE"], fill_value=0)
           .reset_index())
    bad = n[(n[["SYSBP", "DIABP", "PULSE"]] != 3).any(axis=1)].copy()
    counts = ("SYSBP " + bad["SYSBP"].astype(str) + ", DIABP " + bad["DIABP"].astype(str)
              + ", PULSE " + bad["PULSE"].astype(str))
    return _findings(bad, "R02", "VS_INCOMPLETE_SET", "MEDIUM",
                     where="visit " + bad["VISIT"].astype(str),
                     detail="readings found (expected 3 each): " + counts,
                     value=bad[["SYSBP", "DIABP", "PULSE"]].sum(axis=1))


# ---- R03 ------------------------------------------------------------------------------
def r03_weight_jump(view):
    """Weight changes by more than WEIGHT_JUMP_PCT between consecutive visits."""
    vs = view["vs"]
    w = vs[(vs["TESTCD"] == "WEIGHT") & vs["RESULT"].notna()].sort_values(["USUBJID", "VISITNUM"]).copy()
    w["PREV"] = w.groupby("USUBJID")["RESULT"].shift()
    w["PCT"] = (w["RESULT"] / w["PREV"] - 1) * 100
    j = w[w["PCT"].abs() > WEIGHT_JUMP_PCT]
    return _findings(j, "R03", "WEIGHT_JUMP", "MEDIUM",
                     where="visit " + j["VISIT"].astype(str),
                     detail=("weight " + j["PREV"].round(1).astype(str) + " -> "
                             + j["RESULT"].round(1).astype(str) + " (" + j["PCT"].round(0).astype(int).astype(str) + "%)"),
                     value=j["PCT"].round(1))


# ---- R04 ------------------------------------------------------------------------------
def r04_duplicate_visit(view):
    """The same subject + visit number appears twice in the visit table."""
    sv = view["sv"]
    d = sv[sv.duplicated(["USUBJID", "VISITNUM"], keep=False)]
    return _findings(d, "R04", "DUPLICATE_VISIT", "MEDIUM",
                     where="visit " + d["VISIT"].astype(str),
                     detail="visit number " + d["VISITNUM"].astype(str) + " recorded more than once")


# ---- R05 ------------------------------------------------------------------------------
def r05_fatal_not_serious(view):
    """A fatal adverse event must be marked serious."""
    ae = view["ae"]
    d = ae[((ae["DIED"] == "Y") | (ae["OUTCOME"] == "FATAL")) & (ae["SERIOUS"] != "Y")]
    return _findings(d, "R05", "FATAL_NOT_SERIOUS", "HIGH",
                     where="AE " + d["AETERM"].astype(str),
                     detail="outcome " + d["OUTCOME"].astype(str) + " but serious = " + d["SERIOUS"].astype(str))


# ---- R06 ------------------------------------------------------------------------------
def _death_dates() -> pd.DataFrame:
    """Death date from the raw DM table (not carried into the clean subjects table)."""
    dm = load_domain("dm")
    out = dm.loc[dm["DTHDTC"].notna(), ["USUBJID", "DTHDTC"]].copy()
    out["DEATH_DATE"] = parse_dtc(out["DTHDTC"])
    return out[["USUBJID", "DEATH_DATE"]]


def r06_visit_after_death(view):
    """No visit may be dated after the subject's death date."""
    v = view["sv"].merge(_death_dates(), on="USUBJID")
    d = v[v["DATE"].notna() & v["DEATH_DATE"].notna() & (v["DATE"] > v["DEATH_DATE"])]
    return _findings(d, "R06", "VISIT_AFTER_DEATH", "HIGH",
                     where="visit " + d["VISIT"].astype(str),
                     detail="visit on " + d["DATE"].dt.strftime("%Y-%m-%d") + ", death on "
                            + d["DEATH_DATE"].dt.strftime("%Y-%m-%d"),
                     value=(d["DATE"] - d["DEATH_DATE"]).dt.days)


# ---- R07 ------------------------------------------------------------------------------
def r07_ae_long_before_dose(view):
    """Adverse event starts far before first dose (treated as history, not trial AE)."""
    ae = view["ae"]
    full = ae["START_PRECISION"].eq("DAY") & (ae["START_DAY"] < -AE_PRE_DOSE_DAYS)
    partial = ae["START_PRECISION"].isin(["MONTH", "YEAR"]) & ae["START_VS_DOSE"].eq("BEFORE_DOSE")
    d = ae[full | partial]
    days = d["START_DAY"].abs().astype("Int64").astype(str)
    detail = np.where(d["START_PRECISION"].eq("DAY"),
                      "started " + days + " days before first dose",
                      "partial date " + d["START_TEXT"].astype(str) + " is entirely before first dose")
    return _findings(d, "R07", "AE_LONG_BEFORE_DOSE", "MEDIUM",
                     where="AE " + d["AETERM"].astype(str),
                     detail=pd.Series(detail, index=d.index), value=d["START_DAY"])


# ---- R08 ------------------------------------------------------------------------------
def r08_lab_flag_mismatch(view):
    """Lab flag (LOW/NORMAL/HIGH) disagrees with result vs reference range."""
    lb = view["lb"]
    l = lb.dropna(subset=["RESULT", "LO", "HI"])
    l = l[l["FLAG"].isin(["NORMAL", "HIGH", "LOW"])]
    calc = np.where(l["RESULT"] < l["LO"], "LOW", np.where(l["RESULT"] > l["HI"], "HIGH", "NORMAL"))
    d = l[l["FLAG"].to_numpy() != calc].copy()
    d["CALC"] = calc[l["FLAG"].to_numpy() != calc]
    return _findings(d, "R08", "LAB_FLAG_MISMATCH", "LOW",
                     where="lab " + d["TESTCD"].astype(str) + ", " + d["VISIT"].astype(str),
                     detail=("flag " + d["FLAG"] + " but " + d["RESULT"].astype(str) + " vs range "
                             + d["LO"].astype(str) + "-" + d["HI"].astype(str) + " is " + d["CALC"]),
                     value=d["RESULT"])


# ---- R09 ------------------------------------------------------------------------------
def r09_duplicate_ae(view):
    """Two different AE events with the same subject, term, start text and severity."""
    ae = view["ae"]
    keys = ["USUBJID", "AETERM", "START_TEXT", "SEVERITY"]
    d = ae[ae.duplicated(keys, keep=False)]
    return _findings(d, "R09", "DUPLICATE_AE", "LOW",
                     where="AE " + d["AETERM"].astype(str),
                     detail="event " + d["AESPID"].astype(str) + " repeats term/start/severity of another event")


@dataclass(frozen=True)
class Rule:
    rule_id: str
    name: str
    func: Callable[[dict], pd.DataFrame]


RULES = [Rule("R01", "ARM_SWAP", r01_arm_swap), Rule("R02", "VS_INCOMPLETE_SET", r02_vs_incomplete),
         Rule("R03", "WEIGHT_JUMP", r03_weight_jump), Rule("R04", "DUPLICATE_VISIT", r04_duplicate_visit),
         Rule("R05", "FATAL_NOT_SERIOUS", r05_fatal_not_serious), Rule("R06", "VISIT_AFTER_DEATH", r06_visit_after_death),
         Rule("R07", "AE_LONG_BEFORE_DOSE", r07_ae_long_before_dose), Rule("R08", "LAB_FLAG_MISMATCH", r08_lab_flag_mismatch),
         Rule("R09", "DUPLICATE_AE", r09_duplicate_ae)]


def run_all_rules(view: dict) -> pd.DataFrame:
    """Run every rule and return one findings table, worst severity first."""
    parts = [r.func(view) for r in RULES]
    parts = [p for p in parts if not p.empty]
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=FINDING_COLUMNS)
    out["_s"] = out["SEVERITY"].map(SEVERITY_ORDER)
    return out.sort_values(["_s", "RULE_ID", "USUBJID"]).drop(columns="_s").reset_index(drop=True)


def rule_summary(findings: pd.DataFrame) -> pd.DataFrame:
    """One row per rule: how many findings and how many distinct subjects (0 if none)."""
    base = pd.DataFrame({"RULE_ID": [r.rule_id for r in RULES], "RULE_NAME": [r.name for r in RULES]})
    g = findings.groupby("RULE_ID").agg(FINDINGS=("USUBJID", "size"), SUBJECTS=("USUBJID", "nunique"))
    return base.merge(g, on="RULE_ID", how="left").fillna({"FINDINGS": 0, "SUBJECTS": 0}).astype({"FINDINGS": int, "SUBJECTS": int})