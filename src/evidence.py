"""Step 8 (part 1): turn a flag into an EVIDENCE PACKET - a small dict of facts computed by our code.

Claude only ever sees these packets. It never sees the raw tables and never decides what is an error.
The 'rule_meaning' and 'innocent_explanations' texts are written by us (not by the model), so the
explanations stay anchored to things we have checked.
"""
import numpy as np
import pandas as pd

from src.config import AE_PRE_DOSE_DAYS, WEIGHT_JUMP_PCT
from src.site_stats import KRI_LABELS

RULE_NOTES = {
    "R01": ("The subject received a different treatment arm than the one they were randomised to.",
            "A dose change after a side effect can be recorded this way; compare with the protocol's allowed dose reductions."),
    "R02": ("A visit has fewer or more than the expected three readings per test for blood pressure and pulse.",
            "A reading may genuinely not have been taken (equipment problem, patient refusal) and be documented elsewhere."),
    "R03": (f"Weight changed by more than {WEIGHT_JUMP_PCT} percent between two consecutive visits.",
            "A real change, a different scale or clothing, or a decimal or unit slip when typing."),
    "R04": ("The same visit number is recorded twice for one subject.",
            "A repeated assessment entered under the same visit number; check the source record."),
    "R05": ("A fatal adverse event is not marked as serious.",
            "By definition a fatal event is serious, so this is nearly always a missing or wrong flag in the record."),
    "R06": ("A visit is dated after the subject's recorded death.",
            "Either the visit date or the death date may be mistyped (often off by a day), or the visit sits under the wrong subject."),
    "R07": (f"An adverse event starts more than {AE_PRE_DOSE_DAYS} days before the first dose, or has a partial date that falls entirely before it.",
            "Medical history may have been entered as an adverse event, or the start date was mistyped."),
    "R08": ("The lab flag (low, normal, high) disagrees with the result compared with the reference range.",
            "Often a rounding difference at the boundary: the stored result may be rounded differently from the value used to set the flag."),
    "R09": ("Two separate adverse events share the same term, start date and severity.",
            "A genuine repeat event or an accidental duplicate entry."),
}

KRI_NOTES = {
    "DROPOUT_RATE": "Dropout depends on how sick patients are and on the mix of study arms; a small site can drift by chance.",
    "ANY_AE_RATE": "A low rate can mean under-reporting, but also a different mix of placebo and active patients (not adjusted for here) or a small sample.",
    "OUT_OF_WINDOW_PATIENT_RATE": "Travel distance, staffing or holidays can delay visits; check whether one or two patients drive the rate.",
    "DQ_FLAG_RATE": "More rule findings per patient points to entry problems at the site; small sites move a lot with a few patients.",
    "DIGIT_0_5_SHARE": "Staff or devices may round readings; check which device is used and who takes the measurement.",
}


def _clean(v):
    """Make a value JSON-friendly: NaN -> None, numpy numbers -> Python numbers."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if np.isnan(v) else float(v)
    return v


def rule_packets(findings: pd.DataFrame, severities=("HIGH",)) -> list:
    """One packet per rule finding with the chosen severities."""
    packets = []
    chosen = findings[findings["SEVERITY"].isin(severities)]
    for i, r in enumerate(chosen.itertuples(index=False)):
        others = findings[(findings["USUBJID"] == r.USUBJID) & ~((findings["RULE_ID"] == r.RULE_ID) & (findings["WHERE"] == r.WHERE))]
        meaning, innocent = RULE_NOTES[r.RULE_ID]
        packets.append({
            "card_id": f"RULE-{r.RULE_ID}-{r.USUBJID}-{i + 1}",
            "kind": "RULE_FINDING", "severity": r.SEVERITY,
            "rule_id": r.RULE_ID, "rule_name": r.RULE_NAME,
            "subject": r.USUBJID, "site": r.SITEID, "where": r.WHERE, "detail": r.DETAIL, "value": _clean(r.VALUE),
            "rule_meaning": meaning, "innocent_explanations": innocent,
            "same_subject_other_flags": sorted(set(others["RULE_NAME"])),
        })
    return packets


def site_packets(kris: pd.DataFrame, statuses=("WATCH", "ALERT")) -> list:
    """One packet per site signal (a KRI that is WATCH or ALERT)."""
    scored = int(kris["Z"].notna().sum())
    packets = []
    chosen = kris[kris["STATUS"].isin(statuses)].copy()
    chosen["_a"] = chosen["Z"].abs()
    for r in chosen.sort_values("_a", ascending=False).itertuples(index=False):
        digit = r.KRI == "DIGIT_0_5_SHARE"
        packets.append({
            "card_id": f"SITE-{r.KRI}-{r.SITEID}", "kind": "SITE_SIGNAL", "severity": r.STATUS,
            "kri": r.KRI, "kri_meaning": KRI_LABELS[r.KRI], "site": r.SITEID,
            "unit": "patients", "n_units": int(r.N), "events": _clean(r.EVENTS),
            "site_rate_pct": round(float(r.RATE) * 100, 1), "rest_of_study_rate_pct": round(float(r.REF_RATE) * 100, 1),
            "z_score": round(float(r.Z), 2), "direction": r.DIRECTION,
            "tests_scored": scored, "expected_flags_by_chance": round(scored * 0.0455, 1),
            "expected_if_no_rounding_pct": 20 if digit else None,
            "innocent_explanations": KRI_NOTES[r.KRI],
        })
    return packets