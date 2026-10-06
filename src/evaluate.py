"""Step 6: score the detectors against the answer key."""
import pandas as pd

DETECTED_STATUSES = ["WATCH", "ALERT"]


def evaluate_records(key: pd.DataFrame, base_findings: pd.DataFrame, new_findings: pd.DataFrame) -> pd.DataFrame:
    """Add DETECTED (did the expected rule fire on exactly this record?) and ALREADY_FLAGGED to each record error."""
    rec = key[key["LEVEL"] == "RECORD"].copy()
    new_keys = set(zip(new_findings["RULE_ID"], new_findings["USUBJID"], new_findings["WHERE"]))
    base_keys = set(zip(base_findings["RULE_ID"], base_findings["USUBJID"], base_findings["WHERE"]))
    ids = list(zip(rec["EXPECTED_DETECTOR"], rec["USUBJID"], rec["WHERE"]))
    rec["DETECTED"] = [i in new_keys for i in ids]
    rec["ALREADY_FLAGGED"] = [i in base_keys for i in ids]
    return rec


def recall_by(rec: pd.DataFrame, by: list) -> pd.DataFrame:
    g = rec.groupby(by)["DETECTED"].agg(INJECTED="size", DETECTED="sum").reset_index()
    g["RECALL"] = (g["DETECTED"] / g["INJECTED"]).round(2)
    return g


def rule_scorecard(rec: pd.DataFrame, base_findings: pd.DataFrame, new_findings: pd.DataFrame) -> pd.DataFrame:
    """Per rule: recall on injected errors, and what the NEW alarms were (matching the key, or collateral)."""
    cols = ["RULE_ID", "USUBJID", "WHERE"]
    base = set(map(tuple, base_findings[cols].to_numpy()))
    injected = set(zip(rec["EXPECTED_DETECTOR"], rec["USUBJID"], rec["WHERE"]))
    rows = []
    for rule, g in rec.groupby("EXPECTED_DETECTOR"):
        f = new_findings[new_findings["RULE_ID"] == rule]
        keys = set(map(tuple, f[cols].to_numpy()))      # one alarm per (rule, subject, place)
        new = [k for k in keys if k not in base]
        hits = [k for k in new if k in injected]
        rows.append({"RULE_ID": rule, "INJECTED": len(g), "DETECTED": int(g["DETECTED"].sum()),
                     "RECALL": round(g["DETECTED"].mean(), 2), "NEW_ALARMS": len(new),
                     "NEW_MATCHING_KEY": len(hits), "NEW_COLLATERAL": len(new) - len(hits),
                     "OLD_FINDINGS_BEFORE": len(set(k for k in base if k[0] == rule))})
    return pd.DataFrame(rows)


def evaluate_sites(key: pd.DataFrame, base_kris: pd.DataFrame, new_kris: pd.DataFrame) -> pd.DataFrame:
    """For each site error: z-score before and after, and whether the KRI moved into WATCH/ALERT the right way."""
    rows = []
    for _, r in key[key["LEVEL"] == "SITE"].iterrows():
        _, kri, direction = r["EXPECTED_DETECTOR"].split(":")
        b = base_kris[(base_kris["KRI"] == kri) & (base_kris["SITEID"] == r["SITEID"])].iloc[0]
        a = new_kris[(new_kris["KRI"] == kri) & (new_kris["SITEID"] == r["SITEID"])].iloc[0]
        rows.append({"INJECT_ID": r["INJECT_ID"], "ERROR_TYPE": r["ERROR_TYPE"], "SITEID": r["SITEID"],
                     "STRENGTH": r["PARAM_LABEL"], "KRI": kri,
                     "RATE_BEFORE": b["RATE"], "RATE_AFTER": a["RATE"], "Z_BEFORE": b["Z"], "Z_AFTER": a["Z"],
                     "STATUS_AFTER": a["STATUS"],
                     "DETECTED": bool(a["STATUS"] in DETECTED_STATUSES and a["DIRECTION"] == direction)})
    return pd.DataFrame(rows)


def site_false_alarms(key: pd.DataFrame, base_kris: pd.DataFrame, new_kris: pd.DataFrame) -> pd.DataFrame:
    """Sites that turned from OK into WATCH/ALERT on a KRI we did NOT attack at that site.

    DQ_FLAG_RATE is left out: we planted record errors in many sites, so that KRI is SUPPOSED to react.
    """
    injected = {(r["EXPECTED_DETECTOR"].split(":")[1], r["SITEID"]) for _, r in key[key["LEVEL"] == "SITE"].iterrows()}
    m = base_kris.merge(new_kris, on=["KRI", "SITEID"], suffixes=("_B", "_A"))
    m = m[m["STATUS_B"].eq("OK") & m["STATUS_A"].isin(DETECTED_STATUSES) & (m["KRI"] != "DQ_FLAG_RATE")]
    m = m[[(k, s) not in injected for k, s in zip(m["KRI"], m["SITEID"])]]
    return m[["KRI", "SITEID", "Z_B", "Z_A", "STATUS_A"]].round(2)