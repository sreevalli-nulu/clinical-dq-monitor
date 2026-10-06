"""Step 7: precision and recall for the anomaly scores, using the answer key from Step 6."""
import pandas as pd
from sklearn.metrics import average_precision_score

VISIT_ERROR_TYPES = ["WEIGHT_CHANGE", "VS_READING_DELETED", "DUPLICATE_VISIT", "LAB_FLAG_FLIPPED"]
VISIT_RULES = ["R02", "R03", "R04", "R06", "R08"]


def _visit_name(where: str) -> str:
    """'visit WEEK 2' -> 'WEEK 2'   'lab CA, WEEK 24' -> 'WEEK 24'"""
    return where[len("visit "):] if where.startswith("visit ") else where.split(", ", 1)[1]


def visit_truth(key: pd.DataFrame) -> pd.DataFrame:
    """Planted visit-level errors as (USUBJID, VISIT, ERROR_TYPE, PARAM_LABEL)."""
    k = key[key["ERROR_TYPE"].isin(VISIT_ERROR_TYPES)].copy()
    k["VISIT"] = k["WHERE"].map(_visit_name)
    return k[["USUBJID", "VISIT", "ERROR_TYPE", "PARAM_LABEL"]]


def rule_visit_units(findings: pd.DataFrame) -> set:
    """Visits flagged by the visit-based rules, as a set of (USUBJID, VISIT)."""
    f = findings[findings["RULE_ID"].isin(VISIT_RULES)]
    return {(u, _visit_name(w)) for u, w in zip(f["USUBJID"], f["WHERE"])}


def ae_truth(key: pd.DataFrame, ae_features: pd.DataFrame) -> pd.Series:
    """True for AE rows that were back-dated: same subject + term and start day == minus the planted days."""
    k = key[key["ERROR_TYPE"] == "AE_BACKDATED"]
    days = {(u, t): -int(p.split()[0]) for u, t, p in zip(k["USUBJID"], k["WHERE"].str[3:], k["PARAM_LABEL"])}
    return pd.Series([days.get((u, t)) == d for u, t, d in zip(ae_features["USUBJID"], ae_features["AETERM"],
                                                              ae_features["start_day"])], index=ae_features.index)


def precision_recall_at_k(scores: pd.Series, y: pd.Series, ks: list) -> pd.DataFrame:
    order = scores.sort_values(ascending=False).index
    rows = []
    for k in ks:
        top = y.loc[order[:k]]
        rows.append({"K_FLAGGED": k, "TRUE_ERRORS_FOUND": int(top.sum()),
                     "PRECISION": round(top.mean(), 3), "RECALL": round(top.sum() / y.sum(), 3)})
    return pd.DataFrame(rows)


def average_precision(scores: pd.Series, y: pd.Series) -> float:
    """Area under the precision-recall curve. Random scoring would give about y.mean()."""
    return float(average_precision_score(y, scores))


def label_units(features: pd.DataFrame, truth: pd.DataFrame) -> pd.Series:
    """1 if the (USUBJID, VISIT) of a feature row holds a planted error."""
    t = set(zip(truth["USUBJID"], truth["VISIT"]))
    return pd.Series([(u, v) in t for u, v in zip(features["USUBJID"], features["VISIT"])], index=features.index)