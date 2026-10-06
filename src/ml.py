"""Step 7: unsupervised anomaly detection with Isolation Forest.

Isolation Forest builds many random decision trees that try to cut a single row away from all the
others. Rows that are easy to isolate (few cuts) are unusual. It never sees the answer key: in a real
trial nobody tells you which rows are wrong. We only use the key afterwards, to score it.

Two models, because the two kinds of record look different:
  * VISIT model - one row per (subject, visit): counts and averages of vitals, labs and visit timing
  * AE model    - one row per adverse event: start day, duration, severity, how rare the term is
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

BP = ["SYSBP", "DIABP", "PULSE"]
SEV_ORD = {"MILD": 1, "MODERATE": 2, "SEVERE": 3}
PREC_ORD = {"DAY": 0, "MONTH": 1, "YEAR": 2}
VISIT_FEATURES = ["sv_rows_vs_typical", "vs_rows_vs_typical", "sysbp_count_vs_typical", "diabp_count_vs_typical",
                  "pulse_count_vs_typical", "lab_count_vs_typical", "abs_deviation", "sysbp_sd", "pulse_sd",
                  "sysbp_dev_pct", "diabp_dev_pct", "pulse_dev_pct", "temp_dev_pct", "weight_dev_pct",
                  "frac_flag_abnormal", "frac_calc_abnormal"]
AE_FEATURES = ["start_day", "duration_days", "has_end", "sev_ord", "serious", "n_records",
               "sev_changed", "prec_ord", "term_rarity"]


def _dosed(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["DOSED"]]


def _dev_from_subject_median(df: pd.DataFrame, testcd: str) -> pd.Series:
    """Per (USUBJID, VISIT): how far this visit's average reading is from the subject's own typical value, in %."""
    x = df[(df["TESTCD"] == testcd) & df["RESULT"].notna()].copy()
    x["dev"] = (x["RESULT"] / x.groupby("USUBJID")["RESULT"].transform("median") - 1).abs() * 100
    return x.groupby(["USUBJID", "VISIT"])["dev"].max()


def build_visit_features(view: dict) -> pd.DataFrame:
    """One row per (USUBJID, VISIT) of dosed subjects.

    Two ideas make the features fair:
      * counts are compared with what is TYPICAL FOR THAT KIND OF VISIT (a lab-only visit is normal for some visits)
      * values are compared with the SUBJECT'S OWN usual value (a 60 kg patient is not odd for being light)
    """
    sv, vs, lb = _dosed(view["sv"]), _dosed(view["vs"]), _dosed(view["lb"])
    keys = pd.concat([d[["USUBJID", "SITEID", "VISIT"]] for d in (sv, vs, lb)]).drop_duplicates(["USUBJID", "VISIT"])
    f = keys.set_index(["USUBJID", "VISIT"])
    by = ["USUBJID", "VISIT"]

    f["n_sv_rows"] = sv.groupby(by).size()
    f["abs_deviation"] = sv.groupby(by)["DEVIATION"].apply(lambda s: s.abs().max())
    f["n_vs_rows"] = vs.groupby(by).size()
    for t in BP:
        f[f"n_{t.lower()}"] = vs[vs["TESTCD"] == t].groupby(by).size()
    f["sysbp_sd"] = vs[vs["TESTCD"] == "SYSBP"].groupby(by)["RESULT"].std()
    f["pulse_sd"] = vs[vs["TESTCD"] == "PULSE"].groupby(by)["RESULT"].std()
    for t, name in [("SYSBP", "sysbp"), ("DIABP", "diabp"), ("PULSE", "pulse"), ("TEMP", "temp"), ("WEIGHT", "weight")]:
        f[f"{name}_dev_pct"] = _dev_from_subject_median(vs, t)

    num = lb[lb["RESULT"].notna() & lb["LO"].notna() & lb["HI"].notna()].copy()
    num["calc_abn"] = (num["RESULT"] < num["LO"]) | (num["RESULT"] > num["HI"])
    num["flag_abn"] = num["FLAG"].isin(["HIGH", "LOW"])
    f["n_lab"] = lb.groupby(by).size()
    f["frac_flag_abnormal"] = num.groupby(by)["flag_abn"].mean()
    f["frac_calc_abnormal"] = num.groupby(by)["calc_abn"].mean()

    f = f.reset_index()
    fill0 = ["n_sv_rows", "n_vs_rows", "n_sysbp", "n_diabp", "n_pulse", "n_lab", "abs_deviation", "sysbp_sd", "pulse_sd",
             "sysbp_dev_pct", "diabp_dev_pct", "pulse_dev_pct", "temp_dev_pct", "weight_dev_pct",
             "frac_flag_abnormal", "frac_calc_abnormal"]
    f[fill0] = f[fill0].fillna(0)
    for raw, out in [("n_sv_rows", "sv_rows_vs_typical"), ("n_vs_rows", "vs_rows_vs_typical"),
                     ("n_sysbp", "sysbp_count_vs_typical"), ("n_diabp", "diabp_count_vs_typical"),
                     ("n_pulse", "pulse_count_vs_typical"), ("n_lab", "lab_count_vs_typical")]:
        f[out] = f[raw] - f.groupby("VISIT")[raw].transform("median")
    return f[["USUBJID", "SITEID", "VISIT"] + VISIT_FEATURES]


def build_ae_features(view: dict) -> pd.DataFrame:
    """One row per adverse event of dosed subjects."""
    ae = _dosed(view["ae"]).copy()
    dur = (ae["END_DATE"] - ae["START_DATE"]).dt.days
    ae["has_end"] = dur.notna().astype(int)
    ae["duration_days"] = dur.fillna(dur.median())
    ae["start_day"] = ae["START_DAY"].fillna(ae["START_DAY"].median())
    ae["sev_ord"] = ae["SEVERITY"].map(SEV_ORD).fillna(2)
    ae["serious"] = (ae["SERIOUS"] == "Y").astype(int)
    ae["n_records"] = ae["N_RECORDS"]
    ae["sev_changed"] = ae["SEV_CHANGED"].astype(int)
    ae["prec_ord"] = ae["START_PRECISION"].map(PREC_ORD).fillna(0)
    freq = ae["AEDECOD"].map(ae["AEDECOD"].value_counts(normalize=True))
    ae["term_rarity"] = -np.log(freq)
    return ae[["USUBJID", "SITEID", "AESPID", "AETERM"] + AE_FEATURES].reset_index(drop=True)


def anomaly_scores(features: pd.DataFrame, columns: list, seed: int = 0, n_estimators: int = 300) -> pd.Series:
    """Higher score = easier to isolate = more unusual. No labels are used."""
    model = IsolationForest(n_estimators=n_estimators, contamination="auto", random_state=seed)
    model.fit(features[columns])
    return pd.Series(-model.score_samples(features[columns]), index=features.index, name="SCORE")