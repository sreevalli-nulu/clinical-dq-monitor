"""Step 6: error injector.

Takes the CLEAN trial view, plants known errors in a COPY, and returns the copy plus an
ANSWER KEY listing every planted error. The original tables are never changed.

Two kinds of errors:
  RECORD errors  - one wrong value (typo, missing reading, duplicate visit ...), meant for the Step 4 rules.
  SITE errors    - a whole site behaves oddly (rounding, under-reporting, late visits), meant for the Step 5 KRIs.
Each error has a strength (PARAM_LABEL) so we can see how subtle an error can be before we miss it.
"""
import numpy as np
import pandas as pd

from src.config import VISIT_WINDOW_DAYS

KEY_COLUMNS = ["INJECT_ID", "ERROR_TYPE", "LEVEL", "TABLE", "USUBJID", "SITEID", "WHERE",
               "EXPECTED_DETECTOR", "PARAM_LABEL", "DETAIL"]
BP_TESTS = ["SYSBP", "DIABP", "PULSE"]

WEIGHT_MULTIPLIERS = [1.08, 0.90, 1.20, 0.80, 1.50, 0.50, 10.0, 0.10]   # weight x this
AE_DAYS_BEFORE = [20, 31, 45, 90, 180, 365]                              # days before first dose
SITE_FRACTIONS = [0.15, 0.50]                                            # share of a site's patients affected
LATE_VISIT_SHIFT_DAYS = 10

# (error type, KRI that should react, direction it should move)
SITE_PLAN = [("SITE_DIGIT_PREFERENCE", "DIGIT_0_5_SHARE", "HIGH"),
             ("SITE_AE_UNDERREPORTING", "ANY_AE_RATE", "LOW"),
             ("SITE_LATE_VISITS", "OUT_OF_WINDOW_PATIENT_RATE", "HIGH")]


class _Key:
    def __init__(self):
        self.rows = []

    def add(self, error_type, level, table, usubjid, siteid, where, detector, param, detail):
        self.rows.append({"INJECT_ID": f"X{len(self.rows) + 1:03d}", "ERROR_TYPE": error_type, "LEVEL": level,
                          "TABLE": table, "USUBJID": usubjid, "SITEID": siteid, "WHERE": where,
                          "EXPECTED_DETECTOR": detector, "PARAM_LABEL": param, "DETAIL": detail})

    def frame(self):
        return pd.DataFrame(self.rows, columns=KEY_COLUMNS)


def _pick_sites(rng, base_kris, subjects):
    """For each site error, pick sites that are big enough and currently typical for that KRI."""
    n_dosed = subjects[subjects["DOSED"]].groupby("SITEID").size()
    used, plan = set(), []
    for error_type, kri, direction in SITE_PLAN:
        k = base_kris[base_kris["KRI"] == kri].set_index("SITEID")
        ok = [s for s in k.index if n_dosed.get(s, 0) >= 15 and abs(k.loc[s, "Z"]) < 1.5 and s not in used]
        if len(ok) < len(SITE_FRACTIONS):
            raise ValueError(f"Not enough eligible sites for {error_type}: {ok}")
        chosen = list(rng.choice(sorted(ok), size=len(SITE_FRACTIONS), replace=False))
        used.update(chosen)
        plan += [(error_type, kri, direction, site, frac) for site, frac in zip(chosen, SITE_FRACTIONS)]
    return plan


def _choose_patients(rng, ids, frac):
    ids = sorted(ids)
    k = max(1, int(round(frac * len(ids))))
    return set(rng.choice(ids, size=k, replace=False))


def _one_per_subject(rng, df, n):
    """Randomly pick n rows, at most one per subject."""
    shuffled = df.sample(frac=1.0, random_state=int(rng.integers(1_000_000_000)))
    return shuffled.drop_duplicates("USUBJID").head(n)


def inject_errors(view: dict, base_findings: pd.DataFrame, base_kris: pd.DataFrame,
                  seed: int = 42, n_per_type: int = 24):
    rng = np.random.default_rng(seed)
    v = {name: df.copy() for name, df in view.items()}
    key = _Key()
    subjects = v["subjects"]
    dosed = subjects[subjects["DOSED"]]
    base_keys = set(zip(base_findings["RULE_ID"], base_findings["USUBJID"], base_findings["WHERE"]))
    base_subjects = lambda rule: set(base_findings.loc[base_findings["RULE_ID"] == rule, "USUBJID"])  # noqa: E731

    # ---------------- SITE errors first, so their sites can be kept out of the record pool -------------
    site_rows = _pick_sites(rng, base_kris, subjects)
    site_ids = {s for _, _, _, s, _ in site_rows}
    for error_type, kri, direction, site, frac in site_rows:
        pts = list(dosed.loc[dosed["SITEID"] == site, "USUBJID"])
        label = f"{frac:.0%} of patients"
        if error_type == "SITE_DIGIT_PREFERENCE":
            chosen = _choose_patients(rng, pts, frac)
            m = v["vs"]["USUBJID"].isin(chosen) & v["vs"]["TESTCD"].isin(BP_TESTS) & v["vs"]["RESULT"].notna()
            v["vs"].loc[m, "RESULT"] = np.round(v["vs"].loc[m, "RESULT"] / 5) * 5
            detail = f"{len(chosen)} of {len(pts)} patients: every BP/pulse reading rounded to nearest 5"
        elif error_type == "SITE_AE_UNDERREPORTING":
            ae_pts = [p for p in pts if p in set(v["ae"]["USUBJID"])]
            chosen = _choose_patients(rng, ae_pts, frac * len(pts) / max(len(ae_pts), 1))
            v["ae"] = v["ae"][~v["ae"]["USUBJID"].isin(chosen)]
            detail = f"{len(chosen)} of {len(pts)} patients: all adverse events deleted"
        else:
            chosen = _choose_patients(rng, pts, frac)
            m = v["sv"]["USUBJID"].isin(chosen) & v["sv"]["WINDOW_APPLIES"]
            v["sv"].loc[m, "STUDY_DAY"] += LATE_VISIT_SHIFT_DAYS
            v["sv"].loc[m, "DATE"] += pd.Timedelta(days=LATE_VISIT_SHIFT_DAYS)
            v["sv"].loc[m, "DEVIATION"] += LATE_VISIT_SHIFT_DAYS
            v["sv"].loc[m, "OUT_OF_WINDOW"] = v["sv"].loc[m, "DEVIATION"].abs() > VISIT_WINDOW_DAYS
            detail = f"{len(chosen)} of {len(pts)} patients: every post-baseline visit moved {LATE_VISIT_SHIFT_DAYS} days later"
        key.add(error_type, "SITE", {"SITE_DIGIT_PREFERENCE": "vs", "SITE_AE_UNDERREPORTING": "ae", "SITE_LATE_VISITS": "sv"}[error_type],
                np.nan, site, f"site {site}", f"KRI:{kri}:{direction}", label, detail)

    # ---------------- RECORD errors, only in sites we did not touch above ------------------------------
    pool = set(dosed.loc[~dosed["SITEID"].isin(site_ids), "USUBJID"])

    # 1. WEIGHT_CHANGE ------------------------------------------------------------------
    vs = v["vs"]
    w = vs[(vs["TESTCD"] == "WEIGHT") & vs["RESULT"].notna() & vs["USUBJID"].isin(pool)
           & ~vs["USUBJID"].isin(base_subjects("R03"))].sort_values(["USUBJID", "VISITNUM"])
    w = w[w.groupby("USUBJID").cumcount() > 0]            # needs an earlier weight to compare with
    for i, (idx, row) in enumerate(_one_per_subject(rng, w, n_per_type).iterrows()):
        mult = WEIGHT_MULTIPLIERS[i % len(WEIGHT_MULTIPLIERS)]
        v["vs"].loc[idx, "RESULT"] = row["RESULT"] * mult
        key.add("WEIGHT_CHANGE", "RECORD", "vs", row["USUBJID"], row["SITEID"], f"visit {row['VISIT']}", "R03",
                f"{(mult - 1) * 100:+.0f}%", f"weight {row['RESULT']:.1f} -> {row['RESULT'] * mult:.1f}")

    # 2. VS_READING_DELETED --------------------------------------------------------------
    bp = vs[vs["TESTCD"].isin(BP_TESTS) & vs["USUBJID"].isin(pool)]
    cnt = bp.groupby(["USUBJID", "VISITNUM", "TESTCD"]).size().unstack(fill_value=0).reindex(columns=BP_TESTS, fill_value=0)
    full = cnt[(cnt == 3).all(axis=1)].reset_index()
    pick = _one_per_subject(rng, full, n_per_type).reset_index(drop=True)
    for i, row in pick.iterrows():
        test = BP_TESTS[i % 3]
        cand = v["vs"][(v["vs"]["USUBJID"] == row["USUBJID"]) & (v["vs"]["VISITNUM"] == row["VISITNUM"])
                       & (v["vs"]["TESTCD"] == test)]
        drop = cand.index[int(rng.integers(len(cand)))]
        visit = v["vs"].loc[drop, "VISIT"]
        site = v["vs"].loc[drop, "SITEID"]
        v["vs"] = v["vs"].drop(index=drop)
        key.add("VS_READING_DELETED", "RECORD", "vs", row["USUBJID"], site, f"visit {visit}", "R02",
                f"{test} deleted", f"one {test} reading removed from a complete set of 3")

    # 3. DUPLICATE_VISIT -----------------------------------------------------------------
    sv = v["sv"]
    cand = sv[sv["VISIT_TYPE"].eq("PLANNED") & sv["USUBJID"].isin(pool) & ~sv["USUBJID"].isin(base_subjects("R04"))]
    dups = _one_per_subject(rng, cand, n_per_type)
    for _, row in dups.iterrows():
        key.add("DUPLICATE_VISIT", "RECORD", "sv", row["USUBJID"], row["SITEID"], f"visit {row['VISIT']}", "R04",
                "copy of a visit", f"visit {row['VISITNUM']:g} recorded twice")
    v["sv"] = pd.concat([v["sv"], dups], ignore_index=True)

    # 4. AE_BACKDATED --------------------------------------------------------------------
    ae = v["ae"]
    cand = ae[ae["START_PRECISION"].eq("DAY") & ae["START_VS_DOSE"].eq("AFTER_DOSE") & (ae["START_DAY"] >= 1)
              & ae["USUBJID"].isin(pool) & ~ae["USUBJID"].isin(base_subjects("R07"))]
    for i, (idx, row) in enumerate(_one_per_subject(rng, cand, n_per_type).iterrows()):
        days = AE_DAYS_BEFORE[i % len(AE_DAYS_BEFORE)]
        shift = int(row["START_DAY"]) - 1 + days
        new_date = row["START_DATE"] - pd.Timedelta(days=shift)
        v["ae"].loc[idx, ["START_DAY", "START_DATE", "START_TEXT", "START_VS_DOSE"]] = [
            -float(days), new_date, new_date.strftime("%Y-%m-%d"), "BEFORE_DOSE"]
        key.add("AE_BACKDATED", "RECORD", "ae", row["USUBJID"], row["SITEID"], f"AE {row['AETERM']}", "R07",
                f"{days} days before dose", f"AE start moved from day {int(row['START_DAY'])} to day -{days}")

    # 5. LAB_FLAG_FLIPPED ----------------------------------------------------------------
    lb = v["lb"]
    l = lb[lb["RESULT"].notna() & lb["LO"].notna() & lb["HI"].notna() & lb["FLAG"].isin(["HIGH", "LOW"])
           & lb["USUBJID"].isin(pool)].copy()
    width = l["HI"] - l["LO"]
    dist = np.where(l["RESULT"] > l["HI"], l["RESULT"] - l["HI"], l["LO"] - l["RESULT"])
    l = l[(dist / width >= 0.20) & (width > 0)]
    l["_WHERE"] = "lab " + l["TESTCD"].astype(str) + ", " + l["VISIT"].astype(str)
    l = l[[(("R08", u, w_) not in base_keys) for u, w_ in zip(l["USUBJID"], l["_WHERE"])]]
    for idx, row in _one_per_subject(rng, l, n_per_type).iterrows():
        v["lb"].loc[idx, "FLAG"] = "NORMAL"
        key.add("LAB_FLAG_FLIPPED", "RECORD", "lb", row["USUBJID"], row["SITEID"], row["_WHERE"], "R08",
                "clearly out of range", f"{row['FLAG']} flag changed to NORMAL (result {row['RESULT']:.4g}, range {row['LO']:g}-{row['HI']:g})")

    return v, key.frame()