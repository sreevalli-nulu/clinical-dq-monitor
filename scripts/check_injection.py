import numpy as np
import pandas as pd

from src.evaluate import evaluate_records, evaluate_sites
from src.injector import inject_errors
from src.rules import run_all_rules
from src.site_stats import build_site_kris
from src.trial_view import load_trial_view

clean = load_trial_view()
snapshot = {k: v.copy() for k, v in clean.items()}
base_findings = run_all_rules(clean)
base_kris = build_site_kris(clean, base_findings)
failed = 0


def check(label, ok, detail=""):
    global failed
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    failed += (not ok)


corrupted, key = inject_errors(clean, base_findings, base_kris, seed=42)
rec = key[key["LEVEL"] == "RECORD"]
site = key[key["LEVEL"] == "SITE"]

check("the clean tables were not changed", all(clean[k].equals(snapshot[k]) for k in clean))
check("24 errors of each of the 5 record types",
      rec.groupby("ERROR_TYPE").size().eq(24).all() and rec["ERROR_TYPE"].nunique() == 5)
check("6 site errors (3 types x 2 strengths)", len(site) == 6 and site["ERROR_TYPE"].nunique() == 3)
check("inject ids are unique", key["INJECT_ID"].is_unique)
_, key_again = inject_errors(clean, base_findings, base_kris, seed=42)
check("same seed gives the same answer key", key.equals(key_again))
_, key_other = inject_errors(clean, base_findings, base_kris, seed=7)
check("a different seed gives a different answer key", not key.equals(key_other))
check("record errors avoid the sites chosen for site errors", not (set(rec["SITEID"]) & set(site["SITEID"])))
check("at most one record error per subject within each type",
      all(g["USUBJID"].is_unique for _, g in rec.groupby("ERROR_TYPE")))

# the data really changed
check("24 vital-sign readings were deleted",
      len(clean["vs"]) - len(corrupted["vs"]) == 24, f"{len(clean['vs'])} -> {len(corrupted['vs'])}")
check("24 duplicate visits were added", len(corrupted["sv"]) - len(clean["sv"]) == 24)
check("fewer adverse events after under-reporting", len(corrupted["ae"]) < len(clean["ae"]))
check("24 lab flags were flipped to NORMAL",
      int((clean["lb"]["FLAG"].fillna("") != corrupted["lb"]["FLAG"].fillna("")).sum()) == 24)
w_c = clean["vs"][clean["vs"]["TESTCD"] == "WEIGHT"]["RESULT"]
w_i = corrupted["vs"].loc[w_c.index, "RESULT"]
check("24 weights were changed", int((w_c != w_i).sum()) == 24)

# the detectors behaved as designed
ev = evaluate_records(key, base_findings, run_all_rules(corrupted))
check("every record error was a NEW problem (none already flagged before injection)", not ev["ALREADY_FLAGGED"].any())
for t in ["DUPLICATE_VISIT", "VS_READING_DELETED", "LAB_FLAG_FLIPPED"]:
    check(f"{t}: all detected", ev.loc[ev["ERROR_TYPE"] == t, "DETECTED"].all())
wc = ev[ev["ERROR_TYPE"] == "WEIGHT_CHANGE"]
big = wc["PARAM_LABEL"].str.replace("%", "").astype(float).abs() > 15
check("WEIGHT_CHANGE: every change above 15% was caught", wc.loc[big, "DETECTED"].all())
check("WEIGHT_CHANGE: no change of 10% or less was caught", not wc.loc[~big, "DETECTED"].any())
ab = ev[ev["ERROR_TYPE"] == "AE_BACKDATED"]
check("AE_BACKDATED: caught only when more than 30 days before dose",
      (ab["DETECTED"] == (ab["PARAM_LABEL"].str.extract(r"(\d+)")[0].astype(int) > 30)).all())
sites = evaluate_sites(key, base_kris, build_site_kris(corrupted, run_all_rules(corrupted)))
strong = sites[sites["STRENGTH"] == "50% of patients"]
check("all three 50% site errors were detected", strong["DETECTED"].all())

print("\nALL CHECKS PASSED" if not failed else f"\n{failed} CHECK(S) FAILED")