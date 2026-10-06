import numpy as np
import pandas as pd

from src.config import MIN_SITE_SUBJECTS
from src.rules import run_all_rules
from src.site_stats import KRI_COLUMNS, build_site_kris, rate_zscores, subject_indicators
from src.trial_view import load_trial_view

view = load_trial_view()
findings = run_all_rules(view)
kris = build_site_kris(view, findings)
ind = subject_indicators(view, findings)
failed = 0


def check(label, ok, detail=""):
    global failed
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    failed += (not ok)


# 1. Hand-worked example: site has 10 events in 20; the other 80 subjects have 10 events.
toy = pd.DataFrame({"SITEID": ["A", "B"], "N": [20, 80], "EVENTS": [10, 10]})
z = rate_zscores(toy).set_index("SITEID")["Z"]
expected = (0.5 - 0.125) / np.sqrt(0.125 * 0.875 / 20)
check("z-score matches the hand calculation", abs(z["A"] - expected) < 1e-9, f"{z['A']:.3f} vs {expected:.3f}")

# 2. Small-site guard
tiny = pd.DataFrame({"SITEID": ["A", "B"], "N": [4, 96], "EVENTS": [4, 10]})
r = rate_zscores(tiny).set_index("SITEID")
check("a site below the minimum gets no z-score", pd.isna(r.loc["A", "Z"]) and r.loc["A", "STATUS"] == "TOO_SMALL")

# 3. Real data
check("all five KRIs are present", set(kris["KRI"]) == {"DROPOUT_RATE", "ANY_AE_RATE", "OUT_OF_WINDOW_PATIENT_RATE",
                                                        "DQ_FLAG_RATE", "DIGIT_0_5_SHARE"})
check("standard columns", list(kris.columns) == KRI_COLUMNS)
check("every dosed subject appears once", len(ind) == 254 and ind["USUBJID"].is_unique, f"{len(ind)} rows")
for kri in ["DROPOUT_RATE", "ANY_AE_RATE", "OUT_OF_WINDOW_PATIENT_RATE", "DQ_FLAG_RATE"]:
    k = kris[kris["KRI"] == kri]
    check(f"{kri}: site counts add up to 254 dosed subjects", k["N"].sum() == 254, f"sum {k['N'].sum()}")
small_sites = set(ind.groupby("SITEID").size().loc[lambda s: s < MIN_SITE_SUBJECTS].index)
scored = kris[kris["Z"].notna() & (kris["KRI"] != "DIGIT_0_5_SHARE")]
check("no small site was scored", not (set(scored["SITEID"]) & small_sites), f"small sites: {sorted(small_sites)}")
check("dropout events add up to 144", kris[kris["KRI"] == "DROPOUT_RATE"]["EVENTS"].sum() == 144)
digit = kris[kris["KRI"] == "DIGIT_0_5_SHARE"].set_index("SITEID")
check("sites 705 and 713 are digit-preference ALERTs", {"705", "713"} <= set(digit.index[digit["STATUS"] == "ALERT"]))
check("a LOW digit share is never an ALERT", not ((digit["DIRECTION"] == "LOW") & (digit["STATUS"] == "ALERT")).any())

print("\nALL CHECKS PASSED" if not failed else f"\n{failed} CHECK(S) FAILED")