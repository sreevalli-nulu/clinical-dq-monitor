import numpy as np
import pandas as pd

from src.rules import run_all_rules
from src.site_stats import KRI_LABELS, build_site_kris, kri_matrix
from src.trial_view import PROCESSED_DIR, load_trial_view

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)

view = load_trial_view()
findings = run_all_rules(view)
kris = build_site_kris(view, findings)
kris.to_parquet(PROCESSED_DIR / "site_kris.parquet", index=False)

print("=== What each KRI measures ===")
for k, v in KRI_LABELS.items():
    print(f"  {k:28s} {v}")

print("\n=== Z-score matrix (blank = site too small to score) ===")
print(kri_matrix(kris).round(1).to_string())

print("\n=== Sites needing attention (|z| >= 2), strongest first ===")
flagged = kris[kris["STATUS"].isin(["WATCH", "ALERT"])].copy()
flagged["ABS_Z"] = flagged["Z"].abs()
print(flagged.sort_values("ABS_Z", ascending=False)
      [["KRI", "SITEID", "N", "EVENTS", "RATE", "REF_RATE", "Z", "STATUS", "DIRECTION"]]
      .round(2).to_string(index=False))

tested = int(kris["Z"].notna().sum())
print(f"\nTests actually scored: {tested}.  By pure chance about {tested * 0.0455:.1f} would reach |z| >= 2 "
      f"and {tested * 0.0027:.2f} would reach |z| >= 3.")
print(f"Observed: {int(kris['STATUS'].isin(['WATCH', 'ALERT']).sum())} flagged (|z| >= 2) and {int((kris['STATUS'] == 'ALERT').sum())} alerts (|z| >= 3).")

print("\n=== Scoring status counts ===")
print(kris["STATUS"].value_counts().to_string())