import pandas as pd

from src.evaluate import (evaluate_records, evaluate_sites, recall_by, rule_scorecard, site_false_alarms)
from src.injector import inject_errors
from src.rules import run_all_rules
from src.site_stats import build_site_kris
from src.trial_view import PROCESSED_DIR, load_trial_view

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 70)

SEED = 42
clean = load_trial_view()
base_findings = run_all_rules(clean)
base_kris = build_site_kris(clean, base_findings)

corrupted, key = inject_errors(clean, base_findings, base_kris, seed=SEED)
new_findings = run_all_rules(corrupted)
new_kris = build_site_kris(corrupted, new_findings)

out = PROCESSED_DIR / "injected"
out.mkdir(parents=True, exist_ok=True)
for name, df in corrupted.items():
    df.to_parquet(out / f"{name}.parquet", index=False)
key.to_parquet(PROCESSED_DIR / "answer_key.parquet", index=False)

print(f"Injected {len(key)} errors with seed {SEED}: {(key['LEVEL'] == 'RECORD').sum()} record errors "
      f"and {(key['LEVEL'] == 'SITE').sum()} site errors.")
print(key.groupby(["LEVEL", "ERROR_TYPE"]).size().to_string())

rec = evaluate_records(key, base_findings, new_findings)
print("\n=== RECORD ERRORS: recall by type (did the rule fire on exactly that record?) ===")
print(recall_by(rec, ["ERROR_TYPE"]).to_string(index=False))

print("\n=== RECORD ERRORS: recall by strength ===")
print(recall_by(rec[rec["ERROR_TYPE"].isin(["WEIGHT_CHANGE", "AE_BACKDATED"])], ["ERROR_TYPE", "PARAM_LABEL"]).to_string(index=False))

print("\n=== RULE SCORECARD (new alarms = findings that did not exist before injection) ===")
print(rule_scorecard(rec, base_findings, new_findings).to_string(index=False))

sites = evaluate_sites(key, base_kris, new_kris)
print("\n=== SITE ERRORS: did the KRI react? ===")
print(sites.round(2).to_string(index=False))
print(f"\nSite errors detected: {int(sites['DETECTED'].sum())} of {len(sites)}")

fa = site_false_alarms(key, base_kris, new_kris)
print(f"\n=== Sites that turned OK -> WATCH/ALERT without being injected: {len(fa)} ===")
print(fa.to_string(index=False))

print(f"\nSaved corrupted tables to {out} and the answer key to {PROCESSED_DIR / 'answer_key.parquet'}")