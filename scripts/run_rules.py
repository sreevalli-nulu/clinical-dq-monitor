import pandas as pd

from src.rules import rule_summary, run_all_rules
from src.trial_view import PROCESSED_DIR, load_trial_view

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_colwidth", 70)

view = load_trial_view()
findings = run_all_rules(view)
findings.to_parquet(PROCESSED_DIR / "findings_rules.parquet", index=False)

print("=== Findings per rule ===")
print(rule_summary(findings).to_string(index=False))
print(f"\nTotal findings: {len(findings)}   Distinct subjects: {findings['USUBJID'].nunique()}")

print("\n=== Subjects flagged by the most DIFFERENT rules ===")
multi = (findings.groupby("USUBJID")
         .agg(RULES=("RULE_ID", lambda s: ", ".join(sorted(set(s)))), N_RULES=("RULE_ID", "nunique"),
              FINDINGS=("RULE_ID", "size"))
         .sort_values(["N_RULES", "FINDINGS"], ascending=False))
print(multi.head(8))

print("\n=== HIGH severity findings (read every one) ===")
print(findings[findings["SEVERITY"] == "HIGH"][["RULE_ID", "USUBJID", "WHERE", "DETAIL"]].to_string(index=False))

print("\n=== Findings per site (raw counts - NOT yet fair across site sizes) ===")
print(findings.groupby("SITEID").size().sort_values(ascending=False).head(6))