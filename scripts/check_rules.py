from src.rules import FINDING_COLUMNS, RULES, SEVERITY_ORDER, rule_summary, run_all_rules
from src.trial_view import load_trial_view

view = load_trial_view()
findings = run_all_rules(view)
summary = rule_summary(findings).set_index("RULE_ID")["FINDINGS"].to_dict()

EXPECTED = {"R01": 12, "R02": 7, "R03": 5, "R04": 2, "R05": 3, "R06": 1, "R07": 19, "R08": 162, "R09": 2}
failed = 0


def check(label, ok, detail=""):
    global failed
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    failed += (not ok)


check("every rule returns the standard columns",
      all(list(r.func(view).columns) == FINDING_COLUMNS for r in RULES))
for rid, n in EXPECTED.items():
    check(f"{rid} finding count", summary.get(rid) == n, f"got {summary.get(rid)}, expected {n}")
check("severity values are valid", findings["SEVERITY"].isin(SEVERITY_ORDER).all())
check("no finding without a subject or site", findings[["USUBJID", "SITEID"]].notna().all().all())
subjects = set(view["subjects"]["USUBJID"])
check("every finding's subject exists in the subjects table", set(findings["USUBJID"]) <= subjects)
check("findings are sorted worst severity first",
      findings["SEVERITY"].map(SEVERITY_ORDER).is_monotonic_increasing)
check("3 fatal-not-serious findings are all HIGH",
      (findings[findings["RULE_ID"] == "R05"]["SEVERITY"] == "HIGH").all())

print("\nALL CHECKS PASSED" if not failed else f"\n{failed} CHECK(S) FAILED")