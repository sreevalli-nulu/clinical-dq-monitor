import pandas as pd
from src.load import load_domain
from src.trial_view import arm_column, build_trial_view, load_trial_view, save_trial_view

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


view = build_trial_view()
save_trial_view(view)
loaded = load_trial_view()
subjects, sv, vs, lb, ae = (view[n] for n in ["subjects", "sv", "vs", "lb", "ae"])

print("=== 1. SAVED FILES MATCH WHAT WE BUILT ===")
for name in view:
    try:
        pd.testing.assert_frame_equal(view[name], loaded[name])
        check(f"{name} round trip", True, f"rows={len(loaded[name])}")
    except AssertionError as e:
        check(f"{name} round trip", False, str(e)[:120])

print("\n=== 2. TABLES AGREE WITH EACH OTHER ===")
known = set(subjects["USUBJID"])
for name in ["sv", "vs", "lb", "ae"]:
    unknown = set(view[name]["USUBJID"]) - known
    check(f"{name.upper()} subjects all exist in subjects table", not unknown, f"unknown={len(unknown)}")
dosed = set(subjects.loc[subjects["DOSED"], "USUBJID"])
for name in ["vs", "lb", "ae"]:
    extra = set(view[name]["USUBJID"]) - dosed
    check(f"{name.upper()} has dosed subjects only", not extra, f"screen failures present={len(extra)}")
site_in_id = subjects["USUBJID"].str.split("-").str[1]
mismatch = (site_in_id != subjects["SITEID"]).sum()
check("site inside USUBJID matches SITEID", mismatch == 0, f"mismatches={mismatch}")

print("\n=== 3. ADVERSE EVENTS: RECORDS -> EVENTS ===")
raw = load_domain("ae")
check("every raw record is accounted for", ae["N_RECORDS"].sum() == len(raw),
      f"records={len(raw)} events={len(ae)}")
check("one row per event", not ae.duplicated(["USUBJID", "AESPID"]).any())
terms_per_event = raw.groupby(["USUBJID", "AESPID"])["AEDECOD"].nunique()
check("an event never mixes two medical terms", (terms_per_event == 1).all())
print(f"events whose severity changed between records: {int(ae['SEV_CHANGED'].sum())}")
fatal = ae[(ae["DIED"] == "Y") | (ae["OUTCOME"] == "FATAL")]
check("fatal events kept", len(fatal) == 3, f"found={len(fatal)}")

print("\n=== 4. WHEN DID EACH EVENT START, COMPARED WITH FIRST DOSE? ===")
print(pd.crosstab(ae["START_PRECISION"], ae["START_VS_DOSE"], margins=True), "\n")
partial = ae[ae["START_PRECISION"].isin(["MONTH", "YEAR"])]
print("Partial-date events:", len(partial))
print(partial["START_VS_DOSE"].value_counts().to_string(), "\n")
print("Treatment-emergent (AFTER_DOSE) events by actual arm:")
te = ae[ae["START_VS_DOSE"] == "AFTER_DOSE"]
n_dosed = subjects[subjects["DOSED"]].groupby(arm_column("exposure")).size()
by_arm = te.groupby(arm_column("exposure"))["USUBJID"].nunique()
print(pd.DataFrame({"subjects_with_TEAE": by_arm, "dosed": n_dosed, "pct": (by_arm / n_dosed * 100).round(0)}))

print("\n=== 5. ARM HELPER ===")
check("retention uses planned arm", arm_column("retention") == "ARM")
check("exposure uses actual arm", arm_column("exposure") == "ACTARM")