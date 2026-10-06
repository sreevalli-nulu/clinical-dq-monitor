import pandas as pd
from src.load import load_domain
from src.visits import build_sv, build_vs, build_lb

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


sv, vs, lb = build_sv(), build_vs(), build_lb()

print("=== 1. ROW COUNTS MATCH THE RAW FILES ===")
for name, built in [("sv", sv), ("vs", vs), ("lb", lb)]:
    raw = len(load_domain(name))
    check(f"{name.upper()} rows", len(built) == raw, f"built={len(built)} raw={raw}")

print("\n=== 2. EVERY DATE PARSED ===")
for name, built in [("SV", sv), ("VS", vs), ("LB", lb)]:
    check(f"{name} dates", built["DATE"].isna().sum() == 0, f"missing={built['DATE'].isna().sum()}")

print("\n=== 3. OUR STUDY DAY AGREES WITH THE FILE'S OWN VSDY / LBDY ===")
for name, built in [("VS", vs), ("LB", lb)]:
    both = built.dropna(subset=["STUDY_DAY", "STUDY_DAY_SRC"])
    bad = (both["STUDY_DAY"] != both["STUDY_DAY_SRC"]).sum()
    check(f"{name} study day", bad == 0, f"rows compared={len(both)} disagreements={bad}")

print("\n=== 4. VISIT TYPES (SV) ===")
print(sv["VISIT_TYPE"].value_counts(), "\n")
print("Planned visits per VISIT:")
print(sv[sv["VISIT_TYPE"] == "PLANNED"].groupby(["VISITNUM", "VISIT"]).size().to_string(), "\n")

print("=== 5. VISIT WINDOW ===")
w = sv[sv["WINDOW_APPLIES"]]
print(f"visits checked={len(w)}  out of window={int(w['OUT_OF_WINDOW'].sum())}  "
      f"rate={w['OUT_OF_WINDOW'].mean() * 100:.1f}%")
by_site = w.groupby("SITEID")["OUT_OF_WINDOW"].agg(n="size", out="sum", pct=lambda s: round(s.mean() * 100))
print(by_site.sort_values("pct", ascending=False).head(8), "\n")
old = w[(w["VISITNUM"] % 1 == 0) & (w["VISITNUM"] <= 99)]
print(f"2B's narrower definition (whole-number visits 4-99) gives: n={len(old)} out={int(old['OUT_OF_WINDOW'].sum())}\n")

print("=== 6. VS ===")
check("NOT_DONE rows", vs["NOT_DONE"].sum() == 8, f"found={vs['NOT_DONE'].sum()} (expected 8 from 2C)")
check("NOT_DONE rows have no result", vs.loc[vs["NOT_DONE"], "RESULT"].isna().all())
print(vs[["USUBJID", "VISIT", "TESTCD", "POS", "TPT", "RESULT", "UNIT"]].head(4), "\n")

print("=== 7. LB ===")
print("Text-only results by test:\n", lb[lb["TEXT_ONLY"]].groupby("TESTCD").size().to_string())
print("Censored results by test:\n", lb[lb["CENSORED"]].groupby("TESTCD").size().to_string())
check("text-only + censored = results with no number", (lb["TEXT_ONLY"].sum() + lb["CENSORED"].sum()) == lb["RESULT"].isna().sum())