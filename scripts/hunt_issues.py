import numpy as np
import pandas as pd
from src.load import load_domain

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 60)
pd.set_option("display.max_colwidth", 40)

dm = load_domain("dm")
sv = load_domain("sv")
vs = load_domain("vs")
lb = load_domain("lb")
ae = load_domain("ae")


def header(n, title):
    print(f"\n{'=' * 8} CHECK {n}: {title} {'=' * 8}")


# 1 ---------------------------------------------------------------
header(1, "Planned arm differs from actual arm (DM)")
swap = dm[dm["ARM"] != dm["ACTARM"]]
print(f"{len(swap)} subjects")
print(swap.groupby(["ARM", "ACTARM"]).size())
print("Sites involved:", sorted(swap["SITEID"].unique()))

# 2 ---------------------------------------------------------------
header(2, "Vital-sign visits with missing readings (expected 3 per test)")
bp = vs[vs["VSTESTCD"].isin(["SYSBP", "DIABP", "PULSE"])]
counts = bp.groupby(["USUBJID", "VISITNUM", "VSTESTCD"]).size().unstack(fill_value=0)
bad = counts[(counts != 3).any(axis=1)]
print(f"{len(bad)} of {len(counts)} subject-visits")
print(bad)

# 3 ---------------------------------------------------------------
header(3, "Weight jumps of more than 15% between consecutive visits")
w = vs[vs["VSTESTCD"] == "WEIGHT"].sort_values(["USUBJID", "VISITNUM"]).copy()
w["PREV"] = w.groupby("USUBJID")["VSSTRESN"].shift()
w["PCT"] = ((w["VSSTRESN"] / w["PREV"] - 1) * 100).round(0)
jumps = w[w["PCT"].abs() > 15]
print(f"{len(jumps)} jumps")
print(jumps[["USUBJID", "VISIT", "PREV", "VSSTRESN", "PCT"]])

# 4 ---------------------------------------------------------------
header(4, "Same subject + same visit number recorded twice (SV)")
dup = sv[sv.duplicated(["USUBJID", "VISITNUM"], keep=False)]
print(dup[["USUBJID", "VISITNUM", "VISIT", "SVSTDTC"]])

# 5 ---------------------------------------------------------------
header(5, "Fatal adverse events not marked 'serious'")
fatal = ae[(ae["AESDTH"] == "Y") | (ae["AEOUT"] == "FATAL")]
print(f"{len(fatal)} fatal events, of which AESER = 'N': {(fatal['AESER'] == 'N').sum()}")
print(fatal[["USUBJID", "AESDTH", "AEOUT", "AESER"]])

# 6 ---------------------------------------------------------------
header(6, "Visits dated after the subject's death")
dd = dm[dm["DTHDTC"].notna()][["USUBJID", "DTHDTC"]]
v = sv.merge(dd, on="USUBJID")
v = v[pd.to_datetime(v["SVSTDTC"]) > pd.to_datetime(v["DTHDTC"])]
print(v[["USUBJID", "VISIT", "SVSTDTC", "DTHDTC"]])

# 7 ---------------------------------------------------------------
header(7, "AEs that started long before the first dose")
a = ae.merge(dm[["USUBJID", "RFSTDTC"]], on="USUBJID")
a["START"] = pd.to_datetime(a["AESTDTC"], errors="coerce")      # partial dates become NaT
a["FIRSTDOSE"] = pd.to_datetime(a["RFSTDTC"], errors="coerce")
early = a[(a["FIRSTDOSE"] - a["START"]).dt.days > 30]
print(f"Full dates, more than 30 days before first dose: {len(early)} rows")
print(early[["USUBJID", "AETERM", "AESTDTC", "RFSTDTC"]])
part = a[a["AESTDTC"].str.len() < 10].copy()
part["YEARS_BEFORE"] = part["FIRSTDOSE"].dt.year - part["AESTDTC"].str[:4].astype(int)
print(f"\nPartial dates: {len(part)} rows, {(part['YEARS_BEFORE'] >= 5).sum()} start 5+ years before the trial")
print(part.sort_values("YEARS_BEFORE", ascending=False)
      [["USUBJID", "AETERM", "AESTDTC", "RFSTDTC", "YEARS_BEFORE"]].head(8))

# 8 ---------------------------------------------------------------
header(8, "Lab flag disagrees with the normal range")
l = lb.dropna(subset=["LBSTRESN", "LBSTNRLO", "LBSTNRHI"]).copy()
l = l[l["LBNRIND"].isin(["NORMAL", "HIGH", "LOW"])]
l["CALC"] = np.where(l["LBSTRESN"] < l["LBSTNRLO"], "LOW",
                     np.where(l["LBSTRESN"] > l["LBSTNRHI"], "HIGH", "NORMAL"))
mism = l[l["LBNRIND"] != l["CALC"]].copy()
print(f"{len(mism)} disagreements out of {len(l)} flagged results")
print(pd.crosstab(mism["LBNRIND"], mism["CALC"]))

# Case A: flagged NORMAL but the number is outside the range - how far outside?
a_rows = mism[mism["LBNRIND"] == "NORMAL"].copy()
width = a_rows["LBSTNRHI"] - a_rows["LBSTNRLO"]
dist = np.where(a_rows["CALC"] == "HIGH", a_rows["LBSTRESN"] - a_rows["LBSTNRHI"],
                a_rows["LBSTNRLO"] - a_rows["LBSTRESN"])
a_rows["OUTSIDE_PCT"] = dist / width * 100
print("\nFlagged NORMAL but outside range - distance past the limit (% of range width):")
print(pd.cut(a_rows["OUTSIDE_PCT"], [0, 1, 5, 20, 1000], include_lowest=True,
             labels=["<1%", "1-5%", "5-20%", ">20%"]).value_counts().sort_index())

# Case B: flagged HIGH but the number is inside the range
b_rows = mism[mism["LBNRIND"] == "HIGH"]
print("\nFlagged HIGH but inside the range:")
print(b_rows[["USUBJID", "LBTESTCD", "VISIT", "LBSTRESN", "LBSTNRLO", "LBSTNRHI"]])

# 9 ---------------------------------------------------------------
header(9, "Possible duplicate AE records")
cols = ["USUBJID", "AETERM", "AESTDTC", "AEENDTC", "AESEV", "AEREL"]
dups = ae[ae.duplicated(cols, keep="first")]
print(f"{len(dups)} rows repeat an earlier row on all of: {cols[1:]}")
print(dups.groupby("USUBJID").size().sort_values(ascending=False).head(5))

