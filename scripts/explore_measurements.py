import pandas as pd
from src.load import load_domain

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_rows", 100)

dm = load_domain("dm")
vs = load_domain("vs")
lb = load_domain("lb")
ae = load_domain("ae")
ds = load_domain("ds")

# ---------------- VITAL SIGNS ----------------
print("=== VS: WHAT IS MEASURED ===")
print(vs.groupby(["VSTESTCD", "VSTEST", "VSSTRESU"]).size(), "\n")

print("=== VS: ONE SUBJECT, ONE VISIT (note position and time point) ===")
one = vs[(vs["USUBJID"] == "01-701-1015") & (vs["VISITNUM"] == 4)]
print(one[["VSTESTCD", "VSPOS", "VSTPT", "VSSTRESN", "VSSTRESU"]], "\n")

print("=== VS: SUMMARY PER TEST ===")
print(vs.groupby("VSTESTCD")["VSSTRESN"].describe().round(1), "\n")

print("=== VS: ROWS WITH NO NUMERIC RESULT ===")
print(vs[vs["VSSTRESN"].isna()][["USUBJID", "VSTESTCD", "VISIT", "VSSTAT"]], "\n")

# ---------------- LABS ----------------
print("=== LB: ROWS PER CATEGORY ===")
print(lb["LBCAT"].value_counts(), "\n")

print("=== LB: FLAG (LBNRIND) COUNTS ===")
print(lb["LBNRIND"].value_counts(dropna=False), "\n")

print("=== LB: NUMERIC RESULT MISSING - WHICH TESTS? ===")
miss = lb[lb["LBSTRESN"].isna()]
print(miss["LBTESTCD"].value_counts(), "\n")
print("Text results behind those missing numbers:")
print(miss["LBSTRESC"].value_counts().head(), "\n")

print("=== LB: ALT (LIVER ENZYME) EXAMPLE ===")
alt = lb[lb["LBTESTCD"] == "ALT"]
print(alt["LBSTRESN"].describe().round(1))
print("Normal range used:", alt["LBSTNRLO"].dropna().unique(), "to", alt["LBSTNRHI"].dropna().unique(), "\n")

# ---------------- ADVERSE EVENTS ----------------
print("=== AE: SEVERITY / SERIOUS / OUTCOME ===")
print(ae["AESEV"].value_counts(), "\n")
print(ae["AESER"].value_counts(), "\n")
print(ae["AEOUT"].value_counts(), "\n")

print("=== AE: 8 MOST COMMON EVENTS ===")
print(ae["AEDECOD"].value_counts().head(8), "\n")

print("=== AE: DATE LENGTH (10 = full date, 7 = year-month, 4 = year only) ===")
print(ae["AESTDTC"].str.len().value_counts(), "\n")

print("=== AE: SUBJECTS WITH >= 1 EVENT, BY ARM ===")
ae_arm = ae.merge(dm[["USUBJID", "ARM"]], on="USUBJID")
n_ae = ae_arm.groupby("ARM")["USUBJID"].nunique()
n_all = dm[dm["ARM"] != "Screen Failure"]["ARM"].value_counts()
print(pd.DataFrame({"with_AE": n_ae, "total": n_all, "pct": (n_ae / n_all * 100).round(0)}), "\n")

print("=== AE: APPLICATION SITE ERYTHEMA, SUBJECTS BY ARM ===")
print(ae_arm[ae_arm["AEDECOD"] == "APPLICATION SITE ERYTHEMA"].groupby("ARM")["USUBJID"].nunique(), "\n")

# ---------------- DISPOSITION ----------------
print("=== DS: WHY DID SUBJECTS LEAVE? (one row per subject) ===")
final = ds[ds["DSCAT"] == "DISPOSITION EVENT"]
print(final["DSDECOD"].value_counts(), "\n")

print("=== DS: SAME REASON, DIFFERENT WORDING ===")
print(final[final["DSDECOD"] == "WITHDRAWAL BY SUBJECT"]["DSTERM"].value_counts())
