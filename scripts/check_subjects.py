import pandas as pd
from src.subjects import build_subjects

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

s = build_subjects()

print("SHAPE:", s.shape, "\n")
print("COLUMNS AND TYPES:\n", s.dtypes, "\n")
print("POPULATIONS:")
print("  screened     :", len(s))
print("  dosed        :", s["DOSED"].sum())
print("  completed    :", s["COMPLETED"].sum())
print("  discontinued :", s["DISCONTINUED"].sum())
print("  died         :", s["DIED"].sum(), "\n")

dosed = s[s["DOSED"]]
print("DISPOSITION BY ACTUAL ARM (dosed only):")
print(pd.crosstab(dosed["DISPOSITION"], dosed["ACTARM"], margins=True), "\n")

print("DOSED SUBJECTS PER SITE:")
print(dosed.groupby("SITEID").size().sort_values(), "\n")
print("SMALL SITES:", sorted(s[s["SMALL_SITE"]]["SITEID"].unique()), "\n")

print("ONE ROW PER SUBJECT?", s["USUBJID"].is_unique)
print("MISSING VALUES:\n", s.isna().sum()[lambda x: x > 0])