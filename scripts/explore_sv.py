import pandas as pd
from src.load import load_domain
from src.derive import add_study_day

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 100)

sv = load_domain("sv")
dm = load_domain("dm")

print("COLUMNS:", sv.columns.tolist(), "\n")
print("ONE SUBJECT'S VISITS:\n",
      sv[sv["USUBJID"] == "01-701-1015"][["VISITNUM", "VISIT", "VISITDY", "SVSTDTC"]], "\n")
print("NUMBER OF SUBJECTS AT EACH VISIT:\n", sv.groupby(["VISITNUM", "VISIT"]).size(), "\n")

# ---- Planned vs actual timing (main scheduled visits only) ----
sv = sv.merge(dm[["USUBJID", "SITEID", "RFSTDTC"]], on="USUBJID", how="left")
sv = add_study_day(sv, "SVSTDTC")

sched = sv[(sv["VISITNUM"] % 1 == 0) & (sv["VISITNUM"].between(4, 99))].copy()
sched["DEV"] = sched["SVSTDY"] - sched["VISITDY"]
sched["OUT_OF_WINDOW"] = sched["DEV"].abs() > 7

print("DAYS EARLY (-) / LATE (+):\n", sched["DEV"].describe(), "\n")
print("VISITS PER SITE:\n", sched.groupby("SITEID").size(), "\n")
print("OUT-OF-WINDOW RATE BY SITE (%):\n",
      (sched.groupby("SITEID")["OUT_OF_WINDOW"].mean() * 100).round(0).sort_values(ascending=False), "\n")

worst = sched.reindex(sched["DEV"].abs().sort_values(ascending=False).index)
print("5 MOST EXTREME VISITS:\n",
      worst.head(5)[["USUBJID", "SITEID", "VISIT", "VISITDY", "SVSTDY", "DEV"]])

