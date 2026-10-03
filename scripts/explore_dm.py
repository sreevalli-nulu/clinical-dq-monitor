import pandas as pd
from src.load import load_domain

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

dm = load_domain("dm")

print("COLUMNS:", dm.columns.tolist(), "\n")
print("TREATMENT ARMS:\n", dm["ARM"].value_counts(), "\n")
print("SUBJECTS PER SITE:\n", dm["SITEID"].value_counts().sort_index(), "\n")
print("SITE x ARM:\n", pd.crosstab(dm["SITEID"], dm["ARM"]), "\n")
print("AGE SUMMARY:\n", dm["AGE"].describe(), "\n")
print("FIRST 5 SUBJECTS:\n", dm[["USUBJID", "SITEID", "ARM", "AGE", "SEX", "RFSTDTC", "RFENDTC"]].head())
