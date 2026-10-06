import pandas as pd

from src.config import MIN_SITE_SUBJECTS
from src.load import load_domain


def build_subjects() -> pd.DataFrame:
    """One clean row per screened subject, with population flags and outcome."""
    dm = load_domain("dm")
    ds = load_domain("ds")
    sv = load_domain("sv")

    s = dm[["USUBJID", "SITEID", "ARM", "ACTARM", "AGE", "SEX",
            "RFSTDTC", "RFENDTC", "DTHFL"]].copy()
    s["SITEID"] = s["SITEID"].astype(str)
    s["RFSTDT"] = pd.to_datetime(s["RFSTDTC"], errors="coerce")
    s["RFENDT"] = pd.to_datetime(s["RFENDTC"], errors="coerce")
    s = s.drop(columns=["RFSTDTC", "RFENDTC"])

    # Populations: everyone who was screened, and the subset who actually got a dose
    s["SCREEN_FAIL"] = s["ARM"] == "Screen Failure"
    s["DOSED"] = ~s["SCREEN_FAIL"]

    # Final outcome: exactly one 'DISPOSITION EVENT' row per subject in DS
    final = ds[ds["DSCAT"] == "DISPOSITION EVENT"][["USUBJID", "DSDECOD"]]
    final = final.rename(columns={"DSDECOD": "DISPOSITION"})
    s = s.merge(final, on="USUBJID", how="left", validate="one_to_one")

    s["COMPLETED"] = s["DISPOSITION"] == "COMPLETED"
    s["DISCONTINUED"] = s["DOSED"] & ~s["COMPLETED"]
    s["DIED"] = s["DTHFL"] == "Y"

    # How many visits each subject has in SV
    n_visits = sv.groupby("USUBJID").size().rename("N_VISITS")
    s = s.merge(n_visits, on="USUBJID", how="left")
    s["N_VISITS"] = s["N_VISITS"].fillna(0).astype(int)

    # Site size (dosed subjects only) and the small-site flag
    site_n = s[s["DOSED"]].groupby("SITEID").size().rename("SITE_N_DOSED")
    s = s.merge(site_n, on="SITEID", how="left")
    s["SITE_N_DOSED"] = s["SITE_N_DOSED"].fillna(0).astype(int)
    s["SMALL_SITE"] = s["SITE_N_DOSED"] < MIN_SITE_SUBJECTS

    return s.drop(columns=["DTHFL"])