from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

for f in sorted(RAW_DIR.glob("*.xpt")):
    df = pd.read_sas(f, format="xport", encoding="utf-8")
    print(f"{f.stem.upper():3} rows={len(df):6}  cols={df.shape[1]:3}  "
          f"subjects={df['USUBJID'].nunique()}")