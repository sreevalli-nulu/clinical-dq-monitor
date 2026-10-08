"""Copy the small result files the app needs into app_data/, which IS committed to git.

The server you deploy does not have your raw data or your data/processed folder (they are not committed),
so it reads these copies instead. Run this again after you re-run the pipeline, then commit app_data/.
"""
import shutil

from src.api import APP_DATA_DIR
from src.api_data import OPTIONAL_FILES, REQUIRED_FILES
from src.trial_view import PROCESSED_DIR

APP_DATA_DIR.mkdir(exist_ok=True)
total = 0
for fname in [*REQUIRED_FILES.values(), *OPTIONAL_FILES.values()]:
    src = PROCESSED_DIR / fname
    if not src.exists():
        raise SystemExit(f"{src} is missing - run the earlier steps first (rules, site stats, ML, explanations).")
    shutil.copy2(src, APP_DATA_DIR / fname)
    total += src.stat().st_size
    print(f"copied {fname:28s} {src.stat().st_size / 1024:7.1f} KB")
print(f"\n{len(REQUIRED_FILES) + len(OPTIONAL_FILES)} files, {total / 1024:.0f} KB total, copied to {APP_DATA_DIR}")
