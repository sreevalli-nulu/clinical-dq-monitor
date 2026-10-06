from pathlib import Path

import pandas as pd

from src.events import build_ae
from src.subjects import build_subjects
from src.visits import build_lb, build_sv, build_vs

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

BUILDERS = {
    "subjects": build_subjects,
    "sv": build_sv,
    "vs": build_vs,
    "lb": build_lb,
    "ae": build_ae,
}


def arm_column(purpose: str) -> str:
    """Which arm column to use. 'retention' (dropout, completion, visit compliance) uses the
    planned arm; 'exposure' (what people actually received, e.g. AE rates by dose) uses the actual arm."""
    columns = {"retention": "ARM", "exposure": "ACTARM"}
    if purpose not in columns:
        raise ValueError(f"purpose must be one of {sorted(columns)}, got {purpose!r}")
    return columns[purpose]


def build_trial_view() -> dict[str, pd.DataFrame]:
    """Build every clean table from the raw SDTM files."""
    return {name: build() for name, build in BUILDERS.items()}


def save_trial_view(view: dict[str, pd.DataFrame]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in view.items():
        df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)


def load_trial_view() -> dict[str, pd.DataFrame]:
    """Fast load of the saved tables (run scripts/build_trial_view.py first)."""
    missing = [n for n in BUILDERS if not (PROCESSED_DIR / f"{n}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing} in {PROCESSED_DIR}. Run scripts/build_trial_view.py first.")
    return {name: pd.read_parquet(PROCESSED_DIR / f"{name}.parquet") for name in BUILDERS}