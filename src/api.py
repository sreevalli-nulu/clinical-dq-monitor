"""Step 9: the FastAPI backend (read-only).

Run with scripts/run_api.py, then open http://127.0.0.1:8000/docs for the interactive docs.
The API only serves what the pipeline already produced. Statistics and rules decide what is
flagged, Claude (or the template) only wrote the wording, and this layer never changes either.
"""
import os
from functools import lru_cache
from pathlib import Path as FsPath
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.api_data import MissingDataError, Store
from src.trial_view import PROCESSED_DIR

ROOT = FsPath(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"                        # the dashboard page lives here
APP_DATA_DIR = ROOT / "app_data"              # small copy of the results that is committed for deployment


def default_data_dir() -> FsPath:
    """Where the result files are read from.
    1) DQ_DATA_DIR if set, 2) your fresh local results in data/processed, 3) the committed app_data copy
    (this is what a deployed server uses, because data/processed is not committed)."""
    env = os.environ.get("DQ_DATA_DIR")
    if env:
        return FsPath(env)
    if (PROCESSED_DIR / "findings_rules.parquet").exists():
        return PROCESSED_DIR
    return APP_DATA_DIR


DATA_DIR = default_data_dir()                 # tests point this at another folder
SAFE_ID = r"^[A-Za-z0-9-]{1,20}$"             # site ids and subject ids: letters, digits, dashes only
Severity = Literal["HIGH", "MEDIUM", "LOW"]

app = FastAPI(
    title="Clinical Data Quality Monitor API",
    version="1.0.0",
    description=("Read-only API over the CDISC Pilot 01 data-quality results: rule findings, site risk "
                 "indicators, ML anomaly scores and plain-language explanation cards. "
                 "Everything returned is a lead for a human reviewer, not a verdict."),
)

# The data is the public CDISC pilot set and the API is read-only, so any origin may read it.
# For a deployment, set DQ_CORS_ORIGINS="https://your-dashboard.example" (comma-separated) to restrict it.
origins = [o.strip() for o in os.environ.get("DQ_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET"], allow_headers=["*"])


# The page is one self-contained file (inline CSS and JS, no outside scripts), so the policy can be strict.
PAGE_HEADERS = {
    "Content-Security-Policy": ("default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                                "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
                                "form-action 'none'"),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def dashboard():
    """The dashboard page."""
    page = WEB_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="web/index.html is missing")
    return FileResponse(page, media_type="text/html", headers=PAGE_HEADERS)


@lru_cache(maxsize=4)
def _store_for(path: str) -> Store:
    return Store(path)


def get_store() -> Store:
    try:
        return _store_for(str(DATA_DIR))
    except MissingDataError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


def need(value, script: str):
    """Optional data (explanations, ML scores) that has not been generated yet -> a clear 503."""
    if value is None:
        raise HTTPException(status_code=503, detail=f"This data has not been generated yet. Run scripts/{script} first.")
    return value


@app.get("/health", tags=["meta"])
def health():
    """Is the API up, and which result files exist? Works even when data is missing."""
    try:
        store = _store_for(str(DATA_DIR))
        return {"status": "ok", "files": store.available()}
    except MissingDataError as exc:
        return {"status": "data_missing", "detail": str(exc)}


@app.get("/summary", tags=["overview"])
def summary(store: Store = Depends(get_store)):
    """Headline numbers for the dashboard cards."""
    return store.summary()


@app.get("/rules", tags=["findings"])
def rules(store: Store = Depends(get_store)):
    """The nine edit checks with how many findings each produced."""
    return store.rules()


@app.get("/findings", tags=["findings"])
def findings(severity: Severity | None = None,
             rule_id: str | None = Query(None, pattern=r"^R\d{2}$", description="e.g. R05"),
             site: str | None = Query(None, pattern=SAFE_ID),
             subject: str | None = Query(None, pattern=SAFE_ID),
             limit: int = Query(50, ge=1, le=500),
             offset: int = Query(0, ge=0),
             store: Store = Depends(get_store)):
    """Rule findings, most severe first. Filter by severity, rule, site or subject; page with limit/offset."""
    return store.findings_query(severity, rule_id, site, subject, limit, offset)


@app.get("/sites", tags=["sites"])
def sites(store: Store = Depends(get_store)):
    """One row per site with finding counts and its worst risk-indicator status (worst first)."""
    return store.sites()


@app.get("/sites/matrix", tags=["sites"])
def site_matrix(store: Store = Depends(get_store)):
    """Site-by-indicator z-scores for a heat map. Small sites have null z and status TOO_SMALL."""
    return store.kri_matrix()


@app.get("/sites/{site}", tags=["sites"])
def site_detail(site: str = Path(..., pattern=SAFE_ID), store: Store = Depends(get_store)):
    """Everything about one site: indicators with z-scores, plus its explanation cards."""
    if not store.has_site(site):
        raise HTTPException(status_code=404, detail=f"Unknown site {site!r}")
    return store.site_detail(site)


@app.get("/subjects/{usubjid}", tags=["subjects"])
def subject_detail(usubjid: str = Path(..., pattern=SAFE_ID), store: Store = Depends(get_store)):
    """One subject: summary row, all findings and explanation cards."""
    if not store.has_subject(usubjid):
        raise HTTPException(status_code=404, detail=f"Unknown subject {usubjid!r}")
    return store.subject_detail(usubjid)


@app.get("/explanations", tags=["explanations"])
def explanations(severity: Severity | None = None,
                 site: str | None = Query(None, pattern=SAFE_ID),
                 kind: Literal["RULE_FINDING", "SITE_SIGNAL"] | None = None,
                 store: Store = Depends(get_store)):
    """Explanation cards. 'source' says whether Claude or the plain template wrote the wording."""
    need(store.explanations, "run_explain.py")
    return store.explanations_query(severity, site, kind)


@app.get("/ml/{which}/performance", tags=["ml"])
def ml_performance(which: Literal["visits", "ae"], store: Store = Depends(get_store)):
    """Precision, recall and average precision of the anomaly scores against the known answer key,
    next to the rules. Computed from the stored scores; nothing is re-trained here."""
    need(store.ml_visits if which == "visits" else store.ml_ae, "run_ml.py")
    return store.ml_performance(which)


@app.get("/ml/{which}", tags=["ml"])
def ml_top(which: Literal["visits", "ae"], k: int = Query(20, ge=1, le=200), store: Store = Depends(get_store)):
    """Top-k anomaly scores. These scores are computed on the ERROR-INJECTED copy of the data,
    so IS_ERROR is the answer key from Step 6. It is an evaluation view, not real findings."""
    need(store.ml_visits if which == "visits" else store.ml_ae, "run_ml.py")
    return {"note": "Scores come from the error-injected test data; IS_ERROR is the known answer key.",
            "items": store.ml_top(which, k)}
