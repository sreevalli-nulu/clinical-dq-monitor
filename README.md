# Clinical Data Quality Monitor

A small web app that checks a clinical trial's data for problems the way a data-quality team would, and explains what it finds in plain language.

It runs on the public **CDISC Pilot 01** sample trial (306 subjects, 17 sites). It combines three kinds of checks and a wording layer:

| Layer | What it does | Who decides |
|---|---|---|
| **Rule edit checks** (9 rules) | Fatal event not marked serious, visit after death, duplicate visits, weight jumps, and so on | Code |
| **Site risk indicators** | Compares each site with all the others (dropout, adverse-event rate, out-of-window visits, data-quality flags, digit preference) using leave-one-out z-scores | Code |
| **Anomaly model** | An Isolation Forest scores visits and adverse-event records that look unusual | Code |
| **Explanation cards** | Turns the facts above into plain sentences | Claude, or a plain template if Claude is unavailable |

**Flags are leads for a reviewer, never verdicts.** Statistics detect, the language layer only explains: it cannot change a severity, and a card is replaced by template wording if its text contains a number that is not in the evidence or accusing words such as "fraud".

> Live app: _add your link here after you deploy (see Deploy below)_

## What the numbers say

| | |
|---|---|
| Subjects / sites | 306 / 17 (4 sites have too few dosed patients for statistics) |
| Rule findings | 213: 4 high, 33 medium, 176 low |
| Site tests | 65 scored: 4 alerts and 5 watches (about 3 flags are expected by chance alone) |
| Detection test* | Rules found 94% of planted visit errors and 83% of planted back-dated adverse events |
| Anomaly model* | Average precision 0.12 on visits (random: 0.03) and 0.29 on adverse events (random: 0.03) |

\*Measured on a copy of the data with known errors deliberately injected (a seeded error injector with an answer key), so recall and precision can be computed honestly. The model is weaker than the rules. Precision is a lower bound, because a flagged record that is not in the answer key may still be a real issue in the original data.

## How it fits together

```
raw SDTM files (data/raw)
   -> clean trial view            src/trial_view.py, subjects.py, visits.py, events.py
   -> rule edit checks            src/rules.py            -> findings_rules.parquet
   -> site statistics (KRIs)      src/site_stats.py       -> site_kris.parquet
   -> error injector + answer key src/injector.py, evaluate.py
   -> anomaly model + evaluation  src/ml.py, evaluate_ml.py -> ml_*_scores.parquet
   -> evidence + explanations     src/evidence.py, explain.py -> explanations.json
   -> read-only API               src/api.py, api_data.py (FastAPI)
   -> dashboard                   web/index.html (served by the same app)
```

The API only reads the saved results. It never recomputes a flag, a z-score or a severity, so the dashboard can never disagree with the pipeline.

## Run it on your computer

```
pip install -r requirements.txt

python scripts/download_data.py        # raw CDISC pilot files
python scripts/build_trial_view.py     # clean tables
python scripts/run_rules.py            # rule findings
python scripts/run_site_stats.py       # site indicators
python scripts/run_injection.py        # error injection + answer key
python scripts/run_ml.py               # anomaly scores
python scripts/run_explain.py          # explanation cards (uses Claude if ANTHROPIC_API_KEY is set)

python scripts/run_api.py              # then open http://127.0.0.1:8000
```

Each step has a matching `scripts/check_*.py` that verifies its output (for example `python scripts/check_api.py` runs 45 checks on the API).

To use Claude for the wording, put `ANTHROPIC_API_KEY=sk-ant-...` in a `.env` file in the project root (it is git-ignored) and run `scripts/run_explain.py`. Without a key the app still works and uses template wording.

## Deploy (free, public)

The deployed server needs no raw data and no API key. It reads a small committed copy of the results in `app_data/` (about 200 KB), and the explanation cards are written ahead of time, so visitors cause no Claude calls and no cost.

1. Refresh the committed copy after any pipeline change: `python scripts/export_app_data.py`
2. Commit and push everything, including `app_data/`, `web/`, `render.yaml` and `requirements-deploy.txt`.
3. On [render.com](https://render.com): **New** > **Blueprint**, pick this GitHub repository and apply. Render reads `render.yaml`, installs `requirements-deploy.txt` and starts `uvicorn src.api:app`.
4. When the deploy finishes, open the `onrender.com` link. `/health` should say `ok`, and `/docs` shows every API endpoint.

On the free plan the app goes to sleep after about 15 minutes without visitors, so the first visit afterwards can take up to a minute to wake it. To restrict which websites may call the API from a browser, set the environment variable `DQ_CORS_ORIGINS` to a comma-separated list of origins.

**Updating the live app:** re-run the pipeline steps you changed, run `scripts/export_app_data.py`, commit, push. Render redeploys automatically.

## API

Read-only. Interactive docs at `/docs`.

| Endpoint | Returns |
|---|---|
| `GET /summary` | Headline numbers |
| `GET /findings` | Rule findings, filter by `severity`, `rule_id`, `site`, `subject`; page with `limit` and `offset` |
| `GET /rules` | The nine rules with counts |
| `GET /sites`, `/sites/{site}`, `/sites/matrix` | Site table, one site in detail, heat-map data |
| `GET /subjects/{usubjid}` | One subject with all findings and cards |
| `GET /explanations` | Explanation cards (filter by `severity`, `site`, `kind`) |
| `GET /ml/visits`, `/ml/ae` and their `/performance` | Top anomaly scores and precision/recall against the answer key (test data) |
| `GET /health` | Is the app up and which result files exist |

## Limits and honesty

- This is a learning project on public sample data. It is not validated software and must not be used for clinical, regulatory or monitoring decisions.
- Site z-scores test many indicators at once, so a few flags are expected by chance. A single Watch is a reason to look, not evidence.
- Sites with fewer than 5 dosed patients get no z-score.
- The anomaly model is evaluated on injected errors, which are simpler than real ones. Treat its numbers as a comparison between methods, not as a promise of real-world performance.
- Free-text values from the database are treated as data, never as instructions, when the explanation layer builds its prompt.

## Data

CDISC Pilot 01, a public sample clinical-trial dataset in the SDTM format.
