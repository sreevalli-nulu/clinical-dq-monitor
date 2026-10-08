"""Step 9 checks. Runs the API in-process (no server needed) against the real result files."""
import json
import re
import tempfile

import pandas as pd
from fastapi.testclient import TestClient

from src import api
from src.trial_view import PROCESSED_DIR

client = TestClient(api.app)
results = []


def check(name, ok, extra=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({extra})" if extra else ""))


def get(url):
    return client.get(url)


findings = pd.read_parquet(PROCESSED_DIR / "findings_rules.parquet")
subjects = pd.read_parquet(PROCESSED_DIR / "subjects.parquet")
kris = pd.read_parquet(PROCESSED_DIR / "site_kris.parquet")
cards = json.loads((PROCESSED_DIR / "explanations.json").read_text(encoding="utf-8"))

# 1 health
r = get("/health")
check("health is ok and lists the files", r.status_code == 200 and r.json()["status"] == "ok" and r.json()["files"]["findings"])

# 2 summary agrees with the files
s = get("/summary").json()
check("summary counts match the files",
      s["subjects"] == len(subjects) and s["findings"]["total"] == len(findings)
      and sum(s["findings"]["by_severity"].values()) == len(findings)
      and sum(x["findings"] for x in s["findings"]["by_rule"]) == len(findings),
      f"{s['subjects']} subjects, {s['findings']['total']} findings")
check("summary site signals match the KRI file",
      s["site_signals"]["ALERT"] == int((kris.STATUS == "ALERT").sum())
      and s["site_signals"]["WATCH"] == int((kris.STATUS == "WATCH").sum()))

# 3 findings: pagination, filters, ordering
r = get("/findings?limit=10").json()
check("findings paging: total and page size", r["total"] == len(findings) and len(r["items"]) == 10)
sev_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
all_items = get("/findings?limit=500").json()["items"]
check("findings are ordered most severe first", [sev_order[i["SEVERITY"]] for i in all_items]
      == sorted(sev_order[i["SEVERITY"]] for i in all_items))
page1 = get("/findings?limit=100&offset=0").json()["items"]
page2 = get("/findings?limit=100&offset=100").json()["items"]
check("two pages stitched together equal the first 200 rows", page1 + page2 == all_items[:200])
high = get("/findings?severity=HIGH").json()
check("severity filter", high["total"] == int((findings.SEVERITY == "HIGH").sum()) and all(i["SEVERITY"] == "HIGH" for i in high["items"]))
r05 = get("/findings?rule_id=R05").json()
check("rule filter", r05["total"] == int((findings.RULE_ID == "R05").sum()) and all(i["RULE_ID"] == "R05" for i in r05["items"]))
some_site = findings.SITEID.iloc[0]
by_site = get(f"/findings?site={some_site}").json()
check("site filter", by_site["total"] == int((findings.SITEID == some_site).sum()))
check("findings page contains no NaN text", "NaN" not in get("/findings?limit=500").text)

# 4 input validation
check("bad severity is rejected (422)", get("/findings?severity=BANANA").status_code == 422)
check("limit above 500 is rejected (422)", get("/findings?limit=10000").status_code == 422)
check("odd characters in a site id are rejected (422)", get("/findings?site=../etc").status_code == 422)
check("rule_id must look like R05 (422)", get("/findings?rule_id=DROP").status_code == 422)

# 5 sites
sites = get("/sites").json()
check("every site appears once", sorted(x["site"] for x in sites) == sorted(subjects.SITEID.unique())
      and len({x["site"] for x in sites}) == len(sites), f"{len(sites)} sites")
check("sites are listed worst status first",
      [x["worst_status"] for x in sites] == sorted((x["worst_status"] for x in sites),
                                                   key=lambda t: -{"ALERT": 3, "WATCH": 2, "OK": 1, "TOO_SMALL": 0}[t]))
check("site finding counts add up to the file", sum(sum(x["findings"].values()) for x in sites) == len(findings))
top = sites[0]["site"]
d = get(f"/sites/{top}")
check("site detail returns its indicators", d.status_code == 200 and len(d.json()["kris"]) == kris.KRI.nunique(), f"site {top}")
check("unknown site is 404", get("/sites/999").status_code == 404)
m = get("/sites/matrix").json()
check("matrix has one row per site and one cell per indicator",
      len(m["rows"]) == subjects.SITEID.nunique() and all(len(r["cells"]) == len(m["kris"]) for r in m["rows"]))
check("matrix has no NaN (null instead)", "NaN" not in get("/sites/matrix").text)
too_small = [c for r in m["rows"] for c in r["cells"] if c["status"] == "TOO_SMALL"]
check("TOO_SMALL cells have null z", too_small and all(c["z"] is None for c in too_small), f"{len(too_small)} cells")

# 6 subjects
sid = findings.USUBJID.iloc[0]
sd = get(f"/subjects/{sid}")
check("subject detail returns summary row and findings",
      sd.status_code == 200 and sd.json()["subject"]["USUBJID"] == sid
      and len(sd.json()["findings"]) == int((findings.USUBJID == sid).sum()))
check("unknown subject is 404", get("/subjects/NOPE-1").status_code == 404)

# 7 explanations
e = get("/explanations").json()
check("all explanation cards are served", len(e) == len(cards), f"{len(e)} cards")
check("cards keep severity and source from the pipeline",
      {(c["card_id"], c["severity"], c["source"]) for c in e} == {(c["card_id"], c["severity"], c["source"]) for c in cards})
check("explanation filter by severity", all(c["severity"] == "HIGH" for c in get("/explanations?severity=HIGH").json()))
check("explanation filter by kind",
      all(c["kind"] == "SITE_SIGNAL" for c in get("/explanations?kind=SITE_SIGNAL").json()))

# 8 ML
v = get("/ml/visits?k=15").json()
scores = [i["SCORE"] for i in v["items"]]
check("ML top-k size and order", len(scores) == 15 and scores == sorted(scores, reverse=True))
check("ML response states it is test data", "injected" in v["note"].lower())
check("ML ae endpoint works", len(get("/ml/ae?k=5").json()["items"]) == 5)
check("unknown ML kind is rejected (422)", get("/ml/banana").status_code == 422)

# 8b ML performance agrees with scikit-learn and with the stored scores
from sklearn.metrics import average_precision_score

ml_v = pd.read_parquet(PROCESSED_DIR / "ml_visit_scores.parquet")
ml_a = pd.read_parquet(PROCESSED_DIR / "ml_ae_scores.parquet")
for which, frame in (("visits", ml_v), ("ae", ml_a)):
    pf = get(f"/ml/{which}/performance").json()
    ref_ap = average_precision_score(frame["IS_ERROR"], frame["SCORE"])
    top_k = frame.sort_values("SCORE", ascending=False, kind="stable").head(pf["at_k"][0]["k"])
    check(f"{which}: average precision matches scikit-learn", abs(pf["average_precision"] - ref_ap) < 1e-3,
          f"{pf['average_precision']:.4f} vs {ref_ap:.4f}")
    check(f"{which}: precision@k is hits/k and recall uses the answer key",
          pf["at_k"][0]["hits"] == int(top_k["IS_ERROR"].sum()) and pf["errors"] == int(frame["IS_ERROR"].sum()))
    check(f"{which}: rule precision and recall are consistent",
          pf["rules"]["hits"] == int((frame["RULE_FLAG"] & frame["IS_ERROR"]).sum())
          and abs(pf["rules"]["recall"] - pf["rules"]["hits"] / pf["errors"]) < 1e-9)

# 8c the dashboard page and its headers
page = get("/")
check("dashboard page is served", page.status_code == 200 and "Clinical Data Quality Monitor" in page.text
      and page.headers["content-type"].startswith("text/html"))
check("page has a strict content-security-policy", "default-src 'self'" in page.headers.get("content-security-policy", "")
      and page.headers.get("x-content-type-options") == "nosniff")
check("page loads nothing from other websites", not re.findall(r'(?:src|href)=["\']https?://', page.text) and "<script src" not in page.text)

# 8d the committed app_data copy (what a deployed server reads) serves the same numbers
original = api.DATA_DIR
api.DATA_DIR = api.APP_DATA_DIR
copy_summary = get("/summary").json()
api.DATA_DIR = original
check("app_data copy gives the same summary as data/processed", copy_summary == get("/summary").json(),
      "run scripts/export_app_data.py if this fails")

# 9 missing data gives a clear 503, health still answers
original = api.DATA_DIR
with tempfile.TemporaryDirectory() as empty:
    api.DATA_DIR = type(PROCESSED_DIR)(empty)
    r = get("/summary")
    h = get("/health").json()
    api.DATA_DIR = original
check("missing data -> 503 with a helpful message", r.status_code == 503 and "scripts/" in r.json()["detail"])
check("health reports data_missing instead of crashing", h["status"] == "data_missing")

# 10 read-only
check("write methods are not allowed", client.post("/findings").status_code == 405 and client.delete("/sites/701").status_code == 405)

print(f"\n{sum(results)}/{len(results)} checks passed")
if not all(results):
    raise SystemExit(1)
