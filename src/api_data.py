"""Step 9: the data layer behind the API.

Everything here is plain pandas, with no web code, so it can be tested on its own.
The store only READS the files that Steps 1-8 already wrote. It never recomputes
a flag, a z-score or a severity, so the API can never disagree with the pipeline.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.rules import RULES, SEVERITY_ORDER
from src.site_stats import KRI_LABELS

REQUIRED_FILES = {
    "subjects": "subjects.parquet",
    "findings": "findings_rules.parquet",
    "kris": "site_kris.parquet",
}
OPTIONAL_FILES = {
    "explanations": "explanations.json",
    "ml_visits": "ml_visit_scores.parquet",
    "ml_ae": "ml_ae_scores.parquet",
}
STATUS_RANK = {"ALERT": 3, "WATCH": 2, "OK": 1, "TOO_SMALL": 0}


class MissingDataError(FileNotFoundError):
    """A file the API needs does not exist yet. The message says which script creates it."""


def to_records(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> list of JSON-safe dicts (NaN becomes null, dates become ISO text)."""
    return json.loads(df.to_json(orient="records", date_format="iso"))


class Store:
    def __init__(self, processed_dir: Path):
        self.dir = Path(processed_dir)
        missing = [name for name in REQUIRED_FILES.values() if not (self.dir / name).exists()]
        if missing:
            raise MissingDataError(f"Missing {missing} in {self.dir}. Run scripts/build_trial_view.py, "
                                   f"scripts/run_rules.py and scripts/run_site_stats.py first.")
        self.subjects = pd.read_parquet(self.dir / REQUIRED_FILES["subjects"])
        self.findings = pd.read_parquet(self.dir / REQUIRED_FILES["findings"])
        self.kris = pd.read_parquet(self.dir / REQUIRED_FILES["kris"])
        self.explanations = self._read_json("explanations")
        self.ml_visits = self._read_parquet("ml_visits")
        self.ml_ae = self._read_parquet("ml_ae")

    def _read_json(self, key):
        path = self.dir / OPTIONAL_FILES[key]
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def _read_parquet(self, key):
        path = self.dir / OPTIONAL_FILES[key]
        return pd.read_parquet(path) if path.exists() else None

    def available(self) -> dict:
        names = {**REQUIRED_FILES, **OPTIONAL_FILES}
        return {key: (self.dir / fname).exists() for key, fname in names.items()}

    # ---------------------------------------------------------------- overview
    def summary(self) -> dict:
        f, k, s = self.findings, self.kris, self.subjects
        by_sev = {sev: int((f["SEVERITY"] == sev).sum()) for sev in SEVERITY_ORDER}
        by_rule = (f.groupby("RULE_ID").size().reindex([r.rule_id for r in RULES], fill_value=0))
        sites = self.sites()
        sources = {}
        for card in self.explanations or []:
            sources[card["source"]] = sources.get(card["source"], 0) + 1
        return {
            "subjects": int(len(s)),
            "dosed_subjects": int(s["DOSED"].sum()),
            "sites": int(s["SITEID"].nunique()),
            "findings": {
                "total": int(len(f)),
                "by_severity": by_sev,
                "by_rule": [{"rule_id": r.rule_id, "rule_name": r.name, "findings": int(by_rule[r.rule_id])}
                            for r in RULES],
            },
            "site_signals": {st: int((k["STATUS"] == st).sum()) for st in STATUS_RANK},
            "sites_by_worst_status": {st: sum(1 for x in sites if x["worst_status"] == st) for st in STATUS_RANK},
            "explanation_cards": None if self.explanations is None else len(self.explanations),
            "explanation_sources": sources,
            "note": "Findings and signals are leads for a human reviewer, not verdicts.",
        }

    # ---------------------------------------------------------------- findings
    def findings_query(self, severity=None, rule_id=None, site=None, subject=None, limit=50, offset=0) -> dict:
        f = self.findings
        if severity:
            f = f[f["SEVERITY"] == severity]
        if rule_id:
            f = f[f["RULE_ID"] == rule_id]
        if site:
            f = f[f["SITEID"] == site]
        if subject:
            f = f[f["USUBJID"] == subject]
        f = (f.assign(_o=f["SEVERITY"].map(SEVERITY_ORDER))
              .sort_values(["_o", "RULE_ID", "SITEID", "USUBJID"], kind="stable").drop(columns="_o"))
        page = f.iloc[offset: offset + limit]
        return {"total": int(len(f)), "limit": limit, "offset": offset, "items": to_records(page)}

    def rules(self) -> list[dict]:
        counts = self.findings.groupby("RULE_ID").agg(findings=("USUBJID", "size"), subjects=("USUBJID", "nunique"))
        out = []
        for r in RULES:
            row = counts.loc[r.rule_id] if r.rule_id in counts.index else None
            out.append({"rule_id": r.rule_id, "rule_name": r.name,
                        "findings": int(row["findings"]) if row is not None else 0,
                        "subjects": int(row["subjects"]) if row is not None else 0})
        return out

    # ------------------------------------------------------------------- sites
    def sites(self) -> list[dict]:
        s, k, f = self.subjects, self.kris, self.findings
        base = s.groupby("SITEID").agg(subjects=("USUBJID", "size"), dosed=("DOSED", "sum"),
                                       small_site=("SMALL_SITE", "any"))
        sev = f.groupby(["SITEID", "SEVERITY"]).size().unstack(fill_value=0)
        rows = []
        for site, b in base.iterrows():
            ks = k[k["SITEID"] == site]
            worst = max(ks["STATUS"], key=lambda x: STATUS_RANK[x], default="TOO_SMALL")
            counts = {name: int(sev.loc[site, name]) if site in sev.index and name in sev.columns else 0
                      for name in SEVERITY_ORDER}
            rows.append({"site": site, "subjects": int(b["subjects"]), "dosed": int(b["dosed"]),
                         "small_site": bool(b["small_site"]), "findings": counts,
                         "worst_status": worst,
                         "alert_kris": int((ks["STATUS"] == "ALERT").sum()),
                         "watch_kris": int((ks["STATUS"] == "WATCH").sum())})
        return sorted(rows, key=lambda r: (-STATUS_RANK[r["worst_status"]], -r["findings"]["HIGH"], r["site"]))

    def has_site(self, site: str) -> bool:
        return bool((self.subjects["SITEID"] == site).any())

    def site_detail(self, site: str) -> dict:
        info = next(r for r in self.sites() if r["site"] == site)
        ks = self.kris[self.kris["SITEID"] == site].copy()
        ks["LABEL"] = ks["KRI"].map(KRI_LABELS)
        return {**info, "kris": to_records(ks),
                "explanations": [c for c in (self.explanations or []) if c.get("site") == site]}

    def kri_matrix(self) -> dict:
        """Heat-map data for the dashboard: one row per site, one z-score per KRI."""
        wide = self.kris.pivot(index="SITEID", columns="KRI", values="Z")
        status = self.kris.pivot(index="SITEID", columns="KRI", values="STATUS")
        kris = list(wide.columns)
        return {"kris": [{"kri": c, "label": KRI_LABELS.get(c, c)} for c in kris],
                "rows": [{"site": site,
                          "cells": [{"kri": c,
                                     "z": None if pd.isna(wide.loc[site, c]) else round(float(wide.loc[site, c]), 3),
                                     "status": str(status.loc[site, c])} for c in kris]}
                         for site in wide.index]}

    # ---------------------------------------------------------------- subjects
    def has_subject(self, usubjid: str) -> bool:
        return bool((self.subjects["USUBJID"] == usubjid).any())

    def subject_detail(self, usubjid: str) -> dict:
        row = self.subjects[self.subjects["USUBJID"] == usubjid].iloc[0:1]
        flags = self.findings_query(subject=usubjid, limit=500)["items"]
        cards = [c for c in (self.explanations or []) if c.get("subject") == usubjid]
        return {"subject": to_records(row)[0], "findings": flags, "explanations": cards}

    # ------------------------------------------------------------ explanations
    def explanations_query(self, severity=None, site=None, kind=None) -> list[dict]:
        cards = self.explanations or []
        if severity:
            cards = [c for c in cards if c.get("severity") == severity]
        if site:
            cards = [c for c in cards if c.get("site") == site]
        if kind:
            cards = [c for c in cards if c.get("kind") == kind]
        return cards

    # ---------------------------------------------------------------------- ML
    def ml_top(self, which: str, k: int) -> list[dict]:
        df = self.ml_visits if which == "visits" else self.ml_ae
        cols = {"visits": ["USUBJID", "VISIT", "SCORE", "RULE_FLAG", "IS_ERROR"],
                "ae": ["USUBJID", "SITEID", "AESPID", "AETERM", "SCORE", "RULE_FLAG", "IS_ERROR"]}[which]
        top = df.sort_values("SCORE", ascending=False, kind="stable").head(k)[cols]
        return to_records(top)

    def ml_performance(self, which: str) -> dict:
        """Precision and recall of the anomaly scores against the Step 6 answer key (IS_ERROR).
        Pure pandas/numpy so the deployed server does not need scikit-learn."""
        df = self.ml_visits if which == "visits" else self.ml_ae
        ranked = df.sort_values("SCORE", ascending=False, kind="stable")
        y = ranked["IS_ERROR"].to_numpy(dtype=bool)
        n, n_err = len(y), int(y.sum())
        base = n_err / n
        hits_cum = np.cumsum(y)
        avg_prec = float((hits_cum[y] / (np.flatnonzero(y) + 1)).mean()) if n_err else 0.0
        ks = [k for k in ([25, 50, 100, 200, 400] if which == "visits" else [10, 25, 50, 100]) if k <= n]
        at_k = [{"k": k, "hits": int(hits_cum[k - 1]), "precision": float(hits_cum[k - 1] / k),
                 "recall": float(hits_cum[k - 1] / n_err), "lift": float(hits_cum[k - 1] / k / base)} for k in ks]
        flagged = df["RULE_FLAG"].to_numpy(dtype=bool)
        rule_hits = int((flagged & df["IS_ERROR"].to_numpy(dtype=bool)).sum())
        out = {"units": "visits" if which == "visits" else "adverse events", "n": n, "errors": n_err, "base_rate": base,
               "average_precision": avg_prec, "at_k": at_k,
               "rules": {"flagged": int(flagged.sum()), "hits": rule_hits,
                         "precision": rule_hits / max(int(flagged.sum()), 1), "recall": rule_hits / n_err},
               "rules_plus_model": None}
        if which == "visits" and "ML_TOPK" in df:
            union = flagged | df["ML_TOPK"].to_numpy(dtype=bool)
            out["rules_plus_model"] = {"model_top_k": int(df["ML_TOPK"].sum()), "flagged": int(union.sum()),
                                       "recall": float((union & df["IS_ERROR"].to_numpy(dtype=bool)).sum() / n_err)}
        return out
