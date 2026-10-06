import pandas as pd

from src.evaluate_ml import (VISIT_ERROR_TYPES, ae_truth, average_precision, label_units,
                             precision_recall_at_k, rule_visit_units, visit_truth)
from src.ml import AE_FEATURES, VISIT_FEATURES, anomaly_scores, build_ae_features, build_visit_features
from src.rules import run_all_rules
from src.trial_view import PROCESSED_DIR

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)

NAMES = ["subjects", "sv", "vs", "lb", "ae"]
view = {n: pd.read_parquet(PROCESSED_DIR / "injected" / f"{n}.parquet") for n in NAMES}
key = pd.read_parquet(PROCESSED_DIR / "answer_key.parquet")
findings = run_all_rules(view)

# ======================= VISIT MODEL =======================
feat = build_visit_features(view)
feat["SCORE"] = anomaly_scores(feat, VISIT_FEATURES)
truth = visit_truth(key)
feat["IS_ERROR"] = label_units(feat, truth)
n_err = int(feat["IS_ERROR"].sum())
print(f"=== VISIT MODEL: {len(feat)} subject-visits, {n_err} contain a planted error "
      f"({feat['IS_ERROR'].mean():.1%}). Planted visit errors in key: {len(truth)} ===")

ks = [25, 50, 100, 200, 400]
pr = precision_recall_at_k(feat["SCORE"], feat["IS_ERROR"], ks)
print("\nIsolation Forest - what you get if reviewers read only the top K visits:")
print(pr.to_string(index=False))
print(f"\nAverage precision: {average_precision(feat['SCORE'], feat['IS_ERROR']):.3f}   "
      f"(random guessing would give about {feat['IS_ERROR'].mean():.3f})")

rule_units = rule_visit_units(findings)
feat["RULE_FLAG"] = [(u, v) in rule_units for u, v in zip(feat["USUBJID"], feat["VISIT"])]
tp = int((feat["RULE_FLAG"] & feat["IS_ERROR"]).sum())
print(f"\nRules (R02, R03, R04, R06, R08): flagged {int(feat['RULE_FLAG'].sum())} visits, "
      f"{tp} contain a planted error -> precision {tp / feat['RULE_FLAG'].sum():.3f}, recall {tp / n_err:.3f}")

K = 200
top = set(feat.sort_values("SCORE", ascending=False).head(K).index)
feat["ML_TOPK"] = feat.index.isin(top)
union = feat["RULE_FLAG"] | feat["ML_TOPK"]
print(f"Rules + top {K} from the model: flagged {int(union.sum())} visits, "
      f"recall {(union & feat['IS_ERROR']).sum() / n_err:.3f}")

# recall by error type, for rules vs model vs both
t = truth.merge(feat[["USUBJID", "VISIT", "RULE_FLAG", "ML_TOPK"]], on=["USUBJID", "VISIT"], how="left")
t["EITHER"] = t["RULE_FLAG"] | t["ML_TOPK"]
print(f"\nRecall by error type (model = top {K} visits):")
print(t.groupby("ERROR_TYPE")[["RULE_FLAG", "ML_TOPK", "EITHER"]].mean().round(2)
      .rename(columns={"RULE_FLAG": "RULES", "ML_TOPK": "MODEL", "EITHER": "EITHER"}).to_string())

w = t[t["ERROR_TYPE"] == "WEIGHT_CHANGE"]
print("\nWeight changes: who finds which size? (share found)")
print(w.groupby("PARAM_LABEL")[["RULE_FLAG", "ML_TOPK"]].mean().round(2)
      .rename(columns={"RULE_FLAG": "RULES", "ML_TOPK": "MODEL"}).to_string())

missed = t[~t["RULE_FLAG"]]
print(f"\nPlanted visit errors the RULES missed: {len(missed)}. The model's top {K} caught "
      f"{int(missed['ML_TOPK'].sum())} of them.")

print("\nWhich features push visits to the top 25? (average value: top 25 vs everyone)")
cmp = pd.DataFrame({"TOP_25": feat.sort_values("SCORE", ascending=False).head(25)[VISIT_FEATURES].mean(),
                    "ALL": feat[VISIT_FEATURES].mean()}).round(2)
print(cmp.to_string())

# What are the model's top visits? Planned in-person visits that exist in SV but have no vitals and no labs.
sv_d = view["sv"][view["sv"]["DOSED"] & (view["sv"]["VISIT_TYPE"] == "PLANNED")]
has_data = set(zip(view["vs"]["USUBJID"], view["vs"]["VISIT"])) | set(zip(view["lb"]["USUBJID"], view["lb"]["VISIT"]))
empty = sv_d[[(u, v) not in has_data for u, v in zip(sv_d["USUBJID"], sv_d["VISIT"])]]
in_person = empty[~empty["VISIT"].str.contains(r"\(T\)", regex=True)]
print(f"\nPlanned visits with no vitals and no labs: {len(empty)}; "
      f"{len(empty) - len(in_person)} are telephone visits (normal), {len(in_person)} are in-person visits (suspicious).")
print(f"Of the model's top 25 visits, {int(feat.sort_values('SCORE', ascending=False).head(25).set_index(['USUBJID', 'VISIT']).index.isin(list(zip(in_person['USUBJID'], in_person['VISIT']))).sum())} "
      f"are exactly such in-person visits with no data. None of these is in the answer key.")

# ======================= AE MODEL =======================
ae = build_ae_features(view)
ae["SCORE"] = anomaly_scores(ae, AE_FEATURES)
ae["IS_ERROR"] = ae_truth(key, ae)
print(f"\n=== AE MODEL: {len(ae)} adverse events, {int(ae['IS_ERROR'].sum())} back-dated ===")
print(precision_recall_at_k(ae["SCORE"], ae["IS_ERROR"], [10, 25, 50, 100]).to_string(index=False))
print(f"Average precision: {average_precision(ae['SCORE'], ae['IS_ERROR']):.3f}   (random: {ae['IS_ERROR'].mean():.3f})")
r07 = findings[findings["RULE_ID"] == "R07"]
r07_keys = set(zip(r07["USUBJID"], r07["WHERE"].str[3:]))
ae["RULE_FLAG"] = [(u, t_) in r07_keys for u, t_ in zip(ae["USUBJID"], ae["AETERM"])]
print(f"Rule R07 flagged {int(ae['RULE_FLAG'].sum())} events, {int((ae['RULE_FLAG'] & ae['IS_ERROR']).sum())} of them back-dated; "
      f"model top 50 found {int(ae.sort_values('SCORE', ascending=False).head(50)['IS_ERROR'].sum())}.")

feat.drop(columns=["SITEID"]).to_parquet(PROCESSED_DIR / "ml_visit_scores.parquet", index=False)
ae.to_parquet(PROCESSED_DIR / "ml_ae_scores.parquet", index=False)
print(f"\nSaved scores to {PROCESSED_DIR}")