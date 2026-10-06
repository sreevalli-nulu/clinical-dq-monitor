import pandas as pd

from src.evaluate_ml import (average_precision, ae_truth, label_units, precision_recall_at_k,
                             rule_visit_units, visit_truth)
from src.ml import AE_FEATURES, VISIT_FEATURES, anomaly_scores, build_ae_features, build_visit_features
from src.rules import run_all_rules
from src.trial_view import PROCESSED_DIR

NAMES = ["subjects", "sv", "vs", "lb", "ae"]
view = {n: pd.read_parquet(PROCESSED_DIR / "injected" / f"{n}.parquet") for n in NAMES}
key = pd.read_parquet(PROCESSED_DIR / "answer_key.parquet")
failed = 0


def check(label, ok, detail=""):
    global failed
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    failed += (not ok)


# 1. precision / recall maths on a tiny hand example
s = pd.Series([0.9, 0.8, 0.7, 0.1])
y = pd.Series([True, False, True, False])
t = precision_recall_at_k(s, y, [2, 3]).set_index("K_FLAGGED")
check("top 2: precision 0.5, recall 0.5", t.loc[2, "PRECISION"] == 0.5 and t.loc[2, "RECALL"] == 0.5)
check("top 3: precision 0.667, recall 1.0", abs(t.loc[3, "PRECISION"] - 0.667) < 1e-3 and t.loc[3, "RECALL"] == 1.0)

# 2. features
feat = build_visit_features(view)
check("one feature row per (subject, visit)", not feat.duplicated(["USUBJID", "VISIT"]).any(), f"{len(feat)} rows")
check("visit features have no missing values", not feat[VISIT_FEATURES].isna().any().any())
ae = build_ae_features(view)
check("AE features have no missing values", not ae[AE_FEATURES].isna().any().any(), f"{len(ae)} events")
check("no answer-key information is used as a feature",
      not ({"IS_ERROR", "ERROR_TYPE", "INJECT_ID", "PARAM_LABEL"} & set(VISIT_FEATURES + AE_FEATURES)))

# 3. the model
a1, a2 = anomaly_scores(feat, VISIT_FEATURES, seed=0), anomaly_scores(feat, VISIT_FEATURES, seed=0)
check("same seed gives the same scores", a1.equals(a2))
check("a different seed gives different scores", not a1.equals(anomaly_scores(feat, VISIT_FEATURES, seed=1)))

# 4. scoring against the answer key
feat["SCORE"] = a1
feat["IS_ERROR"] = label_units(feat, visit_truth(key))
prev = feat["IS_ERROR"].mean()
check("visit model beats random guessing by at least 2x", average_precision(feat["SCORE"], feat["IS_ERROR"]) > 2 * prev,
      f"AP {average_precision(feat['SCORE'], feat['IS_ERROR']):.3f} vs random {prev:.3f}")
ae["SCORE"] = anomaly_scores(ae, AE_FEATURES)
ae["IS_ERROR"] = ae_truth(key, ae)
check("AE model beats random guessing by at least 5x",
      average_precision(ae["SCORE"], ae["IS_ERROR"]) > 5 * ae["IS_ERROR"].mean(),
      f"AP {average_precision(ae['SCORE'], ae['IS_ERROR']):.3f} vs random {ae['IS_ERROR'].mean():.3f}")
pr = precision_recall_at_k(feat["SCORE"], feat["IS_ERROR"], [50, 100, 200, 400])
check("recall never goes down as K grows", pr["RECALL"].is_monotonic_increasing)

rules = rule_visit_units(run_all_rules(view))
feat["RULE_FLAG"] = [(u, v) in rules for u, v in zip(feat["USUBJID"], feat["VISIT"])]
rules_recall = (feat["RULE_FLAG"] & feat["IS_ERROR"]).sum() / feat["IS_ERROR"].sum()
model_recall = pr.set_index("K_FLAGGED").loc[200, "RECALL"]
check("for these known error types the rules beat the model", rules_recall > model_recall,
      f"rules {rules_recall:.2f} vs model(top 200) {model_recall:.2f}")

print("\nALL CHECKS PASSED" if not failed else f"\n{failed} CHECK(S) FAILED")