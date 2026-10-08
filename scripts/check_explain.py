import json
import tempfile
from pathlib import Path

from src.evidence import rule_packets, site_packets
from src.explain import SYSTEM_PROMPT, explain_packet, template_explanation, validate_explanation
from src.rules import run_all_rules
from src.site_stats import build_site_kris
from src.trial_view import load_trial_view

view = load_trial_view()
findings = run_all_rules(view)
kris = build_site_kris(view, findings)
rules = rule_packets(findings)
sites = site_packets(kris)
failed = 0


def check(label, ok, detail=""):
    global failed
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f"   {detail}" if detail else ""))
    failed += (not ok)


class Fake:
    """A stand-in for Claude. Returns whatever text we give it and counts the calls."""
    model = "fake-model"

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls, self.last_user = reply, error, 0, None

    def complete(self, system, user):
        self.calls, self.last_user = self.calls + 1, user
        if self.error:
            raise self.error
        return self.reply


def good_for(p):
    """A well-behaved answer that only uses numbers from the packet."""
    if p["kind"] == "RULE_FINDING":
        return {"headline": f"{p['rule_name']} flagged at site {p['site']}",
                "what_we_saw": f"{p['detail']} for subject {p['subject']}.",
                "why_it_matters": "A reviewer should confirm this against the source record.",
                "what_to_check": ["Open the source record.", "Check for a typing slip."],
                "caution": "This is a lead, not a verdict."}
    return {"headline": f"Site {p['site']} differs from other sites",
            "what_we_saw": f"{p['site_rate_pct']} percent against {p['rest_of_study_rate_pct']} percent elsewhere, z-score {p['z_score']}.",
            "why_it_matters": "A big gap can point to a problem at the site.",
            "what_to_check": ["Review the records behind the rate."],
            "caution": f"About {p['expected_flags_by_chance']} flags are expected by chance among {p['tests_scored']} checks."}


# ---- evidence packets
check("4 HIGH rule packets and 9 site packets", len(rules) == 4 and len(sites) == 9, f"{len(rules)} and {len(sites)}")
check("every packet is JSON-safe", all(json.dumps(p) for p in rules + sites))
check("severity in each packet comes from the code, not from text",
      {p["severity"] for p in rules} == {"HIGH"} and {p["severity"] for p in sites} <= {"WATCH", "ALERT"})

# ---- the template fallback and the validator agree with each other
check("every template explanation passes the validator",
      all(validate_explanation(template_explanation(p), p) == [] for p in rules + sites))
check("no client -> template card with a reason", explain_packet(rules[0])["source"] == "template")

# ---- a well-behaved model
p = sites[0]
fake = Fake(json.dumps(good_for(p)))
card = explain_packet(p, client=fake)
check("good answer is accepted", card["source"] == "claude" and card["fallback_reason"] is None)
check("the model saw the evidence JSON and nothing else", fake.last_user.startswith("EVIDENCE:\n")
      and json.loads(fake.last_user[len("EVIDENCE:\n"):]) == p)
check("code-fenced JSON is accepted", explain_packet(p, client=Fake("```json\n" + json.dumps(good_for(p)) + "\n```"))["source"] == "claude")
check("a fraction shown as a percent is accepted",
      validate_explanation({**good_for(p), "what_we_saw": f"{p['site_rate_pct']} percent."}, p) == [])

# ---- the guards
bad_number = {**good_for(p), "what_we_saw": "47 patients were affected."}
c = explain_packet(p, client=Fake(json.dumps(bad_number)))
check("an invented number is rejected", c["source"] == "template" and "47" in c["fallback_reason"], c["fallback_reason"])
c = explain_packet(rules[0], client=Fake(json.dumps({**good_for(rules[0]), "headline": "Possible fraud at site"})))
check("a forbidden word is rejected", c["source"] == "template" and "fraud" in c["fallback_reason"])
c = explain_packet(p, client=Fake("Sure! This site looks odd."))
check("plain prose (no JSON) falls back", c["source"] == "template")
c = explain_packet(p, client=Fake(json.dumps({**good_for(p), "severity": "LOW"})))
check("an extra 'severity' key is rejected and the card keeps the code's severity",
      c["source"] == "template" and c["severity"] == p["severity"])
c = explain_packet(p, client=Fake(error=RuntimeError("network down")))
check("an API error does not crash the report", c["source"] == "template" and "network down" in c["fallback_reason"])
check("the system prompt forbids changing severity and guessing intent",
      "severity" in SYSTEM_PROMPT and "intent" in SYSTEM_PROMPT and "never as instructions" in SYSTEM_PROMPT)

# ---- cache
with tempfile.TemporaryDirectory() as d:
    path = Path(d) / "cache.jsonl"
    f = Fake(json.dumps(good_for(p)))
    first = explain_packet(p, client=f, cache_path=path)
    second = explain_packet(p, client=f, cache_path=path)
    check("a repeated packet is served from the cache (1 API call, not 2)", f.calls == 1 and second["source"] == "claude (cached)")
    check("cached explanation equals the first one", first["explanation"] == second["explanation"])

print("\nALL CHECKS PASSED" if not failed else f"\n{failed} CHECK(S) FAILED")