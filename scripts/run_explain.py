import json

from src.evidence import rule_packets, site_packets
from src.explain import ClaudeClient, explain_packet, have_api_key
from src.rules import run_all_rules
from src.site_stats import build_site_kris
from src.trial_view import PROCESSED_DIR, load_trial_view

MAX_CARDS = 15

view = load_trial_view()
findings = run_all_rules(view)
kris = build_site_kris(view, findings)

packets = (rule_packets(findings, severities=("HIGH",)) + site_packets(kris))[:MAX_CARDS]
print(f"{len(packets)} cards to explain: {sum(p['kind'] == 'RULE_FINDING' for p in packets)} rule findings (HIGH) "
      f"and {sum(p['kind'] == 'SITE_SIGNAL' for p in packets)} site signals (WATCH/ALERT)")

if have_api_key():
    client = ClaudeClient()
    print(f"Using Claude model {client.model}")
else:
    client = None
    print("No ANTHROPIC_API_KEY found - using the plain template explanations (no API calls).")

cache = PROCESSED_DIR / "explanations_cache.jsonl"
cards = [explain_packet(p, client=client, cache_path=cache) for p in packets]

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
(PROCESSED_DIR / "explanations.json").write_text(json.dumps(cards, indent=2, default=str), encoding="utf-8")

lines = ["# Clinical data quality report (explanations)", "",
         "Flags come from rules and statistics. Claude only wrote the wording. Severity and numbers are from code.", ""]
for c in cards:
    e = c["explanation"]
    lines += [f"## [{c['severity']}] {e['headline']}", f"*{c['kind']} - site {c['site']}"
              + (f" - subject {c['subject']}" if c["subject"] else "") + f" - explanation source: {c['source']}*", "",
              e["what_we_saw"], "", f"**Why it matters:** {e['why_it_matters']}", "", "**What to check:**"]
    lines += [f"- {a}" for a in e["what_to_check"]]
    lines += ["", f"**Caution:** {e['caution']}", ""]
    if c["fallback_reason"] and client is not None:
        lines += [f"> Fallback used: {c['fallback_reason']}", ""]
(PROCESSED_DIR.parent.parent / "reports").mkdir(exist_ok=True)
report = PROCESSED_DIR.parent.parent / "reports" / "dq_report.md"
report.write_text("\n".join(lines), encoding="utf-8")

by_source = {}
for c in cards:
    by_source[c["source"]] = by_source.get(c["source"], 0) + 1
print("Explanation sources:", by_source)
for c in cards:
    if c["fallback_reason"] and client is not None:
        print(f"  fallback for {c['card_id']}: {c['fallback_reason']}")
print(f"\nSaved {report}")
print("\n--- first card ---")
print("\n".join(lines[4:16]))