"""Safe API key diagnostic. Never prints the key - only its shape - so the output is safe to paste."""
import os

from dotenv import dotenv_values

from src.trial_view import PROCESSED_DIR

root = PROCESSED_DIR.parents[1]          # the project folder (works in the console, where __file__ does not exist)
env_path = root / ".env"


def describe(label, key):
    if not key:
        print(f"{label}: NOT SET")
        return
    quoted = key[:1] in ('"', "'")
    has_space = " " in key.strip()
    print(f"{label}: set, {len(key)} characters, starts with '{key[:10]}', "
          f"ends with a space/newline: {key != key.strip()}, wrapped in quotes: {quoted}, "
          f"contains a space inside: {has_space}")


print(f"Project folder: {root}")
print(f".env file at project root: {'FOUND' if env_path.exists() else 'NOT FOUND (check the name and the folder)'}")
file_key = dotenv_values(env_path).get("ANTHROPIC_API_KEY") if env_path.exists() else None
env_key = os.environ.get("ANTHROPIC_API_KEY")
describe("Key in .env file          ", file_key)
describe("Key already in environment", env_key)
if file_key and env_key:
    print(f"The two keys are identical: {file_key == env_key}")

key = file_key or env_key
if not key:
    raise SystemExit("\nNo key found anywhere. Create .env in the project root with one line: ANTHROPIC_API_KEY=sk-ant-...")

import anthropic

try:
    client = anthropic.Anthropic(api_key=key.strip().strip("\"'"))
    msg = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=20,
                                 messages=[{"role": "user", "content": "Reply with the single word OK."}])
    print("\nSUCCESS - Claude replied:", "".join(b.text for b in msg.content if getattr(b, "type", "") == "text"))
except Exception as exc:
    print(f"\nFAILED - {type(exc).__name__}: {exc}")