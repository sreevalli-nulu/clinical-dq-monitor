from pathlib import Path
import requests

BASE = "https://github.com/phuse-org/phuse-scripts/blob/master/data/sdtm/cdiscpilot01/{}.xpt?raw=true"
DOMAINS = ["dm", "sv", "vs", "lb", "ae", "ex", "ds"]
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for d in DOMAINS:
        out = RAW_DIR / f"{d}.xpt"
        if out.exists():
            print(f"skip {d} (already downloaded)")
            continue
        print(f"downloading {d} ...")
        r = requests.get(BASE.format(d), timeout=120)
        r.raise_for_status()
        out.write_bytes(r.content)
        print(f"saved {d}: {len(r.content) / 1024:.0f} KB")

if __name__ == "__main__":
    main()
