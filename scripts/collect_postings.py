"""Pull current real postings from the official Bundesagentur für Arbeit
Jobsuche API for a live evaluation set - NOT for training (see
data/seed_postings.py for why: redistributing third-party posting text in a
public training-data file is a real copyright/ToS problem even for an
official public API's content).

This script only needs to run once in a while, locally, to refresh what
"real-world" evaluation looks like. Nothing it fetches is committed to the
repo by default.

Usage:
    python -m scripts.collect_postings --was "AI Engineer" --n 20 --out data/live_postings.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.jobsuche_client import get_posting_detail, search_postings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--was", default="AI Engineer", help="Search term (job title / keywords).")
    parser.add_argument("--wo", default="Deutschland", help="Location.")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--out", default="data/live_postings.jsonl")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w") as f:
        for summary in search_postings(was=args.was, wo=args.wo, max_results=args.n):
            refnr = summary.get("refnr")
            if not refnr:
                continue
            try:
                detail = get_posting_detail(refnr)
            except Exception as exc:  # noqa: BLE001 - one bad posting shouldn't kill the batch
                print(f"  skipping {refnr}: {exc}")
                continue
            f.write(json.dumps(detail, ensure_ascii=False) + "\n")
            written += 1
            print(f"  [{written}] {summary.get('titel')}")

    print(f"Wrote {written} live postings to {out_path}")


if __name__ == "__main__":
    main()
