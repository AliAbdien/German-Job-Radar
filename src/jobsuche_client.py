"""Client for the Bundesagentur für Arbeit's public Jobsuche API.

This is the official, free, public API of the German Federal Employment
Agency (https://jobsuche.api.bund.dev/) - real, current German job postings,
legally obtained (no scraping, no ToS violation, published for public reuse).
Requires only a static, publicly-documented client ID header (no API key
signup, no auth flow) - see the API's own docs for the current value.

This module was written and schema-checked against the documented API, but
could not be live-tested from the sandboxed environment it was written in
(no outbound access to arbeitsagentur.de). Run `python -m src.jobsuche_client`
once on a machine with normal internet access to confirm the response shape
still matches before relying on it.
"""
from __future__ import annotations

import time
from typing import Iterator

import requests

BASE_URL = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/jobs"
# Public, documented client ID for the Jobsuche API (not a secret - it's the
# same value every client of this API uses; see jobsuche.api.bund.dev).
CLIENT_ID = "jobboerse-jobsuche"


def search_postings(
    was: str = "AI Engineer",
    wo: str = "Deutschland",
    max_results: int = 100,
    page_size: int = 25,
) -> Iterator[dict]:
    """Yields raw posting-summary dicts from the Jobsuche API's search endpoint.

    `was` = search term (job title / keywords), `wo` = location. The search
    endpoint returns summaries; use `get_posting_detail` for the full text of
    a specific posting (the summary alone usually isn't enough to extract
    salary/skills from).
    """
    headers = {"X-API-Key": CLIENT_ID}
    fetched = 0
    page = 1
    while fetched < max_results:
        resp = requests.get(
            BASE_URL,
            headers=headers,
            params={"was": was, "wo": wo, "size": min(page_size, max_results - fetched), "page": page},
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("stellenangebote", [])
        if not results:
            break
        for item in results:
            yield item
            fetched += 1
            if fetched >= max_results:
                return
        page += 1
        time.sleep(0.2)  # be a polite client of a free public API


def get_posting_detail(refnr: str) -> dict:
    """Fetch the full posting (including free-text description) by reference number."""
    headers = {"X-API-Key": CLIENT_ID}
    resp = requests.get(f"{BASE_URL}/{refnr}", headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    print("Searching for a handful of AI/ML postings in Germany...")
    for i, posting in enumerate(search_postings(was="AI Engineer", max_results=5)):
        print(f"{i+1}. {posting.get('titel')} — {posting.get('arbeitgeber')} ({posting.get('arbeitsort', {}).get('ort')})")
        print(f"   refnr: {posting.get('refnr')}")
