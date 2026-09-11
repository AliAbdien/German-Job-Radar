"""Client for the Bundesagentur für Arbeit's public Jobsuche API.

This is the official, free, public API of the German Federal Employment
Agency (https://jobsuche.api.bund.dev/) - real, current German job postings,
legally obtained (no scraping, no ToS violation, published for public reuse).
Requires only a static, publicly-documented client ID header (no API key
signup, no auth flow) - see the API's own docs for the current value.

Live-tested 2026-09-11: the original version of this module used a wrong
search path (`/pc/v4/jobs`, which doesn't exist - the gateway returns a 403
"No match found for request for url" for any unmatched route, which reads
misleadingly like an auth failure) and a wrong detail path/shape (job details
are fetched from a *separate* `pc/v4/jobdetails/{refnr}` endpoint, where
`refnr` must be base64-encoded first, not the same path as search with the
refnr appended). Both fixed below against the current spec at
https://github.com/bundesAPI/jobsuche-api - still worth a real run to confirm
the response shape once more before trusting it for daily use.
"""
from __future__ import annotations

import base64
import time
from typing import Iterator
from urllib.parse import quote

import requests

SERVICE_BASE = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service"
SEARCH_URL = f"{SERVICE_BASE}/pc/v6/jobs"
DETAIL_URL = f"{SERVICE_BASE}/pc/v4/jobdetails"
# Public, documented client ID for the Jobsuche API (not a secret - it's the
# same value every client of this API uses; see jobsuche.api.bund.dev).
CLIENT_ID = "jobboerse-jobsuche"
# Some responses from this gateway are inconsistent for requests with a
# generic python-requests User-Agent; sending a plain browser-like one is
# cheap insurance and matches what the reference client examples send.
_HEADERS = {"X-API-Key": CLIENT_ID, "User-Agent": "German-Job-Radar/1.0"}


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
    fetched = 0
    page = 1
    while fetched < max_results:
        resp = requests.get(
            SEARCH_URL,
            headers=_HEADERS,
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
    """Fetch the full posting (including free-text description) by reference number.

    The detail endpoint takes the refnr base64-encoded, on its own path - not
    the same shape as the search endpoint (see module docstring for the bug
    this replaced).
    """
    encoded_refnr = quote(base64.b64encode(refnr.encode()).decode(), safe="")
    resp = requests.get(f"{DETAIL_URL}/{encoded_refnr}", headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


if __name__ == "__main__":
    print("Searching for a handful of AI/ML postings in Germany...")
    for i, posting in enumerate(search_postings(was="AI Engineer", max_results=5)):
        print(f"{i+1}. {posting.get('titel')} — {posting.get('arbeitgeber')} ({posting.get('arbeitsort', {}).get('ort')})")
        print(f"   refnr: {posting.get('refnr')}")
