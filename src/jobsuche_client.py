"""Client for the Bundesagentur für Arbeit's public Jobsuche API.

This is the official, free, public API of the German Federal Employment
Agency (https://jobsuche.api.bund.dev/) - real, current German job postings,
legally obtained (no scraping, no ToS violation, published for public reuse).
Requires only a static, publicly-documented client ID header (no API key
signup, no auth flow) - see the API's own docs for the current value.

Live-tested 2026-09-11, two rounds of real bugs found and fixed:

1. The original version used a wrong search path (`/pc/v4/jobs`, which
   doesn't exist - the gateway returns a 403 "No match found for request for
   url" for any unmatched route, which reads misleadingly like an auth
   failure) and a wrong detail path/shape (job details are fetched from a
   *separate* `pc/v4/jobdetails/{refnr}` endpoint, where `refnr` must be
   base64-encoded first, not the same path as search with the refnr
   appended). Fixed against the spec at
   https://github.com/bundesAPI/jobsuche-api.
2. After that fix, a live search returned 0 results with no error. Cause:
   `wo` ("Beschäftigungsort") is a *free-text place search*, not a country
   filter - "Deutschland" doesn't match any place name, so it silently
   filtered out every result instead of erroring. Fixed by making `wo`
   optional and omitting it from the request entirely when unset, which
   searches nationwide.
3. Still 0 results after fix #2 - the actual bug. A raw response dump showed
   the v6 response's results live under `ergebnisliste`, not `stellenangebote`
   (the third-party docs I could reach without live access were simply wrong
   on this key - a reminder that a written spec and a live response are not
   the same source of truth). Also renamed field reads to match the real
   payload: `stellenangebotsTitel` (not `titel`), `firma` (not `arbeitgeber`),
   `stellenlokationen[0].adresse.ort` (not `arbeitsort.ort`), and critically
   `referenznummer` (not `refnr`) - `get_posting_detail` takes this value.
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
    wo: str = "",
    max_results: int = 100,
    page_size: int = 25,
) -> Iterator[dict]:
    """Yields raw posting-summary dicts from the Jobsuche API's search endpoint.

    `was` = search term (job title / keywords). `wo` = free-text place name
    (e.g. "Berlin", "Augsburg") - NOT a country; leave it empty (default) for
    a nationwide search. Passing "Deutschland" or similar silently returns
    zero results instead of erroring, since `wo` only matches actual place
    names - see the module docstring for how that was found.
    The search endpoint returns summaries; use `get_posting_detail` for the
    full text of a specific posting (the summary alone usually isn't enough
    to extract salary/skills from).
    """
    fetched = 0
    page = 1
    while fetched < max_results:
        params = {"was": was, "size": min(page_size, max_results - fetched), "page": page}
        if wo:
            params["wo"] = wo
        resp = requests.get(SEARCH_URL, headers=_HEADERS, params=params, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("ergebnisliste", [])
        if not results:
            break
        for item in results:
            yield item
            fetched += 1
            if fetched >= max_results:
                return
        page += 1
        time.sleep(0.2)  # be a polite client of a free public API


def get_posting_detail(referenznummer: str) -> dict:
    """Fetch the full posting (including free-text description) by reference number.

    `referenznummer` is the value under that key in a search result (NOT a
    key called `refnr` - the search endpoint doesn't use that name; see the
    module docstring). It's base64-encoded before hitting the detail
    endpoint, which is a separate path from search, not the same shape with
    the reference number appended.

    NOTE: this endpoint itself is still unverified live as of the last fix -
    the search endpoint fix was confirmed against real data, but nobody has
    yet confirmed get_posting_detail's response shape against a real
    referenznummer. Worth a live check before trusting its output shape.
    """
    encoded_refnr = quote(base64.b64encode(referenznummer.encode()).decode(), safe="")
    resp = requests.get(f"{DETAIL_URL}/{encoded_refnr}", headers=_HEADERS, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _location(posting: dict) -> str | None:
    """Best-effort city name out of a search result's stellenlokationen list."""
    locations = posting.get("stellenlokationen") or []
    if not locations:
        return None
    return locations[0].get("adresse", {}).get("ort")


if __name__ == "__main__":
    print("Searching for a handful of AI/ML postings in Germany...")
    for i, posting in enumerate(search_postings(was="AI Engineer", max_results=5)):
        print(f"{i+1}. {posting.get('stellenangebotsTitel')} — {posting.get('firma')} ({_location(posting)})")
        print(f"   referenznummer: {posting.get('referenznummer')}")
