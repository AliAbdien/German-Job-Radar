"""One-off diagnostic script - NOT part of the project, just for debugging the
live 0-results issue. Prints the raw API response so we can see exactly what
came back, instead of guessing. Safe to delete after use."""
import json

import requests

from src.jobsuche_client import CLIENT_ID, SEARCH_URL, _HEADERS

for label, params in [
    ("was=AI Engineer, no wo", {"was": "AI Engineer", "size": 5, "page": 1}),
    ("was=Data Scientist, no wo", {"was": "Data Scientist", "size": 5, "page": 1}),
    ("was=Softwareentwickler, no wo", {"was": "Softwareentwickler", "size": 5, "page": 1}),
    ("no was, no wo (everything)", {"size": 5, "page": 1}),
]:
    print(f"\n=== {label} ===")
    print(f"URL: {SEARCH_URL}")
    print(f"headers: {_HEADERS}")
    print(f"params: {params}")
    resp = requests.get(SEARCH_URL, headers=_HEADERS, params=params, timeout=15)
    print(f"status: {resp.status_code}")
    print(f"content-type: {resp.headers.get('content-type')}")
    try:
        payload = resp.json()
        print(f"top-level keys: {list(payload.keys())}")
        print(json.dumps(payload, ensure_ascii=False, indent=2)[:2000])
    except Exception as exc:
        print(f"not JSON ({exc}), raw text (first 1000 chars):")
        print(resp.text[:1000])
