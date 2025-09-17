# app/fetch_minutes.py
"""
Fetch recent FOMC minutes (HTML) and convert them to plain text.
- Scrapes the FOMC calendar for minutes links (HTML only).
- Or accept explicit minutes URLs via --urls (comma separated).
- Saves original HTML under data/raw/ and text under data/text/.
- Appends metadata to index/meta.json.
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup
import html2text

BASE = "https://www.federalreserve.gov"
CAL_URL = f"{BASE}/monetarypolicy/fomccalendars.htm"
HEADERS = {"User-Agent": "Mozilla/5.0 (FOMC-Minutes-LLM/0.1)"}

RAW_DIR = Path("data/raw"); RAW_DIR.mkdir(parents=True, exist_ok=True)
TXT_DIR = Path("data/text"); TXT_DIR.mkdir(parents=True, exist_ok=True)
META_PATH = Path("index/meta.json")


def abs_url(u: str) -> str:
    """Return absolute URL for Fed site links."""
    if not u:
        return ""
    if u.startswith("http"):
        return u
    if u.startswith("/"):
        return BASE + u
    return f"{BASE}/{u}"


def find_minutes_urls(limit: int = 6) -> List[str]:
    """
    Scrape the calendar page for minutes HTML links.
    This version is robust: it looks for anchors whose href contains 'fomcminutes'
    and ends with .htm/.html.
    """
    print(f"[info] fetching calendar page: {CAL_URL}")
    r = requests.get(CAL_URL, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    urls: List[str] = []

    # Primary: href contains 'fomcminutes'
    for a in soup.select('a[href*="fomcminutes"]'):
        href = a.get("href") or ""
        if re.search(r"\.htm(l)?$", href):
            urls.append(abs_url(href))

    # Fallback: anchors whose text contains 'Minutes' and link looks like monetarypolicy/*.htm
    if not urls:
        for a in soup.find_all("a"):
            text = (a.get_text() or "").strip().lower()
            href = a.get("href") or ""
            if "minute" in text and re.search(r"/monetarypolicy/.*\.htm(l)?$", href):
                urls.append(abs_url(href))

    # De-duplicate, keep order, then trim to limit
    seen = set()
    uniq = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)

    if not uniq:
        print("[warn] no minutes HTML links found on the calendar page.")
    return uniq[:limit]


def html_to_text(html: str) -> str:
    """Convert HTML to readable Markdown-like plain text."""
    h = html2text.HTML2Text()
    h.ignore_images = True
    h.ignore_links = True
    h.body_width = 0
    txt = h.handle(html)
    txt = re.sub(r"\n{3,}", "\n\n", txt).strip()
    return txt


def fetch_and_save(url: str) -> str:
    """Download a minutes page and save HTML + extracted text. Return text path or ''."""
    print(f"[info] downloading minutes: {url}")
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()

    content_type = (resp.headers.get("Content-Type") or "").lower()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", url).strip("-")[:100]

    # Save raw HTML (or non-HTML binary as .bin)
    if "html" in content_type or url.lower().endswith((".htm", ".html")):
        (RAW_DIR / f"{slug}.html").write_text(resp.text, encoding="utf-8")
        text = html_to_text(resp.text)
        text_path = TXT_DIR / f"{slug}.txt"
        text_path.write_text(text, encoding="utf-8")
        print(f"[ok] saved text -> {text_path}")
        return str(text_path)
    else:
        (RAW_DIR / f"{slug}.bin").write_bytes(resp.content)
        print(f"[warn] non-HTML content saved; no text extracted.")
        return ""


def save_meta(items: List[dict]) -> None:
    """Append metadata to index/meta.json."""
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if META_PATH.exists():
        try:
            existing = json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing.extend(items)
    META_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ok] meta updated -> {META_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch FOMC minutes and save as text.")
    parser.add_argument("--limit", type=int, default=4, help="how many recent minutes to fetch from the calendar")
    parser.add_argument(
        "--urls",
        type=str,
        default="",
        help="comma-separated minutes HTML URLs to fetch (skip calendar scraping)",
    )
    args = parser.parse_args()

    if args.urls:
        targets = [u.strip() for u in args.urls.split(",") if u.strip()]
    else:
        targets = find_minutes_urls(limit=args.limit)

    meta_items = []
    for u in targets:
        try:
            txt_path = fetch_and_save(u)
            if txt_path:
                meta_items.append({"source": u, "text_path": txt_path})
        except Exception as e:
            print(f"[err] {u} -> {e}")

    if meta_items:
        save_meta(meta_items)
    else:
        print("[warn] no text files produced. Try providing --urls with explicit HTML minutes links.")


if __name__ == "__main__":
    main()
