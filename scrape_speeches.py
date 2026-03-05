"""
Scrape all Bank of Canada speeches from the archive.

Output: data/speeches_raw.json
Each entry: {id, date, speaker, title, url, text, text_length}
"""

import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

BASE_URL = "https://www.bankofcanada.ca"
LISTING_URL = "https://www.bankofcanada.ca/press/speeches/"
OUTPUT_PATH = Path("data/speeches_raw.json")
DELAY = 1.0  # seconds between requests


def get_listing_page(page_num: int, session: requests.Session):
    """Fetch one page of speech listings. Returns soup or None if no speeches found."""
    params = {"content_type[]": "4", "mt_page": page_num}
    resp = session.get(LISTING_URL, params=params, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")
    cards = soup.select("article.media")
    return soup if cards else None


def parse_listing_page(soup: BeautifulSoup) -> list[dict]:
    """Extract speech metadata from a listing page."""
    speeches = []
    for card in soup.select("article.media"):
        title_el = card.select_one("h3.media-heading a")
        if not title_el:
            continue
        title = title_el.get_text(strip=True)
        url = title_el.get("href", "")
        if url and not url.startswith("http"):
            url = BASE_URL + url

        date_el = card.select_one("span.media-date")
        date_str = normalize_date(date_el.get_text(strip=True)) if date_el else ""

        speaker_els = card.select("span.media-authors a")
        speaker = ", ".join(el.get_text(strip=True) for el in speaker_els)

        speeches.append({"title": title, "url": url, "date": date_str, "speaker": speaker})
    return speeches


def normalize_date(raw: str) -> str:
    """Convert 'March 4, 2026' style dates to YYYY-MM-DD."""
    import datetime
    raw = raw.strip()
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw


def fetch_speech_text(url: str, session: requests.Session) -> str:
    """Fetch and extract the full text of a speech page."""
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")
    content = soup.select_one("#main-content")
    if not content:
        return ""
    paragraphs = [p.get_text(separator=" ", strip=True) for p in content.find_all("p")]
    text = "\n\n".join(p for p in paragraphs if len(p) > 40)
    return text


def make_id(speech: dict) -> str:
    """Generate a stable slug ID from date + title."""
    slug = re.sub(r"[^a-z0-9]+", "-", speech["title"].lower()).strip("-")[:60]
    return f"{speech['date']}-{slug}"


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    existing = {}
    if OUTPUT_PATH.exists():
        for s in json.loads(OUTPUT_PATH.read_text()):
            existing[s["url"]] = s
        print(f"Loaded {len(existing)} existing speeches from cache")

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (Globe and Mail data journalism)"

    all_metadata = []
    print("Phase 1: Scraping listing pages...")
    page = 1
    with tqdm(desc="Listing pages") as pbar:
        while True:
            soup = get_listing_page(page, session)
            if soup is None:
                print(f"No more results at page {page}, stopping.")
                break
            page_speeches = parse_listing_page(soup)
            if not page_speeches:
                print(f"No speeches parsed at page {page}, stopping.")
                break
            all_metadata.extend(page_speeches)
            pbar.update(1)
            pbar.set_postfix(total=len(all_metadata), page=page)
            page += 1
            time.sleep(DELAY)

    print(f"Found {len(all_metadata)} speeches across {page - 1} pages")

    print("Phase 2: Fetching speech texts...")
    results = []
    for speech in tqdm(all_metadata, desc="Fetching speeches"):
        if speech["url"] in existing and existing[speech["url"]].get("text"):
            results.append(existing[speech["url"]])
            continue
        try:
            text = fetch_speech_text(speech["url"], session)
            entry = {
                "id": make_id(speech),
                "date": speech["date"],
                "speaker": speech["speaker"],
                "title": speech["title"],
                "url": speech["url"],
                "text": text,
                "text_length": len(text),
            }
            results.append(entry)
        except Exception as e:
            print(f"\nError fetching {speech['url']}: {e}")
            results.append({**speech, "id": make_id(speech), "text": "", "text_length": 0})

        if len(results) % 50 == 0:
            OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        time.sleep(DELAY)

    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nDone. Saved {len(results)} speeches to {OUTPUT_PATH}")

    short = [s for s in results if s.get("text_length", 0) < 200]
    print(f"Speeches with <200 chars of text: {len(short)}")
    for s in short[:10]:
        print(f"  {s['date']} {s['speaker']}: {s['url']}")


if __name__ == "__main__":
    main()
