# Bank of Canada Speech Hawkishness Scoring — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Score every Bank of Canada speech since 1995 on a hawkish–dovish spectrum using a pairwise LLM tournament, producing `data/speeches_scored.csv`.

**Architecture:** Four sequential scripts — scrape → tournament → score — plus a verification notebook. The TrueSkill library handles rating math; Gemini 2.0 Flash judges pairwise comparisons of anonymized speech text.

**Tech Stack:** Python 3.12, uv, requests, beautifulsoup4, trueskill, google-genai, pandas, tqdm, jupyter

---

## Task 0: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `data/` directory

**Step 1: Initialize uv project**

```bash
cd /Users/DMcMillan@globeandmail.com/Documents/Code/2026-tgam-bocspeeches
uv init --no-readme .
echo "3.12" > .python-version
```

**Step 2: Add dependencies**

```bash
uv add requests beautifulsoup4 trueskill google-genai pandas tqdm ipykernel
```

**Step 3: Create data directory and gitignore**

```bash
mkdir -p data
cat > .gitignore << 'EOF'
data/
gemini_api_key.txt
*.pyc
__pycache__/
.ipynb_checkpoints/
EOF
```

**Step 4: Verify environment**

```bash
uv run python --version
# Expected: Python 3.12.x
```

**Step 5: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore
git commit -m "feat: scaffold project with uv and dependencies"
```

---

## Task 1: Speech Scraper

**Files:**
- Create: `scrape_speeches.py`

The BoC speech archive lives at:
`https://www.bankofcanada.ca/press/speeches/?content_type%5B%5D=4&mt_page=N`

Pages go from 1 (newest) up to ~267+ (oldest, 1995). Each page lists ~10 speeches as cards with title, speaker, date, and a link to the full speech page.

**Step 1: Write `scrape_speeches.py`**

```python
"""
Scrape all Bank of Canada speeches from the archive.

Output: data/speeches_raw.json
Each entry: {id, date, speaker, title, url, text}
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
PARAMS_BASE = {"content_type[]": "4"}
OUTPUT_PATH = Path("data/speeches_raw.json")
DELAY = 1.0  # seconds between requests


def get_listing_page(page_num: int, session: requests.Session) -> BeautifulSoup | None:
    """Fetch one page of speech listings. Returns None if no speeches found."""
    params = {**PARAMS_BASE, "mt_page": page_num}
    resp = session.get(LISTING_URL, params=params, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    # Speech cards are in <article> or list items — inspect the actual page structure
    cards = soup.select("article.card") or soup.select(".views-row") or soup.select("li.view-row")
    if not cards:
        # Try a more generic selector for BoC's CMS
        cards = soup.select(".tease-item") or soup.select(".field-items .field-item")
    return soup if cards else None


def parse_listing_page(soup: BeautifulSoup) -> list[dict]:
    """Extract speech metadata from a listing page."""
    speeches = []

    # BoC uses article cards — adapt selectors to actual HTML structure
    cards = (
        soup.select("article.card")
        or soup.select(".tease-item")
        or soup.select("li.view-row")
        or soup.select(".views-row")
    )

    for card in cards:
        # Title and URL
        title_el = card.select_one("h3 a, h2 a, .card-title a, .title a")
        if not title_el:
            continue
        title = title_el.get_text(strip=True)
        url = title_el.get("href", "")
        if url and not url.startswith("http"):
            url = BASE_URL + url

        # Date
        date_el = card.select_one("time, .date, .field-name-post-date")
        date_str = ""
        if date_el:
            date_str = date_el.get("datetime", "") or date_el.get_text(strip=True)
            # Normalize to YYYY-MM-DD
            date_str = normalize_date(date_str)

        # Speaker
        speaker_el = card.select_one(".speakers, .field-name-field-speakers, .author, .byline")
        speaker = speaker_el.get_text(strip=True) if speaker_el else ""

        speeches.append({
            "title": title,
            "url": url,
            "date": date_str,
            "speaker": speaker,
        })

    return speeches


def normalize_date(raw: str) -> str:
    """Convert various date formats to YYYY-MM-DD."""
    import datetime
    raw = raw.strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y"):
        try:
            dt = datetime.datetime.strptime(raw[:len(fmt) + 2], fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw  # return as-is if unparseable


def fetch_speech_text(url: str, session: requests.Session) -> str:
    """Fetch and extract the full text of a speech page."""
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    # Remove navigation, headers, footers, sidebars, footnotes
    for tag in soup.select("nav, header, footer, .sidebar, .breadcrumb, "
                            ".share-tools, .related, .footnotes, script, style"):
        tag.decompose()

    # Main content area — BoC uses article or .page-content
    content = (
        soup.select_one("article .page-content")
        or soup.select_one(".page-content")
        or soup.select_one("article")
        or soup.select_one("main")
    )

    if not content:
        return ""

    # Extract paragraphs, preserving order
    paragraphs = [p.get_text(separator=" ", strip=True) for p in content.find_all("p")]
    text = "\n\n".join(p for p in paragraphs if len(p) > 40)
    return text


def make_id(speech: dict) -> str:
    """Generate a stable slug ID from date + title."""
    slug = re.sub(r"[^a-z0-9]+", "-", speech["title"].lower()).strip("-")[:60]
    return f"{speech['date']}-{slug}"


def main():
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    # Load existing data to allow resuming
    existing = {}
    if OUTPUT_PATH.exists():
        for s in json.loads(OUTPUT_PATH.read_text()):
            existing[s["url"]] = s
        print(f"Loaded {len(existing)} existing speeches from cache")

    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (Globe and Mail data journalism)"

    all_metadata = []

    # Phase 1: Collect all metadata from listing pages
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

    # Phase 2: Fetch full text for each speech
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

        # Save incrementally every 50 speeches
        if len(results) % 50 == 0:
            OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

        time.sleep(DELAY)

    OUTPUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nDone. Saved {len(results)} speeches to {OUTPUT_PATH}")

    # Report on short/empty texts
    short = [s for s in results if s.get("text_length", 0) < 200]
    print(f"Speeches with <200 chars of text (may need manual review): {len(short)}")
    for s in short[:10]:
        print(f"  {s['date']} {s['speaker']}: {s['url']}")


if __name__ == "__main__":
    main()
```

**Step 2: Run scraper on a single page to verify selectors work**

```bash
uv run python -c "
import requests
from bs4 import BeautifulSoup

resp = requests.get('https://www.bankofcanada.ca/press/speeches/?content_type%5B%5D=4&mt_page=1', timeout=30)
soup = BeautifulSoup(resp.content, 'html.parser')
# Print first 3000 chars of body to inspect structure
print(soup.body.prettify()[:3000])
"
```

Inspect the output. If `article.card` doesn't match, adjust the `parse_listing_page()` selectors in `scrape_speeches.py` to match the actual HTML structure before running the full scrape.

**Step 3: Run full scrape**

```bash
uv run python scrape_speeches.py
```

Expected: a progress bar iterating pages, ending with "Saved N speeches to data/speeches_raw.json". Expect 500–1000+ speeches.

**Step 4: Verify output**

```bash
uv run python -c "
import json
from pathlib import Path
speeches = json.loads(Path('data/speeches_raw.json').read_text())
print(f'Total speeches: {len(speeches)}')
print(f'Date range: {speeches[-1][\"date\"]} to {speeches[0][\"date\"]}')
print(f'Avg text length: {sum(s[\"text_length\"] for s in speeches) / len(speeches):.0f} chars')
short = [s for s in speeches if s[\"text_length\"] < 200]
print(f'Short/empty: {len(short)}')
print('Sample:', speeches[0]['speaker'], speeches[0]['date'], speeches[0]['title'][:60])
"
```

**Step 5: Commit**

```bash
git add scrape_speeches.py
git commit -m "feat: add BoC speech scraper"
```

---

## Task 2: Tournament Engine

**Files:**
- Create: `build_tournament.py`

**Step 1: Write `build_tournament.py`**

```python
"""
Run a pairwise TrueSkill tournament to score BoC speeches on hawkishness.

Methodology:
- Each speech is compared against random opponents using Gemini 2.0 Flash
- Speaker names are stripped from speech text before comparison
- TrueSkill ratings converge; speeches exit the pool once sigma < 2.0
- State is checkpointed every 100 comparisons for resumability

Output:
- data/tournament_state.json  (resumable state)
- data/tournament_results.json (final ratings per speech)
"""

import json
import os
import random
import re
import time
from pathlib import Path

import trueskill
from google import genai
from google.genai import types
from tqdm import tqdm

# --- Config ---
SPEECHES_PATH = Path("data/speeches_raw.json")
STATE_PATH = Path("data/tournament_state.json")
RESULTS_PATH = Path("data/tournament_results.json")
SIGMA_THRESHOLD = 2.0       # Stop comparing a speech once uncertainty < this
MAX_TEXT_CHARS = 3000       # Truncate speeches to this length for API efficiency
CHECKPOINT_EVERY = 100      # Save state every N comparisons
DELAY = 2.0                 # Seconds between Gemini API calls
YEAR_WINDOW = 3             # Prefer opponents within ±N years (soft constraint)


# --- Gemini setup ---
def load_gemini_api_key() -> str:
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key.strip()
    key_file = Path("gemini_api_key.txt")
    if key_file.exists():
        return key_file.read_text().strip()
    raise ValueError("GEMINI_API_KEY not found. Set env var or create gemini_api_key.txt")


# --- Anonymization ---
def anonymize(text: str, speaker_name: str) -> str:
    """
    Strip speaker identity from speech text.

    Removes:
    - The speaker's full name and component parts (first/last name)
    - Title+name patterns: 'Governor Smith', 'Deputy Governor Jones'
    - Self-referential 'I' is left intact (too broad to replace)
    """
    if not text:
        return text

    result = text

    # Replace full name and name parts
    if speaker_name:
        parts = speaker_name.strip().split()
        # Full name first (longest match first to avoid partial replacements)
        name_variants = [speaker_name] + parts
        for variant in name_variants:
            if len(variant) > 2:  # skip initials like "J."
                result = re.sub(re.escape(variant), "[SPEAKER]", result, flags=re.IGNORECASE)

    # Replace title+name patterns
    title_pattern = r"\b(Governor|Deputy Governor|Senior Deputy Governor|Chair|President)\s+\[SPEAKER\]"
    result = re.sub(title_pattern, "[THE SPEAKER]", result, flags=re.IGNORECASE)

    # Replace any remaining bare title+capitalized-word that looks like a name reference
    title_name_pattern = r"\b(Governor|Deputy Governor|Senior Deputy Governor)\s+[A-Z][a-z]+"
    result = re.sub(title_name_pattern, "[THE SPEAKER]", result)

    return result


def truncate(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    # Truncate at a sentence boundary near the limit
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:
        return truncated[:last_period + 1]
    return truncated


# --- Gemini comparison ---
COMPARISON_PROMPT = """You are evaluating two central bank speeches on monetary policy.

SPEECH A:
{speech_a}

---

SPEECH B:
{speech_b}

---

Which speech takes a more hawkish position on monetary policy — meaning a stronger preference for tighter monetary policy, higher interest rates, or greater concern about inflation risks relative to growth risks?

Reply with ONLY one of: A, B, or EQUAL"""


def compare_speeches(client, speech_a: dict, speech_b: dict) -> str:
    """
    Ask Gemini which speech is more hawkish.
    Returns 'A', 'B', or 'EQUAL'.
    """
    text_a = anonymize(truncate(speech_a["text"]), speech_a["speaker"])
    text_b = anonymize(truncate(speech_b["text"]), speech_b["speaker"])

    prompt = COMPARISON_PROMPT.format(speech_a=text_a, speech_b=text_b)

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=10,
        ),
    )

    result = response.text.strip().upper()
    if result not in ("A", "B", "EQUAL"):
        # Try to extract from longer response
        for token in ("EQUAL", "A", "B"):
            if token in result:
                return token
        return "EQUAL"  # fallback
    return result


# --- TrueSkill helpers ---
def ratings_to_dict(ratings: dict) -> dict:
    """Serialize TrueSkill Rating objects to plain dicts."""
    return {k: {"mu": v.mu, "sigma": v.sigma} for k, v in ratings.items()}


def dict_to_ratings(d: dict) -> dict:
    """Deserialize plain dicts back to TrueSkill Rating objects."""
    return {k: trueskill.Rating(mu=v["mu"], sigma=v["sigma"]) for k, v in d.items()}


# --- Opponent selection ---
def select_pair(active_ids: list[str], speeches_by_id: dict, year_window: int = YEAR_WINDOW):
    """
    Select two speeches to compare.
    Soft preference for similar years; falls back to random if no nearby match.
    """
    if len(active_ids) < 2:
        return None, None

    a_id = random.choice(active_ids)
    a_year = int(speeches_by_id[a_id]["date"][:4]) if speeches_by_id[a_id]["date"] else 2000

    # Try to find a nearby-year opponent
    nearby = [
        sid for sid in active_ids
        if sid != a_id
        and abs(int(speeches_by_id[sid]["date"][:4] or 2000) - a_year) <= year_window
    ]
    b_id = random.choice(nearby) if nearby else random.choice([s for s in active_ids if s != a_id])
    return a_id, b_id


# --- Main ---
def main():
    speeches = json.loads(SPEECHES_PATH.read_text())

    # Filter out speeches with no text
    speeches = [s for s in speeches if s.get("text_length", 0) >= 200]
    print(f"Running tournament on {len(speeches)} speeches with sufficient text")

    speeches_by_id = {s["id"]: s for s in speeches}

    # Load or initialize TrueSkill ratings
    env = trueskill.TrueSkill(draw_probability=0.1)

    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text())
        ratings = dict_to_ratings(state["ratings"])
        comparison_log = state["comparison_log"]
        total_comparisons = state["total_comparisons"]
        print(f"Resuming from checkpoint: {total_comparisons} comparisons done")
    else:
        ratings = {s["id"]: env.create_rating() for s in speeches}
        comparison_log = []
        total_comparisons = 0

    # Active pool: speeches not yet converged
    active_ids = [sid for sid, r in ratings.items() if r.sigma >= SIGMA_THRESHOLD]
    print(f"Active pool: {len(active_ids)} speeches remaining (sigma >= {SIGMA_THRESHOLD})")

    api_key = load_gemini_api_key()
    client = genai.Client(api_key=api_key)

    with tqdm(desc="Comparisons", total=len(speeches) * 15) as pbar:
        pbar.update(total_comparisons)

        while len(active_ids) >= 2:
            a_id, b_id = select_pair(active_ids, speeches_by_id)
            if not a_id:
                break

            try:
                result = compare_speeches(client, speeches_by_id[a_id], speeches_by_id[b_id])
            except Exception as e:
                print(f"\nAPI error: {e}. Retrying after 10s...")
                time.sleep(10)
                continue

            # Update TrueSkill ratings
            r_a, r_b = ratings[a_id], ratings[b_id]
            if result == "A":
                ratings[a_id], ratings[b_id] = env.rate_1vs1(r_a, r_b)
            elif result == "B":
                ratings[b_id], ratings[a_id] = env.rate_1vs1(r_b, r_a)
            else:  # EQUAL
                ratings[a_id], ratings[b_id] = env.rate_1vs1(r_a, r_b, drawn=True)

            comparison_log.append({
                "a_id": a_id, "b_id": b_id, "result": result,
                "a_sigma_after": ratings[a_id].sigma,
                "b_sigma_after": ratings[b_id].sigma,
            })
            total_comparisons += 1
            pbar.update(1)

            # Remove converged speeches from active pool
            active_ids = [sid for sid in active_ids if ratings[sid].sigma >= SIGMA_THRESHOLD]

            # Checkpoint
            if total_comparisons % CHECKPOINT_EVERY == 0:
                state = {
                    "ratings": ratings_to_dict(ratings),
                    "comparison_log": comparison_log,
                    "total_comparisons": total_comparisons,
                    "active_remaining": len(active_ids),
                }
                STATE_PATH.write_text(json.dumps(state, indent=2))
                pbar.set_postfix(active=len(active_ids), comparisons=total_comparisons)

            time.sleep(DELAY)

    # Save final state
    STATE_PATH.write_text(json.dumps({
        "ratings": ratings_to_dict(ratings),
        "comparison_log": comparison_log,
        "total_comparisons": total_comparisons,
        "active_remaining": 0,
    }, indent=2))

    # Build results
    results = []
    for s in speeches:
        sid = s["id"]
        r = ratings[sid]
        comps = sum(1 for c in comparison_log if c["a_id"] == sid or c["b_id"] == sid)
        results.append({
            "id": sid,
            "date": s["date"],
            "speaker": s["speaker"],
            "title": s["title"],
            "url": s["url"],
            "trueskill_mu": r.mu,
            "trueskill_sigma": r.sigma,
            "comparisons": comps,
        })

    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nTournament complete. {total_comparisons} total comparisons.")
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
```

**Step 2: Test anonymization logic in isolation before running**

```bash
uv run python -c "
import re, sys
sys.path.insert(0, '.')
from build_tournament import anonymize

text = 'Governor Macklem said the Bank of Canada is watching inflation closely. Tiff Macklem noted that rates may rise.'
speaker = 'Tiff Macklem'
print(anonymize(text, speaker))
# Expected: [THE SPEAKER] said the Bank of Canada is watching inflation closely. [SPEAKER] noted that rates may rise.
"
```

**Step 3: Test a single comparison with Gemini**

```bash
uv run python -c "
import json
from pathlib import Path
from build_tournament import load_gemini_api_key, compare_speeches, anonymize
from google import genai

speeches = json.loads(Path('data/speeches_raw.json').read_text())
# Pick two speeches with text
with_text = [s for s in speeches if s.get('text_length', 0) > 500]
a, b = with_text[0], with_text[10]
print(f'Comparing: {a[\"speaker\"]} ({a[\"date\"]}) vs {b[\"speaker\"]} ({b[\"date\"]})')

client = genai.Client(api_key=load_gemini_api_key())
result = compare_speeches(client, a, b)
print(f'Result: {result}')
"
```

Expected: prints "A", "B", or "EQUAL" without error.

**Step 4: Run full tournament** (this will take hours for 500+ speeches — use screen/tmux or run in background)

```bash
uv run python build_tournament.py
```

The tournament checkpoints every 100 comparisons to `data/tournament_state.json`. If interrupted, re-running will resume from the last checkpoint automatically.

**Step 5: Commit**

```bash
git add build_tournament.py
git commit -m "feat: add pairwise TrueSkill tournament engine"
```

---

## Task 3: Scoring & CSV Export

**Files:**
- Create: `score_speeches.py`

**Step 1: Write `score_speeches.py`**

```python
"""
Convert TrueSkill tournament results to normalized hawkishness scores.

Methodology:
1. Raw score: normalize TrueSkill mu to 0-100 scale (min=0, max=100, median~50)
2. Era-adjusted score: subtract quarterly mean, re-center to 50
   This isolates individual disposition from the prevailing policy environment.

Output: data/speeches_scored.csv
"""

import json
from pathlib import Path

import pandas as pd

RESULTS_PATH = Path("data/tournament_results.json")
OUTPUT_PATH = Path("data/speeches_scored.csv")


def normalize_0_100(series: pd.Series) -> pd.Series:
    """Min-max normalize to 0-100."""
    min_val, max_val = series.min(), series.max()
    if max_val == min_val:
        return pd.Series(50.0, index=series.index)
    return (series - min_val) / (max_val - min_val) * 100


def era_adjust(df: pd.DataFrame, score_col: str) -> pd.Series:
    """
    Era-adjust scores by subtracting the quarterly mean and re-centering to 50.

    For each speech:
        era_adjusted = 50 + (raw_score - mean_raw_score_in_same_quarter)

    This separates dispositional hawkishness from the era's overall policy stance.
    A score above 50 means more hawkish than contemporaries; below 50 means more dovish.
    """
    df = df.copy()
    df["quarter"] = df["date"].dt.to_period("Q")
    quarterly_mean = df.groupby("quarter")[score_col].transform("mean")
    return 50 + (df[score_col] - quarterly_mean)


def main():
    results = json.loads(RESULTS_PATH.read_text())
    df = pd.DataFrame(results)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    # Raw score: normalize mu to 0-100
    df["raw_score"] = normalize_0_100(df["trueskill_mu"])

    # Era-adjusted score
    df["era_adjusted_score"] = era_adjust(df, "raw_score")

    # Round for readability
    df["raw_score"] = df["raw_score"].round(2)
    df["era_adjusted_score"] = df["era_adjusted_score"].round(2)
    df["trueskill_mu"] = df["trueskill_mu"].round(4)
    df["trueskill_sigma"] = df["trueskill_sigma"].round(4)

    # Output columns
    output = df[[
        "date", "speaker", "title", "url",
        "trueskill_mu", "trueskill_sigma", "comparisons",
        "raw_score", "era_adjusted_score",
    ]]
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")

    output.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(output)} scored speeches to {OUTPUT_PATH}")

    # Summary stats
    print(f"\nScore summary:")
    print(f"  Raw score    — mean: {df['raw_score'].mean():.1f}, "
          f"std: {df['raw_score'].std():.1f}, "
          f"min: {df['raw_score'].min():.1f}, max: {df['raw_score'].max():.1f}")
    print(f"  Era-adjusted — mean: {df['era_adjusted_score'].mean():.1f}, "
          f"std: {df['era_adjusted_score'].std():.1f}")
    print(f"\nTop 5 most hawkish (era-adjusted):")
    print(output.nlargest(5, "era_adjusted_score")[["date", "speaker", "title", "era_adjusted_score"]].to_string(index=False))
    print(f"\nTop 5 most dovish (era-adjusted):")
    print(output.nsmallest(5, "era_adjusted_score")[["date", "speaker", "title", "era_adjusted_score"]].to_string(index=False))

    # Per-speaker averages
    speaker_avgs = (
        df.groupby("speaker")["era_adjusted_score"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "avg_era_adjusted", "count": "n_speeches"})
        .sort_values("avg_era_adjusted", ascending=False)
    )
    print(f"\nSpeaker averages (min 3 speeches):")
    print(speaker_avgs[speaker_avgs["n_speeches"] >= 3].to_string())


if __name__ == "__main__":
    main()
```

**Step 2: Run scoring**

```bash
uv run python score_speeches.py
```

Expected: prints summary stats with top hawks/doves and per-speaker averages.

**Step 3: Verify output CSV**

```bash
uv run python -c "
import pandas as pd
df = pd.read_csv('data/speeches_scored.csv')
print(df.head())
print(df.dtypes)
print(f'Rows: {len(df)}')
"
```

**Step 4: Commit**

```bash
git add score_speeches.py
git commit -m "feat: add scoring and era-adjustment script"
```

---

## Task 4: Verification Notebook

**Files:**
- Create: `analysis.ipynb`

**Step 1: Create notebook**

```bash
uv run jupyter notebook
```

Create `analysis.ipynb` with cells that:

1. Load `data/speeches_scored.csv`
2. Plot raw_score distribution (histogram)
3. Plot era_adjusted_score over time (line chart colored by score)
4. Show per-speaker box plots of era-adjusted scores
5. Show the 10 most hawkish and 10 most dovish speeches
6. Check that sigma < 2.0 for all speeches (tournament convergence verification)

**Step 2: Run all cells and verify no errors**

```bash
uv run jupyter nbconvert --to notebook --execute analysis.ipynb --output analysis.ipynb
```

**Step 3: Commit**

```bash
git add analysis.ipynb
git commit -m "feat: add verification notebook"
```

---

## Task 5: Methodology Documentation

**Files:**
- Create: `METHODOLOGY.md`

**Step 1: Write `METHODOLOGY.md`**

```markdown
# Methodology: Bank of Canada Speech Hawkishness Scoring

## Overview

This project scores every Bank of Canada speech since 1995 on a hawkish–dovish
spectrum using a pairwise LLM tournament. The approach is adapted from
[FedLock](https://jnathan9.github.io/fedlock/) by Joe Weisenthal, which applied
the same methodology to Federal Reserve speeches.

The scores are not based on keyword counts, dictionary methods, or pre-trained
classifiers. They come from a pairwise tournament: Gemini 2.0 Flash reads two
anonymized speeches side-by-side and decides which takes the more hawkish
position on monetary policy.

## Data Source

Speeches were scraped from the Bank of Canada's public archive:
https://www.bankofcanada.ca/press/speeches/

Coverage spans [START_DATE] to [END_DATE], comprising [N] speeches.

## Anonymization

Before any speech enters a comparison, speaker identity is removed from the
full body text. This matters because the model has been trained on text about
the Bank of Canada and "knows" who the historical hawks and doves are. Without
anonymization, it could let that prior knowledge influence judgments rather than
reading the text on its merits.

Anonymization steps:
1. The speaker's name and name variants are replaced with `[SPEAKER]`
2. Title+name patterns (e.g., "Governor Macklem") are replaced with `[THE SPEAKER]`
3. Institutional references ("Bank of Canada", "Governing Council") are kept,
   as these don't identify individuals

## Pairwise Tournament

Each comparison presents two anonymized speeches (truncated to 3,000 characters
if longer). The prompt asks Gemini 2.0 Flash which speech takes a more hawkish
position on monetary policy — meaning a stronger preference for tighter policy,
higher rates, or greater concern about inflation relative to growth risks.
The model replies with A, B, or EQUAL.

Opponent selection slightly favors speeches from similar time periods (within
±3 years) to improve comparability, but cross-era comparisons are included to
establish a global ranking.

## TrueSkill Rating System

TrueSkill is a Bayesian rating system developed by Microsoft Research that
tracks both a mean rating (μ) and an uncertainty (σ) for each item. Each
comparison updates both speeches' ratings.

- Starting values: μ = 25.0, σ = 8.333 (TrueSkill defaults)
- EQUAL outcomes are treated as draws (draw_probability = 0.1)
- A speech exits the active tournament pool once σ < 2.0 (converged)
- The tournament ends when all speeches have converged

## Scoring

**Raw score:** TrueSkill μ values are normalized to a 0–100 scale using
min-max normalization. A score of 50 is approximately the median speech.
Higher scores indicate more hawkish positions; lower scores indicate more
dovish positions.

**Era-adjusted score:** Raw scores conflate a speaker's dispositional
hawkishness with the era they spoke in. The same language urging caution on
rate cuts means something very different when inflation is at 2% versus 8%.

Era-adjustment:
1. For each speech, compute the mean raw score of all speeches in the same
   calendar quarter
2. Subtract that quarterly mean from the speech's raw score
3. Re-center to 50: `era_adjusted = 50 + (raw_score - quarterly_mean)`

An era-adjusted score above 50 means the speech was more hawkish than its
contemporaries; below 50 means more dovish than contemporaries.

## Output

`data/speeches_scored.csv` contains one row per speech with:
- date, speaker, title, url
- trueskill_mu, trueskill_sigma (raw rating and final uncertainty)
- comparisons (number of head-to-head matchups)
- raw_score (normalized 0–100)
- era_adjusted_score (normalized relative to quarterly peers)

## Limitations

- **LLM judgment is not ground truth.** Scores reflect Gemini 2.0 Flash's
  reading of the texts, not a definitive characterization of any official's
  views or the Bank of Canada's policy intentions.
- **Anonymization is imperfect.** Some speeches may contain indirect identity
  signals (references to previous statements, distinctive phrasing) that
  survive the name-stripping process.
- **Era-adjustment is a heuristic.** It assumes the distribution of hawkishness
  within any quarter centers around the prevailing policy stance. This is
  reasonable across rate cycles but may be noisy in quarters with very few
  speeches.
- **Text quality varies.** Older speeches (pre-2005) may be formatted
  differently or incompletely digitized on the BoC website.
- **No macro context.** Unlike FedLock, this version does not provide
  macroeconomic indicators alongside each comparison. Hawkishness is judged
  on textual tone alone, not tone relative to economic conditions.
```

**Step 2: Fill in [START_DATE], [END_DATE], [N] after scraping is complete**

**Step 3: Commit**

```bash
git add METHODOLOGY.md
git commit -m "docs: add detailed methodology documentation"
```

---

## Running Order Summary

```bash
# 1. Scrape all speeches (~1-2 hours, rate-limited)
uv run python scrape_speeches.py

# 2. Run tournament (many hours — use tmux or screen)
uv run python build_tournament.py

# 3. Score and export CSV
uv run python score_speeches.py

# 4. Verify in notebook
uv run jupyter notebook
```

The tournament can be interrupted and resumed at any time — state is
checkpointed every 100 comparisons to `data/tournament_state.json`.
