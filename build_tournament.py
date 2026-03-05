"""
Run a pairwise TrueSkill tournament to score BoC speeches on hawkishness.

Methodology:
- Each speech is compared against random opponents using Gemini 2.0 Flash
- Speaker names are stripped from speech text before comparison
- TrueSkill ratings converge; speeches exit the pool once sigma < 2.0
- State is checkpointed every 100 comparisons for resumability

Output:
- data/tournament_state.json  (resumable checkpoint)
- data/tournament_results.json (final ratings per speech)
"""

import json
import os
import random
import re
import string
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
SIGMA_THRESHOLD = 2.0
MAX_TEXT_CHARS = 3000
CHECKPOINT_EVERY = 100
DELAY = 2.0
YEAR_WINDOW = 3


def load_gemini_api_key() -> str:
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key:
        return env_key.strip()
    key_file = Path("gemini_api_key.txt")
    if key_file.exists():
        return key_file.read_text().strip()
    raise ValueError("GEMINI_API_KEY not found. Set env var or create gemini_api_key.txt")


def anonymize(text: str, speaker_name: str) -> str:
    """Strip speaker identity from speech text body."""
    if not text:
        return text
    result = text
    if speaker_name:
        parts = speaker_name.strip().split()
        # Strip punctuation from parts to handle comma-separated multi-speaker strings
        # e.g. "Tiff Macklem, Carolyn Rogers" -> parts include "Macklem," which must be cleaned
        clean_parts = [p.strip(string.punctuation) for p in parts]
        name_variants = [speaker_name] + [p for p in clean_parts if len(p) > 2]
        for variant in name_variants:
            result = re.sub(re.escape(variant), "[SPEAKER]", result, flags=re.IGNORECASE)
    title_pattern = r"\b(Governor|Deputy Governor|Senior Deputy Governor|Chair|President)\s+\[SPEAKER\]"
    result = re.sub(title_pattern, "[THE SPEAKER]", result, flags=re.IGNORECASE)
    title_name_pattern = r"\b(Governor|Deputy Governor|Senior Deputy Governor)\s+[A-Z][a-z]+"
    result = re.sub(title_name_pattern, "[THE SPEAKER]", result)
    return result


def truncate(text: str, max_chars: int = MAX_TEXT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars * 0.8:
        return truncated[:last_period + 1]
    return truncated


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
    """Ask Gemini which speech is more hawkish. Returns 'A', 'B', or 'EQUAL'."""
    text_a = anonymize(truncate(speech_a["text"]), speech_a["speaker"])
    text_b = anonymize(truncate(speech_b["text"]), speech_b["speaker"])
    prompt = COMPARISON_PROMPT.format(speech_a=text_a, speech_b=text_b)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=10),
    )
    result = response.text.strip().upper()
    if result not in ("A", "B", "EQUAL"):
        for token in ("EQUAL", "A", "B"):
            if token in result:
                return token
        return "EQUAL"
    return result


def ratings_to_dict(ratings: dict) -> dict:
    return {k: {"mu": v.mu, "sigma": v.sigma} for k, v in ratings.items()}


def dict_to_ratings(d: dict) -> dict:
    return {k: trueskill.Rating(mu=v["mu"], sigma=v["sigma"]) for k, v in d.items()}


def select_pair(active_ids: list[str], speeches_by_id: dict, year_window: int = YEAR_WINDOW):
    """Select two speeches. Soft preference for similar years."""
    if len(active_ids) < 2:
        return None, None
    a_id = random.choice(active_ids)
    a_year = int(speeches_by_id[a_id]["date"][:4]) if speeches_by_id[a_id].get("date") else 2000
    nearby = [
        sid for sid in active_ids
        if sid != a_id
        and abs(int(speeches_by_id[sid].get("date", "2000")[:4]) - a_year) <= year_window
    ]
    b_id = random.choice(nearby) if nearby else random.choice([s for s in active_ids if s != a_id])
    return a_id, b_id


def main():
    speeches = json.loads(SPEECHES_PATH.read_text())
    speeches = [s for s in speeches if s.get("text_length", 0) >= 200]
    print(f"Running tournament on {len(speeches)} speeches with sufficient text")

    speeches_by_id = {s["id"]: s for s in speeches}
    env = trueskill.TrueSkill(mu=25, sigma=8.333, draw_probability=0.1)

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

            r_a, r_b = ratings[a_id], ratings[b_id]
            if result == "A":
                ratings[a_id], ratings[b_id] = env.rate_1vs1(r_a, r_b)
            elif result == "B":
                ratings[b_id], ratings[a_id] = env.rate_1vs1(r_b, r_a)
            else:
                ratings[a_id], ratings[b_id] = env.rate_1vs1(r_a, r_b, drawn=True)

            comparison_log.append({
                "a_id": a_id, "b_id": b_id, "result": result,
                "a_sigma_after": ratings[a_id].sigma,
                "b_sigma_after": ratings[b_id].sigma,
            })
            total_comparisons += 1
            pbar.update(1)
            active_ids = [sid for sid in active_ids if ratings[sid].sigma >= SIGMA_THRESHOLD]

            pbar.set_postfix(active=len(active_ids), result=result)

            if total_comparisons % CHECKPOINT_EVERY == 0:
                state = {
                    "ratings": ratings_to_dict(ratings),
                    "comparison_log": comparison_log,
                    "total_comparisons": total_comparisons,
                    "active_remaining": len(active_ids),
                }
                tmp = STATE_PATH.with_suffix(".tmp")
                tmp.write_text(json.dumps(state, indent=2))
                os.replace(tmp, STATE_PATH)

            time.sleep(DELAY)

    final_state = {
        "ratings": ratings_to_dict(ratings),
        "comparison_log": comparison_log,
        "total_comparisons": total_comparisons,
        "active_remaining": 0,
    }
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(final_state, indent=2))
    os.replace(tmp, STATE_PATH)

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
