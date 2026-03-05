# Design: Bank of Canada Speech Hawkishness Scoring

**Date:** 2026-03-05
**Project:** `2026-tgam-bocspeeches`
**Goal:** Score every Bank of Canada speech since 1995 on a hawkish–dovish spectrum using a pairwise LLM tournament, producing a CSV dataset for use in charts and articles.

---

## Overview

This project adapts the [FedLock methodology](https://jnathan9.github.io/fedlock/) — originally applied to Federal Reserve speeches — to the Bank of Canada corpus. An LLM reads pairs of anonymized speeches and judges which takes a more hawkish position on monetary policy. TrueSkill ratings aggregate these judgments into continuous scores for each speech.

The output is a single scored CSV file. Visualization is out of scope for this phase.

---

## Architecture

Four sequential scripts plus a verification notebook:

| Script | Purpose | Output |
|--------|---------|--------|
| `scrape_speeches.py` | Crawl BoC archive, fetch full text | `data/speeches_raw.json` |
| `build_tournament.py` | Run pairwise TrueSkill tournament | `data/tournament_state.json`, `data/tournament_results.json` |
| `score_speeches.py` | Normalize ratings, apply era-adjustment | `data/speeches_scored.csv` |
| `analysis.ipynb` | Verify scores, inspect distributions | — |

---

## Step 1: Scraping

**Source:** `https://www.bankofcanada.ca/press/speeches/`

The BoC speech archive paginates via the `mt_page` query parameter, going back to 1995. The scraper will:

- Iterate pages until no further results are returned
- Extract from each listing: date, speaker name, title, URL
- Fetch the full text of each individual speech page
- Strip HTML boilerplate: navigation, headers, footers, footnote markers, related links
- Apply a ~1 second delay between requests to be a polite crawller
- Save all raw text to `data/speeches_raw.json` so the tournament can be re-run without re-scraping

**Raw JSON schema (per speech):**
```json
{
  "id": "unique-slug",
  "date": "YYYY-MM-DD",
  "speaker": "Tiff Macklem",
  "title": "Speech title",
  "url": "https://...",
  "text": "Full speech text..."
}
```

---

## Step 2: Anonymization

Before any speech enters a comparison, speaker identity is stripped from the **full text body**, not just the prompt header. This matches the FedLock design choice: without anonymization, the LLM can draw on prior knowledge about known hawks and doves rather than reading the text on its merits.

Anonymization steps applied to each speech text:
1. Replace the speaker's name (and common variants, e.g. first name only) with `[SPEAKER]`
2. Replace positional references that reveal identity — e.g. "Governor Macklem", "Deputy Governor Rogers" — with `[THE SPEAKER]`
3. Leave institutional references ("Bank of Canada", "Governing Council") intact, as these don't identify individuals

In the comparison prompt, the two speeches are labelled `SPEECH A` and `SPEECH B`.

---

## Step 3: Tournament Engine

### Model
Gemini 2.0 Flash via the Gemini API.

### Comparison prompt structure
Each API call presents two anonymized speech excerpts (truncated to a token limit if necessary) and asks:

> "You are evaluating two central bank speeches on monetary policy. Which speech takes a more hawkish position — meaning a stronger preference for tighter monetary policy, higher interest rates, or greater concern about inflation relative to growth risks? Reply with only: A, B, or EQUAL."

### Rating system: TrueSkill
- Each speech starts with μ = 25.0, σ = 8.333 (TrueSkill defaults)
- Each comparison updates both speeches' μ and σ
- `EQUAL` outcomes are treated as draws
- The `trueskill` Python library handles all rating math

### Opponent selection
- At each round, randomly select two speeches from the pool of speeches where σ ≥ 2.0 (not yet converged)
- Prefer pairing speeches from similar time periods (within ±3 years) to improve comparability, but allow cross-era comparisons to establish a global ranking

### Early stopping
- A speech exits the active pool once σ < 2.0
- This threshold matches FedLock's convergence criterion
- Speeches with more extreme scores (very hawkish or very dovish) converge faster; ambiguous speeches near 50 receive more comparisons
- Tournament ends when all speeches have σ < 2.0

### Checkpointing
- Tournament state (all μ, σ values + comparison log) saved to `data/tournament_state.json` every 100 comparisons
- Script can be resumed from checkpoint if interrupted
- Full comparison log preserved for auditability

### Rate limiting
- ~2 second delay between API calls (consistent with existing 2026 project pattern)

---

## Step 4: Scoring & Era-Adjustment

### Raw score normalization
TrueSkill μ values are normalized to a 0–100 scale:
- Min μ → 0, Max μ → 100
- 50 = neutral (median speech)

### Era-adjustment
Raw scores conflate a speaker's dispositional hawkishness with the era they spoke in. A speech urging caution when inflation is 2% is meaningfully hawkish; the same language when inflation is 8% is unremarkable.

Era-adjustment method (matching FedLock):
1. For each speech, compute the mean raw score of all speeches in the same quarter
2. Subtract that quarterly mean from the speech's raw score
3. Re-center to 50: `era_adjusted = 50 + (raw_score - quarterly_mean)`

This isolates dispositional hawkishness from the prevailing policy environment.

### Output CSV columns

| Column | Description |
|--------|-------------|
| `date` | Speech date (YYYY-MM-DD) |
| `speaker` | Speaker name |
| `title` | Speech title |
| `url` | Source URL |
| `trueskill_mu` | Raw TrueSkill mean rating |
| `trueskill_sigma` | Final uncertainty (all < 2.0) |
| `comparisons` | Number of head-to-head matchups |
| `raw_score` | Normalized 0–100 score |
| `era_adjusted_score` | Score relative to quarterly average, re-centered to 50 |

---

## Methodology Notes & Limitations

- **LLM judgment is not ground truth.** Scores reflect Gemini 2.0 Flash's reading of the texts, not a definitive characterization of any official's views or the Bank of Canada's intentions.
- **Anonymization is imperfect.** Some speeches may contain indirect identity signals (references to the speaker's previous statements, distinctive phrasing) that survive anonymization.
- **Era-adjustment is a heuristic.** The quarterly averaging assumes that the distribution of hawkishness within any quarter is centered around the prevailing policy stance. This holds reasonably well across rate cycles but may be noisy in quarters with few speeches.
- **Text quality varies.** Older speeches (pre-2005) may have been OCR'd or formatted differently on the BoC website. The scraper should flag any speeches with suspiciously short text.
- **No macro context.** Unlike FedLock, this version does not provide macroeconomic indicators alongside each comparison. This simplifies the pipeline but means hawkishness is judged purely on textual tone rather than tone relative to conditions.

---

## Out of Scope (Phase 1)

- Interactive visualization
- Automated updates / GitHub Actions
- Macro context in comparisons
- Cross-central-bank comparisons (Fed vs. BoC)
