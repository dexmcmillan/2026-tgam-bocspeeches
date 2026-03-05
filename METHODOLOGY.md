# Methodology: Bank of Canada Speech Hawkishness Scoring

## Overview

This project scores every Bank of Canada speech since 1995 on a hawkish–dovish spectrum using a pairwise LLM tournament. The approach is adapted from [FedLock](https://jnathan9.github.io/fedlock/) by Joe Weisenthal, which applied the same methodology to Federal Reserve speeches.

Scores are not based on keyword counts, dictionary methods, or pre-trained classifiers. They come from a pairwise tournament: Gemini 2.0 Flash reads two anonymized speeches side-by-side and decides which takes the more hawkish position on monetary policy.

## Data Source

Speeches were scraped from the Bank of Canada's public archive:
https://www.bankofcanada.ca/press/speeches/

The archive was scraped in full across all paginated listing pages. The scraper collected 853 speeches covering 1995-03-30 to 2026-03-04.

Speeches with fewer than 200 characters of extracted text were excluded from the tournament (typically YouTube-hosted speeches or PDF-only pages where text could not be extracted).

## Anonymization

Before any speech enters a comparison, speaker identity is removed from the full body text. This matches FedLock's design: without anonymization, the LLM can draw on prior knowledge about known hawks and doves rather than reading the text on its merits.

Anonymization steps applied to each speech before it is sent to Gemini:

1. The speaker's full name and each name component (first name, last name) are replaced with `[SPEAKER]`. Components shorter than 3 characters (initials) are skipped to avoid replacing common words.
2. Any occurrence of `[SPEAKER]` immediately following a title (Governor, Deputy Governor, Senior Deputy Governor, Chair, President) is further replaced with `[THE SPEAKER]`.
3. A fallback regex catches any remaining title + capitalized-name patterns not caught by step 1.
4. Institutional references ("Bank of Canada", "Governing Council") are left intact — these identify the institution, not the individual.

## Pairwise Tournament

### Comparison structure

Each comparison presents two anonymized speeches (truncated to 3,000 characters at the nearest sentence boundary). The prompt is:

> "Which speech takes a more hawkish position on monetary policy — meaning a stronger preference for tighter monetary policy, higher interest rates, or greater concern about inflation risks relative to growth risks? Reply with ONLY one of: A, B, or EQUAL"

The model responds with `A`, `B`, or `EQUAL`.

### Opponent selection

At each round, two speeches are drawn from the active pool (all speeches not yet converged). Opponent selection has a soft time-era preference: the algorithm first looks for an opponent within ±3 years of the selected speech. If none is available in the active pool, it falls back to any remaining speech. This improves comparability within rate cycles while still establishing a global ranking across eras.

### TrueSkill rating system

[TrueSkill](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/) is a Bayesian rating system developed by Microsoft Research that tracks both a mean rating (μ) and an uncertainty (σ) for each item.

Settings used:
- Initial μ: 25.0
- Initial σ: 8.333
- Draw probability: 0.1
- `EQUAL` outcomes treated as draws

Each comparison updates both speeches' μ and σ according to the TrueSkill update rules.

### Early stopping (convergence criterion)

A speech exits the active tournament pool once its σ falls below 2.0. This is the same convergence threshold used by FedLock. Speeches with more extreme scores (very hawkish or very dovish) converge faster; speeches near the neutral centre require more comparisons.

The tournament ends when all speeches have σ < 2.0.

### Checkpointing

Tournament state (all ratings + full comparison log) is saved every 100 comparisons using an atomic write (write to `.tmp`, then `os.replace()`). The tournament can be interrupted and resumed at any point.

## Scoring

### Raw score

TrueSkill μ values are normalized to a 0–100 scale using min-max normalization:

```
raw_score = (μ - μ_min) / (μ_max - μ_min) × 100
```

A raw score of 0 is the most dovish speech in the corpus; 100 is the most hawkish.

### Era-adjusted score

Raw scores conflate a speaker's dispositional hawkishness with the era they spoke in. A speech urging caution on rate cuts when inflation is at 2% is meaningfully hawkish; the same language when inflation is at 8% is unremarkable.

Era-adjustment method (matching FedLock):

1. For each speech, compute the mean raw score of all speeches delivered in the same calendar quarter.
2. Subtract that quarterly mean from the speech's raw score.
3. Re-center to 50:

```
era_adjusted = 50 + (raw_score − quarterly_mean)
```

An era-adjusted score above 50 means the speech was more hawkish than its contemporaries in that quarter. Below 50 means more dovish than contemporaries.

Note: Quarters containing only a single speech receive an era-adjusted score of exactly 50.0, since there are no contemporaries to compare against. Because this formula is unbounded, era-adjusted scores can in rare cases fall slightly below 0 or above 100 for extreme outlier speeches within a quarter.

## Output

`data/speeches_scored.csv` contains one row per speech with these columns:

| Column | Description |
|--------|-------------|
| `date` | Speech date (YYYY-MM-DD) |
| `speaker` | Speaker name |
| `title` | Speech title |
| `url` | Source URL on bankofcanada.ca |
| `trueskill_mu` | Raw TrueSkill mean rating (4 decimal places) |
| `trueskill_sigma` | Final uncertainty — all values < 2.0 (4 decimal places) |
| `comparisons` | Number of head-to-head matchups this speech participated in |
| `raw_score` | Normalized 0–100 score (2 decimal places) |
| `era_adjusted_score` | Score relative to quarterly peers, re-centered to 50 (2 decimal places) |

## Limitations

- **LLM judgment is not ground truth.** Scores reflect Gemini 2.0 Flash's reading of the texts, not a definitive characterization of any official's views or the Bank of Canada's policy intentions.

- **Anonymization is imperfect.** Some speeches may contain indirect identity signals — references to the speaker's previous remarks, distinctive phrasing, or context that survived the name-stripping process — that could allow the model to infer who is speaking. Conversely, the title-based fallback regex may *over-anonymize* by replacing references to third-party governors cited within a speech (e.g., a Macklem speech that quotes a Poloz-era decision would have "Governor Poloz" replaced with "[THE SPEAKER]").

- **Era-adjustment is a heuristic.** The quarterly averaging assumes that the distribution of hawkishness within any quarter is centered around the prevailing policy stance. This holds reasonably well across rate cycles but may be noisy in quarters with very few speeches.

- **Text quality varies.** Older speeches (pre-2005) may be formatted differently or have inconsistent digitization on the BoC website. Speeches with fewer than 200 characters of extracted text were excluded.

- **No macro context.** Unlike FedLock, this version does not provide macroeconomic indicators (inflation, unemployment, GDP) alongside each comparison. Hawkishness is judged on textual tone alone, not tone relative to economic conditions at the time of the speech.

- **Truncation.** Speeches longer than 3,000 characters are truncated at the nearest sentence boundary. Very long speeches may have substantive content in later sections that is not considered.

## Reproducibility

The full pipeline can be re-run from scratch:

```bash
uv run python scrape_speeches.py     # ~1–2 hours (rate-limited)
uv run python build_tournament.py    # many hours; checkpoints every 100 comparisons
uv run python score_speeches.py      # seconds
```

The tournament is resumable if interrupted — it reads `data/tournament_state.json` on startup and continues from where it left off.
