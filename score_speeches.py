"""
Convert TrueSkill tournament results to normalized hawkishness scores.

Methodology:
1. Raw score: normalize TrueSkill mu to 0-100 scale (min=0, max=100)
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

    A score above 50 means more hawkish than contemporaries.
    A score below 50 means more dovish than contemporaries.
    """
    df = df.copy()
    df["quarter"] = df["date"].dt.to_period("Q")
    quarterly_mean = df.groupby("quarter")[score_col].transform("mean")
    return 50 + (df[score_col] - quarterly_mean)


def main():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"{RESULTS_PATH} not found. Run build_tournament.py first."
        )

    results = json.loads(RESULTS_PATH.read_text())
    df = pd.DataFrame(results)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    df["raw_score"] = normalize_0_100(df["trueskill_mu"])
    df["era_adjusted_score"] = era_adjust(df, "raw_score")

    df["raw_score"] = df["raw_score"].round(2)
    df["era_adjusted_score"] = df["era_adjusted_score"].round(2)
    df["trueskill_mu"] = df["trueskill_mu"].round(4)
    df["trueskill_sigma"] = df["trueskill_sigma"].round(4)

    output = df[[
        "date", "speaker", "title", "url",
        "trueskill_mu", "trueskill_sigma", "comparisons",
        "raw_score", "era_adjusted_score",
    ]].copy()
    output["date"] = output["date"].dt.strftime("%Y-%m-%d")

    output.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(output)} scored speeches to {OUTPUT_PATH}")

    print(f"\nScore summary:")
    print(f"  Raw score    — mean: {df['raw_score'].mean():.1f}, "
          f"std: {df['raw_score'].std():.1f}, "
          f"min: {df['raw_score'].min():.1f}, max: {df['raw_score'].max():.1f}")
    print(f"  Era-adjusted — mean: {df['era_adjusted_score'].mean():.1f}, "
          f"std: {df['era_adjusted_score'].std():.1f}")

    print(f"\nTop 5 most hawkish (era-adjusted):")
    print(output.nlargest(5, "era_adjusted_score")[
        ["date", "speaker", "title", "era_adjusted_score"]
    ].to_string(index=False))

    print(f"\nTop 5 most dovish (era-adjusted):")
    print(output.nsmallest(5, "era_adjusted_score")[
        ["date", "speaker", "title", "era_adjusted_score"]
    ].to_string(index=False))

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
