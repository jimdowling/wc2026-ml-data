#!/usr/bin/env python3
"""Feature pipeline: build the derived match-features feature group.

One row per historical match, from the listed team's perspective, holding every
model input plus the label. Team ratings are PIT-joined from ``elo_ratings`` /
``fifa_ratings`` on the team's name; the opponent's on the opponent's name
(alias map applied so e.g. Türkiye -> Turkey resolves). The diff features use the
*exact same arithmetic* as ``src/app/features.py`` so training and serving agree.

The feature view ``wc2026_match_result`` (created in train.py) select_all's this
feature group with ``labels=["result"]``, so this FG defines the model's input
schema.

Run:  python src/feature/build_match_features.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

# Import the shared feature definitions (single source of truth).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import features as F  # noqa: E402

MATCH_FEATURES_FG = "wc2026_match_features"


def latest_as_of(matches_keyed, ratings_df, value_cols, prefix):
    """PIT (as-of) join: for each match row attach the rating in effect on the
    match date for the row's ``_key`` country.

    ``matches_keyed`` must have columns ``_key`` (ratings-spelling country) and
    ``date`` (datetime). ``ratings_df`` has ``country``, ``date`` and value
    columns. Returns ``matches_keyed`` with ``<prefix><value_col>`` columns added.
    """
    left = matches_keyed.sort_values("date").reset_index()  # keep original index
    right = ratings_df.rename(columns={"country": "_key"}).sort_values("date")
    merged = pd.merge_asof(
        left,
        right[["_key", "date"] + value_cols].sort_values("date"),
        on="date",
        by="_key",
        direction="backward",
    )
    # Fall back to the earliest available rating when the match predates all
    # snapshots for that country (merge_asof backward returns NaN there).
    earliest = (
        ratings_df.sort_values("date").groupby("country")[value_cols].first()
    )
    for col in value_cols:
        fb = merged["_key"].map(earliest[col])
        merged[col] = merged[col].fillna(fb)
    merged = merged.set_index("index").sort_index()
    return merged[value_cols].rename(columns={c: prefix + c for c in value_cols})


def main():
    import hopsworks

    project = hopsworks.login()
    fs = project.get_feature_store()

    rr = fs.get_feature_group("wc2026_recent_results", version=1).read()
    elo = fs.get_feature_group("elo_ratings", version=1).read()
    fifa = fs.get_feature_group("fifa_ratings", version=1).read()

    rr["date"] = pd.to_datetime(rr["date"])
    elo["date"] = pd.to_datetime(elo["date"])
    fifa["date"] = pd.to_datetime(fifa["date"])

    # Ratings-spelling keys for the team and the opponent.
    rr["_team_key"] = rr["country"].map(F.ratings_key)
    rr["_opp_key"] = rr["opposition_country"].map(F.ratings_key)

    elo_cols = ["elo_rating"]
    fifa_cols = ["ranking", "points"]

    team = rr[["_team_key", "date"]].rename(columns={"_team_key": "_key"})
    opp = rr[["_opp_key", "date"]].rename(columns={"_opp_key": "_key"})

    team_elo = latest_as_of(team, elo, elo_cols, "t_")
    team_fifa = latest_as_of(team, fifa, fifa_cols, "t_")
    opp_elo = latest_as_of(opp, elo, elo_cols, "o_")
    opp_fifa = latest_as_of(opp, fifa, fifa_cols, "o_")

    df = rr.copy()
    df["team_elo"] = team_elo["t_elo_rating"].fillna(F.DEFAULT_ELO)
    df["opp_elo"] = opp_elo["o_elo_rating"].fillna(F.DEFAULT_ELO)
    df["team_rank"] = team_fifa["t_ranking"].fillna(F.DEFAULT_RANK)
    df["opp_rank"] = opp_fifa["o_ranking"].fillna(F.DEFAULT_RANK)
    df["team_points"] = team_fifa["t_points"].fillna(F.DEFAULT_POINTS)
    df["opp_points"] = opp_fifa["o_points"].fillna(F.DEFAULT_POINTS)

    # Diff features — identical arithmetic to F.build_feature_row.
    df["elo_diff"] = df["team_elo"] - df["opp_elo"]
    df["rank_diff"] = df["opp_rank"] - df["team_rank"]
    df["points_diff"] = df["team_points"] - df["opp_points"]

    df["home"] = (df["home_away"] == "home").astype(int)
    df["is_friendly"] = (df["match_type"] == "friendly").astype(int)

    # Keep keys/helpers (country, opposition_country, date) for splitting plus
    # match_type so train.py can optionally drop friendlies.
    out_cols = (
        ["country", "opposition_country", "date", "match_type"]
        + F.FEATURE_COLUMNS
        + ["result"]
    )
    out = df[out_cols].copy()

    # Sanity: no NaNs in model inputs / label.
    assert not out[F.FEATURE_COLUMNS + ["result"]].isna().any().any(), \
        "NaNs in feature columns"

    print(f"Built {len(out):,} match-feature rows; "
          f"result dist:\n{out['result'].value_counts()}", file=sys.stderr)

    fg = fs.get_or_create_feature_group(
        name=MATCH_FEATURES_FG,
        version=1,
        description=(
            "Per-match model features (from the listed team's perspective) with "
            "PIT-joined elo/fifa ratings + label. Source of the wc2026_match_result "
            "feature view / model input schema."
        ),
        primary_key=["country", "opposition_country", "date"],
        event_time="date",
        online_enabled=False,
    )
    fg.insert(out)
    print(f"  -> feature group '{MATCH_FEATURES_FG}' (v1), {len(out):,} rows",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
