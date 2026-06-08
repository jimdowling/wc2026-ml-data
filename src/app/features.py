"""Single source of truth for WC2026 match-prediction features.

Both the feature pipeline (``src/feature/build_match_features.py``) and the
Streamlit app (``src/app/app.py``) import this module, so the feature row for a
historical match (training) and for a hypothetical fixture (serving) are built
by *the same code* — no training/serving skew. The Hopsworks feature view
``wc2026_match_result`` is select_all over the feature group this logic produces,
so the feature view *is* the model's input schema.

It is deliberately dependency-light (only the std-lib + numbers) so it can be
imported inside the deployed Streamlit app environment.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Team-name aliases: any schedule / results spelling -> ratings-table spelling.
# The ratings feature groups (elo_ratings / fifa_ratings) use the right-hand
# spellings; the schedule and recent-results tables use the left-hand ones.
# Names not listed match the ratings spelling directly.
# ---------------------------------------------------------------------------
ALIAS_TO_RATINGS = {
    "USA": "United States",
    "Türkiye": "Turkey",
    "Korea Republic": "South Korea",
    "IR Iran": "Iran",
    "Czechia": "Czech Republic",
    "Côte d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "Congo DR": "DR Congo",
}

# Host nations get group-stage home advantage.
HOSTS = {"USA", "Canada", "Mexico", "United States"}

# Weak-team defaults for any team without a rating (e.g. New Caledonia).
DEFAULT_ELO = 1300.0
DEFAULT_RANK = 210.0
DEFAULT_POINTS = 700.0

# The exact model-input columns, in order. The feature view yields these (minus
# the label) as X; the app must build its rows with these same columns.
FEATURE_COLUMNS = [
    "home",
    "is_friendly",
    "team_elo",
    "opp_elo",
    "elo_diff",
    "team_rank",
    "opp_rank",
    "rank_diff",
    "team_points",
    "opp_points",
    "points_diff",
]

# Label encoding (class order: 0=loss, 1=draw, 2=win, == model.classes_).
RESULT_TO_LABEL = {"loss": 0, "draw": 1, "win": 2}
LABEL_TO_RESULT = {0: "loss", 1: "draw", 2: "win"}


def ratings_key(name):
    """Map a schedule/results team name to its ratings-table spelling."""
    return ALIAS_TO_RATINGS.get(name, name)


def lookup_team(name, ratings):
    """Return (elo, rank, points) for a team, falling back to weak defaults.

    ``ratings`` is a dict: ratings_spelling -> {"elo", "rank", "points"}.
    """
    rec = ratings.get(ratings_key(name))
    if rec is None:
        return DEFAULT_ELO, DEFAULT_RANK, DEFAULT_POINTS
    return (
        float(rec.get("elo", DEFAULT_ELO)),
        float(rec.get("rank", DEFAULT_RANK)),
        float(rec.get("points", DEFAULT_POINTS)),
    )


def build_feature_row(team, opponent, home, is_friendly, ratings):
    """Build the ordered model-input feature dict for ``team`` vs ``opponent``.

    ``home``/``is_friendly`` are 0/1. Ratings are looked up live (via alias) from
    the supplied ``ratings`` dict — never from a frozen snapshot bundled with the
    model. This is the single definition used in both training and serving.
    """
    team_elo, team_rank, team_points = lookup_team(team, ratings)
    opp_elo, opp_rank, opp_points = lookup_team(opponent, ratings)
    return {
        "home": int(home),
        "is_friendly": int(is_friendly),
        "team_elo": team_elo,
        "opp_elo": opp_elo,
        "elo_diff": team_elo - opp_elo,
        "team_rank": team_rank,
        "opp_rank": opp_rank,
        # rank: lower is better, so rank_diff = opp_rank - team_rank is positive
        # when the listed team is the stronger (better-ranked) side.
        "rank_diff": opp_rank - team_rank,
        "team_points": team_points,
        "opp_points": opp_points,
        "points_diff": team_points - opp_points,
    }
