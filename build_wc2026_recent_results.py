#!/usr/bin/env python3
"""
Load the raw World Cup 2026 CSV datasets into the Hopsworks feature store and
build the recent-match-results feature group.

Two things happen:

1. Every ``data/*.csv`` file is written to its own offline feature group, named
   after the CSV file (e.g. ``data/elo_ratings.csv`` -> feature group
   ``elo_ratings``).
2. The 20 most recent men's senior international matches for every team
   participating in the FIFA World Cup 2026 are downloaded from
   openfootball/internationals and written to the ``wc2026_recent_results``
   feature group.

Usage:
  python build_wc2026_recent_results.py
  python build_wc2026_recent_results.py --data-dir data --fg-version 1
  python build_wc2026_recent_results.py --skip-recent-results   # CSVs only

Requires:
  pip install requests pandas hopsworks
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

# Per-file feature group configuration. Files not listed here fall back to the
# heuristics in ``infer_fg_config`` (event_time from a "date" column, primary
# key from country/date if present).
FG_CONFIG: dict[str, dict] = {
    "elo_ratings.csv": {
        "primary_key": ["country", "date"],
        "event_time": "date",
        "description": "Pre-match Elo ratings per country over time.",
    },
    "fifa_ratings.csv": {
        "primary_key": ["country", "date"],
        "event_time": "date",
        "description": "FIFA world ranking points per country over time.",
    },
    "FIFA2026_schedule_Fixtures.csv": {
        "primary_key": ["match_number"],
        "event_time": "date_dt",
        "description": "FIFA World Cup 2026 fixture schedule.",
    },
}

# ---------------------------------------------------------------------------
# Recent results configuration
# ---------------------------------------------------------------------------

RECENT_RESULTS_FG = "wc2026_recent_results"
REPO_ZIP = "https://github.com/openfootball/internationals/archive/refs/heads/master.zip"

# Names chosen to match openfootball/internationals team names where possible.
WC_TEAMS = [
    "Mexico", "South Africa", "South Korea", "Czech Republic",
    "Canada", "Bosnia and Herzegovina", "Qatar", "Switzerland",
    "Brazil", "Morocco", "Haiti", "Scotland",
    "United States", "Paraguay", "Australia", "Türkiye",
    "Germany", "Curaçao", "Ivory Coast", "Ecuador",
    "Netherlands", "Japan", "Sweden", "Tunisia",
    "Belgium", "Egypt", "Iran", "New Zealand",
    "Spain", "Cape Verde", "Saudi Arabia", "Uruguay",
    "France", "Senegal", "Iraq", "Norway",
    "Argentina", "Algeria", "Austria", "Jordan",
    "Portugal", "DR Congo", "Uzbekistan", "Colombia",
    "England", "Croatia", "Ghana", "Panama",
]

ALIASES = {
    "USA": "United States",
    "USMNT": "United States",
    "United States of America": "United States",
    "Korea Republic": "South Korea",
    "Republic of Korea": "South Korea",
    "Côte d'Ivoire": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Cabo Verde": "Cape Verde",
    "IR Iran": "Iran",
    "Turkey": "Türkiye",
    "Czechia": "Czech Republic",
    "Democratic Republic of the Congo": "DR Congo",
    "D. R. Congo": "DR Congo",
}

MONTH = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}
DATE_RE = re.compile(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+([A-Z][a-z]{2})\s+(\d{1,2})\s*$")
# Example: "  Argentina              4-1 Brazil                  @ Buenos Aires, Argentina"
MATCH_RE = re.compile(r"^\s+(.+?)\s+(\d+)\s*-\s*(\d+)\s+(.+?)(?:\s+@\s+(.+?))?(?:\s+\[.*\])?\s*$")
TITLE_RE = re.compile(r"^=\s*(.+?)\s*(?:#.*)?$")
YEAR_RE = re.compile(r"(18|19|20)\d{2}")


@dataclass
class LoadResult:
    fg_name: str
    rows: int
    columns: list[str] = field(default_factory=list)


@dataclass
class Match:
    date: datetime
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    tournament: str
    venue: str | None
    source_file: str


# ---------------------------------------------------------------------------
# CSV -> feature group helpers
# ---------------------------------------------------------------------------


def sanitize_fg_name(filename: str) -> str:
    """Turn a CSV filename into a valid (lowercase) feature group name."""
    stem = Path(filename).stem.lower()
    stem = re.sub(r"[^a-z0-9_]+", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_")
    return stem


def infer_fg_config(df: pd.DataFrame, filename: str) -> dict:
    """Look up the configured feature group settings, or infer sensible ones."""
    if filename in FG_CONFIG:
        return FG_CONFIG[filename]

    cols = list(df.columns)
    event_time = "date" if "date" in cols else None
    if {"country", "date"}.issubset(cols):
        primary_key = ["country", "date"]
    else:
        primary_key = [cols[0]] if cols else []
    return {
        "primary_key": primary_key,
        "event_time": event_time,
        "description": f"World Cup 2026 dataset loaded from {filename}.",
    }


def prepare_dataframe(df: pd.DataFrame, event_time: str | None) -> pd.DataFrame:
    """Coerce the event-time column to a proper datetime for the feature store."""
    df = df.copy()
    if event_time and event_time in df.columns:
        df[event_time] = pd.to_datetime(df[event_time], errors="coerce")
    return df


def load_csv_to_feature_group(fs, csv_path: Path, fg_version: int) -> LoadResult:
    df = pd.read_csv(csv_path)
    cfg = infer_fg_config(df, csv_path.name)
    df = prepare_dataframe(df, cfg.get("event_time"))

    fg_name = sanitize_fg_name(csv_path.name)
    print(f"  -> feature group '{fg_name}' (v{fg_version}), "
          f"{len(df):,} rows, columns: {list(df.columns)}", file=sys.stderr)

    fg = fs.get_or_create_feature_group(
        name=fg_name,
        version=fg_version,
        description=cfg.get("description", f"Loaded from {csv_path.name}."),
        primary_key=cfg.get("primary_key") or None,
        event_time=cfg.get("event_time"),
        online_enabled=False,
    )
    fg.insert(df)
    return LoadResult(fg_name=fg_name, rows=len(df), columns=list(df.columns))


# ---------------------------------------------------------------------------
# Recent results: download + parse openfootball/internationals
# ---------------------------------------------------------------------------


def canonical(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip()
    return ALIASES.get(name, name)


def year_from_path(path: str) -> int | None:
    m = YEAR_RE.search(path)
    return int(m.group(0)) if m else None


def clean_tournament(title: str, path: str) -> str:
    title = re.sub(r"\s*#.*$", "", title).strip()
    if title:
        return title
    parent = Path(path).parent.name.replace("_", " ").title()
    return parent


def iter_txt_files_from_zip(zip_bytes: bytes) -> Iterable[tuple[str, str]]:
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            if not info.filename.endswith(".txt"):
                continue
            # skip notes/team indexes, keep tournament/year files
            base = Path(info.filename).name
            if not YEAR_RE.search(base):
                continue
            text = zf.read(info).decode("utf-8", errors="replace")
            yield info.filename, text


def parse_file(path: str, text: str) -> list[Match]:
    y = year_from_path(path)
    if y is None:
        return []
    tournament = Path(path).parent.name.replace("_", " ").title()
    current_date: datetime | None = None
    out: list[Match] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line:
            continue
        mt = TITLE_RE.match(line)
        if mt:
            tournament = clean_tournament(mt.group(1), path)
            continue
        md = DATE_RE.match(line.strip())
        if md:
            _, mon, day = md.groups()
            current_date = datetime(y, MONTH[mon], int(day))
            continue
        if current_date is None:
            continue
        mm = MATCH_RE.match(line)
        if not mm:
            continue
        home, hs, a_s, away, venue = mm.groups()
        # Skip scorer continuation lines that accidentally match badly.
        home = canonical(home)
        away = canonical(away)
        if not home or not away or home.startswith("("):
            continue
        out.append(Match(
            date=current_date,
            home_team=home,
            away_team=away,
            home_score=int(hs),
            away_score=int(a_s),
            tournament=tournament,
            venue=venue,
            source_file=path,
        ))
    return out


def infer_match_type(tournament: str) -> str:
    return "friendly" if "friendly" in tournament.lower() else "competitive"


def build_recent_results(matches: list[Match], teams: list[str], before_date: str | None) -> pd.DataFrame:
    teamset = {canonical(t) for t in teams}
    cutoff = pd.to_datetime(before_date) if before_date else None
    rows = []
    for m in matches:
        if cutoff is not None and pd.Timestamp(m.date.date()) > cutoff:
            continue
        for team in (m.home_team, m.away_team):
            if team not in teamset:
                continue
            is_home = team == m.home_team
            team_score = m.home_score if is_home else m.away_score
            opp_score = m.away_score if is_home else m.home_score
            opp = m.away_team if is_home else m.home_team
            result = "win" if team_score > opp_score else "loss" if team_score < opp_score else "draw"
            rows.append({
                "date": m.date.date().isoformat(),
                "country": team,
                "home_away": "home" if is_home else "away",
                "wc_team_score": team_score,
                "opposition_score": opp_score,
                "result": result,
                "match_type": infer_match_type(m.tournament),
                "opposition_country": opp,
                "home_team": m.home_team,
                "away_team": m.away_team,
                "tournament": m.tournament,
                "source_file": m.source_file,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date_sort"] = pd.to_datetime(df["date"])
    df = df.sort_values(["country", "date_sort"], ascending=[True, False])
    df = df.groupby("country", group_keys=False).head(20)
    df = df.drop(columns=["date_sort"]).reset_index(drop=True)
    return df


def download_recent_results(repo_zip: str, before_date: str | None) -> pd.DataFrame:
    import requests

    print(f"Fetching {repo_zip} ...", file=sys.stderr)
    if repo_zip.startswith("http"):
        r = requests.get(repo_zip, timeout=60)
        r.raise_for_status()
        zip_bytes = r.content
    else:
        zip_bytes = Path(repo_zip).read_bytes()

    matches: list[Match] = []
    for path, text in iter_txt_files_from_zip(zip_bytes):
        matches.extend(parse_file(path, text))
    print(f"Parsed {len(matches):,} matches", file=sys.stderr)

    df = build_recent_results(matches, WC_TEAMS, before_date)
    counts = df.groupby("country").size().sort_values() if not df.empty else pd.Series(dtype=int)
    missing_or_short = counts[counts < 20]
    if not missing_or_short.empty:
        print("WARNING: fewer than 20 matches found for:", file=sys.stderr)
        print(missing_or_short.to_string(), file=sys.stderr)
    return df


def load_recent_results_to_feature_group(fs, df: pd.DataFrame, fg_version: int) -> LoadResult:
    df = prepare_dataframe(df, "date")
    print(f"  -> feature group '{RECENT_RESULTS_FG}' (v{fg_version}), "
          f"{len(df):,} rows, columns: {list(df.columns)}", file=sys.stderr)

    fg = fs.get_or_create_feature_group(
        name=RECENT_RESULTS_FG,
        version=fg_version,
        description=(
            "20 most recent men's senior international matches per WC2026 team, "
            "from openfootball/internationals."
        ),
        primary_key=["country", "date", "opposition_country"],
        event_time="date",
        online_enabled=False,
    )
    fg.insert(df)
    return LoadResult(fg_name=RECENT_RESULTS_FG, rows=len(df), columns=list(df.columns))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="Directory of CSV files to load")
    ap.add_argument("--fg-version", type=int, default=1, help="Hopsworks feature group version")
    ap.add_argument("--before-date", default="2026-06-10",
                    help="Include recent-result matches on or before this date")
    ap.add_argument("--repo-zip", default=REPO_ZIP,
                    help="openfootball/internationals zip URL or local zip path")
    ap.add_argument("--skip-recent-results", action="store_true",
                    help="Only load the data/*.csv feature groups, skip the download")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}/", file=sys.stderr)
        return 1
    print(f"Found {len(csv_files)} CSV file(s) in {data_dir}/", file=sys.stderr)

    # Download the recent results up front so a network failure surfaces before
    # we start writing to the feature store.
    recent_df: pd.DataFrame | None = None
    if not args.skip_recent_results:
        recent_df = download_recent_results(args.repo_zip, args.before_date)

    import hopsworks

    print("Logging in to Hopsworks ...", file=sys.stderr)
    project = hopsworks.login()
    fs = project.get_feature_store()

    results: list[LoadResult] = []
    for csv_path in csv_files:
        print(f"Loading {csv_path} ...", file=sys.stderr)
        results.append(load_csv_to_feature_group(fs, csv_path, args.fg_version))

    if recent_df is not None and not recent_df.empty:
        print(f"Loading recent results into '{RECENT_RESULTS_FG}' ...", file=sys.stderr)
        results.append(load_recent_results_to_feature_group(fs, recent_df, args.fg_version))
    elif recent_df is not None:
        print("WARNING: no recent results parsed; skipping "
              f"'{RECENT_RESULTS_FG}'.", file=sys.stderr)

    print("\nDone. Loaded feature groups:", file=sys.stderr)
    for r in results:
        print(f"  - {r.fg_name}: {r.rows:,} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
