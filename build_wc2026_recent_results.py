#!/usr/bin/env python3
"""
Build the 20 most recent men's senior international football matches for every
team participating in the FIFA World Cup 2026.

Output columns:
  date, country, home_away, wc_team_score, opposition_score, result,
  match_type, opposition_country, home_team, away_team, tournament, source_file

Primary data source:
  openfootball/internationals - a mirror of Mart Jürisoo's international
  football results dataset in Football.TXT format, split by tournament/year.

Usage:
  python build_wc2026_recent_results.py --out wc2026_recent_results.csv
  python build_wc2026_recent_results.py --out wc2026_recent_results.xlsx

Requires:
  pip install requests pandas openpyxl
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

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
class Match:
    date: datetime
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    tournament: str
    venue: str | None
    source_file: str


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


def build_rows(matches: list[Match], teams: list[str], before_date: str | None) -> pd.DataFrame:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="wc2026_recent_results.csv", help="Output .csv or .xlsx path")
    ap.add_argument("--before-date", default="2026-06-10", help="Include matches on or before this date")
    ap.add_argument("--repo-zip", default=REPO_ZIP, help="Repository zip URL or local zip path")
    args = ap.parse_args()

    print(f"Fetching {args.repo_zip} ...", file=sys.stderr)
    if args.repo_zip.startswith("http"):
        r = requests.get(args.repo_zip, timeout=60)
        r.raise_for_status()
        zip_bytes = r.content
    else:
        zip_bytes = Path(args.repo_zip).read_bytes()

    matches: list[Match] = []
    for path, text in iter_txt_files_from_zip(zip_bytes):
        matches.extend(parse_file(path, text))
    print(f"Parsed {len(matches):,} matches", file=sys.stderr)

    df = build_rows(matches, WC_TEAMS, args.before_date)
    counts = df.groupby("country").size().sort_values()
    missing_or_short = counts[counts < 20]
    if not missing_or_short.empty:
        print("WARNING: fewer than 20 matches found for:", file=sys.stderr)
        print(missing_or_short.to_string(), file=sys.stderr)

    out = Path(args.out)
    if out.suffix.lower() == ".xlsx":
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="recent_matches")
            pd.DataFrame({"world_cup_team": WC_TEAMS}).to_excel(writer, index=False, sheet_name="teams")
            src = pd.DataFrame({"source": ["https://github.com/openfootball/internationals", REPO_ZIP]})
            src.to_excel(writer, index=False, sheet_name="sources")
    else:
        df.to_csv(out, index=False)
    print(f"Wrote {out} with {len(df):,} rows", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
