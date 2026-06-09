"""Fetch World Football Elo ratings from eloratings.net and merge them into
``data/elo_ratings.csv``.

The site publishes, per calendar year, a tab-separated results file listing
every A-international played that year together with **both** teams' Elo rating
*after* the match (``https://www.eloratings.net/<YEAR>_results.tsv``). It also
publishes a code -> country-name dictionary (``en.teams.tsv``). We turn each
match into two ``(country, date, elo_rating)`` rows (one per side), restrict to
the countries already tracked in ``data/elo_ratings.csv`` and merge the new
``(country, date)`` pairs in (existing rows are never overwritten).

This is how the 2024 (and any other historical year's) Elo coverage gets added,
so the point-in-time join in the feature view can resolve ratings as of older
match dates.

Usage::

    python src/feature/fetch_elo_ratings.py 2024            # one year
    python src/feature/fetch_elo_ratings.py 2022 2023 2024  # several years
    python src/feature/fetch_elo_ratings.py 2024 --all-teams # don't restrict
"""
from __future__ import annotations

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

BASE = "https://www.eloratings.net"
TEAMS_URL = f"{BASE}/en.teams.tsv"
RESULTS_URL = BASE + "/{year}_results.tsv"

CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "elo_ratings.csv"

# eloratings canonical name -> the spelling used in data/elo_ratings.csv.
# Everything not listed here matches the eloratings name directly.
ELO_TO_CSV = {
    "China": "China PR",
    "Czechia": "Czech Republic",
    "Ireland": "Republic of Ireland",
}


def _get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def load_code_to_csv_name() -> dict[str, str]:
    """code -> CSV country name, using the eloratings team dictionary."""
    text = _get(TEAMS_URL)
    code_to_name: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        code, elo_name = parts[0], parts[1]
        code_to_name[code] = ELO_TO_CSV.get(elo_name, elo_name)
    return code_to_name


def fetch_year(year: int, code_to_name: dict[str, str]) -> dict[tuple[str, str], int]:
    """Return {(country, 'YYYY-MM-DD'): elo} for every side of every match in
    ``year``. Later matches on the same day overwrite earlier ones, so each team
    keeps its most recent rating for that date."""
    text = _get(RESULTS_URL.format(year=year))
    out: dict[tuple[str, str], int] = {}
    for line in text.splitlines():
        c = line.split("\t")
        if len(c) < 12:
            continue
        date = f"{c[0]}-{c[1]}-{c[2]}"
        for code, elo_raw in ((c[3], c[10]), (c[4], c[11])):  # home, away
            name = code_to_name.get(code)
            if name is None:
                continue
            try:
                elo = int(elo_raw)
            except ValueError:
                continue
            out[(name, date)] = elo
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("years", nargs="+", type=int, help="calendar year(s) to fetch")
    ap.add_argument(
        "--all-teams",
        action="store_true",
        help="add every country found, not just those already in the CSV",
    )
    args = ap.parse_args()

    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r]
    tracked = {row[0] for row in rows}
    existing_keys = {(row[0], row[1]) for row in rows}

    code_to_name = load_code_to_csv_name()

    fetched: dict[tuple[str, str], int] = {}
    for year in args.years:
        year_data = fetch_year(year, code_to_name)
        print(f"  {year}: {len(year_data)} (country, date) Elo points", file=sys.stderr)
        fetched.update(year_data)

    added = 0
    seen_countries: set[str] = set()
    for (country, date), elo in sorted(fetched.items()):
        if not args.all_teams and country not in tracked:
            continue
        if (country, date) in existing_keys:
            continue
        rows.append([country, date, str(elo)])
        existing_keys.add((country, date))
        seen_countries.add(country)
        added += 1

    rows.sort(key=lambda x: (x[0], x[1]))
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"Added {added} new Elo rows across {len(seen_countries)} countries.")
    if not args.all_teams:
        missing = sorted(c for c in tracked if c not in seen_countries)
        if missing:
            print(
                f"Tracked countries with no new rows for {args.years}: {missing}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
