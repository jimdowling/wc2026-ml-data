#!/usr/bin/env python3
"""
Load the raw World Cup 2026 CSV datasets into the Hopsworks feature store.

Every ``data/*.csv`` file is written to its own offline feature group, named
after the CSV file (e.g. ``data/elo_ratings.csv`` -> feature group
``elo_ratings``).

Usage:
  python build_wc2026_recent_results.py
  python build_wc2026_recent_results.py --data-dir data --fg-version 1

Requires:
  pip install pandas hopsworks
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

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


@dataclass
class LoadResult:
    fg_name: str
    rows: int
    columns: list[str] = field(default_factory=list)


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="Directory of CSV files to load")
    ap.add_argument("--fg-version", type=int, default=1, help="Hopsworks feature group version")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {data_dir}/", file=sys.stderr)
        return 1
    print(f"Found {len(csv_files)} CSV file(s) in {data_dir}/", file=sys.stderr)

    import hopsworks

    print("Logging in to Hopsworks ...", file=sys.stderr)
    project = hopsworks.login()
    fs = project.get_feature_store()

    results: list[LoadResult] = []
    for csv_path in csv_files:
        print(f"Loading {csv_path} ...", file=sys.stderr)
        results.append(load_csv_to_feature_group(fs, csv_path, args.fg_version))

    print("\nDone. Loaded feature groups:", file=sys.stderr)
    for r in results:
        print(f"  - {r.fg_name}: {r.rows:,} rows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
