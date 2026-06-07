# World Cup 2026 — Feature Group Loader

`build_wc2026_recent_results.py` loads the raw World Cup 2026 CSV datasets in
`data/` into the [Hopsworks](https://www.hopsworks.ai/) feature store and
downloads the recent match results for every WC2026 team. Each `data/*.csv`
file is written to its own **offline feature group**, named after the CSV file,
and the recent results are written to the `wc2026_recent_results` feature group.

## What gets loaded

| Source | Feature group | Primary key | Event time |
|--------|---------------|-------------|------------|
| `data/elo_ratings.csv` | `elo_ratings` | `country, date` | `date` |
| `data/fifa_ratings.csv` | `fifa_ratings` | `country, date` | `date` |
| `data/FIFA2026_schedule_Fixtures.csv` | `fifa2026_schedule_fixtures` | `match_number` | `date_dt` |
| openfootball/internationals (downloaded) | `wc2026_recent_results` | `country, date, opposition_country` | `date` |

The `wc2026_recent_results` feature group holds the **20 most recent** men's
senior international matches per WC2026 team, downloaded from
[openfootball/internationals](https://github.com/openfootball/internationals)
(48 teams × 20 = 960 rows). Pass `--skip-recent-results` to load only the
`data/*.csv` feature groups.

Any other CSV dropped into `data/` is loaded automatically: the feature group
name is derived from the filename, the event time is taken from a `date` column
if present, and the primary key defaults to `country, date` (or the first
column).

## Prerequisites

- Python 3.10+
- A Hopsworks project and an API key.

Install dependencies (a virtualenv is recommended):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Authenticating with Hopsworks

The program calls `hopsworks.login()`. Provide credentials in one of these ways:

- **Environment variables** (recommended for non-interactive runs):

  ```bash
  export HOPSWORKS_HOST="my-instance.hopsworks.ai"
  export HOPSWORKS_PROJECT="my_project"
  export HOPSWORKS_API_KEY="<your-api-key>"
  ```

- **Interactive login** — run the program and paste your API key when prompted.

## Running

From the project root:

```bash
python build_wc2026_recent_results.py
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | `data` | Directory of CSV files to load |
| `--fg-version` | `1` | Feature group version to create / append to |
| `--before-date` | `2026-06-10` | Only include recent-result matches on or before this date |
| `--repo-zip` | openfootball URL | Source zip URL or local path for recent results |
| `--skip-recent-results` | _off_ | Load only the `data/*.csv` feature groups |

Examples:

```bash
# Load CSVs from a different directory
python build_wc2026_recent_results.py --data-dir ./my_csvs

# Write to version 2 of the feature groups
python build_wc2026_recent_results.py --fg-version 2
```

## Output

For each CSV the program prints the target feature group, row count, and
columns, then inserts the rows. A summary is printed at the end:

```
Found 3 CSV file(s) in data/
Logging in to Hopsworks ...
Loading data/FIFA2026_schedule_Fixtures.csv ...
  -> feature group 'fifa2026_schedule_fixtures' (v1), 104 rows, columns: [...]
Loading data/elo_ratings.csv ...
  -> feature group 'elo_ratings' (v1), 768 rows, columns: [...]
Loading data/fifa_ratings.csv ...
  -> feature group 'fifa_ratings' (v1), 454 rows, columns: [...]

Done. Loaded feature groups:
  - fifa2026_schedule_fixtures: 104 rows
  - elo_ratings: 768 rows
  - fifa_ratings: 454 rows
```

The feature groups are then available in the Hopsworks UI under your project's
Feature Store.
