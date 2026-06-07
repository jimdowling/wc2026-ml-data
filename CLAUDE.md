# WC2026 ML Data — Claude instructions

This project builds the source data for a FIFA World Cup 2026 match-prediction
model on Hopsworks. When a session starts (or when the user asks to work on this
project), follow the workflow below.

## Workflow

### Step 1 — Ask whether to build the model

Use **AskUserQuestion** to ask: *"Build a WC 2026 prediction model?"* (Yes / No).

- If **No**, stop and wait for further instructions.
- If **Yes**, continue to Step 2.

### Step 2 — Check the 4 feature groups exist

Use the Hopsworks CLI to check whether all four required feature groups already
exist in the feature store:

| # | Source data table | Feature group |
|---|--------------------|---------------|
| 1 | Recent match results | `wc2026_recent_results` |
| 2 | FIFA ratings | `fifa_ratings` |
| 3 | Elo ratings | `elo_ratings` |
| 4 | WC 2026 schedule | `fifa2026_schedule_fixtures` |

Run:

```bash
hops fg list --json
```

(or check each one with `hops fg info <name> --json`).

- If **all four exist**, proceed to build the prediction model.
- If **one or more are missing**, continue to Step 3.

### Step 3 — Ask whether to generate the missing source data tables

Use **AskUserQuestion** to ask whether to generate the four source data tables:
recent match results, FIFA ratings, Elo ratings, and the WC 2026 schedule.

- If **Yes**, generate the data and load it into the feature groups (see
  "Loading data" below), then proceed to Step 4.
- If **No**, stop — the model cannot be built without the feature groups.

If all four feature groups already existed in Step 2, proceed directly to Step 4.

### Step 4 — Ask how to build the training data and model

Use **AskUserQuestion** three times (one question each) to gather the model
configuration:

1. *"Which games should I select as training data?"* — options: **Last 10**,
   **Last 15**, **Last 20** games (per team).
2. *"Should I include friendly matches?"* — options: **Yes**, **No**.
3. *"How many games should I use as test data to evaluate the model?"* —
   options: **1 game**, **2 games**, **3 games** (the most recent N games per
   team are held out for evaluation).

Then **create a training pipeline** that trains the prediction model based on
the user's answers:

- Read the four feature groups from the Hopsworks feature store.
- Select the last N games per team as configured; filter out friendly matches
  unless the user opted to include them.
- Hold out the most recent N games per team as the test set.
- Train an **XGBoost** classifier (multi-class: Loss / Draw / Win from the
  listed team's perspective), enriched with the pre-match Elo and FIFA ratings.
- Evaluate on the held-out test set and register the model + metrics in the
  Hopsworks model registry.

Record the chosen configuration (training window, friendlies, test size) so the
model and the Streamlit app stay consistent.

## Installing dependencies

Always install the requirements with **`uv pip install`** before running any
Python program in this project:

```bash
uv pip install -r requirements.txt
```

## Loading data

`data/*.csv` files are loaded into one offline feature group each by:

```bash
uv pip install -r requirements.txt
python build_wc2026_recent_results.py
```

This creates `elo_ratings`, `fifa_ratings`, and `fifa2026_schedule_fixtures`
from `data/`. See `README.md` for details and options.

## Visualizing results — Streamlit app

`app.py` is a Streamlit app that visualizes the tournament outlook using
**Monte-Carlo rollouts** of the bracket:

- Group-by-group placing probabilities (win group / runner-up / qualify).
- Each team's route to the final (probability of reaching each knockout round
  and of winning the tournament).
- A **"Re-run simulation"** button to draw a fresh set of rollouts.

Run it with:

```bash
uv pip install -r requirements.txt
streamlit run app.py
```

## Notes

- The Hopsworks CLI is `hops`. Pass `--json` for machine-readable output.
- Authenticate with `hops setup` (interactive) or via the
  `HOPSWORKS_HOST` / `HOPSWORKS_PROJECT` / `HOPSWORKS_API_KEY` env vars.
- The prediction model uses **XGBoost** (see `requirements.txt`).
- Always use **AskUserQuestion** for the decision points in Steps 1, 3, and 4 —
  do not assume the answer.
