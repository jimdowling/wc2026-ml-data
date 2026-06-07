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
the user's answers and using the `hops-train` skill. Put **both the feature view
creation and the model training in the same `train.py` file** — a single script
that creates the feature view first and then trains and registers the model
(do not split them into separate scripts):

- Create a feature view by selecting features from the 3 historical feature groups.
- When reading training data as a Pandas DF filter the last N games per team; filter out friendly matches unless the user opted to include them.
- Hold out the most recent N games per team as the test set.
- Train an **XGBoost** classifier (multi-class: Loss / Draw / Win from the listed team's perspective)
- Evaluate on the held-out test set, save some PNGs (like ROC-AUC and confusion matrix) when registering the model + metrics in the Hopsworks model registry.
  - **Write the PNGs into an `images/` subdirectory of the staging dir**
    (i.e. `<staging_dir>/images/*.png`) so Hopsworks picks them up as the
    model's evaluation images and shows them on the model card. PNGs left at
    the top level of the staging dir do **not** show up as evaluation images.
- After the model is registered, **delete the local staging directory** that
  `train.py` created to hold the model file and evaluation images — those
  artifacts now live in the model registry, so remove the local copy (e.g.
  `shutil.rmtree(model_dir)`) before the script exits.

Then **create a streamlit application** using the `hops-apps` skill. It should include probabilities of points won in the group phase and the usual layout for the knockout rounds. Do monte carlo rollouts and get the average. Add a button so the user can do their own monte carlo rollout to see what the result is.
- Each team's route to the final (probability of reaching each knockout round
  and of winning the tournament).
- A **knockout bracket diagram** showing the knockout results for **all teams**
  (see "Knockout bracket visualization" below).
- A **"Re-run simulation"** button to draw a fresh set of rollouts.
Deploy and start the streamlit app on hopsworks.

#### Knockout bracket visualization

The knockout rounds must be drawn as a **classic left/right-hand bracket diagram**
(a matplotlib figure rendered with `st.pyplot`), not just per-team probability
tables — it shows the full path to the final with the knockout results for every
team. Requirements:

- **Layout:** the Round of 32 ties are the leaves, split into two halves — one
  fanning out down the **left** edge, the other down the **right** edge — and
  both halves converge inward through Round of 16 → Quarter-finals →
  Semi-finals to the **FINAL** in the **center** column. Round headers
  ("Round of 32", "Round of 16", "Quarter-finals", "Semi-finals", "FINAL") sit
  across the top.
- **Each tie is a box** listing both teams (one per line); the **winner is
  highlighted** (bold + a distinct colour). Bracket **connector lines** join each
  pair of feeder matches to the match they feed into. Derive the bracket
  structure (which match feeds which, and the column each match sits in) from the
  schedule's knockout slot strings (e.g. "Group A Winner", "Match 74 Winner") —
  do not hard-code teams.
- A **"CHAMPION: <team>"** banner above the final, and the **third-place
  play-off** drawn as a separate box below the final.
- Two viewing modes via a radio/toggle: **"Most-likely (chalk)"** — every group
  seeding and knockout tie resolved to the higher-probability side (draws counted
  as 50/50 penalties) — and **"One random simulation"** — a single Monte-Carlo
  rollout with upsets included, redrawn when the seed changes or the user
  re-runs.

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


## Notes

- The Hopsworks CLI is `hops`. Pass `--json` for machine-readable output.
- Always use **AskUserQuestion** for the decision points in Steps 1, 3, and 4 —
  do not assume the answer.
