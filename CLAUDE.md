# WC2026 ML Data — Claude instructions

This project builds the source data for a FIFA World Cup 2026 match-prediction
model on Hopsworks. When a session starts (or when the user asks to work on this
project), follow the workflow below.

## Fast path (read this first)

The slow parts of this build are **rediscovering the data model** and the
**deploy-debug loop on the Streamlit app**. Avoid both:

1. **Don't re-explore the data.** The schemas, the team-name alias map, the
   playoff slots and the knockout bracket structure are all written down in
   **"Reference data"** at the bottom of this file. Read that section once and
   write `train.py` / `app.py` straight from it — do not spend turns sampling
   feature groups to rediscover columns or country-name spellings.
2. **The 4 feature groups and `train.py` are quick.** The data loader and a
   `python train.py` run take a couple of minutes total.
3. **The app is the long pole.** Its env clone takes ~5 min and a bad deploy
   wastes another cycle, so **write `app.py` to be testable offline and
   self-test it before deploying** (see "Build the Streamlit app fast"). Kick
   off the env clone *in parallel* with writing/ testing the app so the 5-min
   install overlaps your work.
4. **Heed the gotchas** in "Gotchas that cost time" before writing code — each
   one cost a wasted run the first time.

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

- **The feature view contains every model-input feature *plus* the label — it
  *is* the model's input schema** (load hops-fv skill). Do not compute model
  features ad-hoc in `train.py` or the app; engineer them into the feature store
  so the same columns flow through the FV in both training and serving.
  - In the feature pipeline, build a derived **match-features feature group**
    (one row per historical match, from the listed team's perspective) holding
    **all** model inputs and the label. Compute the team's ratings by PIT-joining
    `elo_ratings`/`fifa_ratings` on `country` and the **opponent's** by joining on
    `opposition_country`; apply the alias map (Reference) so e.g. `Türkiye`/`USA`
    resolve. Columns: `home`, `is_friendly`, `team_elo`, `opp_elo`, `elo_diff`,
    `team_rank`, `opp_rank`, `rank_diff`, `team_points`, `opp_points`,
    `points_diff`, label `result`, plus keys/helpers `country`,
    `opposition_country`, `date` for splitting (not model inputs). This set
    worked: accuracy ≈ 0.50, ROC-AUC ≈ 0.65 on the 3-way label.
  - The feature view `select_all`s that FG with `labels=["result"]` (event-time
    keys as `training_helper_columns`). `train.py` then reads
    `X, y = fv.training_data()` where **X already is exactly the model inputs and
    y the label** — no feature engineering in the training script. The model's
    input schema = the FV schema minus the label.
  - For scoring new fixtures the app must produce the **identical** feature row
    through the same feature view (register the rating-lookup + diff logic as
    **on-demand transformations** on the FV so two team names map to the engineered
    features), so there is one definition of the features and no training/serving
    skew.
- When reading training data as a Pandas DF filter the last N games per team; filter out friendly matches unless the user opted to include them.
- Hold out the most recent N games per team as the test set.
- Train an **XGBoost** classifier (multi-class: Loss / Draw / Win from the listed team's perspective)
- **Nothing is bundled with the model** (no `ratings_lookup.json` /
  `feature_names.json` / `class_names.json` sidecars). The **feature view is the
  model's input schema** — feature names, order and types come from it; the class
  order comes from `model.classes_` (`0=loss, 1=draw, 2=win`). The app builds its
  feature rows by going **through the feature view** (its on-demand transforms
  read ratings live from the feature store), never from a frozen snapshot.
- Evaluate on the held-out test set, save some PNGs (like ROC-AUC and confusion matrix) when registering the model + metrics in the Hopsworks model registry.
  - **Write the PNGs into an `images/` subdirectory of the staging dir**
    (i.e. `<staging_dir>/images/*.png`) so Hopsworks picks them up as the
    model's evaluation images and shows them on the model card. PNGs left at
    the top level of the staging dir do **not** show up as evaluation images.
- After the model is registered, **delete the local staging directory** that
  `train.py` created to hold the model file and evaluation images — those
  artifacts now live in the model registry, so remove the local copy (e.g.
  `shutil.rmtree(model_dir)`) before the script exits.

Then **create a streamlit application** using the `hops-app` skill. It should include probabilities of points won in the group phase and the usual layout for the knockout rounds. Do monte carlo rollouts and get the average. Add a button so the user can do their own monte carlo rollout to see what the result is.
- Each team's route to the final (probability of reaching each knockout round
  and of winning the tournament).
- A **knockout bracket diagram** showing the knockout results for **all teams**
  (see "Knockout bracket visualization" below).
- A **"Re-run simulation"** button to draw a fresh set of rollouts.
Deploy and start the streamlit app on hopsworks.

#### Build the Streamlit app fast

The app is the long pole; these steps avoid the slow deploy-debug loop:

1. **Embedded model, features through the feature view.** Download
   `wc2026_match_predictor` (`mr.get_best_model(..., metric="accuracy")`) and
   predict locally, but build feature rows **through the model's feature view**
   (its on-demand transforms read ratings live from the feature store) — not a
   bundled snapshot. The feature view defines the feature names/order/types; the
   class order is `model.classes_` (`0=loss,1=draw,2=win`). Read the schedule
   from `fifa2026_schedule_fixtures` for groups + KO slot strings.
2. **Pre-compute once, sample many.** Build the full ordered-pair probability
   table for all 48 teams (both `home=0/1`) in **one** `predict_proba` call,
   then each Monte-Carlo rollout is pure-Python sampling from that table — 1000+
   rollouts run in seconds. Cache it with `@st.cache_resource`.
3. **Make `app.py` testable offline and self-test BEFORE deploying.** Put all
   Streamlit calls inside `def main()` guarded by `if __name__ == "__main__"`,
   and keep the simulation/bracket functions pure (pass `strength`/`table` in,
   no module globals). Then a tiny `_selftest.py` can `import app`, call
   `load_model()`/`parse_tournament()`/`monte_carlo()`/`draw_bracket()` against
   the live cluster and assert invariants (48 teams, bracket cols
   16/8/4/2/1, `P(Champion)` sums to 1.0, round-reach monotone, bracket renders
   to a PNG). Fix everything here — one render bug found offline saves a full
   deploy cycle.
4. **Custom env (start it early).** The pickled model needs matching libs, so
   clone `python-app-pipeline` into `wc2026-app-env` and install
   `app/app-requirements.txt` pinned to the **training** versions (see
   "Reference data"). The clone takes ~5 min — **kick it off in the background
   first**, then write + self-test the app while it installs.
5. **App layout on FUSE.** Put `app.py`, `app-requirements.txt` and
   `.streamlit/config.toml` (with `fileWatcherType="none"`) in `app/`. Deploy
   with `create_app(app_path="Users/meb10000/wc2026-ml-data/app/app.py",
   environment="wc2026-app-env")` then `app.run(await_serving=True)`.
6. **matplotlib glyphs:** DejaVu Sans (the default) has no 🏆 — use a star `★`
   or plain text in figure labels. Emoji is fine in Streamlit markdown.

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


## Gotchas that cost time (read before coding)

1. **`fs.get_feature_view(...)` returns `None` when missing — it does not raise.**
   Guard with `if fv is None:` before creating; a `try/except` around it never
   fires and you'll call methods on `None`.
2. **Rating FGs must be keyed by `country` only (event_time `date`)** for the
   PIT join to work. With `date` in the primary key the join fails with
   `errorCode 270249 ... join lacks a key which is part of the primary key`.
   The loader (`build_wc2026_recent_results.py`, `FG_CONFIG`) is already fixed to
   do this — **do not** revert `elo_ratings`/`fifa_ratings` to a `[country,date]`
   PK, and if they somehow exist with the old PK, delete and reload those two.
3. **Country names differ between the schedule/results and the ratings tables.**
   Use the alias map (Reference) or both teams' ratings silently fall back to the
   weak default. Missing aliases is a model-quality bug, not an error.
4. **Group fixtures contain unresolved playoff slots** like
   `"Kosovo/Romania/Slovakia/Türkiye"` (6 of them). Resolve each `/`-token to its
   best-rated candidate so every group has 4 concrete teams.
5. **Read training data in-memory with `fv.training_data()`** (returns
   `(X, y)`); it's fine for this small dataset. No need to materialize a training
   dataset with a Spark job.
6. **`uv pip install`** targets the shared `/srv/hops/venv`, which is the same
   package set the cluster uses — so pinning the app env to the versions you see
   locally (`python -c "import xgboost,sklearn,numpy"`) resolves cleanly.

## Reference data (don't rediscover)

**Feature-group schemas** (as stored, all lowercased):

- `wc2026_recent_results` — `date` (ts), `country`, `home_away`
  (`home`/`away`), `wc_team_score`, `opposition_score`, `result`
  (`win`/`draw`/`loss`, from `country`'s view), `match_type`
  (`competitive`/`friendly`), `opposition_country`, `home_team`, `away_team`,
  `tournament`, `source_file`. PK `[country,date,opposition_country]`, event
  time `date`. 960 rows = 48 teams × 20 most-recent matches. `wc_team_score`/
  `opposition_score` are **leaky** — exclude from features.
- `elo_ratings` — `country`, `date` (ts), `elo_rating`. PK `[country]`, event
  time `date`.
- `fifa_ratings` — `country`, `date` (ts), `ranking`, `points`. PK `[country]`,
  event time `date`.
- `fifa2026_schedule_fixtures` — `date` (str), `match_number` (`"Match N"`),
  `teams`, `group` (`"Group A"`… or empty for KO), `stadium`, `date_dt`. PK
  `[match_number]`. 104 matches.

**Team-name alias map** (schedule/results name → ratings name). Apply before any
ratings lookup; unlisted names match directly:

```
USA → United States      Türkiye → Turkey         Korea Republic → South Korea
IR Iran → Iran           Czechia → Czech Republic  Côte d'Ivoire → Ivory Coast
Cabo Verde → Cape Verde  Congo DR → DR Congo
```

(`New Caledonia` has no rating — it falls to the default; harmless.)
Hosts with group-stage home advantage: **USA, Canada, Mexico**.

**Schedule structure:** 12 groups A–L × 4 teams (matches 1–72). Knockout:
**R32 = matches 73–88, R16 = 89–96, QF = 97–100, SF = 101–102, 3rd-place = 103,
FINAL = 104.** Parse the `teams` slot strings (don't hard-code teams):

- `Group X winners` → group X 1st · `Group X runners-up` → 2nd
- `Group A/B/C/D/F third place` → one of the 8 best third-placed teams, from the
  listed allowed groups (assign via a small bipartite match)
- `Winner match N` → winner of match N · `Runner-up match N` → loser of match N
  (used by the 3rd-place play-off, match 103, whose two slots are both
  `Runner-up match …`)

Qualification: top 2 of each group + the **8 best third-placed** teams (of 12)
reach the R32. KO ties: `P(advance) = P(win) + 0.5·P(draw)`.

**App env pins** (`app/app-requirements.txt`, match training versions):
`xgboost==3.2.0`, `scikit-learn==1.8.0`, `numpy==2.4.6`, `joblib==1.5.3`,
`pandas==2.3.3`, `matplotlib==3.10.9`. Base env `python-app-pipeline` →
clone `wc2026-app-env`. Re-check with `python -c "import …; print(__version__)"`
if a future run sees different local versions.

## Notes

- The Hopsworks CLI is `hops`. Pass `--json` for machine-readable output.
- Always use **AskUserQuestion** for the decision points in Steps 1, 3, and 4 —
  do not assume the answer.
- Artifact names this build produced (reuse them): feature view
  `wc2026_match_result`, model `wc2026_match_predictor`, app `wc2026_simulator`,
  app env `wc2026-app-env`.
