# World Cup 2026 — ML Predictions

A machine-learning pipeline that predicts FIFA World Cup 2026 results. An XGBoost
model is trained on the last competitive matches of every qualified team
(enriched with pre-match Elo and FIFA ratings), then used to run a Monte-Carlo
simulation of the whole tournament — group stage through the final.

The project runs on [Hopsworks](https://www.hopsworks.ai/) and follows the
**Feature / Training / Inference (FTI)** pipeline architecture for MLOps.

## Architecture

```
                ┌─────────────────┐
  raw CSVs ───► │ FEATURE pipeline │ ──► Feature Store (offline feature groups)
                └─────────────────┘            │
                                               ▼
                                     ┌──────────────────┐
                                     │ TRAINING pipeline │ ──► Model Registry
                                     └──────────────────┘   (wc2026_match_result_xgb)
                                               │
                        ┌──────────────────────┴───────────────────────┐
                        ▼                                               ▼
              ┌────────────────────┐                        ┌─────────────────────┐
              │ INFERENCE pipeline │ ──► prediction          │  Streamlit app       │
              │  (batch job)       │     feature groups       │  (Monte-Carlo sim)   │
              └────────────────────┘            │            └─────────────────────┘
                                                ▼
                                       ┌──────────────────┐
                                       │ Superset dashboard│
                                       └──────────────────┘
```

- **Feature store** holds historical match data and pre-match ratings as offline
  feature groups (`wc2026_qualified_teams_matches`,
  `wc2026_qualified_teams_match_ratings`, `wc2026_schedule`).
- **Model registry** holds the trained classifier `wc2026_match_result_xgb`
  (multi-class: Loss / Draw / Win, from the perspective of the listed team).
- **Inference** writes predictions back to the feature store
  (`wc2026_match_predictions`, `wc2026_group_standings`,
  `wc2026_championship_odds`) for consumption by the dashboard.

## Repository layout

```
.
├── data/                       # Raw input CSVs (qualified-team match history + ratings)
├── pipelines/
│   ├── feature/                # Build raw datasets → load into feature groups
│   │   ├── build_wc2026_recent_results.py   # source recent results from openfootball
│   │   └── load_wc2026_feature_groups.py    # write CSVs to offline feature groups
│   ├── training/
│   │   └── train_match_result.py            # train + register the XGBoost model
│   └── inference/
│       └── batch_inference_wc2026.py        # Monte-Carlo rollout → prediction FGs
├── app/
│   └── app.py                  # Streamlit Monte-Carlo tournament simulator
├── dashboard/
│   └── build_wc2026_dashboard.py            # build the Superset dashboard
├── deployment/                 # Hopsworks deployment / orchestration scripts
│   ├── setup_env.py            #   create the app Python environment
│   ├── deploy_app.py           #   create + start the Streamlit app
│   └── deploy_batch_inference_job.py        #   register the batch-inference job
├── requirements/
│   └── inference-requirements.txt           # extra libs for the app environment
└── README.md
```

## Pipelines

### 1. Feature pipeline — `pipelines/feature/`
- `build_wc2026_recent_results.py` — fetches the most recent men's senior
  international matches per qualified team from the openfootball dataset and
  writes a CSV.
- `load_wc2026_feature_groups.py` — reads the prepared CSVs and writes them to
  three offline feature groups.

### 2. Training pipeline — `pipelines/training/train_match_result.py`
Builds a feature view (match history joined with Elo/FIFA ratings on
`(qualified_team, match_id)`), engineers pre-match-only features (no score
leakage), trains an XGBoost multi-class classifier, and registers it along with
evaluation plots and metrics in the model registry.

### 3. Inference pipeline — `pipelines/inference/batch_inference_wc2026.py`
Downloads the model, predicts W/D/L probabilities for every fixture, runs an
N-iteration Monte-Carlo rollout of the tournament, and materialises the results
into the prediction feature groups. Runs headless as a Hopsworks PYTHON job.

### Serving
- `app/app.py` — Streamlit app that renders the live Monte-Carlo simulation
  (group qualification probabilities + knockout bracket).
- `dashboard/build_wc2026_dashboard.py` — builds a published Superset dashboard
  over the prediction feature groups (read via Trino).

## Running on Hopsworks

> **Note:** The scripts authenticate with `hopsworks.login()` and reference
> paths inside the Hopsworks project (e.g. `Users/meb10000/app.py`,
> `/hopsfs/Users/meb10000/...`). Upload the data files and scripts to your
> Hopsworks project's dataset and adjust those path constants to match your
> username before running.

```bash
# 1. Load the feature groups
python pipelines/feature/load_wc2026_feature_groups.py

# 2. Train and register the model
python pipelines/training/train_match_result.py

# 3. (Batch inference) register and run the job
python deployment/deploy_batch_inference_job.py
hops job run wc2026-batch-inference --wait
hops job schedule wc2026-batch-inference "0 0 6 * * ?"   # daily 06:00 UTC

# 4. (App) create the environment, then deploy the Streamlit app
python deployment/setup_env.py
python deployment/deploy_app.py

# 5. (Dashboard) build the Superset dashboard
python dashboard/build_wc2026_dashboard.py
```

## Requirements

- A Hopsworks project and the `hopsworks` Python client (`pip install hopsworks`).
- The app environment additionally needs the libraries in
  `requirements/inference-requirements.txt` (streamlit, xgboost 3.2.0,
  scikit-learn, matplotlib, joblib) — installed via `deployment/setup_env.py`.
