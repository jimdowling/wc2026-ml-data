#!/usr/bin/env python3
"""Training pipeline: create the feature view, then train + register the model.

Single script (per project convention):
  1. Create feature view ``wc2026_match_result`` = select_all over
     ``wc2026_match_features`` with ``labels=["result"]`` — this FV *is* the
     model's input schema.
  2. Read ``X, y = fv.training_data(...)``; filter per the chosen config; hold out
     the most recent game(s) per team as a test set.
  3. Train an XGBoost 3-way classifier (0=loss, 1=draw, 2=win).
  4. Evaluate, save eval PNGs into ``<staging>/images/``.
  5. Register ``wc2026_match_predictor`` (metrics + plots) with FV provenance.
  6. Delete the local staging dir.

Config (the choices made for this build):
  TRAIN_LAST_N=10  INCLUDE_FRIENDLY=True  N_TEST=1
"""
from __future__ import annotations

import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
import features as F  # noqa: E402

# ---- Config (user choices) ----
TRAIN_LAST_N = 10
INCLUDE_FRIENDLY = True
N_TEST = 1

FV_NAME = "wc2026_match_result"
MODEL_NAME = "wc2026_match_predictor"
MATCH_FEATURES_FG = "wc2026_match_features"


def find_col(cols, suffix, exclude=None):
    """Find a (possibly FV-prefixed) column by exact name or ``_<suffix>`` end."""
    exclude = exclude or []
    exact = [c for c in cols if c == suffix and c not in exclude]
    if exact:
        return exact[0]
    cand = [c for c in cols if c.endswith("_" + suffix) and c not in exclude]
    return cand[0] if cand else None


def main():
    import hopsworks

    project = hopsworks.login()
    fs = project.get_feature_store()

    # 1. Feature view = select_all over the match-features FG, label = result.
    fg = fs.get_feature_group(MATCH_FEATURES_FG, version=1)
    fv = fs.get_feature_view(name=FV_NAME, version=1)  # returns None if missing
    if fv is None:
        query = fg.select_all()
        fv = fs.create_feature_view(
            name=FV_NAME,
            version=1,
            description=(
                "WC2026 match result prediction inputs (from the listed team's "
                "perspective) + label. The model's input schema."
            ),
            query=query,
            labels=["result"],
            training_helper_columns=["match_type"],
        )
        print(f"Created feature view '{FV_NAME}' v1", file=sys.stderr)
    else:
        print(f"Feature view '{FV_NAME}' v1 already exists", file=sys.stderr)

    # 2. Training data (with keys/event-time/helpers so we can split per team).
    X_full, y = fv.training_data(
        primary_keys=True, event_time=True, training_helper_columns=True
    )
    cols = list(X_full.columns)
    print(f"training_data columns: {cols}", file=sys.stderr)

    oppo_col = find_col(cols, "opposition_country")
    team_col = find_col(cols, "country", exclude=[oppo_col])
    date_col = find_col(cols, "date")
    mt_col = find_col(cols, "match_type")
    print(f"detected -> team:{team_col} oppo:{oppo_col} date:{date_col} "
          f"match_type:{mt_col}", file=sys.stderr)

    work = X_full.copy()
    work["__result"] = y["result"].values
    work["__team"] = work[team_col].values
    work["__date"] = pd.to_datetime(work[date_col].values)
    work["__mt"] = work[mt_col].values

    # rank games per team, most recent = rank 0
    work["__rank"] = (
        work.sort_values("__date", ascending=False)
        .groupby("__team")
        .cumcount()
    )

    # select the last-N window per team
    sel = work["__rank"] < TRAIN_LAST_N
    if not INCLUDE_FRIENDLY:
        sel = sel & (work["__mt"] != "friendly")
    window = work[sel].copy()

    test_mask = window["__rank"] < N_TEST          # most recent N per team
    train_df = window[~test_mask].copy()
    test_df = window[test_mask].copy()

    X_train = train_df[F.FEATURE_COLUMNS].astype(float)
    X_test = test_df[F.FEATURE_COLUMNS].astype(float)
    y_train = train_df["__result"].map(F.RESULT_TO_LABEL).astype(int)
    y_test = test_df["__result"].map(F.RESULT_TO_LABEL).astype(int)

    print(f"train rows: {len(X_train)}  test rows: {len(X_test)}", file=sys.stderr)
    print(f"train label dist: {y_train.value_counts().to_dict()}", file=sys.stderr)

    # 3. Train XGBoost 3-way classifier.
    from xgboost import XGBClassifier

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    print(f"model.classes_ = {model.classes_}", file=sys.stderr)

    # 4. Evaluate.
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        roc_auc_score,
        roc_curve,
    )

    proba = model.predict_proba(X_test)
    preds = model.predict(X_test)
    acc = float(accuracy_score(y_test, preds))
    try:
        auc = float(
            roc_auc_score(y_test, proba, multi_class="ovr", average="macro",
                          labels=[0, 1, 2])
        )
    except Exception as e:  # pragma: no cover
        print(f"ROC-AUC failed: {e}", file=sys.stderr)
        auc = float("nan")
    metrics = {"accuracy": acc, "roc_auc": auc}
    print(f"metrics: {metrics}", file=sys.stderr)

    model_dir = os.path.join(os.path.dirname(__file__), "_staging_model")
    images_dir = os.path.join(model_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    labels = ["loss", "draw", "win"]
    ax.set_xticks(range(3)); ax.set_xticklabels(labels)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (acc={acc:.3f})")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(f"{images_dir}/confusion_matrix.png", dpi=120)
    plt.close(fig)

    # One-vs-rest ROC curves
    fig, ax = plt.subplots(figsize=(5, 4.5))
    yt = y_test.values
    for k, name in enumerate(labels):
        yk = (yt == k).astype(int)
        if yk.sum() == 0 or yk.sum() == len(yk):
            continue
        fpr, tpr, _ = roc_curve(yk, proba[:, k])
        ax.plot(fpr, tpr, label=f"{name}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC (OvR, macro-AUC={auc:.3f})"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{images_dir}/roc_curve.png", dpi=120)
    plt.close(fig)

    # Feature importance
    fig, ax = plt.subplots(figsize=(6, 4))
    imp = model.feature_importances_
    order = np.argsort(imp)
    ax.barh([F.FEATURE_COLUMNS[i] for i in order], imp[order], color="#2b8cbe")
    ax.set_title("XGBoost feature importance")
    fig.tight_layout(); fig.savefig(f"{images_dir}/feature_importance.png", dpi=120)
    plt.close(fig)

    # 5. Register the model.
    import joblib

    joblib.dump(model, f"{model_dir}/model.pkl")

    mr = project.get_model_registry()
    hw_model = mr.python.create_model(
        name=MODEL_NAME,
        metrics=metrics,
        description=(
            "XGBoost 3-way classifier (0=loss, 1=draw, 2=win) for WC2026 matches, "
            f"from the listed team's perspective. Trained on the last {TRAIN_LAST_N} "
            f"games/team (friendlies {'included' if INCLUDE_FRIENDLY else 'excluded'}), "
            f"{N_TEST} most-recent game(s)/team held out for test."
        ),
        input_example=X_train.head(1),
        feature_view=fv,
    )
    hw_model.save(model_dir, keep_original_files=True)
    print(f"Registered model '{MODEL_NAME}' v{hw_model.version}", file=sys.stderr)

    # Verify the model is actually retrievable from the registry before we
    # delete the local staging copy (a registration that didn't commit must not
    # leave us with nothing).
    check = mr.get_model(MODEL_NAME, version=hw_model.version)
    if check is None:
        print("ERROR: model not retrievable after save(); leaving staging dir.",
              file=sys.stderr)
        return 1
    print(f"Verified model retrievable: metrics={check.training_metrics}",
          file=sys.stderr)

    # 6. Delete local staging dir (artifacts now live in the registry).
    shutil.rmtree(model_dir, ignore_errors=True)
    print("Removed local staging dir", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
