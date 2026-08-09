"""
Individual-level diabetes risk model.

Two models, on purpose
----------------------
A survey-weighted logistic regression is the estimator an epidemiology reviewer
expects, and its coefficients are directly interpretable as adjusted odds
ratios. A histogram gradient boosting classifier will usually beat it on
discrimination because it picks up interactions the additive model cannot.
Shipping only the second would be a machine-learning answer to a public-health
question; shipping only the first would leave accuracy on the table. We fit
both, report both, and let the app switch between them so a user can see
whether a conclusion depends on the model class. It mostly does not, which is
itself worth showing.

Every predictor in this dataset is categorical, so there is no scaling or
imputation to do. HistGradientBoosting handles categories natively via
`categorical_features`, which avoids the dimensionality blow-up of one-hot on
a tree.

Calibration
-----------
The intervention simulator reads predicted probabilities as if they were
prevalences, and averages them. That is only legitimate if the model is
calibrated, so we compute a reliability curve and the Brier score, and both are
shown on the Methods page. An uncalibrated model would make the simulator
confidently wrong rather than usefully uncertain.

Sample weights are passed to `fit` for both models. Without them the model
learns the sample, not the population.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from .config import FEATURES, OUTCOME_COL, OUTCOME_POSITIVE, WEIGHT_COL

RANDOM_STATE = 20260224


@dataclass
class TrainedModels:
    """Everything the app needs at runtime, with nothing it does not."""

    gbm: Pipeline
    logit: Pipeline
    features: list[str]
    background: pd.DataFrame          # national reference sample for Shapley
    background_weights: np.ndarray
    metrics: dict = field(default_factory=dict)

    def predict(self, X: pd.DataFrame, which: str = "gbm") -> np.ndarray:
        model = self.gbm if which == "gbm" else self.logit
        return model.predict_proba(X[self.features])[:, 1]


def _gbm_pipeline(features: list[str]) -> Pipeline:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                         encoded_missing_value=-1)
    return Pipeline(
        [
            ("encode", ColumnTransformer([("cat", enc, features)],
                                         remainder="drop")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    categorical_features=list(range(len(features))),
                    max_iter=260,
                    learning_rate=0.06,
                    max_leaf_nodes=24,
                    min_samples_leaf=40,
                    l2_regularization=1.0,
                    early_stopping=True,
                    validation_fraction=0.15,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _logit_pipeline(features: list[str]) -> Pipeline:
    return Pipeline(
        [
            (
                "encode",
                ColumnTransformer(
                    [("cat", OneHotEncoder(drop="first", handle_unknown="ignore"),
                      features)],
                    remainder="drop",
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, C=1.0)),
        ]
    )


def _reliability(y: np.ndarray, p: np.ndarray, w: np.ndarray, bins: int = 10):
    """Weighted reliability curve. Returns (mean predicted, observed, weight)."""
    edges = np.linspace(0, max(p.max(), 1e-6), bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if not m.any():
            continue
        ww = w[m]
        out.append(
            {
                "predicted": float(np.average(p[m], weights=ww)),
                "observed": float(np.average(y[m], weights=ww)),
                "weight": float(ww.sum()),
                "n": int(m.sum()),
            }
        )
    return out


def train(df: pd.DataFrame, features: list[str] | None = None,
          test_size: float = 0.25, background_n: int = 600) -> TrainedModels:
    features = features or FEATURES
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"training frame is missing features: {missing}")

    X = df[features].copy()
    y = (df[OUTCOME_COL] == OUTCOME_POSITIVE).astype(int).to_numpy()
    w = df[WEIGHT_COL].to_numpy(dtype=float)

    X_tr, X_te, y_tr, y_te, w_tr, w_te = train_test_split(
        X, y, w, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    gbm = _gbm_pipeline(features).fit(X_tr, y_tr, clf__sample_weight=w_tr)
    logit = _logit_pipeline(features).fit(X_tr, y_tr, clf__sample_weight=w_tr)

    metrics = {"n_train": int(len(X_tr)), "n_test": int(len(X_te)),
               "prevalence_test": float(np.average(y_te, weights=w_te))}
    for name, model in (("gbm", gbm), ("logit", logit)):
        p = model.predict_proba(X_te)[:, 1]
        metrics[name] = {
            "auc": float(roc_auc_score(y_te, p, sample_weight=w_te)),
            "brier": float(brier_score_loss(y_te, p, sample_weight=w_te)),
            "reliability": _reliability(y_te, p, w_te),
        }

    # Background sample for the Shapley reference distribution. Drawn with
    # probability proportional to survey weight so it represents the
    # population, not the sample.
    rng = np.random.default_rng(RANDOM_STATE)
    n_bg = min(background_n, len(X_tr))
    probs = w_tr / w_tr.sum()
    take = rng.choice(len(X_tr), size=n_bg, replace=False, p=probs)

    return TrainedModels(
        gbm=gbm,
        logit=logit,
        features=list(features),
        background=X_tr.iloc[take].reset_index(drop=True),
        background_weights=np.ones(n_bg),
        metrics=metrics,
    )


def odds_ratios(models: TrainedModels) -> pd.DataFrame:
    """
    Adjusted odds ratios from the logistic model, one row per non-reference
    category. This is the table an epidemiology reader will look for first.
    """
    ohe = models.logit.named_steps["encode"].named_transformers_["cat"]
    names = ohe.get_feature_names_out(models.features)
    coefs = models.logit.named_steps["clf"].coef_[0]
    rows = []
    for name, c in zip(names, coefs):
        feature, _, level = name.partition("_")
        rows.append({"feature": feature, "level": level,
                     "odds_ratio": float(np.exp(c)), "log_odds": float(c)})
    return pd.DataFrame(rows).sort_values("odds_ratio", ascending=False)


def model_card(models: TrainedModels) -> str:
    return json.dumps(
        {
            "features": models.features,
            "metrics": models.metrics,
            "note": "Cross-sectional BDHS data. Predictions are associational.",
        },
        indent=2,
    )
