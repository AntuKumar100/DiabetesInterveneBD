"""
Exact Shapley attribution — no external explainer library.

Why not just import shap
------------------------
Two reasons, one practical and one substantive.

Practical: KernelSHAP approximates, TreeSHAP is exact but tied to a tree, and
both give you per-row attributions. The question this app asks is not per-row.
It is: *why is this district's risk different from the national average?* That
is a comparison of two distributions, not two rows, and it is the question a
targeting exercise actually needs answered.

Substantive: with ten features, the full coalition lattice is 2^10 = 1024
subsets. That is small enough to enumerate exactly, so there is no reason to
approximate anything. What follows is the Shapley value computed by definition.

The value function
------------------
For a coalition S of features, define

    v(S) = E[ f(x) ]  where features in S are drawn from the district's joint
                      distribution and features outside S from the national
                      distribution.

This is the interventional (marginal) formulation — Janzing, Minorics &
Blöbaum (2020), "Feature relevance quantification in explainable AI: a causal
problem", AISTATS — chosen over the conditional formulation because we want to
attribute the effect of *changing* a district's composition, and the
conditional version leaks credit to correlated features that were never
touched.

Then the Shapley value of feature i is the usual weighted average of its
marginal contributions across all coalitions:

    phi_i = sum_{S subset N\\{i}} |S|!(n-|S|-1)!/n! * [v(S u {i}) - v(S)]

By efficiency, sum_i phi_i = v(N) - v(empty) = district mean risk - national
mean risk. The app asserts this identity at runtime and shows the residual, so
a reader can verify the decomposition is complete rather than taking it on
faith.

Cost: 1024 coalitions x B background rows predictions, done in one batched
call. At B=300 that is ~300k rows, well under a second, and it is cached.
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd


def _shapley_weights(n: int) -> dict[int, float]:
    """Weight attached to a coalition of size s when adding one more feature."""
    return {s: factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)}


def district_shapley(
    models,
    national: pd.DataFrame,
    district_rows: pd.DataFrame,
    which: str = "gbm",
    n_background: int = 300,
    seed: int = 20260224,
) -> pd.DataFrame:
    """
    Decompose (district mean risk - national mean risk) across features.

    Returns one row per feature with its Shapley contribution in percentage
    points, plus an attached check of the efficiency identity.
    """
    features = list(models.features)
    n = len(features)
    if n > 14:
        raise ValueError(
            f"exact enumeration is 2^{n} coalitions; use a sampling estimator above 14"
        )
    if len(district_rows) == 0:
        raise ValueError("district has no rows")

    rng = np.random.default_rng(seed)
    B = min(n_background, len(national))
    base = national[features].iloc[
        rng.choice(len(national), size=B, replace=False)
    ].reset_index(drop=True)

    # Draw the district's marginal values with replacement so the two frames
    # align row-for-row; swapping a subset of columns then produces a genuine
    # interventional mixture rather than a reshuffle of real people.
    dist = district_rows[features].iloc[
        rng.integers(0, len(district_rows), size=B)
    ].reset_index(drop=True)

    subsets = [frozenset(c) for r in range(n + 1)
               for c in combinations(range(n), r)]

    # Build every coalition's counterfactual frame in one stack, predict once.
    frames = []
    for S in subsets:
        f = base.copy()
        for i in S:
            f[features[i]] = dist[features[i]].to_numpy()
        frames.append(f)
    stacked = pd.concat(frames, ignore_index=True)
    preds = models.predict(stacked, which=which)

    v = {S: float(preds[k * B:(k + 1) * B].mean())
         for k, S in enumerate(subsets)}

    w = _shapley_weights(n)
    phi = np.zeros(n)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        for r in range(n):
            for combo in combinations(others, r):
                S = frozenset(combo)
                phi[i] += w[r] * (v[S | {i}] - v[S])

    total = v[frozenset(range(n))] - v[frozenset()]
    out = pd.DataFrame(
        {"feature": features, "contribution": phi * 100}
    ).sort_values("contribution", key=abs, ascending=False).reset_index(drop=True)

    out.attrs["national_risk"] = v[frozenset()] * 100
    out.attrs["district_risk"] = v[frozenset(range(n))] * 100
    out.attrs["gap"] = total * 100
    out.attrs["efficiency_error"] = float(abs(phi.sum() - total) * 100)
    return out


def instance_shapley(models, profile: dict, which: str = "gbm",
                     n_background: int = 300, seed: int = 20260224) -> pd.DataFrame:
    """
    Same machinery for a single hypothetical person, used on the model page so
    a user can build someone and see the decomposition of their risk.
    """
    row = pd.DataFrame([profile])[models.features]
    return district_shapley(
        models, models.background, row, which=which,
        n_background=n_background, seed=seed,
    )
