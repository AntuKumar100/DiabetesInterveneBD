"""
Intervention simulator.

What this does
--------------
Take the real people the survey observed in a district. Pick a modifiable risk
factor and a coverage level. Move that fraction of the eligible people to the
target state, re-score everyone with the fitted model, and report the change in
weighted mean predicted risk.

Two things make this more honest than the usual "what-if" slider.

1. Coverage is explicit and partial. Real programmes never reach everyone.
   Setting coverage to 100% is available but labelled as a ceiling, not a plan.
   Who gets reached is drawn at random among the eligible, repeated across
   replicates, so the reported effect carries the sampling variability of the
   rollout as well as of the survey.

2. The interval is a genuine interval. Each replicate resamples the district's
   *clusters* (matching the estimation scheme in survey.py) and re-draws which
   eligible people are covered. So the band widens for small districts, which
   is where a naive simulator would otherwise hand a planner a crisp number
   built on forty respondents.

What this is not
----------------
It is not a causal effect. The model was fitted on cross-sectional data, so
"people who exercise have lower predicted risk" cannot be turned into
"exercise lowers risk" without assumptions the data cannot test — no temporal
ordering, and reverse causation is entirely plausible for BMI and activity in a
diabetic population. The output is the risk of a population that *looks like*
the district would after the intervention. That framing is repeated in the UI
next to every number this module produces.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import INTERVENTIONS, PSU_COL, WEIGHT_COL


def eligible_mask(rows: pd.DataFrame, factor: str) -> np.ndarray:
    spec = INTERVENTIONS[factor]
    return rows[factor].isin(spec["eligible_from"]).to_numpy()


def apply_interventions(rows: pd.DataFrame, plan: dict[str, float],
                        rng: np.random.Generator) -> pd.DataFrame:
    """
    plan: factor -> coverage in [0, 1]. Returns a modified copy.

    Interventions are applied independently. A person eligible for two of them
    can be reached by both, which is the intended semantics of a combined
    programme, and is why the combined effect is not the sum of the singles.
    """
    out = rows.copy()
    for factor, coverage in plan.items():
        if factor not in INTERVENTIONS:
            raise KeyError(f"unknown intervention: {factor!r}")
        coverage = float(np.clip(coverage, 0.0, 1.0))
        if coverage == 0:
            continue
        elig = np.flatnonzero(eligible_mask(out, factor))
        if elig.size == 0:
            continue
        k = int(round(coverage * elig.size))
        if k == 0:
            continue
        chosen = rng.choice(elig, size=k, replace=False)
        out.iloc[chosen, out.columns.get_loc(factor)] = INTERVENTIONS[factor]["target"]
    return out


def simulate(
    models,
    rows: pd.DataFrame,
    plan: dict[str, float],
    which: str = "gbm",
    n_rep: int = 120,
    seed: int = 20260224,
) -> dict:
    """
    Returns baseline risk, post-intervention risk, absolute and relative
    reduction, a 95% interval on the reduction, and the reached headcount.

    All risks are weighted means of predicted probability, in percentage
    points. Uses the survey weight so the district's own sampling design is
    respected inside the simulation.
    """
    if len(rows) == 0:
        raise ValueError("no rows to simulate on")

    rng = np.random.default_rng(seed)
    w = rows[WEIGHT_COL].to_numpy(dtype=float)

    baseline = float(np.average(models.predict(rows, which=which), weights=w)) * 100

    treated = apply_interventions(rows, plan, rng)
    post = float(np.average(models.predict(treated, which=which), weights=w)) * 100

    # Replicates: resample clusters, then re-draw who is covered.
    codes, _ = pd.factorize(rows[PSU_COL])
    n_psu = codes.max() + 1
    deltas = []
    if n_psu >= 2:
        idx_by_psu = [np.flatnonzero(codes == c) for c in range(n_psu)]
        for _ in range(n_rep):
            draw = rng.integers(0, n_psu, size=n_psu)
            take = np.concatenate([idx_by_psu[c] for c in draw])
            sub = rows.iloc[take]
            ww = sub[WEIGHT_COL].to_numpy(dtype=float)
            b = np.average(models.predict(sub, which=which), weights=ww)
            t = np.average(
                models.predict(apply_interventions(sub, plan, rng), which=which),
                weights=ww,
            )
            deltas.append((b - t) * 100)

    deltas = np.array(deltas, dtype=float)
    if deltas.size:
        lo, hi = (float(x) for x in np.percentile(deltas, [2.5, 97.5]))
    else:
        lo = hi = float("nan")

    reached = {
        factor: int(round(float(np.clip(cov, 0, 1)) * eligible_mask(rows, factor).sum()))
        for factor, cov in plan.items() if cov > 0
    }

    return {
        "baseline": baseline,
        "post": post,
        "reduction": baseline - post,
        "reduction_lo": lo,
        "reduction_hi": hi,
        "relative": (baseline - post) / baseline * 100 if baseline > 0 else np.nan,
        "reached": reached,
        "n_rows": int(len(rows)),
        "n_clusters": int(n_psu),
        "certain": bool(deltas.size and lo > 0),
    }


def leverage_table(models, rows: pd.DataFrame, coverage: float = 0.5,
                   which: str = "gbm", n_rep: int = 60,
                   seed: int = 20260224) -> pd.DataFrame:
    """
    One row per intervention at a fixed coverage, ranked by expected reduction.

    This is the tornado chart's data. Ranking by *reduction per person reached*
    as well as by total reduction matters: tobacco cessation often wins on
    efficiency and loses on total impact simply because fewer people smoke.
    """
    rows_out = []
    for factor, spec in INTERVENTIONS.items():
        n_elig = int(eligible_mask(rows, factor).sum())
        if n_elig == 0:
            rows_out.append({"factor": factor, "label": spec["label"],
                             "eligible": 0, "reached": 0, "reduction": 0.0,
                             "lo": 0.0, "hi": 0.0, "per_1000_reached": 0.0})
            continue
        res = simulate(models, rows, {factor: coverage}, which=which,
                       n_rep=n_rep, seed=seed)
        reached = res["reached"].get(factor, 0)
        rows_out.append(
            {
                "factor": factor,
                "label": spec["label"],
                "eligible": n_elig,
                "reached": reached,
                "reduction": res["reduction"],
                "lo": res["reduction_lo"],
                "hi": res["reduction_hi"],
                "per_1000_reached": (res["reduction"] / reached * 1000
                                     if reached else 0.0),
            }
        )
    return pd.DataFrame(rows_out).sort_values("reduction", ascending=False)
