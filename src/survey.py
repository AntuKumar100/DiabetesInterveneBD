"""
Design-based estimation for a stratified two-stage cluster sample.

The single thing this module exists to prevent
----------------------------------------------
`df.groupby("District")["y"].mean()` is wrong twice over on BDHS data. It
ignores the sampling weight, so the point estimate is biased toward
over-sampled strata. And it computes a standard error under a
simple-random-sample assumption, which understates the true error by the square
root of the design effect — typically 1.3x to 2x for a clustered health survey.
The second error is the dangerous one, because it produces a map where noise
looks like signal.

What we do instead
------------------
Point estimate: Hájek ratio estimator, sum(w*y) / sum(w).

Interval: nonparametric bootstrap that resamples *primary sampling units*
(enumeration areas) with replacement, independently within each stratum
(division). Resampling clusters rather than individuals is what preserves the
intra-cluster correlation that inflates the variance. Percentile intervals,
because the district-level estimates are proportions near a boundary and the
bootstrap distribution is visibly skewed for the sparse districts.

Districts nest inside divisions in Bangladesh, so a district's clusters all
belong to one stratum. We resample within the district's own stratum, which
keeps the replicate sample size stable.

Reference: Rust & Rao (1996), "Variance estimation for complex surveys using
replication techniques", Statistical Methods in Medical Research 5(3).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (
    BOOTSTRAP_SEED,
    N_BOOTSTRAP,
    OUTCOME_COL,
    OUTCOME_POSITIVE,
    PSU_COL,
    RELIABILITY_TIERS,
    WEIGHT_COL,
)


def hajek(y: np.ndarray, w: np.ndarray) -> float:
    """Weighted proportion. Returns nan for an empty group rather than raising."""
    tot = w.sum()
    if tot <= 0 or len(y) == 0:
        return float("nan")
    return float((w * y).sum() / tot)


def _bootstrap_group(y, w, psu, rng, n_rep):
    """
    One district. Resample its clusters with replacement, n_rep times.

    Vectorised over replicates by building an index array rather than looping
    in Python: 64 districts x 400 replicates is 25,600 resamples, and the naive
    loop takes minutes where this takes under a second.
    """
    codes, _ = pd.factorize(psu)
    n_psu = codes.max() + 1
    if n_psu < 2:
        # A single cluster carries no information about between-cluster
        # variance. Report the estimate, refuse to invent an interval.
        return np.full(n_rep, np.nan)

    # Rows grouped by cluster, so a resampled cluster pulls all of its rows.
    order = np.argsort(codes, kind="stable")
    y_s, w_s, codes_s = y[order], w[order], codes[order]
    starts = np.searchsorted(codes_s, np.arange(n_psu), side="left")
    ends = np.searchsorted(codes_s, np.arange(n_psu), side="right")

    wy_by_psu = np.array([(w_s[a:b] * y_s[a:b]).sum() for a, b in zip(starts, ends)])
    w_by_psu = np.array([w_s[a:b].sum() for a, b in zip(starts, ends)])

    draws = rng.integers(0, n_psu, size=(n_rep, n_psu))
    num = wy_by_psu[draws].sum(axis=1)
    den = w_by_psu[draws].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def district_estimates(
    df: pd.DataFrame,
    outcome_col: str = OUTCOME_COL,
    positive: str = OUTCOME_POSITIVE,
    n_rep: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """
    Weighted prevalence per district with bootstrap CI, SE, and design effect.

    All rates are returned in percentage points, because that is the unit a
    health planner reads and converting once here beats converting in nine
    different charts.
    """
    required = {outcome_col, WEIGHT_COL, PSU_COL, "District"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"survey frame is missing columns: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    work = df.copy()
    work["_y"] = (work[outcome_col] == positive).astype(float)

    rows = []
    for district, g in work.groupby("District", sort=True):
        y = g["_y"].to_numpy()
        w = g[WEIGHT_COL].to_numpy(dtype=float)
        psu = g[PSU_COL].to_numpy()

        point = hajek(y, w)
        reps = _bootstrap_group(y, w, psu, rng, n_rep)
        finite = reps[np.isfinite(reps)]

        if finite.size >= n_rep * 0.5:
            se = float(finite.std(ddof=1))
            lo, hi = (float(x) for x in np.percentile(finite, [2.5, 97.5]))
        else:
            se, lo, hi = float("nan"), float("nan"), float("nan")

        # Design effect: how much worse our clustered SE is than the SE a
        # simple random sample of the same size would have given. Anything
        # above ~1.5 is the map's way of saying "these clusters are alike".
        p = np.average(y, weights=w)
        srs_se = np.sqrt(max(p * (1 - p), 1e-12) / len(g))
        deff = (se / srs_se) ** 2 if np.isfinite(se) and srs_se > 0 else np.nan

        rows.append(
            {
                "district": district,
                "division": g["Division"].iloc[0],
                "n": int(len(g)),
                "n_clusters": int(pd.Series(psu).nunique()),
                "prevalence": point * 100,
                "ci_low": lo * 100,
                "ci_high": hi * 100,
                "se": se * 100,
                "deff": deff,
                "unweighted": float(y.mean()) * 100,
            }
        )

    out = pd.DataFrame(rows)
    out["ci_width"] = out["ci_high"] - out["ci_low"]
    out["reliability"] = out["se"].apply(reliability_tier)
    out["reliability_label"] = out["reliability"].map(
        {0: "High", 1: "Moderate", 2: "Low", 3: "Very low"}
    )
    return out.sort_values("prevalence", ascending=False).reset_index(drop=True)


def reliability_tier(se: float, tiers=None) -> int:
    tiers = tiers or RELIABILITY_TIERS
    if not np.isfinite(se):
        return len(tiers)
    for i, cut in enumerate(tiers):
        if se < cut:
            return i
    return len(tiers)


def subgroup_estimates(df: pd.DataFrame, by: str, outcome_col: str = OUTCOME_COL,
                       positive: str = OUTCOME_POSITIVE) -> pd.DataFrame:
    """
    Weighted prevalence by any single categorical column, with a
    Korn-Graubard style interval approximated from the cluster bootstrap.

    Used for the national subgroup strips, where n per cell is large enough
    that 200 replicates is plenty.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    work = df.copy()
    work["_y"] = (work[outcome_col] == positive).astype(float)

    rows = []
    for level, g in work.groupby(by, sort=False):
        y = g["_y"].to_numpy()
        w = g[WEIGHT_COL].to_numpy(dtype=float)
        reps = _bootstrap_group(y, w, g[PSU_COL].to_numpy(), rng, 200)
        finite = reps[np.isfinite(reps)]
        lo, hi = (np.percentile(finite, [2.5, 97.5]) * 100
                  if finite.size else (np.nan, np.nan))
        rows.append(
            {
                "level": str(level),
                "n": int(len(g)),
                "prevalence": hajek(y, w) * 100,
                "ci_low": float(lo),
                "ci_high": float(hi),
            }
        )
    return pd.DataFrame(rows).sort_values("prevalence", ascending=False)
