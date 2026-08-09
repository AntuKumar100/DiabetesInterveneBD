"""
District socio-structural profiles and the peer-similarity metric.

The idea
--------
Ranking districts by prevalence tells you where the burden is. It does not tell
you where the burden is *surprising*. Narayanganj at 34% is unremarkable if
every heavily urban, wealthy district sits near 34%. It is a finding if its
structural twins sit at 20%.

So we place every district in a ten-dimensional profile space built from
weighted composition shares — urban share, wealth mix, education mix, age
structure, behavioural prevalences — and find each district's nearest
neighbours in that space. The residual between a district and its peers is the
part its structure does not explain.

Why Gower and not Euclidean
---------------------------
Every axis here is a proportion, but they have wildly different spreads: urban
share ranges over roughly 80 points across districts, current-smoker share over
maybe 25. Raw Euclidean distance would let urban share dominate the similarity
purely because it has more room to vary. Gower's coefficient normalises each
axis by its observed range before averaging, which puts them on equal footing
and — importantly here — lets us attach a per-axis weight that the user
controls directly.

Gower (1971), "A general coefficient of similarity and some of its properties",
Biometrics 27(4). For all-numeric inputs it reduces to range-normalised
weighted Manhattan distance, which is exactly what we want.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PROFILE_AXES, WEIGHT_COL


def build_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per district, one column per profile axis, values in percent.

    Shares are survey-weighted. Using unweighted composition here while using
    weighted prevalence elsewhere would put the district's profile and its
    outcome in two different populations.
    """
    rows = []
    for district, g in df.groupby("District", sort=True):
        w = g[WEIGHT_COL].to_numpy(dtype=float)
        total = w.sum()
        rec = {"district": district, "division": g["Division"].iloc[0]}
        for label, col, wanted in PROFILE_AXES:
            targets = wanted if isinstance(wanted, tuple) else (wanted,)
            mask = g[col].isin(targets).to_numpy()
            rec[label] = float(w[mask].sum() / total * 100) if total > 0 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows).set_index("district")


def gower_distance(
    profiles: pd.DataFrame,
    target: str,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """
    Distance from `target` to every district, in [0, 1].

    weights: axis label -> non-negative importance. Missing axes default to 1.
    An axis weighted 0 is genuinely excluded, which is how the UI lets a user
    say "I don't care how urban they are, match me on wealth and age."
    """
    axes = [label for label, _, _ in PROFILE_AXES if label in profiles.columns]
    if target not in profiles.index:
        raise KeyError(f"unknown district: {target!r}")

    w = np.array([float((weights or {}).get(a, 1.0)) for a in axes])
    if np.any(w < 0):
        raise ValueError("axis weights must be non-negative")
    if w.sum() <= 0:
        w = np.ones_like(w)  # user zeroed everything; fall back to uniform

    X = profiles[axes].to_numpy(dtype=float)
    ranges = np.nanmax(X, axis=0) - np.nanmin(X, axis=0)
    ranges[ranges == 0] = 1.0  # a constant axis contributes nothing, not nan

    t = profiles.loc[target, axes].to_numpy(dtype=float)
    d = np.abs(X - t) / ranges
    return pd.Series((d * w).sum(axis=1) / w.sum(), index=profiles.index,
                     name="distance").sort_values()


def peers(profiles: pd.DataFrame, target: str, k: int = 6,
          weights: dict[str, float] | None = None,
          within_division: bool = False) -> list[str]:
    """The k nearest districts to `target`, excluding itself."""
    d = gower_distance(profiles, target, weights)
    d = d.drop(index=target, errors="ignore")
    if within_division:
        div = profiles.loc[target, "division"]
        keep = profiles.index[profiles["division"] == div]
        d = d[d.index.isin(keep)]
    return list(d.head(max(1, k)).index)


def structural_residual(estimates: pd.DataFrame, profiles: pd.DataFrame,
                        k: int = 6, weights: dict[str, float] | None = None
                        ) -> pd.DataFrame:
    """
    For every district: its prevalence minus the mean prevalence of its k peers.

    Positive residual = worse than structurally comparable districts, which is
    the shortlist a targeting exercise actually wants. Districts whose own
    estimate is unreliable are kept but flagged, never silently dropped —
    dropping them would quietly bias the shortlist toward large districts.
    """
    prev = estimates.set_index("district")["prevalence"]
    out = []
    for district in profiles.index:
        nb = peers(profiles, district, k=k, weights=weights)
        nb_prev = prev.reindex(nb).dropna()
        expected = float(nb_prev.mean()) if len(nb_prev) else np.nan
        out.append(
            {
                "district": district,
                "observed": float(prev.get(district, np.nan)),
                "peer_expected": expected,
                "residual": float(prev.get(district, np.nan) - expected),
                "peers": ", ".join(nb),
            }
        )
    res = pd.DataFrame(out).merge(
        estimates[["district", "se", "reliability", "reliability_label", "n"]],
        on="district", how="left",
    )
    return res.sort_values("residual", ascending=False).reset_index(drop=True)
