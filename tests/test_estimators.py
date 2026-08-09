"""
Invariant tests for the two pieces of statistics that everything else sits on.

These are not coverage theatre. Each one encodes a mistake that is easy to make
and hard to notice in a dashboard, because a wrong number still renders.

Run: python -m pytest tests/ -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.profiles import gower_distance
from src.survey import district_estimates, hajek, reliability_tier


@pytest.fixture
def toy():
    """
    Two districts, six clusters each. District A has a weight pattern that
    makes the weighted and unweighted means differ on purpose.
    """
    rng = np.random.default_rng(7)
    rows = []
    for district, base in (("alpha", 0.30), ("beta", 0.10)):
        for c in range(6):
            for _ in range(25):
                y = rng.random() < base
                rows.append({
                    "District": district,
                    "Division": "testdiv",
                    "cluster": f"{district}-{c}",
                    "wt": 2.0 if c < 3 else 0.5,
                    "Diabetes": "Yes" if y else "No",
                })
    return pd.DataFrame(rows)


def test_hajek_matches_manual_weighted_mean():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    w = np.array([3.0, 1.0, 1.0, 1.0])
    assert hajek(y, w) == pytest.approx(4.0 / 6.0)


def test_hajek_handles_empty_group():
    assert np.isnan(hajek(np.array([]), np.array([])))


def test_weighted_differs_from_unweighted(toy):
    """
    The whole reason survey.py exists. If these two ever agree exactly on this
    fixture, the weight has stopped being applied somewhere.
    """
    est = district_estimates(toy, n_rep=80)
    assert not np.allclose(est["prevalence"], est["unweighted"])


def test_intervals_contain_point_estimate(toy):
    est = district_estimates(toy, n_rep=200)
    assert (est["ci_low"] <= est["prevalence"] + 1e-9).all()
    assert (est["ci_high"] >= est["prevalence"] - 1e-9).all()


def test_single_cluster_district_gets_no_interval():
    """
    One cluster carries no between-cluster information. Reporting an interval
    there would be inventing precision, so we return nan and the UI shows the
    district as lowest reliability.
    """
    df = pd.DataFrame({
        "District": ["solo"] * 20,
        "Division": ["d"] * 20,
        "cluster": ["only"] * 20,
        "wt": [1.0] * 20,
        "Diabetes": ["Yes"] * 5 + ["No"] * 15,
    })
    est = district_estimates(df, n_rep=50)
    assert np.isnan(est.loc[0, "se"])
    assert est.loc[0, "prevalence"] == pytest.approx(25.0)


def test_clustering_widens_the_interval():
    """
    Perfectly correlated clusters must produce a wider interval than the same
    observations spread across many clusters. This is the design effect, and
    getting it backwards is the classic survey-analysis bug.
    """
    n = 240
    rng = np.random.default_rng(3)
    spread = pd.DataFrame({
        "District": ["x"] * n, "Division": ["d"] * n,
        "cluster": [f"c{i}" for i in range(n)], "wt": [1.0] * n,
        "Diabetes": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
    })
    # Same marginal outcome, but every cluster is internally homogeneous.
    lumps = spread.copy()
    lumps["cluster"] = [f"c{i // 20}" for i in range(n)]
    lumps["Diabetes"] = ["Yes" if (i // 20) % 3 == 0 else "No" for i in range(n)]

    se_spread = district_estimates(spread, n_rep=300).loc[0, "se"]
    se_lumped = district_estimates(lumps, n_rep=300).loc[0, "se"]
    assert se_lumped > se_spread


def test_reliability_tiers_are_ordered():
    assert reliability_tier(1.0) == 0
    assert reliability_tier(3.0) == 1
    assert reliability_tier(5.0) == 2
    assert reliability_tier(9.0) == 3
    assert reliability_tier(float("nan")) == 3


# --- similarity --------------------------------------------------------------
@pytest.fixture
def profiles():
    return pd.DataFrame(
        {
            "Urban share": [10.0, 12.0, 90.0, 50.0],
            "Poorest quintile": [40.0, 38.0, 5.0, 20.0],
            "division": ["a", "a", "b", "b"],
        },
        index=["rural1", "rural2", "city", "mixed"],
    )


def test_distance_to_self_is_zero(profiles, monkeypatch):
    import src.profiles as P
    monkeypatch.setattr(P, "PROFILE_AXES",
                        [("Urban share", "x", "y"), ("Poorest quintile", "x", "y")])
    d = P.gower_distance(profiles, "city")
    assert d["city"] == pytest.approx(0.0)


def test_nearest_neighbour_is_the_structural_twin(profiles, monkeypatch):
    import src.profiles as P
    monkeypatch.setattr(P, "PROFILE_AXES",
                        [("Urban share", "x", "y"), ("Poorest quintile", "x", "y")])
    d = P.gower_distance(profiles, "rural1").drop("rural1")
    assert d.index[0] == "rural2"


def test_zero_weight_axis_is_ignored(profiles, monkeypatch):
    """
    Zeroing an axis must genuinely exclude it. If it only shrinks its influence,
    the 'socioeconomic only' preset in the UI is quietly lying.
    """
    import src.profiles as P
    monkeypatch.setattr(P, "PROFILE_AXES",
                        [("Urban share", "x", "y"), ("Poorest quintile", "x", "y")])
    full = P.gower_distance(profiles, "city")
    only_wealth = P.gower_distance(profiles, "city",
                                   {"Urban share": 0.0, "Poorest quintile": 1.0})
    assert not np.allclose(full.values, only_wealth.values)

    manual = (profiles["Poorest quintile"] - profiles.loc["city", "Poorest quintile"]).abs()
    manual = manual / (profiles["Poorest quintile"].max() - profiles["Poorest quintile"].min())
    assert np.allclose(only_wealth.sort_index().values, manual.sort_index().values)


def test_unknown_district_raises(profiles):
    with pytest.raises(KeyError):
        gower_distance(profiles, "atlantis")
