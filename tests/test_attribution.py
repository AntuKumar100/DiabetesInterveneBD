"""
Axiom tests for the Shapley decomposition, and behavioural tests for the
intervention simulator.

The efficiency test is the important one. It is the property that lets the
waterfall chart claim to be a decomposition rather than a ranking, and it is
checked here against a model whose behaviour we control completely.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.counterfactual import apply_interventions, eligible_mask, simulate
from src.shapley import district_shapley


class ToyModel:
    """
    Stand-in for TrainedModels with a known additive risk function, so the
    Shapley values have an analytic answer to compare against.
    """

    def __init__(self, features, coefs):
        self.features = list(features)
        self.coefs = coefs
        self.background = None

    def predict(self, X, which="gbm"):
        p = np.zeros(len(X))
        for f, mapping in self.coefs.items():
            p += X[f].map(mapping).fillna(0.0).to_numpy()
        return np.clip(p, 0, 1)


@pytest.fixture
def toy():
    features = ["a", "b", "c"]
    coefs = {
        "a": {"lo": 0.05, "hi": 0.25},
        "b": {"lo": 0.02, "hi": 0.10},
        "c": {"lo": 0.01, "hi": 0.01},   # null player: no effect either way
    }
    model = ToyModel(features, coefs)
    rng = np.random.default_rng(11)
    national = pd.DataFrame({
        f: rng.choice(["lo", "hi"], 400, p=[0.7, 0.3]) for f in features
    })
    district = pd.DataFrame({
        "a": ["hi"] * 200, "b": ["hi"] * 200,
        "c": rng.choice(["lo", "hi"], 200),
    })
    return model, national, district


def test_efficiency_identity_holds(toy):
    model, national, district = toy
    phi = district_shapley(model, national, district, n_background=200)
    assert phi.attrs["efficiency_error"] < 1e-9
    assert phi["contribution"].sum() == pytest.approx(phi.attrs["gap"], abs=1e-9)


def test_null_player_gets_zero(toy):
    """Feature 'c' has the same value in both states, so it moved nothing."""
    model, national, district = toy
    phi = district_shapley(model, national, district, n_background=200)
    c = float(phi.loc[phi["feature"] == "c", "contribution"].iloc[0])
    assert abs(c) < 1e-9


def test_additive_model_gives_exact_marginal_effects(toy):
    """
    For a purely additive value function, Shapley values collapse to each
    feature's own marginal effect. Any deviation means the coalition weighting
    is wrong.
    """
    model, national, district = toy
    phi = district_shapley(model, national, district, n_background=300)
    got = dict(zip(phi["feature"], phi["contribution"]))

    for f, mapping in model.coefs.items():
        nat_mean = national[f].map(mapping).mean()
        dist_mean = district[f].map(mapping).mean()
        assert got[f] == pytest.approx((dist_mean - nat_mean) * 100, abs=1.5)


def test_ordering_is_by_absolute_magnitude(toy):
    model, national, district = toy
    phi = district_shapley(model, national, district, n_background=200)
    mags = phi["contribution"].abs().to_numpy()
    assert (np.diff(mags) <= 1e-9).all()


# --- intervention simulator --------------------------------------------------
@pytest.fixture
def rows():
    return pd.DataFrame({
        "Smoker": ["Yes"] * 20 + ["No"] * 30,
        "Physical_Exercise": ["No"] * 25 + ["Yes"] * 25,
        "BMI_level": ["Obesity"] * 10 + ["Overweight"] * 10 + ["No"] * 30,
        "Hypertension": ["Yes"] * 15 + ["No"] * 35,
        "wt": [1.0] * 50,
        "cluster": [f"c{i // 5}" for i in range(50)],
    })


def test_eligibility_matches_the_spec(rows):
    assert eligible_mask(rows, "Smoker").sum() == 20
    assert eligible_mask(rows, "BMI_level").sum() == 20      # obese + overweight
    assert eligible_mask(rows, "Physical_Exercise").sum() == 25


def test_zero_coverage_changes_nothing(rows):
    rng = np.random.default_rng(0)
    out = apply_interventions(rows, {"Smoker": 0.0}, rng)
    pd.testing.assert_frame_equal(out, rows)


def test_full_coverage_moves_every_eligible_person(rows):
    rng = np.random.default_rng(0)
    out = apply_interventions(rows, {"Smoker": 1.0}, rng)
    assert (out["Smoker"] == "No").all()
    assert eligible_mask(out, "Smoker").sum() == 0


def test_partial_coverage_moves_the_right_count(rows):
    rng = np.random.default_rng(0)
    out = apply_interventions(rows, {"Smoker": 0.5}, rng)
    assert (out["Smoker"] == "Yes").sum() == 10


def test_intervention_never_touches_ineligible_people(rows):
    """
    Someone already exercising must not be 'moved' into exercising and counted
    as reached. Coverage denominators depend on this.
    """
    rng = np.random.default_rng(0)
    before = (rows["Physical_Exercise"] == "Yes").sum()
    out = apply_interventions(rows, {"Physical_Exercise": 1.0}, rng)
    assert (out["Physical_Exercise"] == "Yes").sum() == len(rows)
    assert before == 25


def test_unknown_intervention_raises(rows):
    with pytest.raises(KeyError):
        apply_interventions(rows, {"Horoscope": 0.5}, np.random.default_rng(0))


def test_simulator_reports_a_reduction_for_a_harmful_factor():
    """
    End-to-end: with a model where smoking strictly raises risk, removing
    smoking must strictly lower the simulated prevalence.
    """
    model = ToyModel(["Smoker"], {"Smoker": {"Yes": 0.4, "No": 0.1}})
    rows = pd.DataFrame({
        "Smoker": ["Yes"] * 40 + ["No"] * 40,
        "wt": [1.0] * 80,
        "cluster": [f"c{i // 8}" for i in range(80)],
    })
    res = simulate(model, rows, {"Smoker": 1.0}, n_rep=40)
    assert res["reduction"] == pytest.approx(15.0, abs=0.01)
    assert res["reduction_lo"] > 0
    assert res["certain"] is True
