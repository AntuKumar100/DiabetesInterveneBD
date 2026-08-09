"""
Single source of truth for paths, column semantics, and domain constants.

Nothing in this file computes anything. If a module needs to know what counts
as a modifiable risk factor, or which column holds the survey weight, it asks
here rather than hard-coding a string. That is the difference between a
notebook and something you can hand to somebody else.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"

# --- inputs ------------------------------------------------------------------
SURVEY_XLS = RAW / "bdhs_diabetes.xls"
RAW_GEOJSON = RAW / "bd_districts_raw.geojson"

# --- build outputs -----------------------------------------------------------
GEOJSON_OUT = PROCESSED / "bd_districts.geojson"
CENTROIDS_OUT = PROCESSED / "bd_centroids.json"
ESTIMATES_OUT = PROCESSED / "district_estimates.csv"
PROFILES_OUT = PROCESSED / "district_profiles.csv"
MODEL_OUT = PROCESSED / "risk_model.joblib"
MODEL_CARD_OUT = PROCESSED / "model_card.json"

# --- survey design -----------------------------------------------------------
# BDHS is a stratified two-stage cluster sample. `wt` is the sampling weight,
# `cluster` identifies the primary sampling unit (the enumeration area), and we
# treat Division as the stratum. Ignoring any of the three biases both the point
# estimate and — much more severely — the standard error.
WEIGHT_COL = "wt"
PSU_COL = "cluster"
STRATUM_COL = "Division"
OUTCOME_COL = "Diabetes"
OUTCOME_POSITIVE = "Yes"
DISTRICT_COL = "District"

# --- feature semantics -------------------------------------------------------
# Split matters: you cannot intervene on someone's age, and pretending you can
# turns a policy tool into a fantasy. Only MODIFIABLE features are exposed in
# the intervention lab.
MODIFIABLE = ["Smoker", "Physical_Exercise", "BMI_level", "Hypertension"]

STRUCTURAL = [
    "Residence",
    "Wealth_Index",
    "Gender",
    "Education",
    "Age",
    "Marital_Status",
]

FEATURES = STRUCTURAL + MODIFIABLE

# The state each intervention moves people *to*, and who is eligible to move.
# `label` is what a health planner would call the programme, not what the
# column is called.
INTERVENTIONS = {
    "Smoker": {
        "label": "Tobacco cessation",
        "target": "No",
        "eligible_from": ["Yes"],
        "note": "Moves current smokers to non-smoking status.",
    },
    "Physical_Exercise": {
        "label": "Physical activity programme",
        "target": "Yes",
        "eligible_from": ["No"],
        "note": "Moves inactive adults into regular activity.",
    },
    "BMI_level": {
        "label": "Weight reduction",
        "target": "No",
        "eligible_from": ["Obesity", "Overweight"],
        "note": "Moves overweight and obese adults to normal BMI.",
    },
    "Hypertension": {
        "label": "Blood pressure control",
        "target": "No",
        "eligible_from": ["Yes"],
        "note": "Brings hypertensive adults under control. Comorbidity, not a "
                "clean exposure — read the caveat on the Methods page.",
    },
}

# --- district profile axes ---------------------------------------------------
# Each entry is (display name, column, category counted as the numerator).
# These are the axes of the peer-similarity space and the parallel coordinates
# plot. Deliberately socio-structural: we look for districts that *look alike*
# demographically, then ask why their outcomes differ.
PROFILE_AXES = [
    ("Urban share", "Residence", "urban"),
    ("Poorest quintile", "Wealth_Index", "poorest"),
    ("Richest quintile", "Wealth_Index", "richest"),
    ("No education", "Education", "no education"),
    ("Secondary+", "Education", "Secondary or above"),
    ("Aged 35-64", "Age", "35-64"),
    ("Overweight/obese", "BMI_level", ("Overweight", "Obesity")),
    ("Hypertensive", "Hypertension", "Yes"),
    ("Physically active", "Physical_Exercise", "Yes"),
    ("Current smoker", "Smoker", "Yes"),
]

# --- estimation settings -----------------------------------------------------
N_BOOTSTRAP = 400          # cluster-resampled replicates per district
BOOTSTRAP_SEED = 20260224  # fixed so the published numbers are reproducible
RELIABILITY_TIERS = [2.5, 4.5, 7.0]  # SE cut-points in percentage points

# --- name reconciliation -----------------------------------------------------
# Boundary-file spelling -> BDHS spelling. See docs/district_name_map.md for
# why each row exists. Keys are lowercase.
DISTRICT_ALIASES = {
    "barisal": "barishal",
    "bogra": "bogura",
    "brahamanbaria": "brahmanbaria",   # typo in the source boundary file
    "chittagong": "chattogram",
    "comilla": "cumilla",
    "jessore": "jashore",
    "nawabganj": "chapai nawabganj",   # short form in the boundary file
}
