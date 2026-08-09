"""
Cached loading, and a lazy build so a cold deploy still works.

Streamlit Community Cloud restarts containers freely, and a fresh clone has no
processed artifacts. Rather than making the first visitor stare at a stack
trace, `bootstrap()` checks for the artifacts and builds them on demand behind
a spinner. Locally you would run `scripts/02_build_artifacts.py` once and never
hit that path.

Cache choices are deliberate: `cache_data` for frames (hashable, serialisable),
`cache_resource` for the fitted models and the parsed GeoJSON (large, shared,
not worth copying per session).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from . import model as model_mod
from . import profiles as profiles_mod
from . import survey as survey_mod
from .config import (
    ESTIMATES_OUT,
    GEOJSON_OUT,
    MODEL_CARD_OUT,
    MODEL_OUT,
    PROCESSED,
    PROFILES_OUT,
    SURVEY_XLS,
)


class DataMissing(RuntimeError):
    pass


@st.cache_data(show_spinner=False)
def load_survey() -> pd.DataFrame:
    if not SURVEY_XLS.exists():
        raise DataMissing(
            f"Survey file not found at {SURVEY_XLS}. "
            "Place the BDHS extract there and reload."
        )
    # xlrd is required for the legacy .xls container; openpyxl rejects it.
    df = pd.read_excel(SURVEY_XLS, engine="xlrd")
    df.columns = [c.strip() for c in df.columns]
    # pandas 3 splits str out of the object dtype; check the values instead
    # of the dtype label so this behaves the same on pandas 2 and 3.
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("str"):
            df[col] = df[col].astype(str).str.strip()
    return df


def artifacts_exist() -> bool:
    return all(p.exists() for p in (ESTIMATES_OUT, PROFILES_OUT, MODEL_OUT))


def build_artifacts(progress=None) -> None:
    """Run the full pipeline and write everything under data/processed."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = load_survey()

    if progress:
        progress("Estimating district prevalence (cluster bootstrap)…")
    survey_mod.district_estimates(df).to_csv(ESTIMATES_OUT, index=False)

    if progress:
        progress("Building district profiles…")
    profiles_mod.build_profiles(df).to_csv(PROFILES_OUT)

    if progress:
        progress("Fitting risk models…")
    models = model_mod.train(df)
    joblib.dump(models, MODEL_OUT)
    MODEL_CARD_OUT.write_text(model_mod.model_card(models))


def bootstrap() -> None:
    """Call once at the top of every page."""
    if artifacts_exist():
        return
    box = st.empty()
    with st.spinner("First run — building analysis artifacts. Takes about a minute."):
        build_artifacts(progress=lambda m: box.caption(m))
    box.empty()


@st.cache_data(show_spinner=False)
def load_estimates() -> pd.DataFrame:
    return pd.read_csv(ESTIMATES_OUT)


@st.cache_data(show_spinner=False)
def load_profiles() -> pd.DataFrame:
    return pd.read_csv(PROFILES_OUT, index_col=0)


@st.cache_resource(show_spinner=False)
def load_models():
    return joblib.load(MODEL_OUT)


@st.cache_resource(show_spinner=False)
def load_geojson() -> dict:
    if not GEOJSON_OUT.exists():
        raise DataMissing(
            f"District geometry not found at {GEOJSON_OUT}. "
            "Run: python scripts/01_prepare_geo.py"
        )
    return json.loads(Path(GEOJSON_OUT).read_text())


@st.cache_data(show_spinner=False)
def load_model_card() -> dict:
    return json.loads(MODEL_CARD_OUT.read_text()) if MODEL_CARD_OUT.exists() else {}


def district_rows(df: pd.DataFrame, district: str) -> pd.DataFrame:
    rows = df[df["District"] == district]
    if rows.empty:
        raise DataMissing(f"No survey rows for district {district!r}.")
    return rows
