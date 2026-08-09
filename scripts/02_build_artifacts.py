"""
Build every artifact the app reads at runtime.

Run once after cloning:
    python scripts/01_prepare_geo.py
    python scripts/02_build_artifacts.py

Writes to data/processed/:
    district_estimates.csv   weighted prevalence + bootstrap CI per district
    district_profiles.csv    64 x 10 socio-structural profile matrix
    risk_model.joblib        fitted logistic + gradient boosting pipelines
    model_card.json          metrics, for the Methods page

The app will do this lazily on first load if the files are missing, but doing
it here keeps cold starts on Streamlit Cloud fast and makes the numbers
reproducible from the command line.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import model as model_mod          # noqa: E402
from src import profiles as profiles_mod    # noqa: E402
from src import survey as survey_mod        # noqa: E402
from src.config import (                    # noqa: E402
    ESTIMATES_OUT, MODEL_CARD_OUT, MODEL_OUT, PROCESSED, PROFILES_OUT, SURVEY_XLS,
)


def main() -> int:
    import joblib
    import pandas as pd

    if not SURVEY_XLS.exists():
        print(f"Survey file not found: {SURVEY_XLS}", file=sys.stderr)
        return 1

    PROCESSED.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = pd.read_excel(SURVEY_XLS, engine="xlrd")
    df.columns = [c.strip() for c in df.columns]
    # pandas 3 splits str out of the object dtype; check the values instead
    # of the dtype label so this behaves the same on pandas 2 and 3.
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith("str"):
            df[col] = df[col].astype(str).str.strip()
    print(f"survey: {df.shape[0]:,} rows x {df.shape[1]} cols, "
          f"{df['cluster'].nunique()} clusters, {df['District'].nunique()} districts")

    print("→ district estimates (cluster bootstrap)…", flush=True)
    est = survey_mod.district_estimates(df)
    est.to_csv(ESTIMATES_OUT, index=False)
    print(f"  prevalence {est['prevalence'].min():.1f}–{est['prevalence'].max():.1f}%  "
          f"median SE ±{est['se'].median():.2f} pp  "
          f"median deff {est['deff'].median():.2f}")

    print("→ district profiles…", flush=True)
    prof = profiles_mod.build_profiles(df)
    prof.to_csv(PROFILES_OUT)
    print(f"  {prof.shape[0]} districts x {prof.shape[1] - 1} axes")

    print("→ risk models…", flush=True)
    models = model_mod.train(df)
    joblib.dump(models, MODEL_OUT)
    MODEL_CARD_OUT.write_text(model_mod.model_card(models))
    print(f"  gbm   AUC {models.metrics['gbm']['auc']:.3f}  "
          f"Brier {models.metrics['gbm']['brier']:.4f}")
    print(f"  logit AUC {models.metrics['logit']['auc']:.3f}  "
          f"Brier {models.metrics['logit']['brier']:.4f}")

    print(f"done in {time.time() - t0:.1f}s → {PROCESSED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
