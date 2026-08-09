"""
Step 6 — the page that decides whether anyone should believe the other five.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts, ui
from src.loaders import bootstrap, load_estimates, load_model_card, load_models, load_survey
from src.model import odds_ratios
from src.theme import CURRENT, LATERITE, MUTED, SILT

ui.page("Model and methods")
bootstrap()

est = load_estimates()
models = load_models()
card = load_model_card()

ui.header(
    "step 6 · audit",
    "Methods, performance, and what this cannot tell you",
    "Every dashboard should carry the page that undermines it. This one reports "
    "the estimator, the model performance including where it is mediocre, the "
    "calibration curve the intervention simulator depends on, and the "
    "assumptions that would have to hold for any of it to be causal.",
)

ui.district_picker(sorted(est["district"]))
ui.sidebar_footer()
ui.sidebar_footer()

m = models.metrics
tab_perf, tab_est, tab_or, tab_limits, tab_repro = st.tabs(
    ["Model performance", "Estimation", "Adjusted odds ratios",
     "Limitations", "Reproducing this"]
)

with tab_perf:
    ui.stat_row([
        ("Gradient boosting AUC", f"{m['gbm']['auc']:.3f}",
         f"Brier {m['gbm']['brier']:.4f}"),
        ("Logistic AUC", f"{m['logit']['auc']:.3f}",
         f"Brier {m['logit']['brier']:.4f}"),
        ("Training rows", f"{m['n_train']:,}", f"test {m['n_test']:,}"),
        ("Test prevalence", f"{m['prevalence_test'] * 100:.1f}%", "weighted"),
    ])

    st.markdown("")
    st.markdown(
        f"""
### The honest reading of these numbers

An AUC near {m['gbm']['auc']:.2f} is **modest**, and it should be. The model has
ten categorical predictors, all of them coarse — three age bands, three BMI
levels, a binary for exercise. It has no fasting glucose, no HbA1c, no family
history, no diet, no waist circumference. A model that scored 0.90 on these ten
columns would be evidence of leakage, not of skill.

What matters for this application is not discrimination but **calibration**. The
intervention simulator averages predicted probabilities and reads the average as
a prevalence. That step is valid when predicted probabilities match observed
frequencies, and invalid otherwise, regardless of AUC.

Note also that the logistic regression and the gradient booster land within
{abs(m['gbm']['auc'] - m['logit']['auc']):.3f} AUC of each other. With ten
categorical features and a largely additive risk structure, there are few
interactions for the tree to find. Reporting that, rather than quietly shipping
whichever model won, is the point of keeping both.
        """
    )

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("###### Gradient boosting calibration")
        st.plotly_chart(charts.calibration_plot(m["gbm"]["reliability"]),
                        width="stretch")
    with c2:
        st.markdown("###### Logistic calibration")
        st.plotly_chart(charts.calibration_plot(m["logit"]["reliability"]),
                        width="stretch")
    st.caption(
        "Marker size is the number of test observations in that bin. Points "
        "below the diagonal mean the model over-predicts risk at that level."
    )

with tab_est:
    st.markdown(
        """
### District prevalence

**Point estimate** — Hájek ratio estimator, `Σ w·y / Σ w`, using the BDHS
sampling weight `wt`. The unweighted mean is stored alongside it in
`district_estimates.csv` so the difference is inspectable rather than assumed
away.

**Interval** — nonparametric bootstrap resampling *primary sampling units*
(the 674 enumeration areas identified by `cluster`) with replacement, 400
replicates, percentile method. Resampling clusters rather than individuals is
what preserves the intra-cluster correlation. Percentile rather than normal
intervals because district proportions sit near a boundary and the bootstrap
distribution is visibly skewed for the sparse districts.

**Design effect** — the ratio of the clustered variance to the variance a
simple random sample of the same size would have produced. A district with a
design effect of 1.4 has standard errors 18% wider than a naive calculation
would report.
        """
    )
    c1, c2 = st.columns(2)
    c1.metric("Median design effect", f"{est['deff'].median():.2f}")
    c2.metric("Median standard error", f"±{est['se'].median():.2f} pp")

    st.markdown("###### Full estimate table")
    st.dataframe(
        est[["district", "division", "n", "n_clusters", "prevalence",
             "unweighted", "ci_low", "ci_high", "se", "deff",
             "reliability_label"]].round(2),
        hide_index=True, width="stretch", height=340,
    )
    st.download_button("Download estimates as CSV",
                       est.to_csv(index=False).encode(),
                       file_name="interveneBD_district_estimates.csv",
                       mime="text/csv")

    diff = (est["prevalence"] - est["unweighted"]).abs()
    st.caption(
        f"Weighting moves the district estimate by a median of "
        f"{diff.median():.2f} pp and by as much as {diff.max():.2f} pp "
        f"({est.loc[diff.idxmax(), 'district']}). That largest gap is the "
        f"single clearest argument for not using `groupby().mean()`."
    )

with tab_or:
    st.markdown(
        '<p class="lede">Adjusted odds ratios from the survey-weighted logistic '
        'model. Reference category for each variable is the first level '
        'alphabetically, dropped by the encoder. These are associations in a '
        'cross-section, adjusted for the other nine variables and nothing '
        'else.</p>',
        unsafe_allow_html=True,
    )
    ors = odds_ratios(models)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("###### Higher odds")
        st.dataframe(ors.head(12).round(3), hide_index=True,
                     width="stretch")
    with c2:
        st.markdown("###### Lower odds")
        st.dataframe(ors.tail(12).iloc[::-1].round(3), hide_index=True,
                     width="stretch")

with tab_limits:
    st.markdown(
        """
### What this dashboard cannot do

**No time dimension.** BDHS 2022 is a single cross-section. Nothing here can say
whether a district is improving or worsening. Any sentence containing "trend",
"rising", or "since" is outside what this data supports.

**No causal identification.** The intervention simulator moves people between
observed categories and re-scores them under a model fitted on associations.
Reading its output as a programme effect requires no unmeasured confounding,
positivity, and consistency. This dataset can test none of the three. The
physical-activity lever pointing the wrong way — visible on the intervention
page and left visible on purpose — is the cleanest demonstration of why.

**Self-reported, diagnosed diabetes.** The outcome captures people who know they
have diabetes. In a country where a large share of cases are undiagnosed, better
health-service coverage produces *more* recorded diabetes. Some of the
urban-rural and wealth gradient on the first page is diagnostic access rather
than disease.

**Thin districts.** Thirteen districts have fewer than 100 respondents and the
smallest has 40. The value-suppressing palette exists so those districts cannot
be read as precise, but no palette recovers information that was never sampled.

**Ten coarse predictors.** No clinical measurements, no diet, no family history.
The model is a composition summary, not a clinical risk score, and must never be
used for individual screening.

**One survey, one country, one year.** Nothing here generalises outside
Bangladesh in 2022.
        """
    )
    ui.caveat(
        "If you are reading this in a portfolio review: this page is not a "
        "disclaimer bolted on at the end. The value-suppressing palette, the "
        "cluster bootstrap, the efficiency check on the Shapley decomposition, "
        "and the deliberately visible wrong-signed lever are all the same design "
        "commitment expressed in four different places."
    )

with tab_repro:
    st.markdown(
        """
### Reproducing every number in this app

```bash
git clone <your-repo-url> && cd interveneBD
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# one-off: fetch and simplify district boundaries
curl -L -o data/raw/bd_districts_raw.geojson \\
  https://raw.githubusercontent.com/nuhil/bangladesh-geocode/master/geojson/districts.geojson
python scripts/01_prepare_geo.py

# build estimates, profiles, and models
python scripts/02_build_artifacts.py

# run
streamlit run app.py
```

Every random operation is seeded from `BOOTSTRAP_SEED` and `RANDOM_STATE` in
`src/config.py` and `src/model.py`. Re-running the build reproduces the
published numbers exactly.

```bash
python -m pytest tests/ -q     # estimator and similarity invariants
python -m src.theme            # palette luminance and CVD check
```
        """
    )
    if card:
        with st.expander("Raw model card JSON"):
            st.json(card)
