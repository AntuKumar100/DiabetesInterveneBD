"""
Step 3 — decompose a district's gap against the national average.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts, ui
from src.config import MODIFIABLE
from src.loaders import (
    bootstrap, district_rows, load_estimates, load_models, load_survey,
)
from src.shapley import district_shapley, instance_shapley
from src.theme import CURRENT, LATERITE, MUTED, REED, SILT

ui.page("Risk drivers")
bootstrap()

est = load_estimates()
df = load_survey()
models = load_models()

ui.header(
    "step 3 · explain",
    "What accounts for the gap?",
    "This is a Shapley decomposition of a difference between two populations, "
    "not of one person's prediction. The question it answers: if you rebuilt the "
    "national population one attribute at a time until it matched this "
    "district's composition, how much of the risk gap would each attribute "
    "carry? With ten features the coalition lattice is 1,024 subsets, small "
    "enough to enumerate exactly — nothing here is approximated.",
)

district = ui.district_picker(sorted(est["district"]))
which = ui.model_picker()

st.sidebar.markdown("### Attribution")
n_bg = st.sidebar.select_slider(
    "Reference sample size", options=[150, 200, 300, 400], value=200,
    help="Rows drawn from the national background for each of the 1,024 "
         "coalitions. Larger is smoother and slower.",
)
ui.sidebar_footer()
ui.sidebar_footer()


@st.cache_data(show_spinner="Enumerating 1,024 coalitions…", ttl=1800)
def _phi(district: str, which: str, n_bg: int):
    rows = district_rows(df, district)
    out = district_shapley(models, models.background, rows,
                           which=which, n_background=n_bg)
    return out, dict(out.attrs)


phi, meta = _phi(district, which, n_bg)
phi.attrs.update(meta)

row = est[est["district"] == district].iloc[0]

ui.stat_row([
    ("National predicted risk", f"{meta['national_risk']:.2f}%",
     "model on background sample"),
    ("District predicted risk", f"{meta['district_risk']:.2f}%",
     f"observed {row['prevalence']:.1f}%"),
    ("Gap to decompose", f"{meta['gap']:+.2f} pp", "sum of all contributions"),
    ("Efficiency error", f"{meta['efficiency_error']:.2e}",
     "Σφ − gap · should be ≈ 0"),
])

st.markdown("")
left, right = st.columns([1.7, 1], gap="large")

with left:
    st.markdown(f"### From national to {district.title()}")
    st.plotly_chart(charts.shapley_waterfall(phi, height=430),
                    width="stretch")
    st.caption(
        "Red bars push risk up, green pull it down, and the two teal bars are "
        "the endpoints. Because Shapley values satisfy efficiency, the bars sum "
        "exactly to the gap — the residual above is the arithmetic proof."
    )

with right:
    st.markdown("### Contributions")
    show = phi.copy()
    show["share of gap"] = (show["contribution"] / meta["gap"] * 100
                            if abs(meta["gap"]) > 1e-9 else 0)
    show["lever"] = show["feature"].isin(MODIFIABLE).map(
        {True: "modifiable", False: "structural"}
    )
    st.dataframe(
        show[["feature", "contribution", "share of gap", "lever"]].round(2),
        hide_index=True, width="stretch", height=390,
    )

    mod_share = float(phi[phi["feature"].isin(MODIFIABLE)]["contribution"].sum())
    st.markdown(
        f'<div class="stat" style="border-left-color:{SILT}">'
        f'<div class="k">Modifiable share of the gap</div>'
        f'<div class="v">{mod_share:+.2f} pp</div>'
        f'<div class="sub">of {meta["gap"]:+.2f} pp total — the rest is '
        f'structure no health programme can move</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

tab_compare, tab_person, tab_method = st.tabs(
    ["Does the model class matter?", "Build a person", "How the value function works"]
)

with tab_compare:
    st.markdown(
        '<p class="lede">The same decomposition under both model classes. If a '
        'feature swaps sign between them, the finding is an artefact of the '
        'fitting procedure, not a property of the population.</p>',
        unsafe_allow_html=True,
    )
    if st.button("Run the other model and compare"):
        other = "logit" if which == "gbm" else "gbm"
        phi_b, meta_b = _phi(district, other, n_bg)
        merged = phi.merge(phi_b, on="feature", suffixes=("_a", "_b"))
        merged.columns = ["feature", f"{which} (pp)", f"{other} (pp)"]
        merged["sign agrees"] = (
            (merged[f"{which} (pp)"] > 0) == (merged[f"{other} (pp)"] > 0)
        )
        st.dataframe(merged.round(3), hide_index=True, width="stretch")
        n_flip = int((~merged["sign agrees"]).sum())
        if n_flip:
            st.warning(
                f"{n_flip} feature(s) change direction between model classes. "
                f"Do not build a recommendation on those."
            )
        else:
            st.success("All features agree in direction across both models.")

with tab_person:
    st.markdown(
        '<p class="lede">The same machinery on a single hypothetical adult. '
        'Useful for sanity-checking that the model behaves the way clinical '
        'knowledge says it should before you trust it on a district.</p>',
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    profile = {}
    for i, feat in enumerate(models.features):
        opts = sorted(df[feat].dropna().unique())
        profile[feat] = cols[i % 5].selectbox(feat, opts, key=f"p_{feat}")

    if st.button("Explain this person"):
        with st.spinner("Enumerating coalitions…"):
            ip = instance_shapley(models, profile, which=which, n_background=n_bg)
        risk = float(models.predict(pd.DataFrame([profile]), which=which)[0]) * 100
        ui.stat_row([
            ("Predicted risk", f"{risk:.1f}%", "this individual"),
            ("National baseline", f"{ip.attrs['national_risk']:.1f}%", ""),
            ("Difference", f"{ip.attrs['gap']:+.1f} pp", "decomposed below"),
        ])
        st.plotly_chart(charts.shapley_waterfall(ip, height=380),
                        width="stretch")

with tab_method:
    st.markdown(
        """
For a coalition **S** of features, the value function is

```
v(S) = E[ f(x) ]   with features in S drawn from the district's joint
                   distribution, and features outside S from the national one
```

and the Shapley value of feature *i* is the standard weighted average of its
marginal contribution across every coalition that excludes it.

This is the **interventional** formulation (Janzing, Minorics & Blöbaum, AISTATS
2020) rather than the conditional one. The distinction matters here: under the
conditional formulation, a feature that was never touched still receives credit
whenever it correlates with one that was. Since the whole purpose of this page
is to feed an intervention decision, credit has to follow what actually changed.

Three properties make the chart above readable as a decomposition rather than a
ranking:

- **Efficiency** — the contributions sum exactly to the gap. Reported above.
- **Symmetry** — two features with identical marginal contributions get identical values.
- **Null player** — a feature that never changes any prediction gets exactly zero.

Cost is `2^n × B` model evaluations. At n=10 and B=200 that is roughly 205,000
rows per district, batched into a single `predict_proba` call and cached for
thirty minutes.
        """
    )

ui.caveat(
    "<b>Attribution is not causation.</b> A large Shapley value for wealth means "
    "the wealth composition of this district accounts for much of its risk gap "
    "under this model. It does not mean making people poorer would reduce "
    "diabetes. The decomposition describes the model's behaviour on the "
    "observed distribution — nothing more."
)
