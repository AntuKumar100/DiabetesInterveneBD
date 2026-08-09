"""
Step 4 — simulate a programme and see what it is worth, with its uncertainty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src import charts, ui
from src.config import INTERVENTIONS
from src.counterfactual import eligible_mask, leverage_table, simulate
from src.loaders import bootstrap, district_rows, load_estimates, load_models, load_survey
from src.theme import CURRENT, LATERITE, MUTED, REED, SILT

ui.page("Intervention lab")
bootstrap()

est = load_estimates()
df = load_survey()
models = load_models()

ui.header(
    "step 4 · simulate",
    "What would a programme actually buy you here?",
    "Pick the levers and the coverage you can realistically fund. Every eligible "
    "person reached is moved to the target state, everyone is re-scored, and the "
    "district's weighted mean predicted risk is recomputed. The interval is a "
    "real one: each replicate resamples the district's survey clusters and "
    "re-draws who the programme reaches, so a small district gets a wide band "
    "instead of a falsely crisp number.",
)

district = ui.district_picker(sorted(est["district"]))
which = ui.model_picker()
rows = district_rows(df, district)

st.sidebar.markdown("### Programme design")
plan: dict[str, float] = {}
for factor, spec in INTERVENTIONS.items():
    n_elig = int(eligible_mask(rows, factor).sum())
    plan[factor] = st.sidebar.slider(
        spec["label"], 0, 100, 0, 5, format="%d%%",
        help=f"{spec['note']}  Eligible in this district: {n_elig} of {len(rows)}.",
        key=f"cov_{factor}",
    ) / 100.0

n_rep = st.sidebar.select_slider("Bootstrap replicates", [40, 80, 120, 200],
                                 value=120)
ui.sidebar_footer()
ui.sidebar_footer()


active = {k: v for k, v in plan.items() if v > 0}

row = est[est["district"] == district].iloc[0]
ui.stat_row([
    ("District", district.title(), f"{int(row['n']):,} adults surveyed"),
    ("Observed prevalence", f"{row['prevalence']:.1f}%",
     f"±{row['se']:.1f} pp"),
    ("Clusters", f"{int(row['n_clusters'])}",
     "resampled in every replicate"),
    ("Levers active", f"{len(active)} of {len(INTERVENTIONS)}",
     "set coverage in the sidebar"),
])

st.markdown("")

if not active:
    st.info(
        "Set a coverage above zero on at least one lever to run a simulation. "
        "The leverage ranking below runs regardless and is the fastest way to "
        "see which lever is worth funding here."
    )
    result = None
else:
    with st.spinner("Running replicates…"):
        result = simulate(models, rows, active, which=which, n_rep=n_rep)

if result:
    left, right = st.columns([1, 1.35], gap="large")

    with left:
        st.markdown("### Before and after")
        st.plotly_chart(charts.before_after(result, height=280),
                        width="stretch")

    with right:
        st.markdown("### What the model says you bought")
        sig = result["certain"]
        colour = REED if sig else SILT
        st.markdown(
            f'<div class="stat" style="border-left-color:{colour}">'
            f'<div class="k">Absolute reduction</div>'
            f'<div class="v">{result["reduction"]:.2f} pp</div>'
            f'<div class="sub">95% interval '
            f'{result["reduction_lo"]:.2f} to {result["reduction_hi"]:.2f} pp · '
            f'{result["relative"]:.1f}% relative</div></div>',
            unsafe_allow_html=True,
        )
        reached = ", ".join(
            f"{INTERVENTIONS[k]['label'].lower()} {v:,}"
            for k, v in result["reached"].items()
        )
        st.markdown(
            f'<p class="lede" style="margin-top:0.8rem">People reached in the '
            f'survey sample: {reached}.</p>',
            unsafe_allow_html=True,
        )

        if not sig:
            ui.caveat(
                "The interval crosses zero. On this district's sample, a "
                "programme at this coverage cannot be distinguished from doing "
                "nothing. That is a finding about the evidence, not a reason to "
                "hide the result."
            )

    st.markdown("---")

st.markdown("### Which lever, at equal coverage")
cov_bench = st.slider("Benchmark coverage applied to every lever", 10, 100, 50, 5,
                      format="%d%%") / 100.0

with st.spinner("Scoring each lever…"):
    lev = leverage_table(models, rows, coverage=cov_bench, which=which, n_rep=60)

c1, c2 = st.columns([1.4, 1], gap="large")
with c1:
    st.plotly_chart(charts.tornado(lev, height=340), width="stretch")
    st.caption(
        f"Each bar is that lever alone at {cov_bench:.0%} coverage, with its "
        f"cluster-bootstrap interval. Bars are not additive — a person eligible "
        f"for two programmes gets counted once by each."
    )
with c2:
    st.markdown("###### Ranked by efficiency, not total")
    disp = lev.copy()
    disp = disp[["label", "eligible", "reached", "reduction", "per_1000_reached"]]
    disp.columns = ["programme", "eligible", "reached", "pp saved",
                    "pp per 1,000 reached"]
    st.dataframe(disp.round(3).sort_values("pp per 1,000 reached",
                                           ascending=False),
                 hide_index=True, width="stretch")
    st.caption(
        "Total reduction rewards whichever lever has the most eligible people. "
        "Reduction per thousand reached is the number a budget holder needs."
    )

negatives = lev[lev["reduction"] < -0.05]
if not negatives.empty:
    names = ", ".join(negatives["label"].str.lower())
    ui.caveat(
        f"<b>A lever is pointing the wrong way: {names}.</b> Taken literally, "
        f"the model says increasing this would raise predicted risk. Do not "
        f"report that as a finding — it is almost certainly reverse causation. "
        f"In a cross-section, people already diagnosed with diabetes are the "
        f"people most likely to have been told to exercise and to have lost "
        f"weight, so the exposure carries the diagnosis rather than preventing "
        f"it. This is the clearest evidence on the whole dashboard that these "
        f"simulations are associational, and it is displayed rather than "
        f"suppressed for exactly that reason."
    )

st.markdown("---")

with st.expander("Compare a district against the national programme", expanded=False):
    st.markdown(
        '<p class="lede">Same plan, applied nationally and to this district. If '
        'the district gains far more than the country does, the lever is '
        'well-matched to local composition — which is the entire argument for '
        'targeting rather than a uniform national rollout.</p>',
        unsafe_allow_html=True,
    )
    if active and st.button("Run national comparison"):
        with st.spinner("Simulating nationally…"):
            nat = simulate(models, df, active, which=which, n_rep=60)
        comp = pd.DataFrame([
            {"scope": "Nationwide", "baseline": nat["baseline"],
             "after": nat["post"], "reduction": nat["reduction"],
             "relative %": nat["relative"]},
            {"scope": district.title(), "baseline": result["baseline"],
             "after": result["post"], "reduction": result["reduction"],
             "relative %": result["relative"]},
        ])
        st.dataframe(comp.round(2), hide_index=True, width="stretch")
        delta = result["relative"] - nat["relative"]
        st.markdown(
            f'<p class="lede">Targeting premium: <b>{delta:+.1f}</b> percentage '
            f'points of relative reduction versus the national average.</p>',
            unsafe_allow_html=True,
        )
    elif not active:
        st.caption("Set a coverage above zero first.")

ui.caveat(
    "<b>The standing caveat, restated because this page is the one people "
    "screenshot.</b> BDHS 2022 is a single cross-section. The model learns "
    "association, and this simulator moves people between observed categories "
    "and re-scores them. It answers 'what would a population that looks like "
    "this instead score?' — never 'what would happen if we ran this programme?' "
    "Turning the first into the second needs no-unmeasured-confounding, "
    "positivity, and consistency, and this data can test none of them."
)
