"""
Step 2 — compare a district against its structural twins.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src import charts, ui
from src.config import PROFILE_AXES
from src.loaders import bootstrap, load_estimates, load_profiles
from src.profiles import gower_distance, peers, structural_residual
from src.theme import LATERITE, MUTED, REED, SILT, VIOLET

ui.page("Peer districts")
bootstrap()

est = load_estimates()
prof = load_profiles()

ui.header(
    "step 2 · compare",
    "Is this district unusual, or just unusually urban?",
    "Ranking districts by prevalence finds the biggest burden. It does not find "
    "the most surprising one. Here every district is placed in a ten-dimensional "
    "profile space — wealth mix, education, urbanisation, age structure, "
    "behaviour — and compared against the districts that most resemble it. The "
    "gap between a district and its structural twins is the part its composition "
    "does not explain.",
)

district = ui.district_picker(sorted(est["district"]))

st.sidebar.markdown("### Similarity")
k = st.sidebar.slider("Number of peer districts", 3, 12, 6)
within_div = st.sidebar.toggle(
    "Restrict peers to same division", value=False,
    help="Off by default: the most informative twin is often in another "
         "division entirely.",
)

with st.sidebar.expander("What counts as similar", expanded=False):
    st.caption(
        "Each axis is normalised to its national range, then weighted. Set an "
        "axis to zero to exclude it. This is Gower's coefficient — the weights "
        "are the definition of similarity, so they belong to you, not to me."
    )
    preset = st.radio("Preset", ["Balanced", "Socioeconomic only",
                                 "Behavioural only", "Custom"], index=0)

axis_labels = [a[0] for a in PROFILE_AXES]
PRESETS = {
    "Balanced": {a: 1.0 for a in axis_labels},
    "Socioeconomic only": {
        a: (1.0 if a in {"Urban share", "Poorest quintile", "Richest quintile",
                         "No education", "Secondary+", "Aged 35-64"} else 0.0)
        for a in axis_labels
    },
    "Behavioural only": {
        a: (1.0 if a in {"Overweight/obese", "Hypertensive",
                         "Physically active", "Current smoker"} else 0.0)
        for a in axis_labels
    },
}

if preset == "Custom":
    weights = {}
    with st.sidebar.expander("Axis weights", expanded=True):
        for a in axis_labels:
            weights[a] = st.slider(a, 0.0, 3.0, 1.0, 0.25, key=f"w_{a}")
else:
    weights = PRESETS[preset]

st.session_state[ui.STATE_WEIGHTS] = weights
ui.sidebar_footer()
ui.sidebar_footer()

nb = peers(prof, district, k=k, weights=weights, within_division=within_div)
dist = gower_distance(prof, district, weights)

e = est.set_index("district")
own = e.loc[district]
peer_prev = e.reindex(nb)["prevalence"]
gap = float(own["prevalence"] - peer_prev.mean())

ui.stat_row([
    ("District", district.title(), f"{own['division']} division"),
    ("Observed", f"{own['prevalence']:.1f}%",
     f"95% CI {own['ci_low']:.1f} – {own['ci_high']:.1f}"),
    ("Peer expectation", f"{peer_prev.mean():.1f}%",
     f"mean of {len(peer_prev)} structural twins"),
    ("Unexplained gap", f"{gap:+.1f} pp",
     "worse than its twins" if gap > 0 else "better than its twins"),
])

if abs(gap) < own["se"]:
    ui.caveat(
        f"That gap of {gap:+.1f} pp is smaller than this district's own standard "
        f"error of ±{own['se']:.1f} pp. Treat it as noise until a larger sample "
        f"says otherwise."
    )

st.markdown("")
left, right = st.columns([1.55, 1], gap="large")

with left:
    st.markdown("### Structural profile against peers")
    st.plotly_chart(
        charts.parallel_profile(prof, district, nb, height=460),
        width="stretch",
    )
    st.caption(
        f"Gold is {district.title()}. Violet lines are its {len(nb)} nearest "
        f"districts in profile space. Where gold departs from the violet band, "
        f"the similarity match is weak on that axis."
    )

with right:
    st.markdown("### Outcome against peers")
    st.plotly_chart(charts.peer_dumbbell(est, district, nb, height=330),
                    width="stretch")
    st.markdown("###### Closest matches")
    table = pd.DataFrame({
        "district": nb,
        "distance": [round(float(dist[p]), 3) for p in nb],
        "prevalence": [round(float(e.loc[p, "prevalence"]), 1) for p in nb],
        "n": [int(e.loc[p, "n"]) for p in nb],
    })
    st.dataframe(table, hide_index=True, width="stretch")

st.markdown("---")
st.markdown("### National shortlist: where structure fails to explain the burden")
st.markdown(
    '<p class="lede">Every district scored the same way, ranked by how far it '
    'sits above its own peer expectation. This is the list a targeting exercise '
    'should start from — not the raw prevalence ranking, which mostly '
    'rediscovers that cities are wealthier.</p>',
    unsafe_allow_html=True,
)

res = structural_residual(est, prof, k=k, weights=weights)

c1, c2 = st.columns([1, 1], gap="large")
with c1:
    st.markdown("###### Worse than their twins")
    st.dataframe(
        res.head(12)[["district", "observed", "peer_expected", "residual",
                      "reliability_label", "n"]]
        .rename(columns={"reliability_label": "precision"})
        .round(1),
        hide_index=True, width="stretch",
    )
with c2:
    st.markdown("###### Better than their twins")
    st.dataframe(
        res.tail(12).iloc[::-1][["district", "observed", "peer_expected",
                                 "residual", "reliability_label", "n"]]
        .rename(columns={"reliability_label": "precision"})
        .round(1),
        hide_index=True, width="stretch",
    )

hi_conf = res[(res["residual"] > 0) & (res["se"] < 4.5)].head(3)
if not hi_conf.empty:
    names = ", ".join(f"{r.district.title()} ({r.residual:+.1f} pp)"
                      for r in hi_conf.itertuples())
    ui.caveat(
        f"<b>Surviving the precision filter:</b> {names}. These are the "
        f"positive residuals that are not explained away by a wide interval. "
        f"Everything else on the left-hand list needs more data before it "
        f"justifies a budget line."
    )

ui.caveat(
    "<b>What a residual is not.</b> A positive residual says a district does "
    "worse than districts that look like it on ten measured axes. It does not "
    "identify a cause. Unmeasured differences — diet, health service coverage, "
    "diagnostic intensity — sit entirely inside this residual, and diagnostic "
    "intensity in particular can make a better-served district look sicker."
)
