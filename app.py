"""
InterveneBD — entry page.

Where the burden is, and how much of what you are looking at is real.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from src import charts, ui
from src.loaders import bootstrap, load_estimates, load_geojson, load_survey
from src.survey import subgroup_estimates
from src.theme import CURRENT, LATERITE, MUTED, SILT

ui.page("Risk map")
bootstrap()

est = load_estimates()
geo = load_geojson()

ui.header(
    "step 1 · locate",
    "Diabetes risk across 64 districts",
    "Bangladesh's district diabetes prevalence spans a sevenfold range. Some of "
    "that spread is real and some of it is survey noise, and a plain choropleth "
    "cannot tell you which is which. This map encodes both at once: colour "
    "carries prevalence, and districts the survey cannot pin down lose their "
    "colour rather than pretending to a precision they do not have.",
)

# --- sidebar controls --------------------------------------------------------
st.sidebar.markdown("### Map controls")

divisions = sorted(est["division"].unique())
picked_div = st.sidebar.multiselect("Divisions", divisions, default=divisions)

max_se = st.sidebar.slider(
    "Hide districts less precise than ±", 1.0, 12.0, 12.0, 0.5,
    format="%.1f pp",
    help="Filters on the bootstrap standard error. Drag left to keep only "
         "districts whose estimate is well supported.",
)

uncertainty_on = st.sidebar.toggle(
    "Value-suppressing palette", value=True,
    help="Off = conventional sequential choropleth. On = uncertainty suppresses "
         "colour resolution. Switching between the two is the comparison the "
         "usability study measures.",
)

view = st.sidebar.radio("Ranking view", ["All districts", "Top and bottom 20"],
                        horizontal=False)

shown = est[est["division"].isin(picked_div) & (est["se"].fillna(99) <= max_se)]
if shown.empty:
    st.warning("No districts match those filters. Widen the precision cutoff.")
    st.stop()

district = ui.district_picker(sorted(est["district"]))
ui.sidebar_footer()
ui.sidebar_footer()

# --- headline numbers --------------------------------------------------------
national = float(np.average(est["prevalence"], weights=est["n"]))
worst = est.iloc[0]
best = est.iloc[-1]
unreliable = int((est["se"] >= 4.5).sum())

ui.stat_row([
    ("National prevalence", f"{national:.1f}%", "survey-weighted, all adults"),
    ("Highest district", f"{worst['prevalence']:.1f}%",
     f"{worst['district']} · ±{worst['se']:.1f} pp"),
    ("Lowest district", f"{best['prevalence']:.1f}%",
     f"{best['district']} · ±{best['se']:.1f} pp"),
    ("Thin evidence", f"{unreliable} of 64",
     "districts with SE ≥ 4.5 pp"),
])

st.markdown("")

# --- map + legend ------------------------------------------------------------
left, right = st.columns([2.35, 1], gap="large")

with left:
    fig = charts.risk_map(shown, geo, selected=district,
                          uncertainty_on=uncertainty_on)
    event = st.plotly_chart(
        fig, width="stretch", key="risk_map",
        on_select="rerun", selection_mode="points",
    )
    clicked = ui.selected_from_plotly(event, shown["district"].tolist())
    if clicked and clicked != district:
        ui.set_district(clicked)
        st.rerun()
    st.caption("Click a district to carry it through every other page.")

with right:
    if uncertainty_on:
        st.markdown("###### How to read the colour")
        st.plotly_chart(charts.vsup_legend(shown), width="stretch",
                        config={"displayModeBar": False})
        st.markdown(
            f'<p class="lede" style="font-size:0.83rem">Across the top row the '
            f'survey supports eight distinguishable levels. By the bottom row it '
            f'supports one. A grey district is not low risk — it is a district '
            f'we have not measured well enough to place.</p>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown("###### Conventional palette")
        st.markdown(
            f'<p class="lede" style="font-size:0.83rem">Every district now reads '
            f'as equally certain. Compare Munshiganj (n=72) against Dhaka '
            f'(n=692): identical visual authority, six times the evidence behind '
            f'one of them. This is the failure mode the other palette exists to '
            f'prevent.</p>',
            unsafe_allow_html=True,
        )

    row = est[est["district"] == district].iloc[0]
    st.markdown("###### In focus")
    st.markdown(
        f'<div class="stat" style="border-left-color:{SILT}">'
        f'<div class="k">{row["division"]} division</div>'
        f'<div class="v">{row["district"].title()}</div>'
        f'<div class="sub">{row["prevalence"]:.1f}%  '
        f'({row["ci_low"]:.1f} – {row["ci_high"]:.1f})<br>'
        f'n={int(row["n"]):,} · {int(row["n_clusters"])} clusters · '
        f'{row["reliability_label"].lower()} precision</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# --- ranking + national context ---------------------------------------------
tab_rank, tab_groups, tab_design = st.tabs(
    ["Ranked estimates", "National gradients", "Why the intervals are wide"]
)

with tab_rank:
    st.markdown(
        '<p class="lede">The map answers "where". This answers "how sure". '
        'Most adjacent pairs have overlapping intervals, which means their '
        'ranking is not established by this survey — a ranked bar chart would '
        'have hidden that.</p>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        charts.caterpillar(shown, highlight=district,
                           top=40 if view.startswith("Top") else None,
                           height=720 if not view.startswith("Top") else 560),
        width="stretch",
    )

with tab_groups:
    df = load_survey()
    st.markdown(
        '<p class="lede">Before reaching for geography, look at the social '
        'gradients. Diabetes in Bangladesh tracks wealth and urbanisation '
        'strongly — which is precisely why a district-level map needs the peer '
        'comparison on the next page to be interpretable at all.</p>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.subgroup_strip(
            subgroup_estimates(df, "Wealth_Index"), "By wealth quintile"),
            width="stretch")
        st.plotly_chart(charts.subgroup_strip(
            subgroup_estimates(df, "Age"), "By age band"),
            width="stretch")
    with c2:
        st.plotly_chart(charts.subgroup_strip(
            subgroup_estimates(df, "Education"), "By education"),
            width="stretch")
        st.plotly_chart(charts.subgroup_strip(
            subgroup_estimates(df, "Residence"), "By residence"),
            width="stretch")

with tab_design:
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("##### The design effect, district by district")
        st.markdown(
            '<p class="lede">BDHS samples enumeration areas, then households '
            'inside them. People in one area resemble each other, so each extra '
            'respondent adds less information than an independent draw would. '
            'The design effect is how much variance that costs. Treating the '
            'sample as if it were simple random — the default in almost every '
            'student dashboard — understates the standard error by the square '
            'root of this number.</p>',
            unsafe_allow_html=True,
        )
        med_deff = float(est["deff"].median())
        ui.stat_row([
            ("Median design effect", f"{med_deff:.2f}",
             f"SE understated by {np.sqrt(med_deff):.2f}× if ignored"),
            ("Median SE", f"±{est['se'].median():.2f} pp", "cluster bootstrap"),
        ])
    with c2:
        st.markdown("##### Sample size against precision")
        import plotly.graph_objects as go
        from src.theme import LINE, plotly_layout
        f = go.Figure()
        f.add_trace(go.Scatter(
            x=est["n"], y=est["se"], mode="markers+text",
            marker=dict(size=9, color=[SILT if d == district else CURRENT
                                       for d in est["district"]],
                        line=dict(color="#122730", width=1)),
            text=[d if (d == district or n < 90) else "" for d, n in
                  zip(est["district"], est["n"])],
            textposition="top center",
            textfont=dict(size=9, color=MUTED),
            hovertemplate="<b>%{text}</b><br>n=%{x:,} · SE ±%{y:.2f} pp<extra></extra>",
        ))
        f.add_hline(y=4.5, line=dict(color=LATERITE, width=1, dash="dot"),
                    annotation_text="low-precision threshold",
                    annotation_font=dict(color=LATERITE, size=10))
        f.update_layout(**plotly_layout(height=380,
                                        xaxis_title="respondents in district",
                                        yaxis_title="standard error (pp)"))
        st.plotly_chart(f, width="stretch")

st.markdown("---")
ui.caveat(
    "<b>Read this before quoting any number here.</b> These are cross-sectional "
    "2022 survey estimates of self-reported diagnosed diabetes. They describe "
    "the surveyed population, not clinical incidence, and they carry no time "
    "dimension — nothing on this page can tell you whether a district is "
    "improving or worsening."
)
