"""
Step 5 — turn the analysis into a defensible shortlist.

This page exists because the usability study needs a decision task, and a
decision task needs a decision. The measured question is: "you can fund ten
districts — which ten, and can you justify each one?"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui
from src.loaders import bootstrap, load_estimates, load_profiles
from src.profiles import structural_residual
from src.theme import (
    CURRENT, INK_2, LATERITE, LINE, MUTED, REED, SILT, VIOLET, plotly_layout,
)

ui.page("Priority shortlist")
bootstrap()

est = load_estimates()
prof = load_profiles()

ui.header(
    "step 5 · decide",
    "Ten districts, and a reason for each",
    "Three criteria pull in different directions. Burden says go where "
    "prevalence is highest. Excess says go where prevalence is worse than "
    "structurally similar districts. Confidence says go where the survey "
    "actually supports a claim. Set the weights, watch the shortlist move, and "
    "notice how much of the ranking is a value judgement rather than a finding.",
)

district = ui.district_picker(sorted(est["district"]))

st.sidebar.markdown("### Prioritisation weights")
w_burden = st.sidebar.slider("Burden — raw prevalence", 0.0, 1.0, 0.4, 0.05)
w_excess = st.sidebar.slider("Excess over structural peers", 0.0, 1.0, 0.4, 0.05)
w_conf = st.sidebar.slider("Evidence confidence", 0.0, 1.0, 0.2, 0.05)

n_pick = st.sidebar.slider("Shortlist size", 5, 20, 10)
k_peers = st.sidebar.slider("Peers used for the excess score", 3, 12, 6)
hard_filter = st.sidebar.toggle(
    "Exclude districts with SE ≥ 4.5 pp", value=False,
    help="A hard exclusion rather than a soft weight. Compare the two — "
         "they produce visibly different shortlists, and which one you defend "
         "is a policy stance, not a statistical one.",
)
ui.sidebar_footer()
ui.sidebar_footer()

res = structural_residual(est, prof, k=k_peers,
                          weights=st.session_state.get(ui.STATE_WEIGHTS))
merged = res.merge(est[["district", "prevalence", "ci_low", "ci_high",
                        "division", "n_clusters"]],
                   on="district", how="left")


def unit(s: pd.Series) -> pd.Series:
    """Min-max to [0,1]; a constant column becomes 0.5 rather than nan."""
    lo, hi = s.min(), s.max()
    return pd.Series(0.5, index=s.index) if hi - lo < 1e-9 else (s - lo) / (hi - lo)


merged["s_burden"] = unit(merged["prevalence"])
merged["s_excess"] = unit(merged["residual"])
merged["s_conf"] = 1 - unit(merged["se"].fillna(merged["se"].max()))

wsum = max(w_burden + w_excess + w_conf, 1e-9)
merged["priority"] = (
    w_burden * merged["s_burden"]
    + w_excess * merged["s_excess"]
    + w_conf * merged["s_conf"]
) / wsum

pool = merged[merged["se"] < 4.5] if hard_filter else merged
short = pool.sort_values("priority", ascending=False).head(n_pick)

ui.stat_row([
    ("Shortlist size", f"{len(short)}", f"from {len(pool)} eligible districts"),
    ("Mean prevalence", f"{short['prevalence'].mean():.1f}%",
     f"national {np.average(est['prevalence'], weights=est['n']):.1f}%"),
    ("Mean excess", f"{short['residual'].mean():+.1f} pp", "over structural peers"),
    ("Adults represented", f"{int(short['n'].sum()):,}", "in the survey sample"),
])

st.markdown("")

left, right = st.columns([1.4, 1], gap="large")

with left:
    st.markdown("### Burden against excess")
    st.markdown(
        '<p class="lede">The top-right quadrant is where both criteria agree. '
        'Marker size is sample size, so a large marker in the top right is a '
        'district you can defend on all three counts at once.</p>',
        unsafe_allow_html=True,
    )
    sel = set(short["district"])
    fig = go.Figure()
    fig.add_hline(y=0, line=dict(color=LINE, width=1, dash="dot"))
    fig.add_vline(x=float(np.average(est["prevalence"], weights=est["n"])),
                  line=dict(color=LINE, width=1, dash="dot"))
    fig.add_trace(go.Scatter(
        x=merged["prevalence"], y=merged["residual"], mode="markers",
        marker=dict(
            size=np.clip(merged["n"] / merged["n"].max() * 26, 7, 26),
            color=[SILT if d == district else (CURRENT if d in sel else MUTED)
                   for d in merged["district"]],
            opacity=[1.0 if d in sel or d == district else 0.42
                     for d in merged["district"]],
            line=dict(color=INK_2, width=1),
        ),
        text=merged["district"],
        customdata=np.stack([merged["se"], merged["n"], merged["peer_expected"]],
                            axis=-1),
        hovertemplate=("<b>%{text}</b><br>prevalence %{x:.1f}%<br>"
                       "excess %{y:+.1f} pp (peers expect "
                       "%{customdata[2]:.1f}%)<br>"
                       "±%{customdata[0]:.1f} pp · n=%{customdata[1]:,}"
                       "<extra></extra>"),
        showlegend=False,
    ))
    for _, r in short.head(8).iterrows():
        fig.add_annotation(x=r["prevalence"], y=r["residual"], text=r["district"],
                           showarrow=False, yshift=15,
                           font=dict(size=9, color=CURRENT))
    fig.update_layout(**plotly_layout(
        height=470, xaxis_title="weighted prevalence (%)",
        yaxis_title="excess over structural peers (pp)"))
    st.plotly_chart(fig, width="stretch")

with right:
    st.markdown("### The shortlist")
    disp = short[["district", "division", "prevalence", "residual", "se",
                  "reliability_label", "priority"]].copy()
    disp.columns = ["district", "division", "prev %", "excess", "±SE",
                    "precision", "score"]
    st.dataframe(disp.round(2), hide_index=True, width="stretch",
                 height=430)

    st.download_button(
        "Download shortlist as CSV",
        short.to_csv(index=False).encode(),
        file_name=f"interveneBD_shortlist_{n_pick}.csv",
        mime="text/csv",
    )

st.markdown("---")

st.markdown("### How stable is this shortlist?")
st.markdown(
    '<p class="lede">Re-rank under three defensible weightings and see who '
    'survives all of them. Districts that appear in every column are robust '
    'choices. Districts that appear in one are choices you made with a slider.</p>',
    unsafe_allow_html=True,
)

SCENARIOS = {
    "Burden first": (1.0, 0.0, 0.0),
    "Excess first": (0.0, 1.0, 0.0),
    "Evidence first": (0.2, 0.2, 0.6),
    "Your weights": (w_burden, w_excess, w_conf),
}

lists = {}
for name, (a, b, c) in SCENARIOS.items():
    tot = max(a + b + c, 1e-9)
    score = (a * merged["s_burden"] + b * merged["s_excess"]
             + c * merged["s_conf"]) / tot
    lists[name] = list(merged.assign(s=score)
                       .sort_values("s", ascending=False)
                       .head(n_pick)["district"])

cols = st.columns(len(lists))
everywhere = set.intersection(*(set(v) for v in lists.values()))
for col, (name, items) in zip(cols, lists.items()):
    body = "".join(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.76rem;'
        f'padding:2px 0;color:{"#E8F1EF" if d in everywhere else MUTED}">'
        f'{"● " if d in everywhere else "○ "}{d}</div>'
        for d in items
    )
    col.markdown(
        f'<div class="stat"><div class="k">{name}</div>'
        f'<div style="margin-top:0.5rem">{body}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown(
    f'<p class="lede" style="margin-top:1rem">Filled markers appear in all four '
    f'rankings: <b>{", ".join(sorted(everywhere)) or "none"}</b>. That '
    f'intersection is the part of the shortlist that does not depend on your '
    f'weighting.</p>',
    unsafe_allow_html=True,
)

ui.caveat(
    "<b>The weights are yours, and that is the point.</b> There is no objective "
    "ranking here. Burden, excess and confidence genuinely conflict, and the "
    "tool's job is to make the trade-off visible and reversible rather than to "
    "bake one answer into a score and present it as arithmetic."
)
