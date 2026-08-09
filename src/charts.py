"""
Chart builders. Every figure in the app is constructed here so the visual
grammar stays consistent: one ramp, one set of gridlines, one hover style.

Encoding decisions worth naming
-------------------------------
* The choropleth uses discrete VSUP colours passed per-feature rather than a
  continuous Plotly colorscale. Plotly cannot express a bivariate palette, so
  we bin ourselves and hand it explicit colours. That is why the map's legend
  is hand-drawn as a wedge instead of using `coloraxis`.

* Comparison views lock the y-axis to a shared domain. Letting Plotly
  auto-range each panel is the single most common way a dashboard lies: two
  bars of equal height representing different values.

* Peer comparisons use a fixed role palette — the selected district is always
  `SILT`, its peers always `VIOLET`, the national reference always `MUTED`.
  Colour means role, not identity, and it means the same role on every page.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .theme import (
    CURRENT,
    INK_2,
    INK_3,
    LATERITE,
    LINE,
    MUTED,
    PAPER,
    REED,
    RISK_RAMP,
    SILT,
    VIOLET,
    VSUP,
    plotly_layout,
    plotly_colorscale,
)


# --- map ---------------------------------------------------------------------
def risk_map(estimates: pd.DataFrame, geojson: dict, selected: str | None = None,
             uncertainty_on: bool = True, height: int = 560) -> go.Figure:
    """
    District choropleth. When `uncertainty_on`, colours come from the VSUP and
    unreliable districts visibly dissolve toward the panel background. When
    off, it is a plain sequential map — the comparison is the point of the
    toggle, and the A/B is what the usability study measures.
    """
    lo, hi = float(estimates["prevalence"].min()), float(estimates["prevalence"].max())
    vsup = VSUP((lo, hi))

    if uncertainty_on:
        colors = [vsup.encode(p, se) for p, se in
                  zip(estimates["prevalence"], estimates["se"].fillna(99))]
    else:
        span = max(hi - lo, 1e-9)
        from .theme import ramp_color
        colors = [ramp_color((p - lo) / span) for p in estimates["prevalence"]]

    # Plotly needs a numeric z plus a matching discrete scale; we fake it by
    # giving every district its own scale stop. Ugly, exact, and fast.
    n = len(estimates)
    z = list(range(n))
    scale = []
    for i, c in enumerate(colors):
        scale.append([i / n, c])
        scale.append([(i + 1) / n, c])

    custom = np.stack(
        [
            estimates["prevalence"], estimates["ci_low"], estimates["ci_high"],
            estimates["se"], estimates["n"], estimates["n_clusters"],
            estimates["reliability_label"], estimates["division"],
        ],
        axis=-1,
    )

    fig = go.Figure(
        go.Choropleth(
            geojson=geojson,
            locations=estimates["district"],
            featureidkey="properties.district",
            z=z,
            colorscale=scale,
            showscale=False,
            marker_line_color=INK_2,
            marker_line_width=0.6,
            customdata=custom,
            hovertemplate=(
                "<b>%{location}</b>  <span style='color:#7C9AA3'>%{customdata[7]}</span><br>"
                "prevalence  <b>%{customdata[0]:.1f}%</b><br>"
                "95% CI      %{customdata[1]:.1f} – %{customdata[2]:.1f}<br>"
                "std. error  ±%{customdata[3]:.1f} pp  (%{customdata[6]})<br>"
                "sample      %{customdata[4]:,} adults / %{customdata[5]} clusters"
                "<extra></extra>"
            ),
        )
    )

    if selected and selected in set(estimates["district"]):
        fig.add_trace(
            go.Choropleth(
                geojson=geojson,
                locations=[selected],
                featureidkey="properties.district",
                z=[0],
                colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                showscale=False,
                marker_line_color=SILT,
                marker_line_width=2.4,
                hoverinfo="skip",
            )
        )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="rgba(0,0,0,0)",
        projection_type="mercator",
    )
    fig.update_layout(**plotly_layout(height=height, margin=dict(l=0, r=0, t=0, b=0)))
    return fig


def vsup_legend(estimates: pd.DataFrame, height: int = 210) -> go.Figure:
    """
    The bivariate legend, drawn as a fan.

    Reading it: left to right is prevalence, top to bottom is worsening
    precision. The fan narrows downward because the palette genuinely offers
    fewer distinguishable colours as uncertainty grows. Bottom row is one
    colour: for those districts the survey supports no claim about level at
    all.
    """
    lo, hi = float(estimates["prevalence"].min()), float(estimates["prevalence"].max())
    vsup = VSUP((lo, hi))
    cells = vsup.legend_cells()
    labels = vsup.tier_labels()

    fig = go.Figure()
    n_tiers = len(vsup.bins_by_tier)
    row_h = 1.0 / n_tiers

    for tier, b, n_bins, color in cells:
        # Fan geometry: each row is inset from the edges as tier rises.
        inset = tier * 0.085
        width = (1.0 - 2 * inset) / n_bins
        x0 = inset + b * width
        y1 = 1 - tier * row_h
        fig.add_shape(
            type="rect", x0=x0, x1=x0 + width,
            y0=y1 - row_h * 0.86, y1=y1,
            fillcolor=color, line=dict(color=INK_2, width=1),
        )

    for tier, label in enumerate(labels):
        fig.add_annotation(
            x=-0.02, y=1 - tier * row_h - row_h * 0.43, xref="x", yref="y",
            text=label, showarrow=False, xanchor="right",
            font=dict(family="IBM Plex Mono, monospace", size=9, color=MUTED),
        )

    fig.add_annotation(x=0.5, y=1.14, text="prevalence  →", showarrow=False,
                       font=dict(family="IBM Plex Mono, monospace", size=9.5,
                                 color=MUTED))
    fig.add_annotation(x=0.02, y=-0.13, text=f"{lo:.0f}%", showarrow=False,
                       font=dict(family="IBM Plex Mono, monospace", size=9,
                                 color=MUTED))
    fig.add_annotation(x=0.98, y=-0.13, text=f"{hi:.0f}%", showarrow=False,
                       font=dict(family="IBM Plex Mono, monospace", size=9,
                                 color=MUTED))

    fig.update_xaxes(visible=False, range=[-0.42, 1.05])
    fig.update_yaxes(visible=False, range=[-0.2, 1.25])
    fig.update_layout(**plotly_layout(height=height,
                                      margin=dict(l=0, r=0, t=18, b=0)))
    return fig


# --- ranked estimates --------------------------------------------------------
def caterpillar(estimates: pd.DataFrame, highlight: str | None = None,
                top: int | None = None, height: int = 620) -> go.Figure:
    """
    Ranked point-and-interval plot. This is the honest companion to the map:
    the map shows where, this shows how confidently.

    Intervals are drawn, not implied. If two districts' intervals overlap, the
    reader can see immediately that their ranking is not established — which
    is the case for most adjacent pairs here, and is exactly the finding a
    naive ranked bar chart would hide.
    """
    d = estimates.sort_values("prevalence")
    if top:
        d = pd.concat([d.head(top // 2), d.tail(top // 2)])

    colors = [SILT if x == highlight else CURRENT for x in d["district"]]
    widths = [3.0 if x == highlight else 1.4 for x in d["district"]]

    fig = go.Figure()
    for _, r in d.iterrows():
        c = SILT if r["district"] == highlight else LINE
        fig.add_trace(go.Scatter(
            x=[r["ci_low"], r["ci_high"]], y=[r["district"], r["district"]],
            mode="lines", line=dict(color=c, width=2.2 if r["district"] == highlight else 1.4),
            hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=d["prevalence"], y=d["district"], mode="markers",
        marker=dict(size=[9 if x == highlight else 6 for x in d["district"]],
                    color=colors, line=dict(color=INK_2, width=1)),
        customdata=np.stack([d["ci_low"], d["ci_high"], d["n"],
                             d["reliability_label"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>%{x:.1f}%  "
                       "(%{customdata[0]:.1f} – %{customdata[1]:.1f})<br>"
                       "n=%{customdata[2]:,} · %{customdata[3]} precision"
                       "<extra></extra>"),
        showlegend=False,
    ))

    national = np.average(estimates["prevalence"], weights=estimates["n"])
    fig.add_vline(x=national, line=dict(color=MUTED, width=1, dash="dot"),
                  annotation_text=f"national {national:.1f}%",
                  annotation_position="top",
                  annotation_font=dict(color=MUTED, size=10))

    fig.update_layout(**plotly_layout(
        height=height,
        xaxis_title="weighted prevalence (%)",
        yaxis=dict(gridcolor="rgba(0,0,0,0)", tickfont=dict(size=10)),
    ))
    return fig


# --- peer comparison ---------------------------------------------------------
def parallel_profile(profiles: pd.DataFrame, target: str, peer_names: list[str],
                     height: int = 430) -> go.Figure:
    """
    Parallel coordinates over the profile axes.

    Each axis is independently range-normalised to [0, 1] across all 64
    districts, so a line's height on an axis is that district's percentile-like
    position, and the raw value is in the hover. Without normalisation the
    urban-share axis alone would flatten everything else.
    """
    axes = [c for c in profiles.columns if c != "division"]
    X = profiles[axes]
    mn, mx = X.min(), X.max()
    norm = (X - mn) / (mx - mn).replace(0, 1)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=axes, y=norm.mean().values, mode="lines",
        line=dict(color=MUTED, width=1.6, dash="dot"),
        name="national mean", hoverinfo="skip",
    ))

    for p in peer_names:
        if p not in norm.index:
            continue
        fig.add_trace(go.Scatter(
            x=axes, y=norm.loc[p].values, mode="lines",
            line=dict(color=VIOLET, width=1.5), opacity=0.65, name=p,
            customdata=X.loc[p].values,
            hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{customdata:.1f}%<extra></extra>",
        ))

    if target in norm.index:
        fig.add_trace(go.Scatter(
            x=axes, y=norm.loc[target].values, mode="lines+markers",
            line=dict(color=SILT, width=3.2),
            marker=dict(size=7, color=SILT, line=dict(color=INK_2, width=1)),
            name=target, customdata=X.loc[target].values,
            hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{customdata:.1f}%<extra></extra>",
        ))

    fig.update_layout(**plotly_layout(
        height=height,
        yaxis=dict(title="position within national range", range=[-0.05, 1.05],
                   showticklabels=False, gridcolor=LINE),
        xaxis=dict(tickangle=-32, tickfont=dict(size=10), gridcolor="rgba(0,0,0,0)"),
        legend=dict(orientation="h", y=-0.34, font=dict(size=10)),
    ))
    return fig


def peer_dumbbell(estimates: pd.DataFrame, target: str, peer_names: list[str],
                  height: int = 300) -> go.Figure:
    """Target vs each peer, with intervals. The gap is the whole story."""
    e = estimates.set_index("district")
    order = [target] + [p for p in peer_names if p in e.index]
    fig = go.Figure()
    for name in order:
        r = e.loc[name]
        c = SILT if name == target else VIOLET
        fig.add_trace(go.Scatter(
            x=[r["ci_low"], r["ci_high"]], y=[name, name], mode="lines",
            line=dict(color=c, width=1.6), opacity=0.55,
            hoverinfo="skip", showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            x=[r["prevalence"]], y=[name], mode="markers",
            marker=dict(size=11 if name == target else 8, color=c,
                        line=dict(color=INK_2, width=1)),
            hovertemplate=f"<b>{name}</b><br>%{{x:.1f}}%<extra></extra>",
            showlegend=False,
        ))

    peer_mean = e.reindex([p for p in peer_names if p in e.index])["prevalence"].mean()
    fig.add_vline(x=peer_mean, line=dict(color=VIOLET, width=1, dash="dash"),
                  annotation_text=f"peer mean {peer_mean:.1f}%",
                  annotation_font=dict(color=VIOLET, size=10))
    fig.update_layout(**plotly_layout(height=height,
                                      xaxis_title="weighted prevalence (%)",
                                      yaxis=dict(gridcolor="rgba(0,0,0,0)")))
    return fig


# --- attribution -------------------------------------------------------------
def shapley_waterfall(phi: pd.DataFrame, height: int = 400) -> go.Figure:
    """
    Waterfall from national risk to district risk.

    A waterfall rather than a bar chart because the quantity being decomposed
    is a *difference*, and the reader needs to see the parts summing back to
    the whole. The efficiency check printed beside it is what makes that claim
    checkable.
    """
    national = phi.attrs.get("national_risk", 0.0)
    district = phi.attrs.get("district_risk", 0.0)

    d = phi.sort_values("contribution", ascending=False)
    measures = ["absolute"] + ["relative"] * len(d) + ["total"]
    x = ["National"] + list(d["feature"]) + ["District"]
    y = [national] + list(d["contribution"]) + [district]

    fig = go.Figure(go.Waterfall(
        orientation="v", measure=measures, x=x, y=y,
        connector=dict(line=dict(color=LINE, width=1)),
        increasing=dict(marker=dict(color=LATERITE)),
        decreasing=dict(marker=dict(color=REED)),
        totals=dict(marker=dict(color=CURRENT)),
        text=[f"{v:+.2f}" if m == "relative" else f"{v:.1f}"
              for v, m in zip(y, measures)],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono, monospace", size=10, color=MUTED),
        hovertemplate="<b>%{x}</b><br>%{y:+.2f} pp<extra></extra>",
    ))
    fig.update_layout(**plotly_layout(
        height=height, yaxis_title="predicted risk (pp)",
        xaxis=dict(tickangle=-30, tickfont=dict(size=10),
                   gridcolor="rgba(0,0,0,0)"),
    ))
    return fig


# --- intervention ------------------------------------------------------------
def tornado(table: pd.DataFrame, height: int = 330) -> go.Figure:
    """Interventions ranked by expected reduction, with bootstrap intervals."""
    d = table.sort_values("reduction")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d["reduction"], y=d["label"], orientation="h",
        marker=dict(color=[REED if v > 0 else LATERITE for v in d["reduction"]],
                    line=dict(color=INK_2, width=1)),
        error_x=dict(type="data", symmetric=False,
                     array=(d["hi"] - d["reduction"]).clip(lower=0),
                     arrayminus=(d["reduction"] - d["lo"]).clip(lower=0),
                     color=MUTED, thickness=1.2, width=4),
        customdata=np.stack([d["eligible"], d["reached"],
                             d["per_1000_reached"]], axis=-1),
        hovertemplate=("<b>%{y}</b><br>reduction %{x:.2f} pp<br>"
                       "eligible %{customdata[0]:,} · reached %{customdata[1]:,}<br>"
                       "%{customdata[2]:.2f} pp per 1,000 reached<extra></extra>"),
        showlegend=False,
    ))
    fig.add_vline(x=0, line=dict(color=LINE, width=1))
    fig.update_layout(**plotly_layout(
        height=height, xaxis_title="reduction in predicted prevalence (pp)",
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    ))
    return fig


def before_after(result: dict, height: int = 240) -> go.Figure:
    """Two bars, one shared axis, interval on the change. Deliberately plain."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Current", "After programme"],
        y=[result["baseline"], result["post"]],
        marker=dict(color=[MUTED, CURRENT], line=dict(color=INK_2, width=1)),
        text=[f"{result['baseline']:.2f}%", f"{result['post']:.2f}%"],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono, monospace", color=PAPER, size=12),
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>", showlegend=False,
    ))
    top = max(result["baseline"], result["post"])
    fig.update_layout(**plotly_layout(
        height=height, yaxis=dict(title="predicted prevalence (%)",
                                  range=[0, top * 1.25], gridcolor=LINE),
        xaxis=dict(gridcolor="rgba(0,0,0,0)"),
    ))
    return fig


# --- national context --------------------------------------------------------
def subgroup_strip(df: pd.DataFrame, title: str, height: int = 230) -> go.Figure:
    """Weighted prevalence by subgroup level, with intervals."""
    d = df.sort_values("prevalence")
    fig = go.Figure()
    for _, r in d.iterrows():
        fig.add_trace(go.Scatter(
            x=[r["ci_low"], r["ci_high"]], y=[r["level"], r["level"]],
            mode="lines", line=dict(color=LINE, width=1.6),
            hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=d["prevalence"], y=d["level"], mode="markers",
        marker=dict(size=10, color=CURRENT, line=dict(color=INK_2, width=1)),
        customdata=d["n"],
        hovertemplate="<b>%{y}</b><br>%{x:.1f}%  ·  n=%{customdata:,}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(**plotly_layout(
        height=height, title=dict(text=title, font=dict(size=12, color=MUTED)),
        xaxis_title="", yaxis=dict(gridcolor="rgba(0,0,0,0)"),
    ))
    return fig


def calibration_plot(reliability: list[dict], height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 0.6], y=[0, 0.6], mode="lines",
                             line=dict(color=LINE, width=1, dash="dash"),
                             name="perfect", hoverinfo="skip"))
    if reliability:
        d = pd.DataFrame(reliability)
        fig.add_trace(go.Scatter(
            x=d["predicted"], y=d["observed"], mode="lines+markers",
            line=dict(color=CURRENT, width=2),
            marker=dict(size=np.clip(d["n"] / d["n"].max() * 16, 5, 16),
                        color=CURRENT, line=dict(color=INK_2, width=1)),
            name="observed",
            hovertemplate="predicted %{x:.3f}<br>observed %{y:.3f}<extra></extra>",
        ))
    fig.update_layout(**plotly_layout(
        height=height, xaxis_title="predicted probability",
        yaxis_title="observed frequency",
        legend=dict(orientation="h", y=1.1),
    ))
    return fig
