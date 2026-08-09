"""
Design tokens and the value-suppressing uncertainty palette.

Palette rationale
-----------------
The subject is a river delta, and the tool is about silt-level detail on a
national map. So the ramp is built from delta materials rather than a stock
colormap: standing water at the low end, river current through the middle,
exposed silt above that, and laterite brick — the material every rural clinic
in Bangladesh is built from — at the top. It reads as a landscape, which is
the point: a planner should feel they are looking at terrain, not at a
spreadsheet that happens to be coloured.

Lightness rises monotonically across the ramp so it survives greyscale
printing, and the hue path is blue-green -> gold -> red, which stays separable
under deuteranopia and protanopia. That is verified, not asserted: run
`python -m src.theme` to print the simulated ramp and the minimum perceptual
gap between adjacent stops.

VSUP
----
The map uses a value-suppressing uncertainty palette (Correll, Moritz & Heer,
CHI 2018). Ordinary practice is to encode uncertainty on a second channel —
opacity, texture — which readers reliably ignore. A VSUP instead removes the
ability to make fine distinctions where the data cannot support them: at high
uncertainty the palette collapses to fewer, greyer bins, so an unreliable
district *cannot* be read as precisely high or precisely low. You lose
resolution exactly where resolution was fictional.

Concretely: four uncertainty tiers with 8 / 4 / 2 / 1 value bins respectively,
and progressive desaturation toward the panel background.
"""

from __future__ import annotations

import numpy as np

# --- core tokens -------------------------------------------------------------
INK = "#0C1B22"        # page background — deep riverbed
INK_2 = "#122730"      # panel
INK_3 = "#1A3A45"      # raised panel / input background
LINE = "#24505C"       # borders, gridlines
PAPER = "#E8F1EF"      # primary text
MUTED = "#7C9AA3"      # secondary text, axis labels
CURRENT = "#35C4B5"    # primary interactive — river current
SILT = "#E8B44A"       # secondary accent / caution
LATERITE = "#DC5B3E"   # high risk
REED = "#63C08A"       # improvement, reduction
VIOLET = "#9A7BD1"     # peer districts, comparison series

# Sequential risk ramp, low -> high. Six stops that walk the delta from the
# bottom of a channel up onto a dry sandbar. Luminance rises at every step
# (verified below), which is what makes it survive greyscale and dichromacy.
#
# First draft ended on saturated laterite #DC5B3E. It looked better in
# isolation and failed the test: luminance dropped from the gold stop, so the
# two highest risk bands collapsed under deuteranopia to a gap of 26/255.
# Ending on exposed sand instead raised that gap to 53/255. The saturated
# laterite survives as a categorical alert colour, where it has no ordering to
# break.
RISK_RAMP = [
    "#0E3038",  # standing water
    "#15637A",  # deep channel
    "#22A08F",  # current
    "#79C07C",  # reed bank
    "#E6B24A",  # wet silt
    "#FFCE9A",  # exposed char / sandbar
]

# Diverging ramp for "change vs. baseline" views. Anchored on the panel colour
# at zero so no-change is genuinely invisible.
DIVERGE_RAMP = ["#DC5B3E", "#B08556", "#2A424A", "#4E9E86", "#63C08A"]

FONT_DISPLAY = "'Bricolage Grotesque', 'Trebuchet MS', sans-serif"
FONT_BODY = "'Public Sans', 'Segoe UI', system-ui, sans-serif"
FONT_MONO = "'IBM Plex Mono', 'SFMono-Regular', monospace"


# --- colour utilities --------------------------------------------------------
def hex_to_rgb(h: str) -> np.ndarray:
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)


def rgb_to_hex(rgb) -> str:
    r, g, b = (int(round(min(255, max(0, c)))) for c in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def ramp_color(t: float, ramp=None) -> str:
    """Sample the ramp at t in [0, 1] with linear interpolation in sRGB."""
    ramp = ramp or RISK_RAMP
    t = float(np.clip(t, 0.0, 1.0))
    pos = t * (len(ramp) - 1)
    i = int(np.floor(pos))
    if i >= len(ramp) - 1:
        return ramp[-1]
    frac = pos - i
    a, b = hex_to_rgb(ramp[i]), hex_to_rgb(ramp[i + 1])
    return rgb_to_hex(a + (b - a) * frac)


def plotly_colorscale(ramp=None):
    ramp = ramp or RISK_RAMP
    n = len(ramp) - 1
    return [[i / n, c] for i, c in enumerate(ramp)]


def relative_luminance(h: str) -> float:
    c = hex_to_rgb(h) / 255.0
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def simulate_cvd(h: str, kind: str = "deuteranopia") -> str:
    """
    Brettel-style dichromat simulation, matrix form (Viénot et al. 1999).

    Used at build time to check that adjacent ramp stops stay distinguishable.
    Not exact colour science, but good enough to catch a ramp that collapses.
    """
    m = {
        "deuteranopia": np.array([[0.625, 0.375, 0.0],
                                  [0.700, 0.300, 0.0],
                                  [0.0, 0.300, 0.700]]),
        "protanopia": np.array([[0.567, 0.433, 0.0],
                                [0.558, 0.442, 0.0],
                                [0.0, 0.242, 0.758]]),
        "tritanopia": np.array([[0.950, 0.050, 0.0],
                                [0.0, 0.433, 0.567],
                                [0.0, 0.475, 0.525]]),
    }[kind]
    return rgb_to_hex(m @ hex_to_rgb(h))


# --- value-suppressing uncertainty palette -----------------------------------
class VSUP:
    """
    Bivariate (value, uncertainty) palette.

    tiers        SE cut-points in percentage points, ascending.
    bins_by_tier How many value bins survive at each tier. Monotonically
                 decreasing: this *is* the value suppression.
    """

    def __init__(self, value_domain, tiers=None, bins_by_tier=(8, 4, 2, 1),
                 ramp=None, ground=INK_2):
        self.lo, self.hi = float(value_domain[0]), float(value_domain[1])
        self.tiers = list(tiers or [2.5, 4.5, 7.0])
        self.bins_by_tier = list(bins_by_tier)
        self.ramp = ramp or RISK_RAMP
        self.ground = hex_to_rgb(ground)
        if len(self.bins_by_tier) != len(self.tiers) + 1:
            raise ValueError("bins_by_tier must have one more entry than tiers")

    def tier_of(self, se: float) -> int:
        for i, cut in enumerate(self.tiers):
            if se < cut:
                return i
        return len(self.tiers)

    def encode(self, value: float, se: float) -> str:
        """Return the hex colour for one (estimate, standard error) pair."""
        tier = self.tier_of(se)
        n_bins = self.bins_by_tier[tier]

        span = max(self.hi - self.lo, 1e-9)
        t = np.clip((value - self.lo) / span, 0.0, 1.0)

        if n_bins <= 1:
            t_binned = 0.5  # one bin: everything reads as "middle, unknown"
        else:
            idx = min(int(t * n_bins), n_bins - 1)
            t_binned = (idx + 0.5) / n_bins

        base = hex_to_rgb(ramp_color(t_binned, self.ramp))
        # Desaturate toward the panel colour as tier rises.
        pull = tier / max(len(self.bins_by_tier) - 1, 1) * 0.72
        return rgb_to_hex(base * (1 - pull) + self.ground * pull)

    def legend_cells(self):
        """
        Wedge geometry for the 2D legend: a list of
        (tier, bin_index, n_bins, hex) tuples, drawn as a fan that narrows
        downward — wide and colourful where we know a lot, narrow and grey
        where we do not.
        """
        cells = []
        for tier, n_bins in enumerate(self.bins_by_tier):
            for b in range(n_bins):
                t_binned = (b + 0.5) / n_bins if n_bins > 1 else 0.5
                base = hex_to_rgb(ramp_color(t_binned, self.ramp))
                pull = tier / max(len(self.bins_by_tier) - 1, 1) * 0.72
                cells.append((tier, b, n_bins,
                              rgb_to_hex(base * (1 - pull) + self.ground * pull)))
        return cells

    def tier_labels(self):
        out = []
        prev = 0.0
        for cut in self.tiers:
            out.append(f"±{prev:.1f}–{cut:.1f} pp")
            prev = cut
        out.append(f"±{prev:.1f}+ pp")
        return out


# --- Streamlit CSS -----------------------------------------------------------
def css() -> str:
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,800&family=Public+Sans:wght@300;400;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

.stApp {{ background: {INK}; color: {PAPER}; font-family: {FONT_BODY}; }}
section[data-testid="stSidebar"] {{
    background: {INK_2}; border-right: 1px solid {LINE};
}}
h1, h2, h3 {{
    font-family: {FONT_DISPLAY}; font-weight: 800;
    letter-spacing: -0.02em; color: {PAPER};
}}
h1 {{ font-size: 2.05rem; line-height: 1.1; }}
h2 {{ font-size: 1.35rem; margin-top: 1.6rem; }}
h3 {{ font-size: 1.02rem; color: {CURRENT}; font-weight: 600; }}

/* Eyebrow label — used to name the analytic step each page belongs to.
   Encodes position in the workflow, which is real information, not decoration. */
.eyebrow {{
    font-family: {FONT_MONO}; font-size: 0.68rem; letter-spacing: 0.18em;
    text-transform: uppercase; color: {MUTED};
    border-left: 2px solid {CURRENT}; padding-left: 0.55rem; margin-bottom: 0.4rem;
}}
.lede {{ color: {MUTED}; font-size: 0.95rem; max-width: 62ch; line-height: 1.55; }}

.stat {{
    background: {INK_2}; border: 1px solid {LINE};
    border-left: 3px solid {CURRENT};
    padding: 0.75rem 0.9rem; border-radius: 3px; height: 100%;
}}
.stat .k {{
    font-family: {FONT_MONO}; font-size: 0.63rem; letter-spacing: 0.13em;
    text-transform: uppercase; color: {MUTED};
}}
.stat .v {{
    font-family: {FONT_DISPLAY}; font-size: 1.55rem; font-weight: 800;
    line-height: 1.25; color: {PAPER};
}}
.stat .sub {{ font-family: {FONT_MONO}; font-size: 0.7rem; color: {MUTED}; }}

.caveat {{
    border: 1px solid {SILT}33; border-left: 3px solid {SILT};
    background: {SILT}0E; padding: 0.7rem 0.9rem; border-radius: 3px;
    font-size: 0.86rem; color: {PAPER}; line-height: 1.5;
}}
.tag {{
    display: inline-block; font-family: {FONT_MONO}; font-size: 0.63rem;
    letter-spacing: 0.1em; text-transform: uppercase;
    border: 1px solid {LINE}; color: {MUTED};
    padding: 0.12rem 0.42rem; border-radius: 2px; margin-right: 0.3rem;
}}

div[data-testid="stMetricValue"] {{ font-family: {FONT_DISPLAY}; color: {PAPER}; }}
.stSlider label, .stSelectbox label, .stRadio label, .stMultiSelect label {{
    font-family: {FONT_MONO} !important; font-size: 0.7rem !important;
    letter-spacing: 0.08em; text-transform: uppercase; color: {MUTED} !important;
}}
.stButton > button {{
    background: {CURRENT}; color: {INK}; border: 0; border-radius: 2px;
    font-weight: 600; letter-spacing: 0.02em;
}}
.stButton > button:hover {{ background: {SILT}; color: {INK}; }}
.stTabs [data-baseweb="tab"] {{
    font-family: {FONT_MONO}; font-size: 0.74rem; letter-spacing: 0.08em;
    text-transform: uppercase;
}}
.stDataFrame {{ border: 1px solid {LINE}; }}
hr {{ border-color: {LINE}; }}
a {{ color: {CURRENT}; }}
#MainMenu, footer {{ visibility: hidden; }}

@media (prefers-reduced-motion: reduce) {{
    * {{ animation: none !important; transition: none !important; }}
}}
</style>
"""


def plotly_layout(height=430, **kw):
    """Shared Plotly layout so every chart in the app agrees on its furniture."""
    base = dict(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Public Sans, sans-serif", color=PAPER, size=12),
        margin=dict(l=8, r=8, t=34, b=8),
        xaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE),
        yaxis=dict(gridcolor=LINE, zerolinecolor=LINE, linecolor=LINE),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=MUTED)),
        hoverlabel=dict(bgcolor=INK_3, bordercolor=LINE,
                        font=dict(family="IBM Plex Mono, monospace",
                                  color=PAPER, size=11)),
    )
    base.update(kw)
    return base


if __name__ == "__main__":
    print("risk ramp — normal / deuteranopia / protanopia")
    for c in RISK_RAMP:
        print(f"  {c}  L={relative_luminance(c):.3f}  "
              f"{simulate_cvd(c, 'deuteranopia')}  {simulate_cvd(c, 'protanopia')}")

    lums = [relative_luminance(c) for c in RISK_RAMP]
    print("\nluminance monotonic:", all(b > a for a, b in zip(lums, lums[1:])))

    for kind in ("deuteranopia", "protanopia"):
        gaps = [
            float(np.linalg.norm(hex_to_rgb(simulate_cvd(a, kind))
                                 - hex_to_rgb(simulate_cvd(b, kind))))
            for a, b in zip(RISK_RAMP, RISK_RAMP[1:])
        ]
        print(f"min adjacent RGB gap under {kind}: {min(gaps):.1f} / 255")
