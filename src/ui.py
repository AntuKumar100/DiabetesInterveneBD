"""
Shared UI furniture.

The one idea worth stating: the selected district lives in `st.session_state`,
not in a per-page widget. Pick Narayanganj on the map, walk to the intervention
lab, and it is still Narayanganj. Streamlit's default is the opposite — every
page gets a fresh widget — and that default quietly breaks the analytic thread
the whole tool is built around. A user should never have to re-answer a
question they already answered.
"""

from __future__ import annotations

import streamlit as st

from .theme import CURRENT, MUTED, PAPER, css

STATE_DISTRICT = "selected_district"
STATE_MODEL = "model_choice"
STATE_WEIGHTS = "profile_weights"


def page(title: str, icon: str = "◈") -> None:
    """Call first on every page. Sets config and injects the stylesheet."""
    st.set_page_config(
        page_title=f"{title} · InterveneBD",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(css(), unsafe_allow_html=True)


def header(step: str, title: str, lede: str) -> None:
    """
    Eyebrow / title / lede.

    The eyebrow carries the workflow position ("step 3 · explain") because this
    tool is a sequence — map, compare, explain, simulate, decide — and knowing
    where you are in it is real information, not decoration.
    """
    st.markdown(f'<div class="eyebrow">{step}</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    st.markdown(f'<p class="lede">{lede}</p>', unsafe_allow_html=True)


def stat(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="stat"><div class="k">{label}</div>'
        f'<div class="v">{value}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def stat_row(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, sub) in zip(cols, items):
        col.markdown(stat(label, value, sub), unsafe_allow_html=True)


def caveat(text: str) -> None:
    st.markdown(f'<div class="caveat">{text}</div>', unsafe_allow_html=True)


def tags(*labels: str) -> None:
    st.markdown("".join(f'<span class="tag">{t}</span>' for t in labels),
                unsafe_allow_html=True)


def get_district(options: list[str], default: str | None = None) -> str:
    cur = st.session_state.get(STATE_DISTRICT)
    if cur not in options:
        cur = default if default in options else options[0]
        st.session_state[STATE_DISTRICT] = cur
    return cur


def set_district(name: str) -> None:
    if name and name != st.session_state.get(STATE_DISTRICT):
        st.session_state[STATE_DISTRICT] = name


def district_picker(options: list[str], label: str = "District in focus") -> str:
    """Sidebar selector bound to shared state. Present on every page."""
    current = get_district(options)
    choice = st.sidebar.selectbox(
        label, options, index=options.index(current), key="_district_widget"
    )
    set_district(choice)
    return choice


def model_picker() -> str:
    """
    Model switch, exposed rather than hidden.

    Being able to ask "does this conclusion survive a different model class?"
    is a basic robustness check, and burying it in a config file means nobody
    ever runs it.
    """
    labels = {"gbm": "Gradient boosting", "logit": "Logistic regression"}
    current = st.session_state.get(STATE_MODEL, "gbm")
    choice = st.sidebar.radio(
        "Risk model", list(labels), index=list(labels).index(current),
        format_func=labels.get, key="_model_widget",
        help="Both are fitted on the same weighted training split. If a finding "
             "flips between them, treat it as unresolved.",
    )
    st.session_state[STATE_MODEL] = choice
    return choice


def sidebar_footer() -> None:
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;'
        f'color:{MUTED};line-height:1.6;letter-spacing:0.04em">'
        f'BDHS cross-section · 14,167 adults<br>674 clusters · 64 districts<br>'
        f'<span style="color:{CURRENT}">estimates are design-weighted</span></div>',
        unsafe_allow_html=True,
    )

def sidebar_credit() -> None:
    """Author credit, shown beneath the data provenance footer on every page."""
    st.sidebar.markdown(
        f'<div style="margin-top:0.75rem;padding-top:0.75rem;'
        f'border-top:1px solid {MUTED}33">'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;'
        f'letter-spacing:0.08em;text-transform:uppercase;color:{MUTED}">Built by</div>'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.85rem;'
        f'font-weight:600;color:{CURRENT};margin-top:0.2rem">Amartay Kumar Dhar</div>'
        f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;'
        f'color:{MUTED};line-height:1.6;margin-top:0.15rem">'
        f'Department of Statistics and Data Science<br>'
        f'antukumardhar100@gmail.com</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def selected_from_plotly(event, locations) -> str | None:
    """
    Pull a district name out of a Plotly selection event.

    Plotly hands back either the feature id or a positional index depending on
    trace type and version, so we check both rather than assuming. Returns None
    when nothing usable came back, and callers fall through to the selectbox —
    the map click is an accelerator, never the only way to do something.
    """
    try:
        points = event.selection["points"]
    except (AttributeError, KeyError, TypeError):
        return None
    if not points:
        return None
    p = points[0]
    for key in ("location", "label", "hovertext"):
        val = p.get(key)
        if isinstance(val, str) and val in set(locations):
            return val
    for key in ("point_index", "pointIndex", "pointNumber", "point_number"):
        idx = p.get(key)
        if isinstance(idx, int) and 0 <= idx < len(locations):
            return list(locations)[idx]
    return None
