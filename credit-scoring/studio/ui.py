"""Shared UI kit: look & feel, small components and model-chart helpers."""
from __future__ import annotations

import inspect

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from crf.ai import LLM

from . import text

PRIMARY = "#3B5BDB"
INK = "#1B2433"
MUTED = "#6B7690"
_W_NEW = "width" in inspect.signature(st.plotly_chart).parameters
PAGES: dict = {}          # filled by app.py so pages can link to each other

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.hero {padding: 1.4rem 1.6rem; border-radius: 16px; background: linear-gradient(135deg,#3B5BDB 0%,#5F3DC4 100%); color: #fff; margin-bottom: 1rem;}
.hero h1 {color:#fff; margin:0 0 .3rem 0; font-size: 2rem;}
.hero p {margin:0; opacity:.92; font-size: 1.02rem;}
.pagehead {margin-bottom: .6rem;}
.pagehead h2 {margin: 0; font-size: 1.7rem;}
.pagehead p {margin: .15rem 0 0 0; color: #6B7690; font-size: .98rem;}
.pill {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 600; color: #fff; white-space: nowrap;}
.card {border: 1px solid rgba(120,130,160,.28); border-radius: 14px; padding: 1rem 1.1rem; height: 100%;}
.card h4 {margin: 0 0 .25rem 0; font-size: 1.05rem;}
.card p {margin: 0; color: #6B7690; font-size: .92rem;}
.small {color:#6B7690; font-size:.85rem;}
div[data-testid="stMetric"] {border: 1px solid rgba(120,130,160,.22); border-radius: 12px; padding: .65rem .9rem;}
.callout {border-left: 4px solid #3B5BDB; background: rgba(59,91,219,.07); padding: .7rem 1rem; border-radius: 8px; margin: .4rem 0 .8rem 0;}
.warn {border-left-color:#ee8434; background: rgba(238,132,52,.09);}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def header(title: str, subtitle: str, icon: str = ""):
    st.markdown(f"<div class='pagehead'><h2>{icon} {title}</h2><p>{subtitle}</p></div>", unsafe_allow_html=True)


def callout(msg: str, warn: bool = False):
    st.markdown(f"<div class='callout{' warn' if warn else ''}'>{msg}</div>", unsafe_allow_html=True)


def guide(page: str, body: str):
    """First-visit guidance that stays out of the way afterwards."""
    seen = st.session_state.setdefault("seen_guides", set())
    first = page not in seen
    with st.expander("💡 How to use this page", expanded=first):
        st.markdown(body)
    seen.add(page)


def pill(label: str, color: str) -> str:
    return f"<span class='pill' style='background:{color}'>{label}</span>"


def band_pill(band: str) -> str:
    return pill(text.BAND_NAMES.get(band, band), text.BAND_COLORS.get(band, "#888"))


def money(x: float, short: bool = True) -> str:
    if x is None or x != x:
        return "-"
    a = abs(x)
    if short and a >= 1e6:
        return f"${x / 1e6:,.2f}M"
    if short and a >= 1e4:
        return f"${x / 1e3:,.0f}K"
    return f"${x:,.0f}"


def pct(x: float, d: int = 1) -> str:
    return "-" if x is None or x != x else f"{x * 100:.{d}f}%"


def show(fig: go.Figure, height: int = 360):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=44, b=8), font=dict(size=13),
                      legend=dict(orientation="h", y=-0.18), title=dict(font=dict(size=15)))
    st.plotly_chart(fig, **({"width": "stretch"} if _W_NEW else {"use_container_width": True}))


def table(df: pd.DataFrame, **kw):
    st.dataframe(df, hide_index=True, **({"width": "stretch"} if _W_NEW else {"use_container_width": True}), **kw)


def get_llm() -> LLM:
    s = st.session_state
    prov = s.get("llm_provider", "auto")
    return LLM("openai" if prov == "openai-compatible" else prov, s.get("llm_model") or None, s.get("llm_key") or None,
               s.get("llm_url") or None)


def sidebar_ai():
    with st.sidebar.expander("🤖 AI settings"):
        st.selectbox("Provider", ["auto", "anthropic", "openai-compatible", "offline"], key="llm_provider")
        st.text_input("Model (optional)", key="llm_model")
        st.text_input("API key (session only)", type="password", key="llm_key")
        st.text_input("Base URL (Ollama / OpenAI-compatible)", key="llm_url", placeholder="http://localhost:11434/v1")
        st.caption("Without a provider every AI feature falls back to built-in templates. Only aggregates and single-customer fact "
                   "sheets are sent, never raw tables.")
    llm = get_llm()
    st.sidebar.caption(f"AI: {llm.provider} · {llm.model}" if llm.available else "AI: offline templates")


# ------------------------------------------------------------------ model charts (analyst views)
def model_charts(res: dict, key: str):
    tabs = st.tabs(["Performance", "Drivers", "Scorecard", "Stability", "Scorecard vs GBM"])
    with tabs[0]:
        table(res["metrics"].round(4))
        which = st.radio("Sample", ["test", "train"], horizontal=True, key=f"{key}_set")
        a, b = st.columns(2)
        with a:
            fig = go.Figure()
            for k in ("train", "test"):
                fig.add_scatter(x=res["roc"][k]["fpr"], y=res["roc"][k]["tpr"], name=k, mode="lines")
            fig.add_scatter(x=[0, 1], y=[0, 1], name="random", mode="lines", line=dict(dash="dot", color="grey"))
            fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
            show(fig, 320)
        with b:
            k = res["ks"][which]
            fig = go.Figure()
            fig.add_scatter(x=k["pop"], y=k["cum_bad"], name="Cumulative bad", mode="lines")
            fig.add_scatter(x=k["pop"], y=k["cum_good"], name="Cumulative good", mode="lines")
            fig.update_layout(title=f"KS separation ({which})", xaxis_title="Population, riskiest first")
            show(fig, 320)
        d = res["deciles"][which]
        fig = go.Figure()
        fig.add_bar(x=d["decile"], y=d["bad_rate"], name="Actual bad rate")
        fig.add_scatter(x=d["decile"], y=d["avg_pd"], name="Predicted PD", mode="lines+markers")
        fig.update_layout(title="Calibration by risk decile (1 = riskiest)", xaxis=dict(dtick=1), yaxis=dict(tickformat=".0%"))
        show(fig, 320)
    with tabs[1]:
        iv = res["iv_table"].head(15)
        col = {"useless": "#bbb", "weak": "#9ecae1", "medium": "#4292c6", "strong": "#08519c", "suspicious": "#d94801"}
        fig = go.Figure(go.Bar(x=iv["iv"], y=[text.label(f) for f in iv["feature"]], orientation="h",
                               marker_color=[col[s] for s in iv["strength"]]))
        fig.update_layout(title="Predictive power (Information Value)", yaxis=dict(autorange="reversed"))
        show(fig, 420)
        lk = res["leakage"]
        flagged = lk[lk["flagged"]]
        st.success("No leakage flags.") if flagged.empty else st.error("Leakage flagged: " + "; ".join(flagged["feature"]))
        feat = st.selectbox("Look inside a feature", [text.label(f) for f in res["iv_table"]["feature"]], key=f"{key}_feat")
        raw = {text.label(f): f for f in res["iv_table"]["feature"]}[feat]
        t = res["binners"][raw].table
        fig = go.Figure()
        fig.add_bar(x=t["bin"], y=t["pct"], name="% of population", opacity=0.5)
        fig.add_scatter(x=t["bin"], y=t["bad_rate"], name="Bad rate", yaxis="y2", mode="lines+markers")
        fig.update_layout(title=f"{feat}: who is in each band, and how risky", yaxis=dict(tickformat=".0%"),
                          yaxis2=dict(overlaying="y", side="right", tickformat=".0%"))
        show(fig, 320)
    with tabs[2]:
        pt = res["points_table"].copy()
        pt["feature"] = pt["feature"].map(text.label)
        f = st.selectbox("Feature", ["(all)"] + sorted(pt["feature"].unique()), key=f"{key}_pts")
        table((pt if f == "(all)" else pt[pt["feature"] == f]).round(3))
    with tabs[3]:
        st.metric("Score PSI, train vs test", f"{res['score_psi']:.3f}", help="Below 0.10 is stable, above 0.25 means the population changed a lot.")
        fp = res["feature_psi"].sort_values("psi", ascending=False).copy()
        fp["feature"] = fp["feature"].map(text.label)
        fig = go.Figure(go.Bar(x=fp["psi"], y=fp["feature"], orientation="h"))
        fig.add_vline(x=0.1, line_dash="dot")
        fig.add_vline(x=0.25, line_dash="dash")
        fig.update_layout(title="Feature stability (PSI)", yaxis=dict(autorange="reversed"))
        show(fig, 360)
    with tabs[4]:
        ch = res.get("challenger")
        if ch:
            m = res["metrics"].set_index("set").loc["test"]
            c = ch["metrics"]
            table(pd.DataFrame({"Model": ["Scorecard (explainable)", "Gradient boosting (benchmark)"],
                                "AUC": [m["auc"], c["auc"]], "KS": [m["ks"], c["ks"]], "Gini": [m["gini"], c["gini"]]}).round(4))
            gap = c["auc"] - m["auc"]
            callout(f"The black-box benchmark is {gap:+.3f} AUC vs the scorecard. "
                    + ("Small gap: keep the transparent scorecard." if gap < 0.02 else "Material gap: consider a hybrid or adding interactions."))
