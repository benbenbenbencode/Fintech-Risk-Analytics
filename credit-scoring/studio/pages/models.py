"""Model studio (for analysts): registry, performance, drivers, stability."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from studio import data, text, ui

CARDS = {
    "application": ("Decides who to approve at the door.", "Trained on customers the current policy approved, so it is least reliable for profiles that policy rarely approved (selection bias). "
                    "Reject inference is not applied."),
    "behavioral": ("Watches existing accounts and flags those likely to become seriously late.", "Rows are account-months, so the same account appears several times; "
                   "the split is by time (latest 30% of months is the test). Accounts that cannot yet be scored (under 3 months) use the application model."),
    "collections": ("Estimates the chance a late account gets back on track this month.", "It describes what happened under the current collections practice. "
                    "Treatment effects come from the randomised subset, not from this model."),
}


def render():
    ui.header("Model studio", "For analysts and model validators: how good are the models, what drives them, are they stable?", "🧪")
    reg = data.registry().copy()
    reg["name"] = reg["id"].str.replace("_", " ").str.title()
    show = reg[["name", "kind", "rows", "bad_rate", "auc", "ks", "challenger_auc", "psi", "features"]].copy()
    show.columns = ["Model", "What it does", "Rows", "Bad rate", "AUC (test)", "KS (test)", "GBM AUC", "Score PSI", "Features"]
    show["Bad rate"] = (show["Bad rate"] * 100).round(1).astype(str) + "%"
    ui.table(show.round(3))
    st.caption("AUC and KS measure how well a model ranks risky above safe (0.5 AUC is a coin flip; 0.7+ is useful; 0.8+ is strong for credit). "
               "PSI under 0.10 means the population did not shift between training and testing.")
    pick = st.selectbox("Open a model", [i for i in reg["id"] if i != "collections_cure"], format_func=lambda x: x.replace("_", " ").title(), key="mdl_pick")
    res = data.models()["results"][pick]
    kind = pick.split("_")[0]
    st.markdown(f"**{CARDS[kind][0]}**")
    ui.callout("Limitations: " + CARDS[kind][1], warn=True)
    m = res["metrics"].set_index("set")
    c = st.columns(4)
    c[0].metric("Test AUC", f"{m.loc['test', 'auc']:.3f}")
    c[1].metric("Test KS", f"{m.loc['test', 'ks']:.3f}")
    c[2].metric("Bad rate", ui.pct(res["population"]["bad_rate"]))
    c[3].metric("Features kept", len(res["selected"]))
    ui.model_charts(res, key=pick)
    cm = data.models()["collections"]["metrics"]
    with st.expander("Collections cure model"):
        st.write(f"AUC {cm['auc']:.3f} · KS {cm['ks']:.3f} on {cm['n']:,} test cases. Used with the randomised treatment table on the Collections page.")
