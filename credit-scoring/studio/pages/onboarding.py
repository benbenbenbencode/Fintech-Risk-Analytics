"""Onboarding desk: strategy simulator + decision queue."""
from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from studio import cached, data, policy, text, ui


def render():
    ui.header("Onboarding", "Decide who to approve. Test how strict the policy should be, then work the queue of waiting applications.", "🆕")
    ui.guide("onboarding", "**Step 1 · Strategy.** Pick a product and move the slider: a higher cut-off approves more people but accepts more risk. "
                           "The marker shows the profit-maximising point.\n\n**Step 2 · Queue.** Applications waiting for a decision, with a suggested "
                           "*Approve / Refer / Decline* and the reasons. Click one to review and record your decision.")
    prod = st.radio("Product", ["CARD", "LOAN", "BNPL"], horizontal=True, format_func=lambda x: f"{text.PRODUCT_ICONS[x]} {text.PRODUCT_NAMES[x]}", key="ob_prod")
    curve = cached.strategy_curve(prod)
    opt = policy.optimal_cutoff(curve)
    st.subheader("1 · Strategy simulator")
    lo, hi = float(curve.cutoff_pd.min()), float(curve.cutoff_pd.max())
    cut = st.slider("Approve applicants whose predicted 12-month risk is at most", lo, hi, float(opt), 0.005, format="%.3f", key=f"ob_cut_{prod}",
                    help="Applicants above this are referred (up to 1.5×) or declined.")
    row = curve.iloc[(curve.cutoff_pd - cut).abs().argmin()]
    best = curve.loc[curve.expected_profit_usd.idxmax()]
    hist = data.q("SELECT decision, avg(pd_app) pd, count(*) n FROM app_scores WHERE product=? AND decision<>'pending' GROUP BY 1", [prod]).set_index("decision")
    legacy_rate = hist.loc["approved", "n"] / hist.n.sum()
    same_rate = curve.iloc[(curve.approval_rate - legacy_rate).abs().argmin()]
    m = st.columns(4)
    m[0].metric("Approval rate", ui.pct(row.approval_rate, 0), f"{(row.approval_rate - legacy_rate) * 100:+.0f} pts vs current policy")
    m[1].metric("Expected bad rate", ui.pct(row.expected_bad_rate), help="Average predicted 12-month risk of those approved.")
    m[2].metric("Expected profit (2 years)", ui.money(row.expected_profit_usd), f"{ui.money(row.expected_profit_usd - best.expected_profit_usd)} vs best")
    m[3].metric("Expected losses", ui.money(row.expected_loss_usd))
    fig = go.Figure()
    fig.add_scatter(x=curve.approval_rate, y=curve.expected_profit_usd, name="Expected profit", mode="lines", line=dict(width=3))
    fig.add_scatter(x=curve.approval_rate, y=curve.expected_bad_rate, name="Expected bad rate", yaxis="y2", mode="lines", line=dict(dash="dot"))
    fig.add_scatter(x=[row.approval_rate], y=[row.expected_profit_usd], mode="markers", marker=dict(size=13, color="#d1495b"), name="Your cut-off")
    fig.add_scatter(x=[best.approval_rate], y=[best.expected_profit_usd], mode="markers", marker=dict(size=13, symbol="star", color="#2e9e6b"), name="Best profit")
    fig.update_layout(xaxis=dict(title="Approval rate", tickformat=".0%"), yaxis=dict(title="Expected profit (USD)"),
                      yaxis2=dict(title="Bad rate", overlaying="y", side="right", tickformat=".0%"))
    ui.show(fig, 340)
    ui.callout(f"At the **current approval rate ({legacy_rate:.0%})** the model would choose applicants with an expected bad rate of "
               f"**{same_rate.expected_bad_rate:.1%}**, versus **{hist.loc['approved', 'pd']:.1%}** for those the current policy approved. "
               "Estimates use the model's own predictions, including for people who were declined, so treat them as directional.")

    st.subheader("2 · Applications waiting for a decision")
    pend = policy.decide(policy.pending_applications(prod), cut)
    dec_counts = pend.decision_suggested.value_counts()
    c = st.columns(4)
    c[0].metric("Waiting", f"{len(pend):,}")
    for col, name in zip(c[1:], ["Approve", "Refer", "Decline"]):
        col.metric(f"Suggest: {name}", f"{int(dec_counts.get(name, 0)):,}")
    f = st.columns(3)
    pick = f[0].multiselect("Suggested decision", ["Approve", "Refer", "Decline"], default=["Refer"], key="ob_dec")
    ctry = f[1].multiselect("Country", sorted(pend.country.unique()), key="ob_ctry")
    sort = f[2].selectbox("Sort", ["Highest risk first", "Largest request first"], key="ob_sort")
    v = pend[pend.decision_suggested.isin(pick or ["Approve", "Refer", "Decline"])]
    if ctry:
        v = v[v.country.isin(ctry)]
    v = v.sort_values("pd_app", ascending=False) if sort.startswith("Highest") else v.sort_values("requested_amount_usd", ascending=False)
    v = v.head(500).reset_index(drop=True)
    show = v.assign(risk=v.pd_app * 100)[["application_id", "customer_id", "decision_suggested", "risk", "risk_band", "requested_amount_usd", "country",
                                           "city", "age", "knockout"]]
    show.columns = ["Application", "Customer", "Suggested", "12m risk", "Band", "Requested (USD)", "Country", "City", "Age", "Policy flag"]
    ev = st.dataframe(show, hide_index=True, on_select="rerun", selection_mode="single-row", key=f"ob_tbl_{prod}", height=300,
                      column_config={"12m risk": st.column_config.ProgressColumn("12m risk", min_value=0, max_value=40, format="%.1f%%"),
                                     "Requested (USD)": st.column_config.NumberColumn(format="$%d")},
                      **({"width": "stretch"} if ui._W_NEW else {"use_container_width": True}))
    sel = ev.selection.rows if ev is not None and hasattr(ev, "selection") else []
    if sel:
        r = v.iloc[sel[0]]
        with st.container(border=True):
            st.markdown(f"### {r.application_id} · {r.customer_id} &nbsp; {ui.band_pill(r.risk_band)}", unsafe_allow_html=True)
            a = st.columns(4)
            a[0].metric("Suggested", r.decision_suggested)
            a[1].metric("12-month risk", ui.pct(r.pd_app))
            a[2].metric("Requested", ui.money(r.requested_amount_usd, False))
            a[3].metric("Bureau score", "No record" if r.bureau_score != r.bureau_score else int(r.bureau_score))
            if r.knockout:
                st.warning(f"Policy flag: {r.knockout}")
            st.markdown("**Main factors pushing the risk up:**")
            for x in policy.reasons(r.reasons)[:4]:
                st.write(f"• {text.reason_phrase(x['f'], x['bin'])}")
            d = st.columns([1, 2, 1])
            choice = d[0].selectbox("Your decision", ["Approve", "Refer", "Decline"], index=["Approve", "Refer", "Decline"].index(r.decision_suggested), key=f"ob_ch_{r.application_id}")
            note = d[1].text_input("Reason (required if you differ from the suggestion)", key=f"ob_note_{r.application_id}")
            if d[2].button("Record decision", key=f"ob_rec_{r.application_id}"):
                if choice != r.decision_suggested and not note.strip():
                    st.error("Please add a reason when overriding the suggestion.")
                else:
                    st.session_state.setdefault("decisions", []).append({"application": r.application_id, "suggested": r.decision_suggested,
                                                                         "final": choice, "override": choice != r.decision_suggested, "reason": note})
                    st.success("Recorded.")
    log = st.session_state.get("decisions", [])
    if log:
        with st.expander(f"Decisions recorded this session ({len(log)})"):
            import pandas as pd
            ui.table(pd.DataFrame(log))
    st.caption("The model was trained only on approved customers, so it is least certain about profiles the current policy rarely approved.")
