"""Credit limit management: who gets more, how much, and what it is worth."""
from __future__ import annotations

import json

import plotly.express as px
import streamlit as st

from studio import cached, data, policy, text, ui


def render():
    ui.header("Credit limits", "Raise limits where it is safe and profitable. Review the riskiest accounts for a decrease.", "📈")
    ui.guide("limits", "Adjust the rules on the left; the list updates. An account is **eligible** when it is old enough, paying cleanly, "
                       "uses its limit, and its predicted risk is low. The increase depends on the risk band and is capped by the customer's income.")
    d = policy.DEFAULT_CLI
    with st.expander("Policy rules", expanded=False):
        a = st.columns(4)
        prods = a[0].multiselect("Products", ["CARD", "BNPL"], default=["CARD", "BNPL"], format_func=lambda x: text.PRODUCT_NAMES[x], key="li_prod")
        min_mob = a[1].slider("Months open at least", 3, 24, d["min_mob"], key="li_mob")
        max_pd = a[2].slider("Max 6-month risk", 0.01, 0.10, d["max_pd"], 0.005, format="%.3f", key="li_pd")
        min_use = a[3].slider("Min usage of limit", 0.0, 0.8, d["min_usage"], 0.05, key="li_use")
        b = st.columns(4)
        incA = b[0].slider("Increase for band A", 0.0, 1.0, d["increase_by_band"]["A"], 0.05, key="li_a")
        incB = b[1].slider("Increase for band B", 0.0, 1.0, d["increase_by_band"]["B"], 0.05, key="li_b")
        incC = b[2].slider("Increase for band C", 0.0, 1.0, d["increase_by_band"]["C"], 0.05, key="li_c")
        cap = b[3].slider("Card limit cap (× monthly income)", 1.0, 8.0, d["capacity_mult"]["CARD"], 0.5, key="li_cap")
    params = {"min_mob": min_mob, "max_pd": max_pd, "min_usage": min_use, "increase_by_band": {"A": incA, "B": incB, "C": incC},
              "capacity_mult": {"CARD": cap, "BNPL": d["capacity_mult"]["BNPL"]}}
    if not prods:
        st.info("Pick at least one product.")
        return
    df = cached.cli_proposals(json.dumps(params), tuple(prods))
    e = df[df.eligible]
    dec = df[df.action == "Review for decrease"]
    m = st.columns(5)
    m[0].metric("Accounts reviewed", f"{len(df):,}")
    m[1].metric("Eligible for increase", f"{len(e):,}", f"{len(e) / max(len(df), 1):.1%} of reviewed")
    m[2].metric("Extra credit offered", ui.money(e.increase_usd.sum()))
    m[3].metric("Expected profit (12m)", ui.money(e.net_value_usd.sum()), help="Extra revenue minus extra expected loss, assuming only part of the increase is used.")
    m[4].metric("Review for decrease", f"{len(dec):,}", help="Very high predicted risk.")
    a, b = st.columns(2)
    with a:
        g = e.groupby("country").agg(n=("account_id", "size"), profit=("net_value_usd", "sum")).reset_index().sort_values("profit", ascending=False)
        fig = px.bar(g, x="country", y="profit", text="n", title="Expected profit by country (label = accounts)")
        ui.show(fig, 300)
    with b:
        g = e.groupby(["risk_band", "product"]).agg(inc=("increase_usd", "sum")).reset_index()
        fig = px.bar(g, x="risk_band", y="inc", color="product", title="Extra credit by risk band", category_orders={"risk_band": ["A", "B", "C"]})
        ui.show(fig, 300)
    view = st.radio("Show", ["Proposed increases", "Review for decrease", "Not eligible (why)"], horizontal=True, key="li_view")
    f = st.columns(3)
    ctry = f[0].multiselect("Country", sorted(df.country.unique()), key="li_ctry")
    band = f[1].multiselect("Risk band", ["A", "B", "C", "D", "E"], key="li_band")
    if view == "Proposed increases":
        v = e.sort_values("net_value_usd", ascending=False)
    elif view == "Review for decrease":
        v = dec.sort_values("pd_6m", ascending=False)
    else:
        v = df[(~df.eligible) & (df.action == "Hold")]
        why = v.rationale.value_counts().reset_index()
        why.columns = ["Reason", "Accounts"]
        ui.table(why)
    if ctry: v = v[v.country.isin(ctry)]
    if band: v = v[v.risk_band.isin(band)]
    show = v.head(1000)[["account_id", "customer_id", "product", "country", "risk_band", "pd_6m", "usage_rate", "credit_limit_usd",
                         "proposed_limit_usd", "net_value_usd", "rationale"]].copy()
    show["pd_6m"] *= 100
    show["usage_rate"] = show["usage_rate"].clip(upper=1) * 100
    show.columns = ["Account", "Customer", "Product", "Country", "Band", "6m risk", "Usage", "Limit now", "Proposed", "Expected profit", "Why"]
    st.dataframe(show, hide_index=True, height=380, column_config={
        "6m risk": st.column_config.NumberColumn(format="%.1f%%"), "Usage": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
        "Limit now": st.column_config.NumberColumn(format="$%d"), "Proposed": st.column_config.NumberColumn(format="$%d"),
        "Expected profit": st.column_config.NumberColumn(format="$%d")}, **({"width": "stretch"} if ui._W_NEW else {"use_container_width": True}))
    st.download_button("Download this list (CSV)", show.to_csv(index=False), "limit_review.csv", "text/csv")
    st.caption("Expected profit uses the average revenue yield and loss rate measured in this portfolio. Increases also respect a total-exposure cap across a customer's products.")
