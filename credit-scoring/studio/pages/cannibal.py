"""Cannibalization: does BNPL replace card spend, loans and card customers, or add new business?"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from studio import data, ui


def render():
    ui.header("Product cannibalization", "Is BNPL adding new business, or taking it from credit cards and term loans?", "🔀")
    ui.guide("cannibal", "**The trap:** people who use BNPL also shop more, so comparing them with non-users hides the effect. "
                         "We compare each BNPL adopter with a *matched look-alike* who had not adopted, over the same calendar months, "
                         "so seasonality and spending habits cancel out. The last section shows how far off the naive methods are.")
    c = data.cannibal()
    gt = data.ground_truth()
    w = data.WORLD
    e = c["card_spend_effect"]
    r = c["revenue_6m"]
    m = st.columns(5)
    m[0].metric("Card spend after BNPL adoption", f"{e['matched_did'] * 100:+.1f}%", help="Matched comparison, 6 months after vs before adoption.")
    m[1].metric("BNPL revenue (6 months)", ui.money(r["bnpl_revenue_gained_usd"]))
    m[2].metric("Card revenue lost", ui.money(r["card_revenue_lost_usd"]))
    m[3].metric("Cannibalization rate", ui.pct(r["cannibalization_rate"], 0), help="Card revenue lost per $1 of BNPL revenue gained.")
    m[4].metric("Net new revenue", ui.money(r["net_incremental_usd"]), help="BNPL revenue minus card revenue lost, among adopters, over 6 months.")
    cc = st.columns(3)
    cc[0].metric("Term-loan take-up", f"{(c['loan_takeup_ratio'] - 1) * 100:+.0f}%", help="Customers with BNPL available, adopters vs matched non-adopters.")
    cc[1].metric("Card closure risk", f"×{c['card_closure_ratio_12m']:.2f}", help="Chance of closing the card in the 12 months after adoption, vs matched customers.")
    cc[2].metric("Adopters analysed", f"{c['n_adopters']:,}", f"vs {c['n_controls']:,} matched controls", delta_color="off")
    tabs = st.tabs(["The answer", "Before and after", "By country", "Risk stacking", "Why naive methods fail"])
    with tabs[0]:
        net = r["net_incremental_usd"]
        ui.callout(f"Among customers who added BNPL, card spend fell by about **{abs(e['matched_did']) * 100:.0f}%** "
                   f"(95% range {abs(e['ci95'][1]) * 100:.0f}% to {abs(e['ci95'][0]) * 100:.0f}%). Each $1 of BNPL revenue came with about "
                   f"**${r['cannibalization_rate']:.2f}** of lost card revenue, so BNPL is **{'still net positive' if net > 0 else 'net negative'}** for these customers "
                   f"({ui.money(net)} over 6 months). Card customers also became {abs(c['card_closure_ratio_12m'] - 1) * 100:.0f}% more likely to close the card.")
        st.markdown("**What to do with this**\n- Protect card revenue where cannibalization is highest (see *By country*): rewards or instalment options on the card itself.\n"
                    "- Do not judge BNPL on its own revenue; judge it on **net** revenue and on the customers it brings that had no card.\n"
                    "- Cap total exposure per customer across products (see *Risk stacking*).")
        if st.button("Explain this for management", key="cn_explain"):
            llm = ui.get_llm()
            facts = {"card_spend_change_pct": e["matched_did"] * 100, "ci95": e["ci95"], "bnpl_revenue_6m": r["bnpl_revenue_gained_usd"],
                     "card_revenue_lost_6m": r["card_revenue_lost_usd"], "net_6m": net, "loan_takeup_ratio": c["loan_takeup_ratio"],
                     "card_closure_ratio": c["card_closure_ratio_12m"]}
            if llm.available:
                try:
                    st.session_state["cn_text"] = llm.complete("Write a 4-sentence executive explanation of a product cannibalization result. Plain language, "
                                                               "no jargon. Use only these numbers.", str(facts), 400)
                except Exception:
                    st.session_state["cn_text"] = "AI request failed; the summary above is the built-in version."
            else:
                st.session_state["cn_text"] = "Connect an AI provider in the sidebar for a custom write-up. The summary above already states the result."
        if "cn_text" in st.session_state:
            st.write(st.session_state["cn_text"])
    with tabs[1]:
        es = pd.read_parquet(w / "cannibal_event_study.parquet")
        fig = px.line(es, x="rel_month", y="spend_index", color="group", markers=True, color_discrete_map={"Adopters": "#d1495b", "Matched controls": "#3B5BDB"})
        fig.add_vline(x=0, line_dash="dash", annotation_text="BNPL adoption")
        fig.update_layout(title="Card spend around adoption (1.0 = average of the 6 months before)", xaxis_title="Months from adoption", yaxis_title="Card spend index")
        ui.show(fig, 380)
        st.caption("Both groups move together before adoption (same trend, same seasonality). They diverge right after: that gap is the effect of BNPL.")
    with tabs[2]:
        bc = pd.DataFrame(c["by_country"]).sort_values("spend_effect")
        a, b = st.columns(2)
        with a:
            fig = go.Figure(go.Bar(x=bc.spend_effect * 100, y=bc.country, orientation="h", marker_color="#d1495b"))
            fig.update_layout(title="Change in card spend after BNPL adoption (%)")
            ui.show(fig, 340)
        with b:
            fig = go.Figure()
            fig.add_bar(x=bc.country, y=bc.bnpl_revenue_6m, name="BNPL revenue gained")
            fig.add_bar(x=bc.country, y=bc.card_revenue_lost_6m, name="Card revenue lost")
            fig.update_layout(title="Revenue gained vs lost, 6 months (USD)", barmode="group")
            ui.show(fig, 340)
        ces = pd.read_parquet(w / "cannibal_country_es.parquet")
        avg = ces.groupby("rel_month").spend_index.agg(["mean", "min", "max"]).reset_index()
        fig = go.Figure()
        fig.add_scatter(x=avg.rel_month, y=avg["max"], line=dict(width=0), showlegend=False)
        fig.add_scatter(x=avg.rel_month, y=avg["min"], fill="tonexty", line=dict(width=0), name="Range across countries", fillcolor="rgba(59,91,219,.18)")
        fig.add_scatter(x=avg.rel_month, y=avg["mean"], name="Average", line=dict(color="#3B5BDB", width=3))
        fig.add_vline(x=0, line_dash="dash", annotation_text="BNPL launch in the country")
        fig.update_layout(title="Staggered launches: card spend per cardholder vs months since BNPL launch (seasonally adjusted)")
        ui.show(fig, 340)
        ad = pd.read_parquet(w / "cannibal_adoption.parquet")
        ad["month"] = ad.open_month_idx.map(data.month_label)
        ad = ad.sort_values("open_month_idx")
        ad["cum"] = ad.groupby("country").n.cumsum()
        fig = px.line(ad, x="month", y="cum", color="country", title="BNPL accounts opened (cumulative), launches are staggered by country")
        fig.update_layout(xaxis=dict(nticks=8))
        ui.show(fig, 320)
    with tabs[3]:
        sk = pd.read_parquet(w / "cannibal_stacking.parquet")
        sk["delinquent_share"] *= 100
        ui.table(sk.rename(columns={"n_products": "Products held", "customers": "Customers", "avg_exposure_usd": "Avg owed (USD)", "avg_limit_to_income": "Limits ÷ monthly income",
                                    "delinquent_share": "Late on any product (%)"}).round(2))
        ui.callout("Customers holding several credit products carry more total credit relative to income. Scores are per product, but the **limit decision should be made per customer**: "
                   "add a total-exposure cap (the Credit limits page already applies one).", warn=True)
    with tabs[4]:
        est = pd.DataFrame({"Method": ["Naive: adopters vs non-adopters", "Before vs after (adopters only)", "Matched look-alikes (this analysis)"],
                            "Effect on card spend": [e["naive_level_gap"], e["before_after"], e["matched_did"]]})
        fig = go.Figure(go.Bar(x=est["Method"], y=est["Effect on card spend"] * 100, marker_color=["#bbb", "#bbb", "#3B5BDB"],
                               error_y=dict(type="data", array=[0, 0, (e["ci95"][1] - e["ci95"][0]) * 50], visible=True)))
        truth = gt.get("tau_card_spend_avg_over_adopters_with_card")
        if truth:
            fig.add_hline(y=-truth * 100, line_dash="dash", line_color="#2e9e6b", annotation_text="Planted truth (demo data only)")
        fig.update_layout(title="Three ways to estimate the same effect (% change in card spend)", yaxis_title="%")
        ui.show(fig, 380)
        st.markdown("- **Naive** compares different people: BNPL users shop more, so it understates the drop.\n"
                    "- **Before vs after** ignores seasonality (holiday spending) and trends, so it overstates the drop.\n"
                    "- **Matched look-alikes** compare like with like over the same months. In this demo world we know the true answer, and this method lands closest.")
        if truth:
            st.caption(f"Truth check: the data was generated with an average card-spend reduction of {truth * 100:.0f}% for adopters. "
                       "With real data you will not have this line; the matched estimate and its range are what you report.")
