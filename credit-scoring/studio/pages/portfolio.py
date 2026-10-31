"""Portfolio overview."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from studio import ai_studio, data, stats, text, ui


def render():
    ui.header("Portfolio", "How healthy is the book, where is the risk, and is it getting better or worse?", "📊")
    ui.guide("portfolio", "Use the two filters to focus on a product or country. **NPL ratio** = share of money owed that is 90+ days late. "
                          "**30+ days late** is the early warning. The map shows where the money is (bubble size) and how bad it is (colour).")
    cc = data.q("SELECT code, name FROM countries ORDER BY name")
    f = st.columns(2)
    prods = f[0].multiselect("Product", ["CARD", "LOAN", "BNPL"], key="pf_prod", format_func=lambda x: text.PRODUCT_NAMES[x])
    ctry = f[1].multiselect("Country", list(cc.code), key="pf_ctry", format_func=lambda x: dict(zip(cc.code, cc.name))[x])
    k = stats.kpis(prods, ctry)
    m = st.columns(6)
    m[0].metric("Active accounts", f"{int(k['accounts']):,}", f"{int(k['accounts'] - k['accounts_ago']):+,} vs 3m ago")
    m[1].metric("Money owed", ui.money(k["exposure"]), f"{(k['exposure'] / k['exposure_ago'] - 1) * 100:+.1f}%")
    m[2].metric("NPL ratio", ui.pct(k["npl_ratio"], 2), f"{(k['npl_ratio'] - k['npl_ago']) * 100:+.2f} pts", delta_color="inverse",
                help="Share of money owed that is 90+ days late.")
    m[3].metric("30+ days late", ui.pct(k["dpd30_ratio"], 2), f"{(k['dpd30_ratio'] - k['dpd30_ago']) * 100:+.2f} pts", delta_color="inverse")
    m[4].metric("Avg 6m risk", ui.pct(k["avg_pd"], 2), help="Average probability of reaching 90+ days late within 6 months.")
    m[5].metric("Watchlist", f"{int(k['watchlist']):,}", help="Accounts whose predicted risk has risen fast in 3 months.")

    with st.container(border=True):
        st.markdown("**🧠 Briefing**")
        if st.button("Write this month's briefing", key="brief_btn"):
            st.session_state["briefing"] = ai_studio.portfolio_briefing(stats.briefing_inputs(prods, ctry), ui.get_llm())
        if "briefing" in st.session_state:
            t, src = st.session_state["briefing"]
            st.markdown(t)
            st.caption(f"Written by: {'AI' if src == 'llm' else 'built-in template'}")
        else:
            st.caption("A short plain-language summary of what changed and what to do.")

    tabs = st.tabs(["Where", "Trend", "Risk mix", "How accounts move", "Vintages"])
    with tabs[0]:
        g = stats.geo(prods, ctry)
        fig = px.scatter_geo(g, lat="lat", lon="lon", size="exposure", color="npl_ratio", hover_name="city",
                             hover_data={"country": True, "accounts": ":,", "exposure": ":$,.0f", "npl_ratio": ":.2%", "lat": False, "lon": False},
                             color_continuous_scale=["#2e9e6b", "#f2c14e", "#d1495b"], size_max=46, projection="natural earth")
        fig.update_layout(coloraxis_colorbar=dict(title="NPL ratio", tickformat=".1%"), geo=dict(showcountries=True, countrycolor="#c8cfdd",
                          showland=True, landcolor="#f2f4f8", showocean=True, oceancolor="#e8eefb"))
        ui.show(fig, 470)
        t = g.groupby("country").agg(accounts=("accounts", "sum"), exposure=("exposure", "sum")).reset_index()
        t["npl_ratio"] = g.assign(x=g.npl_ratio * g.exposure).groupby("country")["x"].sum().values / t["exposure"].values
        t = t.sort_values("npl_ratio", ascending=False)
        ui.table(t.rename(columns={"country": "Country", "accounts": "Accounts", "exposure": "Owed (USD)", "npl_ratio": "NPL ratio"}))
    with tabs[1]:
        t = stats.trend(prods, ctry)
        a, b = st.columns(2)
        for col, metric, title in ((a, "npl_ratio", "NPL ratio (90+ days late)"), (b, "dpd30_ratio", "30+ days late")):
            fig = px.line(t, x="month", y=metric, color="product", color_discrete_map={"CARD": "#3B5BDB", "LOAN": "#2e9e6b", "BNPL": "#ee8434"})
            fig.update_layout(title=title, yaxis=dict(tickformat=".1%"), xaxis=dict(nticks=8))
            with col:
                ui.show(fig, 330)
        fig = px.area(t, x="month", y="exposure", color="product", color_discrete_map={"CARD": "#3B5BDB", "LOAN": "#2e9e6b", "BNPL": "#ee8434"})
        fig.update_layout(title="Money owed by product (USD)", xaxis=dict(nticks=8))
        ui.show(fig, 300)
    with tabs[2]:
        bm = stats.band_mix(prods, ctry)
        fig = px.bar(bm, x="product", y="exposure", color="risk_band", color_discrete_map=text.BAND_COLORS,
                     category_orders={"risk_band": ["A", "B", "C", "D", "E", "NPL"]})
        fig.update_layout(title="Money owed by risk band", barmode="stack")
        ui.show(fig, 340)
        st.caption("Bands come from the behavioural model: A < 1.5% chance of serious default in 6 months, B < 3.5%, C < 8%, D < 18%, E above.")
    with tabs[3]:
        rr = stats.roll_rates(prods, ctry)
        order = ["Current", "1-29", "30-59", "60-89", "90+"]
        pv = rr.pivot(index="from_b", columns="to_b", values="share").reindex(index=order, columns=order + ["Written off"]).fillna(0)
        fig = go.Figure(go.Heatmap(z=pv.values * 100, x=pv.columns, y=pv.index, colorscale="YlOrRd", text=(pv.values * 100).round(1),
                                   texttemplate="%{text}%", zmax=60))
        fig.update_layout(title="Where accounts go next month (last 12 months)", xaxis_title="Next month", yaxis_title="This month",
                          yaxis=dict(autorange="reversed"))
        ui.show(fig, 380)
        st.caption("Read a row: of accounts that are in the stage on the left, what share ends up in each stage a month later. Diagonal = stay; to the right = getting worse.")
    with tabs[4]:
        v = stats.vintage(prods, ctry)
        if v.empty:
            st.info("Not enough data for the selected filters.")
        else:
            fig = px.line(v, x="mob", y="bad_rate", color="cohort", facet_col="product", markers=True)
            fig.update_layout(title="Share of accounts that reached 90+ days late, by months since opening (cohort = opening year)",
                              yaxis=dict(tickformat=".0%"))
            ui.show(fig, 360)
