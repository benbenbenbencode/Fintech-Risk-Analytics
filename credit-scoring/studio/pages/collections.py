"""Collections: who to contact, what works, what to say."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from studio import ai_studio, cached, data, policy, text, ui
from studio.pages.customer360 import render_customer

STAGE = {1: "1-29 days", 2: "30-59 days", 3: "60-89 days", 4: "90+ days"}


def render():
    ui.header("Collections", "Late accounts ranked by value, with the action that is most likely to work for each one.", "📞")
    ui.guide("collections", "**Work list:** accounts ranked by expected net value (chance of recovery × money saved − cost of contact). "
                            "Click a row for the reasoning and a drafted message.\n\n**What works:** results of randomised tests, so you can see "
                            "which actions actually help at each stage. Nothing is sent automatically.")
    plan = cached.collections_plan()
    plan = plan.dropna(subset=["action"])
    m = st.columns(5)
    m[0].metric("Late accounts", f"{len(plan):,}")
    m[1].metric("Money owed", ui.money(plan.balance_usd.sum()))
    m[2].metric("90+ days late", f"{int((plan.bucket == 4).sum()):,}")
    m[3].metric("Expected recoveries", ui.money((plan.p_cure_with_action * plan.balance_usd).sum()), help="Balance × chance each account gets back on track with the recommended action.")
    m[4].metric("Contact cost", ui.money(plan.action_cost_usd.sum()))
    tabs = st.tabs(["Work list", "What works", "Stages"])
    with tabs[0]:
        f = st.columns(4)
        stage = f[0].multiselect("Stage", list(STAGE), format_func=STAGE.get, key="co_stage")
        prod = f[1].multiselect("Product", ["CARD", "LOAN", "BNPL"], format_func=lambda x: text.PRODUCT_NAMES[x], key="co_prod")
        ctry = f[2].multiselect("Country", sorted(plan.country.unique()), key="co_ctry")
        act = f[3].multiselect("Recommended action", sorted(plan.action_label.unique()), key="co_act")
        v = plan
        if stage: v = v[v.bucket.isin(stage)]
        if prod: v = v[v["product"].isin(prod)]
        if ctry: v = v[v.country.isin(ctry)]
        if act: v = v[v.action_label.isin(act)]
        v = v.head(800).reset_index(drop=True)
        show = v[["customer_id", "product", "bucket", "dpd_now", "balance_usd", "country", "p_cure_no_action", "p_cure_with_action", "action_label",
                  "expected_value_usd"]].copy()
        show["bucket"] = show["bucket"].map(STAGE)
        show["p_cure_no_action"] *= 100
        show["p_cure_with_action"] *= 100
        show.columns = ["Customer", "Product", "Stage", "Days late", "Owes", "Country", "Recovery chance now", "With action", "Do this", "Net value"]
        ev = st.dataframe(show, hide_index=True, on_select="rerun", selection_mode="single-row", key="co_tbl", height=360, column_config={
            "Owes": st.column_config.NumberColumn(format="$%d"), "Net value": st.column_config.NumberColumn(format="$%d"),
            "Recovery chance now": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%"),
            "With action": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%")},
            **({"width": "stretch"} if ui._W_NEW else {"use_container_width": True}))
        st.download_button("Download work list (CSV)", show.to_csv(index=False), "collections_worklist.csv", "text/csv")
        sel = ev.selection.rows if ev is not None and hasattr(ev, "selection") else []
        if sel:
            r = v.iloc[sel[0]]
            with st.container(border=True):
                st.markdown(f"**{r.customer_id} · {text.PRODUCT_NAMES[r['product']]} · {int(r.dpd_now)} days late · owes {ui.money(r.balance_usd, False)}**")
                st.info(policy.rationale_text(r))
                if r.alternative:
                    st.caption(f"Next best: {policy.ACTION_LABELS[r.alternative]}")
            render_customer(r.customer_id, key="co360")
    with tabs[1]:
        up = data.q("SELECT * FROM uplift")
        b = st.radio("Stage", list(STAGE), format_func=STAGE.get, horizontal=True, key="co_up_stage")
        u = up[(up.bucket == b) & (up.n >= 100)].sort_values("uplift")
        u["label"] = u.action.map(policy.ACTION_LABELS)
        fig = go.Figure(go.Bar(x=u.uplift * 100, y=u.label, orientation="h", error_x=dict(type="data", array=1.96 * u.se * 100),
                               marker_color=["#2e9e6b" if x > 0 else "#d1495b" for x in u.uplift]))
        fig.update_layout(title=f"How much each action raises the chance of getting back on track ({STAGE[b]} late), vs doing nothing",
                          xaxis_title="Percentage points (bars show 95% range)")
        ui.show(fig, 340)
        ui.table(u[["label", "n", "cure_rate", "uplift", "cost"]].rename(columns={"label": "Action", "n": "Test cases", "cure_rate": "Got back on track",
                                                                                 "uplift": "Lift vs no action", "cost": "Avg cost (USD)"}).round(3))
        st.caption("Source: the ~20% of cases where the action was chosen at random, which removes the bias of 'collectors call the hard cases'.")
        with st.expander("Rules every action follows"):
            for g in policy.GUARDRAILS:
                st.write("• " + g)
    with tabs[2]:
        s = plan.groupby("bucket").agg(accounts=("account_id", "size"), owed=("balance_usd", "sum")).reset_index()
        s["stage"] = s.bucket.map(STAGE)
        a, b2 = st.columns(2)
        with a:
            ui.show(go.Figure(go.Bar(x=s.stage, y=s.accounts, marker_color=["#a8d08d", "#f2c14e", "#ee8434", "#d1495b"])).update_layout(title="Accounts by stage"), 300)
        with b2:
            ui.show(go.Figure(go.Bar(x=s.stage, y=s.owed, marker_color=["#a8d08d", "#f2c14e", "#ee8434", "#d1495b"])).update_layout(title="Money owed by stage (USD)"), 300)
