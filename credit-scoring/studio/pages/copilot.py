"""Ask the portfolio a question in plain English."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from studio import ai_studio, data, ui

EXAMPLES = ["NPL ratio by country", "How much exposure by product?", "Which cities have the most watchlist accounts?",
            "Average risk by risk band for BNPL", "Delinquent accounts in Brazil by product", "Utilization by country for credit cards"]


def _answer(q: str):
    plan, src = ai_studio.make_plan(q, ui.get_llm())
    df = ai_studio.run_plan(plan)
    label = ai_studio.METRICS[plan["metric"]][1]
    return plan, src, df, label


def render():
    ui.header("Ask a question", "Ask about the portfolio in plain English. You always see how the question was understood.", "💬")
    ui.guide("copilot", "Ask things like *NPL ratio by country* or *how much exposure in Lagos by product*. "
                        "The assistant turns your question into a safe, fixed set of calculations (metric, grouping, filters). "
                        "It cannot invent numbers: every answer comes straight from the portfolio data.")
    st.session_state.setdefault("chat", [])
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex, key=f"ex_{i}"):
            st.session_state["pending_q"] = ex
    q = st.chat_input("Ask about the portfolio...") or st.session_state.pop("pending_q", None)
    if q:
        plan, src, df, label = _answer(q)
        st.session_state["chat"].append({"q": q, "plan": plan, "src": src, "df": df, "label": label})
    for msg in st.session_state["chat"][-6:]:
        with st.chat_message("user"):
            st.write(msg["q"])
        with st.chat_message("assistant"):
            df, plan = msg["df"], msg["plan"]
            if df.empty or df["value"].isna().all():
                st.write("No data matches that question.")
            elif plan["group_by"]:
                top = df.iloc[0]
                st.write(f"**{msg['label']}** by {plan['group_by'].replace('_', ' ')}: highest is **{top[plan['group_by']]}** at **{_fmt(plan['metric'], top['value'])}**.")
                fig = px.bar(df, x=plan["group_by"], y="value", labels={"value": msg["label"]})
                if plan["metric"] in ("npl_ratio", "dpd30_ratio", "avg_pd", "avg_utilization"):
                    fig.update_layout(yaxis=dict(tickformat=".1%"))
                ui.show(fig, 300)
                ui.table(df.assign(value=df["value"].round(4)))
            else:
                st.write(f"**{msg['label']}**: **{_fmt(plan['metric'], df['value'].iloc[0])}**")
            st.caption(f"Understood as: {msg['label']} · group by {plan['group_by'] or 'nothing'} · filters: {ai_studio.describe_filters(plan['filters'])} "
                       f"· interpreted by {'AI' if msg['src'] == 'llm' else 'built-in parser'}")


def _fmt(metric, v):
    if metric in ("npl_ratio", "dpd30_ratio", "avg_pd", "avg_utilization"):
        return ui.pct(v, 2)
    if metric == "exposure_usd":
        return ui.money(v)
    return f"{v:,.0f}"
