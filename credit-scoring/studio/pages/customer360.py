"""Customer 360: everything about one customer on one screen, in plain language."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from studio import ai_studio, cached, data, policy, text, ui
from studio.policy import ACTION_LABELS


def _reason_lines(raw: str, n: int = 3) -> list[str]:
    return [text.reason_phrase(r["f"], r["bin"]) for r in policy.reasons(raw)[:n]]


def render_customer(cid: str, key: str = "c360"):
    cu = data.q("SELECT * FROM customers WHERE customer_id = ?", [cid])
    if cu.empty:
        st.warning("Customer not found.")
        return
    c = cu.iloc[0]
    ac = data.q("SELECT * FROM book WHERE customer_id = ? ORDER BY product", [cid])
    if ac.empty:
        st.info("This customer has no active account.")
        return
    order = ["A", "B", "C", "D", "E", "NPL"]
    worst = max(ac["risk_band"], key=order.index)
    st.markdown(f"### {cid} · {c.age} y/o · {c.employment_type.replace('_', ' ')} · {c.city}, {c.country} &nbsp; {ui.band_pill(worst)}",
                unsafe_allow_html=True)
    k = st.columns(5)
    k[0].metric("Products", len(ac))
    k[1].metric("Total owed", ui.money(ac.balance_usd.sum()))
    k[2].metric("Total limit / loan", ui.money(ac.credit_limit_usd.sum()))
    k[3].metric("Worst delay", f"{int(ac.dpd_now.max())} days", help="DPD = days past due on the most overdue account.")
    k[4].metric("Highest 6-month risk", ui.pct(ac.pd_6m.max()), help="Probability of reaching 90+ days late within 6 months (1 = already there).")
    t1, t2, t3, t4 = st.tabs(["Overview", "History", "Recommended action", "AI brief"])

    with t1:
        cols = st.columns(len(ac))
        for col, (_, a) in zip(cols, ac.iterrows()):
            with col, st.container(border=True):
                st.markdown(f"**{text.PRODUCT_ICONS[a['product']]} {text.PRODUCT_NAMES[a['product']]}** &nbsp; {ui.band_pill(a['risk_band'])}", unsafe_allow_html=True)
                st.caption(f"{a.account_id} · open {int(a.mob)} months")
                m = st.columns(2)
                m[0].metric("Owed", ui.money(a.balance_usd, False))
                m[1].metric("Late by", f"{int(a.dpd_now)} d")
                m = st.columns(2)
                m[0].metric("Usage", ui.pct(a.utilization, 0), help="Share of the limit or loan still outstanding.")
                m[1].metric("6m risk", ui.pct(a.pd_6m))
                lines = _reason_lines(a.reasons)
                if lines:
                    st.caption("Why this risk level:")
                    for l in lines:
                        st.caption("• " + l)
                if a.watchlist:
                    st.warning("On watchlist: risk is rising quickly.")
        prof = st.expander("Customer profile")
        with prof:
            p = st.columns(4)
            p[0].metric("Monthly income", ui.money(c.monthly_income_usd, False))
            p[1].metric("Bureau score", "No record" if c.bureau_score != c.bureau_score else int(c.bureau_score))
            p[2].metric("Joined", str(c.join_date)[:7])
            p[3].metric("Channel", c.acquisition_channel.replace("_", " "))

    with t2:
        h = data.q("SELECT account_id, product, month_idx, balance_usd, dpd, status FROM account_month WHERE customer_id = ? ORDER BY month_idx", [cid])
        h["month"] = h["month_idx"].map(data.month_label)
        fig = go.Figure()
        for acc_id, g in h.groupby("account_id"):
            fig.add_scatter(x=g["month"], y=g["balance_usd"], name=f"{text.PRODUCT_NAMES[g['product'].iloc[0]]} balance", mode="lines")
        fig.update_layout(title="Amount owed over time (USD)")
        ui.show(fig, 300)
        fig = go.Figure()
        for acc_id, g in h.groupby("account_id"):
            fig.add_scatter(x=g["month"], y=g["dpd"], name=text.PRODUCT_NAMES[g["product"].iloc[0]], mode="lines+markers")
        fig.add_hline(y=90, line_dash="dash", annotation_text="90 days = NPL")
        fig.update_layout(title="Days past due")
        ui.show(fig, 280)
        ph = data.q(f"SELECT account_id, month_idx, pd_6m FROM pd_history WHERE account_id IN ({','.join('?' * len(ac))}) ORDER BY month_idx",
                    list(ac.account_id))
        if len(ph):
            ph["month"] = ph["month_idx"].map(data.month_label)
            fig = go.Figure()
            for acc_id, g in ph.groupby("account_id"):
                fig.add_scatter(x=g["month"], y=g["pd_6m"], name=acc_id, mode="lines")
            fig.update_layout(title="Predicted 6-month risk over time", yaxis=dict(tickformat=".0%"))
            ui.show(fig, 280)

    rec_text, ctx = "", {}
    with t3:
        late = ac[ac.status != "active"]
        if len(late):
            plan = cached.collections_plan()
            for _, a in late.iterrows():
                row = plan[plan.account_id == a.account_id]
                if row.empty:
                    continue
                r = row.iloc[0]
                with st.container(border=True):
                    st.markdown(f"**{text.PRODUCT_NAMES[a['product']]} · {int(a.dpd_now)} days late · owes {ui.money(a.balance_usd, False)}**")
                    s = st.columns(3)
                    s[0].metric("Recommended", r.action_label)
                    s[1].metric("Chance to recover", ui.pct(r.p_cure_with_action, 0), delta=f"{(r.p_cure_with_action - r.p_cure_no_action) * 100:+.0f} pts vs no action")
                    s[2].metric("Expected net value", ui.money(r.expected_value_usd, False))
                    st.caption(policy.rationale_text(r))
                    rec_text = f"{r.action_label} for the {text.PRODUCT_NAMES[a['product']]}"
                    ctx = {"action": r.action, "product_name": text.PRODUCT_NAMES[a["product"]].lower(), "amount": ui.money(max(a.payment_due_usd, 1), False),
                           "dpd": int(a.dpd_now)}
            if ctx:
                st.markdown("**Draft the outreach**")
                lc = st.columns(3)
                lang = lc[0].selectbox("Language", ["English", "Bahasa Indonesia", "Español", "Português", "Polski", "Tiếng Việt", "Čeština"], key=f"{key}_lang")
                tone = lc[1].selectbox("Tone", ["empathetic", "neutral", "firm but respectful"], key=f"{key}_tone")
                if lc[2].button("Draft message", key=f"{key}_draft"):
                    st.session_state[f"{key}_msg"] = ai_studio.draft_message(ctx["action"], ctx, lang, tone, ui.get_llm())
                if f"{key}_msg" in st.session_state:
                    msg, src = st.session_state[f"{key}_msg"]
                    st.text_area("Message (review before sending)", msg, height=140, key=f"{key}_msgbox")
                    st.caption(f"Written by: {'AI' if src == 'llm' else 'built-in template'} · a human approves before anything is sent")
                with st.expander("Rules every message follows"):
                    for g in policy.GUARDRAILS:
                        st.write("• " + g)
        else:
            cli = cached.cli_proposals(__import__("json").dumps(policy.DEFAULT_CLI), ("CARD", "BNPL"))
            mine = cli[cli.customer_id == cid]
            if mine.empty:
                st.info("Account is performing normally. No credit-limit review applies (loans have no limit).")
            for _, r in mine.iterrows():
                with st.container(border=True):
                    st.markdown(f"**{text.PRODUCT_NAMES[r['product']]}: {r.action}**")
                    if r.action == "Increase":
                        s = st.columns(3)
                        s[0].metric("Limit now → proposed", f"{ui.money(r.credit_limit_usd, False)} → {ui.money(r.proposed_limit_usd, False)}")
                        s[1].metric("Expected 12m profit", ui.money(r.net_value_usd, False))
                        s[2].metric("6m risk", ui.pct(r.pd_6m))
                        rec_text = f"Raise the {text.PRODUCT_NAMES[r['product']]} limit to {ui.money(r.proposed_limit_usd, False)}"
                    st.caption(r.rationale)

    with t4:
        st.caption("A short written summary for managers. Numbers come from the models; the AI only explains them.")
        if st.button("Write the brief", key=f"{key}_brief"):
            facts = {"customer": cid, "age": int(c.age), "employment_type": c.employment_type, "city": c.city, "country": c.country,
                     "income": float(c.monthly_income_usd), "recommendation": rec_text,
                     "reasons": sum([_reason_lines(a.reasons) for _, a in ac.iterrows()], [])[:4],
                     "accounts": [{"product": a["product"], "status": a.status, "risk_band": a.risk_band, "pd_6m": float(a.pd_6m),
                                   "dpd": int(a.dpd_now), "balance": float(a.balance_usd)} for _, a in ac.iterrows()]}
            st.session_state[f"{key}_brief_txt"] = ai_studio.customer_brief(facts, ui.get_llm())
        if f"{key}_brief_txt" in st.session_state:
            t, src = st.session_state[f"{key}_brief_txt"]
            ui.callout(t)
            st.caption(f"Written by: {'AI' if src == 'llm' else 'built-in template'}")
