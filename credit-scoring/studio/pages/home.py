"""Start page: what can I do here, today's picture, glossary."""
from __future__ import annotations

import streamlit as st

from studio import data, stats, ui


def _card(icon, title, body, page_key):
    with st.container(border=True):
        st.markdown(f"#### {icon} {title}")
        st.caption(body)
        if page_key in ui.PAGES:
            st.page_link(ui.PAGES[page_key], label="Open →")


def render():
    m = data.meta()
    st.markdown(f"""<div class='hero'><h1>Credit Risk Studio</h1>
    <p>Decide on new customers, manage limits, collect smarter and watch the portfolio, across credit cards, term loans and BNPL in 10 countries.</p></div>""",
                unsafe_allow_html=True)
    k = stats.kpis()
    c = st.columns(5)
    c[0].metric("Customers", f"{m['rows']['customers']:,}")
    c[1].metric("Active accounts", f"{int(k['accounts']):,}")
    c[2].metric("Money owed", ui.money(k["exposure"]))
    c[3].metric("NPL ratio", ui.pct(k["npl_ratio"], 2), f"{(k['npl_ratio'] - k['npl_ago']) * 100:+.2f} pts vs 3m", delta_color="inverse")
    c[4].metric("Needs attention", f"{int(k['watchlist'] + k['in_collections']):,}", help="Watchlist plus accounts already late.")
    st.markdown("#### What would you like to do?")
    r1 = st.columns(3)
    with r1[0]: _card("🆕", "Decide on new applications", "See who the model would approve, refer or decline, and test different cut-offs.", "onboarding")
    with r1[1]: _card("📈", "Review credit limits", "Who deserves a higher limit, by how much, and what it is worth.", "limits")
    with r1[2]: _card("📞", "Work the collections list", "Late accounts ranked by value, with the best action for each.", "collections")
    r2 = st.columns(3)
    with r2[0]: _card("👥", "Find a customer", "Search in plain English, filter NPL, open any customer.", "customers")
    with r2[1]: _card("📊", "Watch the portfolio", "Map, trends, risk mix, roll rates and vintages.", "portfolio")
    with r2[2]: _card("🔀", "Is BNPL eating card revenue?", "Cannibalization measured the right way, with a known answer to check against.", "cannibal")
    r3 = st.columns(3)
    with r3[0]: _card("💬", "Ask a question", "\"NPL ratio by country\", \"exposure in Lagos by product\"...", "copilot")
    with r3[1]: _card("🧪", "Model studio", "For analysts: performance, drivers, stability, scorecard vs black box.", "models")
    with r3[2]:
        with st.container(border=True):
            st.markdown("#### ℹ️ About the data")
            st.caption(f"Synthetic world, as of {data.month_label(data.as_of_idx())}: {m['rows']['account_month']:,} monthly account records. "
                       "Realistic structure, not real customers.")
    with st.expander("📖 Plain-language glossary"):
        st.markdown("""
- **DPD (days past due)**: how many days a payment is late. **NPL** = non-performing loan, 90+ days late.
- **PD (probability of default)**: the model's chance that an account reaches 90+ days late. We show it for the next **6 months** (existing accounts) or **12 months** (new applications).
- **Risk band**: A (safest) to E (riskiest) groups customers by PD. **NPL** is a separate band for accounts already 90+ days late.
- **Utilization / usage**: how much of a credit limit is in use.
- **Watchlist**: accounts whose predicted risk rose fast over 3 months, even if they are still paying.
- **Cure**: a late account getting back to current.
- **Cannibalization**: a new product taking customers or spending from an existing one instead of adding new business.
- **Scorecard**: an explainable model where every factor adds or removes points.""")
    with st.expander("🧭 A 2-minute tour"):
        st.markdown("1. **Portfolio** to see the overall picture.\n2. **Collections** to see who to contact today and what to say.\n"
                    "3. **Credit limits** for safe growth.\n4. **Onboarding** to test how strict the approval policy should be.\n"
                    "5. **Customers** to open any person: risk, history, reasons, recommended action and a drafted message.")
