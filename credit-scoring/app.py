"""Credit Risk Studio.   Run:  streamlit run app.py
First run builds the synthetic world (about a minute); afterwards it opens instantly."""
from __future__ import annotations

import streamlit as st

from studio import data, ui

st.set_page_config(page_title="Credit Risk Studio", page_icon="🏦", layout="wide")
ui.inject_css()


def builder():
    st.markdown("<div class='hero'><h1>Credit Risk Studio</h1><p>First run: let's build the demo world. It takes about a minute.</p></div>",
                unsafe_allow_html=True)
    n = st.select_slider("How many customers?", [20_000, 50_000, 100_000], 100_000, key="build_n",
                         help="100,000 customers gives about 1.3 million monthly account records.")
    if st.button("Build the demo world", type="primary", key="build_btn"):
        from studio import cannibal, generate_world, models
        with st.status("Building...", expanded=True) as status:
            log = lambda m: st.write(m)
            st.write("Generating customers, accounts and monthly history...")
            generate_world.generate(n, out_dir=str(data.WORLD), verbose=False)
            st.write("Training models and scoring the portfolio...")
            models.build_all(progress=log)
            st.write("Measuring product cannibalization...")
            cannibal.analyse(data.WORLD)
            data.reset()
            status.update(label="Done", state="complete")
        st.rerun()


if not data.ready():
    builder()
    st.stop()

from studio.pages import cannibal, collections, copilot, customers, home, limits, models, onboarding, portfolio  # noqa: E402

P = {
    "home": st.Page(home.render, title="Start here", icon="🏠", url_path="start", default=True),
    "portfolio": st.Page(portfolio.render, title="Portfolio", icon="📊", url_path="portfolio"),
    "customers": st.Page(customers.render, title="Customers", icon="👥", url_path="customers"),
    "onboarding": st.Page(onboarding.render, title="Onboarding", icon="🆕", url_path="onboarding"),
    "limits": st.Page(limits.render, title="Credit limits", icon="📈", url_path="limits"),
    "collections": st.Page(collections.render, title="Collections", icon="📞", url_path="collections"),
    "cannibal": st.Page(cannibal.render, title="Cannibalization", icon="🔀", url_path="cannibalization"),
    "copilot": st.Page(copilot.render, title="Ask a question", icon="💬", url_path="ask"),
    "models": st.Page(models.render, title="Model studio", icon="🧪", url_path="models"),
}
ui.PAGES.update(P)
nav = st.navigation({"": [P["home"]], "Overview": [P["portfolio"], P["customers"]],
                     "Decisions": [P["onboarding"], P["limits"], P["collections"]],
                     "Insights": [P["cannibal"], P["copilot"]], "Analyst": [P["models"]]})
ui.sidebar_ai()
st.sidebar.caption(f"Data as of {data.month_label(data.as_of_idx())} · synthetic")
nav.run()
