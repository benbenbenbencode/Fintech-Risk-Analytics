"""Customer explorer: list everyone, search in plain English, open any customer."""
from __future__ import annotations

import streamlit as st

from studio import ai_studio, data, text, ui
from studio.pages.customer360 import render_customer

QUICK = {"NPL (90+ days late)": {"status": ["npl"]}, "Late payers": {"status": ["delinquent", "npl"]},
         "Watchlist (risk rising)": {"watchlist": True}, "High risk (D/E)": {"risk_bands": ["D", "E"]},
         "Heavy usage (>80%)": {"min_util": 0.8}, "Has 2+ products": {"multi": True}}
RANK = "CASE risk_band WHEN 'A' THEN 1 WHEN 'B' THEN 2 WHEN 'C' THEN 3 WHEN 'D' THEN 4 WHEN 'E' THEN 5 ELSE 6 END"
INV = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "NPL"}
KEYS = {"f_countries": [], "f_cities": [], "f_products": [], "f_bands": [], "f_status": [], "f_dpd": 0, "f_bal": 0, "f_watch": False}


def _apply_nl():
    q = st.session_state.get("nl_q", "")
    if not q.strip():
        return
    f, src = ai_studio.search_filters(q, ui.get_llm())
    st.session_state.update({"f_countries": f.get("countries", []), "f_cities": f.get("cities", []), "f_products": f.get("products", []),
                             "f_bands": f.get("risk_bands", []), "f_status": f.get("status", []), "f_dpd": int(f.get("min_dpd", 0)),
                             "f_bal": int(f.get("min_balance", 0)), "f_watch": bool(f.get("watchlist", False)), "nl_src": src,
                             "nl_util": f.get("min_util", 0)})


def _reset():
    st.session_state.update(KEYS)
    st.session_state["nl_q"] = ""
    st.session_state["nl_util"] = 0


def render():
    ui.header("Customers", "Find any customer or group of customers, then open the full picture.", "👥")
    ui.guide("customers", "1. **Type what you are looking for** in plain English, e.g. *NPL credit card customers in Jakarta over 60 days late*.\n"
                          "2. Or use the **quick filters** and the filter panel.\n3. **Click a row** to open the customer. "
                          "Risk bands go from **A (safest)** to **E (riskiest)**; **NPL** means already 90+ days late.")
    for k, v in KEYS.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("nl_util", 0)
    c1, c2 = st.columns([4, 1])
    c1.text_input("Search", key="nl_q", on_change=_apply_nl, placeholder="Try: high risk BNPL customers in Brazil on the watchlist",
                  label_visibility="collapsed")
    c2.button("Clear all", on_click=_reset, key="clear_filters")
    quick = st.pills("Quick filters", list(QUICK), selection_mode="multi", key="f_quick") or []
    cc = data.q("SELECT code, name FROM countries ORDER BY name")
    with st.expander("More filters", expanded=bool(st.session_state.get("nl_src"))):
        a = st.columns(4)
        a[0].multiselect("Country", list(cc.code), key="f_countries", format_func=lambda x: dict(zip(cc.code, cc.name))[x])
        a[1].multiselect("City", list(data.q("SELECT city FROM cities ORDER BY city").city), key="f_cities")
        a[2].multiselect("Product", ["CARD", "LOAN", "BNPL"], key="f_products", format_func=lambda x: text.PRODUCT_NAMES[x])
        a[3].multiselect("Risk band", ["A", "B", "C", "D", "E", "NPL"], key="f_bands")
        b = st.columns(4)
        b[0].multiselect("Status", ["active", "delinquent", "npl"], key="f_status")
        b[1].number_input("Days late at least", 0, 360, step=15, key="f_dpd")
        b[2].number_input("Owes at least (USD)", 0, 100000, step=500, key="f_bal")
        b[3].checkbox("Watchlist only", key="f_watch")
    f = {"countries": st.session_state.f_countries, "cities": st.session_state.f_cities, "products": st.session_state.f_products,
         "risk_bands": st.session_state.f_bands, "status": st.session_state.f_status, "min_dpd": st.session_state.f_dpd,
         "min_balance": st.session_state.f_bal, "watchlist": st.session_state.f_watch, "min_util": st.session_state.get("nl_util", 0)}
    multi = False
    for qk in quick:
        spec = QUICK[qk]
        multi |= spec.get("multi", False)
        for k2, v in spec.items():
            if k2 == "status": f["status"] = sorted(set(f["status"]) | set(v)) if f["status"] else v
            elif k2 == "risk_bands": f["risk_bands"] = sorted(set(f["risk_bands"]) | set(v)) if f["risk_bands"] else v
            elif k2 == "watchlist": f["watchlist"] = True
            elif k2 == "min_util": f["min_util"] = v
    f = ai_studio._clean({k: v for k, v in f.items() if v})
    st.caption(f"Showing: **{ai_studio.describe_filters(f)}**" + (f" · understood by {'AI' if st.session_state.get('nl_src') == 'llm' else 'built-in parser'}"
                                                                  if st.session_state.get("nl_q") else ""))
    where, params = ai_studio.where_clause(f)
    sort = st.radio("Sort by", ["Highest risk", "Most owed", "Most days late"], horizontal=True, key="cust_sort")
    order = {"Highest risk": "worst_pd DESC", "Most owed": "owed DESC", "Most days late": "max_dpd DESC, worst_pd DESC"}[sort]
    having = "HAVING count(*) > 1" if multi else ""
    df = data.q(f"""SELECT customer_id, any_value(country) country, any_value(city) city, any_value(age) age, any_value(employment_type) employment,
        string_agg(product, ' + ' ORDER BY product) products, sum(balance_usd) owed, sum(credit_limit_usd) limits, max(dpd_now) max_dpd,
        max(pd_6m) worst_pd, max({RANK}) band_rank, bool_or(watchlist) watchlist
        FROM book WHERE customer_id IN (SELECT DISTINCT customer_id FROM book WHERE {where}) GROUP BY customer_id {having}
        ORDER BY {order} LIMIT 3000""", params)
    if df.empty:
        st.info("No customers match. Try removing a filter.")
        return
    df["band"] = df["band_rank"].map(INV)
    m = st.columns(4)
    m[0].metric("Customers", f"{len(df):,}" + ("+" if len(df) == 3000 else ""))
    m[1].metric("Total owed", ui.money(df.owed.sum()))
    m[2].metric("Late 30+ days", f"{int((df.max_dpd >= 30).sum()):,}")
    m[3].metric("On watchlist", f"{int(df.watchlist.sum()):,}")
    view = df.rename(columns={"customer_id": "Customer", "country": "Country", "city": "City", "age": "Age", "employment": "Employment",
                              "products": "Products", "owed": "Owes (USD)", "limits": "Limit / loan (USD)", "max_dpd": "Days late",
                              "worst_pd": "6m risk", "band": "Band", "watchlist": "Watch"})
    view["6m risk"] = view["6m risk"] * 100
    view = view[["Customer", "Band", "6m risk", "Days late", "Owes (USD)", "Limit / loan (USD)", "Products", "Country", "City", "Age", "Employment", "Watch"]]
    cfg = {"6m risk": st.column_config.ProgressColumn("6m risk", min_value=0, max_value=30, format="%.1f%%",
                                                      help="Chance of reaching 90+ days late within 6 months"),
           "Owes (USD)": st.column_config.NumberColumn(format="$%d"), "Limit / loan (USD)": st.column_config.NumberColumn(format="$%d")}
    ev = st.dataframe(view, hide_index=True, on_select="rerun", selection_mode="single-row", column_config=cfg, key="cust_table",
                      **({"width": "stretch"} if ui._W_NEW else {"use_container_width": True}), height=360)
    st.download_button("Download this list (CSV)", view.to_csv(index=False), "customers.csv", "text/csv")
    sel = ev.selection.rows if ev is not None and hasattr(ev, "selection") else []
    if sel:
        st.session_state["open_customer"] = view.iloc[sel[0]]["Customer"]
    st.divider()
    manual = st.text_input("Or open a customer by ID", placeholder="CU0000010", key="manual_cid")
    cid = manual.strip().upper() or st.session_state.get("open_customer")
    if cid:
        render_customer(cid, key="cust")
