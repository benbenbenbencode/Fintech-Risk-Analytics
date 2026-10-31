"""AI layer for the Studio. Every function has a deterministic offline fallback; an LLM (if configured) makes the
language richer but never changes numbers, decisions or the allowed action list.

Only aggregates and one-customer fact sheets (IDs, no names) are sent to the model. For real customer data use a
private deployment or mask PII first."""
from __future__ import annotations

import json
import re

import pandas as pd

from crf.ai import LLM

from . import data
from .policy import ACTION_LABELS, GUARDRAILS

STATUSES = ["active", "delinquent", "npl"]
BANDS = ["A", "B", "C", "D", "E", "NPL"]
PRODUCTS = ["CARD", "LOAN", "BNPL"]
METRICS = {
    "accounts": ("count(*)", "Accounts"),
    "exposure_usd": ("sum(balance_usd)", "Exposure (USD)"),
    "npl_ratio": ("sum(CASE WHEN status='npl' THEN balance_usd ELSE 0 END)/NULLIF(sum(balance_usd),0)", "NPL ratio"),
    "dpd30_ratio": ("sum(CASE WHEN dpd_now>=30 THEN balance_usd ELSE 0 END)/NULLIF(sum(balance_usd),0)", "30+ DPD ratio"),
    "avg_pd": ("avg(pd_6m)", "Average 6-month PD"),
    "watchlist_accounts": ("sum(CASE WHEN watchlist THEN 1 ELSE 0 END)", "Watchlist accounts"),
    "delinquent_accounts": ("sum(CASE WHEN status<>'active' THEN 1 ELSE 0 END)", "Delinquent or NPL accounts"),
    "avg_utilization": ("avg(utilization)", "Average utilization"),
}
DIMS = {"country": "country", "city": "city", "product": "product", "risk_band": "risk_band", "status": "status",
        "employment_type": "employment_type", "channel": "acquisition_channel"}


def _options() -> dict:
    c = data.q("SELECT code, name FROM countries")
    ci = data.q("SELECT country, city FROM cities")
    return {"countries": dict(zip(c.name.str.lower(), c.code)) | dict(zip(c.code.str.lower(), c.code)),
            "cities": {x.lower(): x for x in ci.city}}


# ====================================================================================================
# natural-language search -> validated filters
# ====================================================================================================
def _clean(f: dict) -> dict:
    opt = _options()
    out = {}
    codes = set(opt["countries"].values())
    cities = set(opt["cities"].values())
    if f.get("countries"):
        out["countries"] = [c for c in f["countries"] if c in codes]
    if f.get("cities"):
        out["cities"] = [c for c in f["cities"] if c in cities]
    if f.get("products"):
        out["products"] = [p for p in f["products"] if p in PRODUCTS]
    if f.get("risk_bands"):
        out["risk_bands"] = [b for b in f["risk_bands"] if b in BANDS]
    if f.get("status"):
        out["status"] = [s for s in f["status"] if s in STATUSES]
    for k in ("min_dpd", "min_balance", "min_util"):
        if f.get(k) is not None:
            try:
                out[k] = float(f[k])
            except (TypeError, ValueError):
                pass
    if f.get("watchlist"):
        out["watchlist"] = True
    return {k: v for k, v in out.items() if v not in ([], None)}


def heuristic_filters(q: str) -> dict:
    t = " " + q.lower() + " "
    opt = _options()
    f: dict = {}
    names = {n: c for n, c in opt["countries"].items() if len(n) > 2}
    f["countries"] = sorted({code for name, code in names.items() if re.search(rf"\b{re.escape(name)}\b", t)}
                            | {code for code in set(opt["countries"].values()) if re.search(rf"\b{code}\b", q)})
    f["cities"] = sorted({name for key, name in opt["cities"].items() if re.search(rf"\b{re.escape(key)}\b", t)})
    prods = []
    if re.search(r"credit card|\bcards?\b", t): prods.append("CARD")
    if re.search(r"term loan|\bloans?\b", t): prods.append("LOAN")
    if re.search(r"bnpl|pay later|buy now", t): prods.append("BNPL")
    f["products"] = prods
    status = []
    if re.search(r"\bnpls?\b|non[- ]?performing|90\+", t): status.append("npl")
    if re.search(r"delinquen|overdue|\blate\b|past due|arrears", t): status += ["delinquent", "npl"]
    f["status"] = sorted(set(status))
    bands = []
    if re.search(r"very high risk|highest risk", t): bands = ["E"]
    elif re.search(r"high[- ]risk|risky", t): bands = ["D", "E"]
    elif re.search(r"medium risk", t): bands = ["C"]
    elif re.search(r"low[- ]risk|safe", t): bands = ["A", "B"]
    f["risk_bands"] = bands
    m = re.search(r"(?:dpd|days past due|days late|more than|over|above)\s*(?:>|>=|than)?\s*(\d{2,3})\s*(?:days|dpd)?", t)
    if m and re.search(r"dpd|days", t): f["min_dpd"] = float(m.group(1))
    m = re.search(r"(?:balance|exposure|owes?)\s*(?:over|above|>|more than|at least)?\s*\$?\s*([\d,\.]+)\s*(k)?", t)
    if m:
        try:
            v = float(m.group(1).replace(",", "")) * (1000 if m.group(2) else 1)
            f["min_balance"] = v
        except ValueError:
            pass
    m = re.search(r"utili[sz]ation\s*(?:over|above|>|more than|at least)?\s*(\d{2,3})\s*%", t)
    if m: f["min_util"] = float(m.group(1)) / 100
    if re.search(r"watch ?list|early warning|deteriorat|getting worse|rising risk", t): f["watchlist"] = True
    return _clean(f)


def search_filters(question: str, llm: LLM | None = None) -> tuple[dict, str]:
    if llm is not None and llm.available:
        try:
            opt = _options()
            system = "You convert a credit-portfolio search request into a JSON filter. Use only the allowed values."
            user = f"""Allowed values:
countries (ISO codes): {sorted(set(opt['countries'].values()))}
cities: {sorted(opt['cities'].values())}
products: {PRODUCTS}; risk_bands: {BANDS} (A lowest risk .. E highest, NPL = already 90+ DPD); status: {STATUSES}
Numeric: min_dpd (days), min_balance (USD), min_util (0-1); watchlist (true = risk rising quickly).
Request: {question}
Return JSON with any of: countries, cities, products, risk_bands, status, min_dpd, min_balance, min_util, watchlist."""
            return _clean(llm.complete_json(system, user, 500)), "llm"
        except Exception:
            pass
    return heuristic_filters(question), "offline"


def describe_filters(f: dict) -> str:
    parts = []
    if f.get("products"): parts.append("product: " + ", ".join(f["products"]))
    if f.get("countries"): parts.append("country: " + ", ".join(f["countries"]))
    if f.get("cities"): parts.append("city: " + ", ".join(f["cities"]))
    if f.get("status"): parts.append("status: " + ", ".join(f["status"]))
    if f.get("risk_bands"): parts.append("risk band: " + ", ".join(f["risk_bands"]))
    if f.get("min_dpd"): parts.append(f"DPD ≥ {f['min_dpd']:.0f}")
    if f.get("min_balance"): parts.append(f"balance ≥ ${f['min_balance']:,.0f}")
    if f.get("min_util"): parts.append(f"utilization ≥ {f['min_util']:.0%}")
    if f.get("watchlist"): parts.append("on watchlist")
    return " · ".join(parts) or "no filters"


def where_clause(f: dict) -> tuple[str, list]:
    w, p = [], []

    def inlist(col, vals):
        w.append(f"{col} IN ({','.join('?' * len(vals))})")
        p.extend(vals)
    if f.get("products"): inlist("product", f["products"])
    if f.get("countries"): inlist("country", f["countries"])
    if f.get("cities"): inlist("city", f["cities"])
    if f.get("status"): inlist("status", f["status"])
    if f.get("risk_bands"): inlist("risk_band", f["risk_bands"])
    if f.get("min_dpd"): w.append("dpd_now >= ?"); p.append(f["min_dpd"])
    if f.get("min_balance"): w.append("balance_usd >= ?"); p.append(f["min_balance"])
    if f.get("min_util"): w.append("utilization >= ?"); p.append(f["min_util"])
    if f.get("watchlist"): w.append("watchlist")
    return (" AND ".join(w) or "TRUE"), p


# ====================================================================================================
# copilot: question -> metric/group-by plan, executed with whitelisted SQL
# ====================================================================================================
def heuristic_plan(q: str) -> dict:
    t = q.lower()
    metric = "accounts"
    for key, pat in (("npl_ratio", r"npl ratio|non[- ]?performing"), ("dpd30_ratio", r"30\+|dpd30|early delinquen"),
                     ("exposure_usd", r"exposure|outstanding|balance|how much"), ("avg_pd", r"average pd|avg pd|probability|risk level"),
                     ("watchlist_accounts", r"watch ?list|early warning"), ("delinquent_accounts", r"delinquen|overdue|late"),
                     ("avg_utilization", r"utili[sz]ation")):
        if re.search(pat, t):
            metric = key
            break
    group = None
    for key, pat in (("city", r"by city|per city|each city|which city|cities"), ("country", r"by country|per country|each country|which country|countries"),
                     ("product", r"by product|per product|each product|which product"), ("risk_band", r"by (risk )?band|per band"),
                     ("employment_type", r"by employment"), ("channel", r"by channel|acquisition")):
        if re.search(pat, t):
            group = key
            break
    filt = heuristic_filters(q)
    if metric in ("npl_ratio", "dpd30_ratio", "delinquent_accounts"):
        filt.pop("status", None)          # the metric already measures it
    return {"metric": metric, "group_by": group, "filters": filt, "top_n": 15, "order": "desc"}


def make_plan(question: str, llm: LLM | None = None) -> tuple[dict, str]:
    if llm is not None and llm.available:
        try:
            filt, _ = search_filters(question, llm)
            system = "You turn a portfolio question into a query plan. Use only the allowed metric and group_by values."
            user = f"""metrics: {list(METRICS)}\ngroup_by: {list(DIMS)} or null\nQuestion: {question}
Return JSON {{"metric": ..., "group_by": ..., "top_n": 15, "order": "desc"|"asc"}}"""
            pl = llm.complete_json(system, user, 300)
            if pl.get("metric") in METRICS and (pl.get("group_by") in DIMS or pl.get("group_by") is None):
                return {"metric": pl["metric"], "group_by": pl.get("group_by"), "filters": filt,
                        "top_n": int(pl.get("top_n", 15)), "order": pl.get("order", "desc")}, "llm"
        except Exception:
            pass
    return heuristic_plan(question), "offline"


def run_plan(plan: dict) -> pd.DataFrame:
    expr, label = METRICS[plan["metric"]]
    where, params = where_clause(plan["filters"])
    order = "ASC" if plan.get("order") == "asc" else "DESC"
    if plan.get("group_by"):
        col = DIMS[plan["group_by"]]
        sql = f"SELECT {col} AS {plan['group_by']}, {expr} AS value FROM book WHERE {where} GROUP BY 1 ORDER BY value {order} NULLS LAST LIMIT {int(plan.get('top_n', 15))}"
    else:
        sql = f"SELECT {expr} AS value FROM book WHERE {where}"
    return data.q(sql, params)


# ====================================================================================================
# briefings and explanations
# ====================================================================================================
def portfolio_briefing(stats: dict, llm: LLM | None = None) -> tuple[str, str]:
    if llm is not None and llm.available:
        try:
            system = ("You are a credit-portfolio analyst writing a short morning briefing for business managers. Plain language, "
                      "no jargon, 5 bullets max: what changed, where the risk is, what to do. Use only the JSON facts.")
            return llm.complete(system, json.dumps(stats, default=float), 700), "llm"
        except Exception:
            pass
    s = stats
    lines = [f"**{s['month']}**: {s['accounts']:,} active accounts, exposure **${s['exposure'] / 1e6:,.1f}M**.",
             f"- NPL ratio is **{s['npl_ratio']:.2%}** ({s['npl_delta']:+.2%} vs 3 months ago); 30+ DPD is **{s['dpd30_ratio']:.2%}**.",
             f"- Highest NPL ratio: **{s['worst_country']}** ({s['worst_country_npl']:.2%}); riskiest product: **{s['worst_product']}** ({s['worst_product_npl']:.2%}).",
             f"- **{s['watchlist']:,}** accounts are on the watchlist (predicted risk rising fast) with ${s['watchlist_exposure'] / 1e6:,.2f}M exposure.",
             f"- Collections queue: **{s['in_collections']:,}** accounts; best expected value from: {s['top_action']}."]
    return "\n".join(lines), "offline"


def customer_brief(facts: dict, llm: LLM | None = None) -> tuple[str, str]:
    if llm is not None and llm.available:
        try:
            system = ("You are a senior credit analyst. In under 120 words, summarise this customer's situation for a business user, "
                      "explain in plain words why the risk is what it is, and state the recommended action. Do not invent facts.")
            return llm.complete(system, json.dumps(facts, default=str), 500), "llm"
        except Exception:
            pass
    accts = facts["accounts"]
    held = ", ".join(f"{a['product']} ({a['status']}, risk {a['risk_band']})" for a in accts)
    worst = max(accts, key=lambda a: (a["pd_6m"] if a["pd_6m"] == a["pd_6m"] else 0))
    txt = (f"{facts['age']}-year-old {facts['employment_type']} in {facts['city']}, {facts['country']}, monthly income ≈ ${facts['income']:,.0f}. "
           f"Holds {held}. Highest predicted risk is on the {worst['product']} "
           f"({worst['pd_6m']:.1%} chance of serious default in 6 months)." if worst["pd_6m"] == worst["pd_6m"] else "")
    if facts.get("reasons"):
        txt += " Main drivers: " + "; ".join(facts["reasons"][:3]) + "."
    if facts.get("recommendation"):
        txt += f" Recommended: {facts['recommendation']}"
    return txt, "offline"


TEMPLATES = {
    "sms_reminder": "Hi, this is a friendly reminder that your {product} payment of {amount} is overdue. You can pay anytime in the app. If something changed, reply HELP and we will find a plan that works.",
    "push_reminder": "Your {product} payment is overdue. Tap to pay in a few seconds, or ask for more time.",
    "call": "Open by confirming identity. Say: 'I'm calling about your {product}; I see a payment of {amount} is overdue. Is everything alright? Can we agree on a date this week?' Listen first, log the promise to pay.",
    "call_payment_plan": "Open by confirming identity. Say: 'I'd like to help you get back on track. Based on your situation we can split the overdue {amount} into smaller instalments. Which weekly amount feels manageable?' Confirm the plan in writing.",
    "restructure_offer": "We understand things can get tight. You may qualify for a longer term or a short grace period on your {product}. Reply YES or open the app to review the offer; no extra paperwork needed for the first step.",
    "field_visit": "Visit only after a documented call attempt. Introduce yourself, show ID, speak only with the account holder, offer the payment-plan options, and never discuss the debt with third parties.",
    "legal_notice": "Formal notice: your {product} is {dpd} days overdue ({amount}). Please contact us within 7 days to arrange payment or a repayment plan to avoid further steps.",
    "none": "No outreach recommended at this time; keep monitoring.",
}


def draft_message(action: str, ctx: dict, language: str = "English", tone: str = "empathetic", llm: LLM | None = None) -> tuple[str, str]:
    base = TEMPLATES.get(action, TEMPLATES["none"]).format(product=ctx.get("product_name", "account"), amount=ctx.get("amount", "the amount due"),
                                                         dpd=ctx.get("dpd", ""))
    if llm is not None and llm.available:
        try:
            system = (f"You write collections messages and call scripts. Language: {language}. Tone: {tone}, respectful, never threatening. "
                      "Follow these guardrails strictly:\n- " + "\n- ".join(GUARDRAILS) +
                      "\nDo not promise discounts or terms that are not in the template. Keep it short.")
            return llm.complete(system, f"Action: {ACTION_LABELS.get(action, action)}\nTemplate: {base}\nContext: {json.dumps(ctx, default=str)}", 400), "llm"
        except Exception:
            pass
    note = "" if language == "English" else f"\n\n(Translation into {language} needs an AI provider; showing the English template.)"
    return base + note, "offline"
