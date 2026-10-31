"""Decision layer that turns model scores into business actions. Everything here is deterministic and auditable;
the AI layer only explains and drafts on top of these outputs."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import data
from .generate_world import ACTIONS, ALLOWED, COST

FIXED_COST_PER_ACCOUNT = {"CARD": 110.0, "LOAN": 140.0, "BNPL": 18.0}   # acquisition + 2y servicing, USD
HORIZON_YEARS = 2.0          # profit horizon for an approved account
LIFETIME_FACTOR = 2.0        # 12-month PD -> PD over the profit horizon
LOSS_IF_NOT_CURED = {1: 0.25, 2: 0.45, 3: 0.70, 4: 0.85}      # chance an uncured account ends in write-off, by stage
RULE_ACTION = {1: "sms_reminder", 2: "call", 3: "call_payment_plan"}
ACTION_LABELS = {
    "none": "No action", "sms_reminder": "SMS reminder", "push_reminder": "App push reminder", "call": "Collector call",
    "call_payment_plan": "Call + payment plan", "restructure_offer": "Restructuring offer", "field_visit": "Field visit",
    "legal_notice": "Legal notice"}
GUARDRAILS = [
    "Contact only inside permitted hours and through channels the customer has consented to.",
    "No threats, shaming or contact with third parties / emergency contacts.",
    "Restructuring offers must stay inside the approved hardship policy (term extension, grace period).",
    "Legal notices and field visits need supervisor approval and a documented prior contact attempt.",
    "Every automated recommendation is logged with its inputs; a human decides before anything is sent.",
]


# ====================================================================================================
# 1. Onboarding
# ====================================================================================================
def knockouts(df: pd.DataFrame) -> pd.Series:
    """Hard policy rules applied before the score. Returns a reason string ('' = passed)."""
    r = pd.Series("", index=df.index)
    r[(df["device_shared_apps_30d"] >= 4)] = "Same device used for many applications"
    r[(df["email_age_days"] < 60) & (df["ip_country_match"] == 0)] = "New email address and location mismatch"
    r[df["employment_type"].eq("unemployed") & df["requested_amount_usd"].gt(0)] = "No verifiable income"
    return r


def pending_applications(product: str | None = None) -> pd.DataFrame:
    sql = """SELECT a.*, c.country, c.city, c.age, c.employment_type, c.monthly_income_usd, c.bureau_score,
                    c.acquisition_channel, c.device_shared_apps_30d, c.email_age_days, c.ip_country_match
             FROM app_scores a JOIN customers c USING (customer_id) WHERE a.decision = 'pending'"""
    if product:
        sql += " AND a.product = ?"
    return data.q(sql, [product] if product else [])


def strategy_curve(product: str, points: int = 40) -> pd.DataFrame:
    """Approval cut-off vs approval rate, expected loss and profit, using every historic application's model PD."""
    a = data.q("SELECT application_id, pd_app, requested_amount_usd FROM app_scores WHERE product = ? AND decision <> 'pending'", [product])
    ue = data.unit_economics().loc[product]
    amt = a["requested_amount_usd"].values
    ead = amt * float(ue["ead_default_ratio"])
    loss_rate = a["pd_app"].values * LIFETIME_FACTOR * float(ue["lgd"])
    contribution = amt * float(ue["rev_yield_granted"]) * HORIZON_YEARS - ead * loss_rate - FIXED_COST_PER_ACCOUNT[product]
    cuts = np.quantile(a["pd_app"], np.linspace(0.05, 1.0, points))
    rows = []
    for c in cuts:
        m = a["pd_app"].values <= c
        rows.append({"cutoff_pd": float(c), "approval_rate": float(m.mean()), "expected_bad_rate": float(a["pd_app"].values[m].mean()),
                     "expected_loss_usd": float((ead * loss_rate)[m].sum()), "expected_profit_usd": float(contribution[m].sum()),
                     "approved": int(m.sum())})
    return pd.DataFrame(rows)


def optimal_cutoff(curve: pd.DataFrame) -> float:
    return float(curve.loc[curve["expected_profit_usd"].idxmax(), "cutoff_pd"])


def decide(df: pd.DataFrame, cutoff: float, refer_band: float = 1.5) -> pd.DataFrame:
    out = df.copy()
    ko = knockouts(out)
    out["knockout"] = ko
    dec = np.where(ko != "", "Decline", np.where(out["pd_app"] <= cutoff, "Approve",
                   np.where(out["pd_app"] <= cutoff * refer_band, "Refer", "Decline")))
    out["decision_suggested"] = dec
    thin = out["bureau_score"].isna() & (out["decision_suggested"] == "Approve") & (out["requested_amount_usd"] > 6 * out["monthly_income_usd"])
    out.loc[thin, "decision_suggested"] = "Refer"
    out.loc[thin, "knockout"] = "No bureau record and a large request"
    return out


# ====================================================================================================
# 2. Credit limit management
# ====================================================================================================
DEFAULT_CLI = dict(min_mob=6, max_pd=0.035, min_usage=0.30, max_late_6m=0, increase_by_band={"A": 0.50, "B": 0.30, "C": 0.10},
                   capacity_mult={"CARD": 4.0, "BNPL": 1.0}, total_limit_mult=8.0, take_up=0.35, round_to={"CARD": 50, "BNPL": 10})


def cli_proposals(params: dict | None = None, products=("CARD", "BNPL")) -> pd.DataFrame:
    p = {**DEFAULT_CLI, **(params or {})}
    b = data.q("""SELECT * FROM book WHERE product IN ({}) AND status IN ('active','delinquent')""".format(",".join(f"'{x}'" for x in products)))
    b["usage_rate"] = np.maximum(b["utilization"], b["spend_3m_usd"] / 3 / b["credit_limit_usd"].clip(lower=1))
    tot = data.q("SELECT customer_id, sum(credit_limit_usd) AS total_limit FROM scores_current WHERE product IN ('CARD','BNPL') GROUP BY 1")
    b = b.merge(tot, on="customer_id", how="left")
    ue = data.unit_economics()
    elig = (b["mob"] >= p["min_mob"]) & (b["status"] == "active") & (b["pd_6m"] <= p["max_pd"]) & (b["usage_rate"] >= p["min_usage"]) & \
           (b["max_dpd_6m"] <= p["max_late_6m"]) & (b["other_delinq"] == 0)
    inc = b["risk_band"].map(p["increase_by_band"]).fillna(0.0)
    cap = b["product"].map(p["capacity_mult"]) * b["monthly_income_usd"]
    room_total = p["total_limit_mult"] * b["monthly_income_usd"] - b["total_limit"]
    step = np.array([p["round_to"][x] for x in b["product"]])
    proposed = np.minimum(b["credit_limit_usd"] * (1 + inc), np.minimum(cap, b["credit_limit_usd"] + np.maximum(room_total, 0)))
    proposed = np.floor(proposed / step) * step
    proposed = np.maximum(proposed, b["credit_limit_usd"])
    b["proposed_limit_usd"] = np.where(elig, proposed, b["credit_limit_usd"])
    b["increase_usd"] = b["proposed_limit_usd"] - b["credit_limit_usd"]
    b["eligible"] = elig & (b["increase_usd"] > 0)
    y = b["product"].map(ue["rev_yield_granted"]); lgd = b["product"].map(ue["lgd"]); ead = b["product"].map(ue["ead_default_ratio"])
    used = b["increase_usd"] * p["take_up"]
    b["exp_revenue_12m_usd"] = used * y
    b["exp_loss_12m_usd"] = used * ead * np.minimum(b["pd_6m"] * 1.8, 1) * lgd
    b["net_value_usd"] = b["exp_revenue_12m_usd"] - b["exp_loss_12m_usd"]
    # decrease / freeze review for the riskiest active accounts
    b["action"] = np.where(b["eligible"], "Increase", np.where((b["risk_band"] == "E") & (b["status"] == "active"), "Review for decrease", "Hold"))
    why = np.where(b["eligible"], "Low risk, high usage, clean payment record in the last 6 months",
                   np.where(b["action"] == "Review for decrease", "Very high predicted risk - consider lowering exposure", ""))
    b["rationale"] = why
    hold = ~b["eligible"] & (b["action"] == "Hold")
    reason = np.select(
        [b["mob"] < p["min_mob"], b["status"] != "active", b["pd_6m"] > p["max_pd"], b["usage_rate"] < p["min_usage"],
         b["max_dpd_6m"] > p["max_late_6m"], b["other_delinq"] > 0],
        ["Too new", "Currently late", "Risk above threshold", "Low usage (no demand signal)", "Late payment in last 6 months",
         "Late on another product"], "Capacity or exposure cap reached")
    b.loc[hold, "rationale"] = reason[hold.values]
    return b


# ====================================================================================================
# 3. Collections
# ====================================================================================================
def _effect_lookup(up: pd.DataFrame):
    return {(int(r.bucket), r.action): (float(r.uplift), float(r.se), int(r.n)) for r in up.itertuples()}


def collections_plan() -> pd.DataFrame:
    """Best action per delinquent account = argmax( P(cure | action) x avoided loss - cost )."""
    d = data.q("SELECT * FROM book WHERE status IN ('delinquent','npl')")
    up = data.q("SELECT * FROM uplift")
    eff = _effect_lookup(up)
    lgd = data.unit_economics()["lgd"]
    cost_idx = data.q("SELECT code, cost_index FROM countries").set_index("code")["cost_index"]
    rows = []
    for r in d.itertuples():
        b = int(r.bucket)
        ref = np.mean([eff.get((b, a), (0, 0, 0))[0] for a in ([RULE_ACTION[b]] if b in RULE_ACTION else ["field_visit", "legal_notice"])])
        base = float(r.cure_prob) if pd.notna(r.cure_prob) else 0.3
        benefit = r.balance_usd * lgd[r.product] * LOSS_IF_NOT_CURED[b]
        best = None
        evs = []
        for a in ALLOWED[b]:
            if (b, a) not in eff or eff[(b, a)][2] < 100:
                continue
            u, se, n = eff[(b, a)]
            p_cure = float(np.clip(base + (u - ref), 0.01, 0.95))
            cost = COST[a] * cost_idx[r.country]
            ev = p_cure * benefit - cost
            evs.append((ev, a, p_cure, u, se, cost))
        evs.sort(reverse=True)
        if not evs:
            continue
        ev, a, p_cure, u, se, cost = evs[0]
        none_ev = next((x for x in evs if x[1] == "none"), None)
        rows.append({"account_id": r.account_id, "action": a, "p_cure_with_action": p_cure, "uplift": u, "uplift_se": se, "action_cost_usd": cost,
                     "expected_value_usd": ev, "avoided_loss_if_cured_usd": benefit,
                     "alternative": evs[1][1] if len(evs) > 1 else "", "p_cure_no_action": float(np.clip(base - ref, 0.01, 0.95))})
    plan = pd.DataFrame(rows)
    out = d.merge(plan, on="account_id", how="left")
    out["action_label"] = out["action"].map(ACTION_LABELS)
    out["priority"] = out["expected_value_usd"].rank(ascending=False, method="first")
    return out.sort_values("priority")


def rationale_text(r) -> str:
    if pd.isna(r.get("action")):
        return "No recommendation available."
    a = ACTION_LABELS[r["action"]]
    if r["action"] == "none":
        return "No action: the balance is small, so expected recovery does not cover the cost of contact. Keep monitoring."
    return (f"{a}: lifts the chance of getting back on track by about {r['uplift'] * 100:+.0f} pts versus doing nothing "
            f"(±{1.96 * r['uplift_se'] * 100:.0f}), costs ≈ ${r['action_cost_usd']:.1f}; expected net value ≈ ${r['expected_value_usd']:,.0f}.")


def reasons(raw: str) -> list[dict]:
    try:
        return json.loads(raw) if isinstance(raw, str) else []
    except Exception:
        return []
