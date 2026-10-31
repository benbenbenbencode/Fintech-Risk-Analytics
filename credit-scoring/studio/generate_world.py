"""Synthetic worldwide retail-lending world: 10 countries, 50 cities, 3 products.

Products
    CARD  revolving credit card
    LOAN  term loan (installment)
    BNPL  buy-now-pay-later line (pay-in-3, interest-free to customer, merchant pays MDR)

What is baked in (so analysis can be validated against known truth):
    * a hidden credit-risk latent + macro stress per country-month (inflation shock, FX shock...)
    * staggered BNPL launches per country  -> difference-in-differences is possible
    * BNPL cannibalizes card spend, term-loan originations and raises card attrition (ground_truth.json)
    * selection: BNPL adopters shop more and are younger -> naive comparisons are biased
    * legacy policy: rule-based approvals, quarterly credit-limit reviews, collections playbook with
      ~20% randomised treatments (so treatment effects can be estimated)
    * ~1% application-fraud customers with device / email signals

Everything is synthetic. Structure and correlations are realistic; the numbers are not calibrated to any
real portfolio.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

N_MONTHS = 48
MONTHS = pd.date_range("2022-10-31", periods=N_MONTHS, freq="ME")
SEAS = np.array([0.90, 0.86, 0.93, 0.95, 0.98, 0.95, 0.97, 1.00, 1.00, 1.06, 1.25, 1.35])  # Jan..Dec spend index
PRODUCTS = {"CARD": "Credit Card", "LOAN": "Term Loan", "BNPL": "BNPL"}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    return np.log(p / (1 - p))


def _c(code, name, region, ccy, fx, inc, bur, risk, card, loan, bnpl_m, w, apr_card, apr_loan, mdr, bureau, cost,
       infl, unemp, rate):
    return dict(code=code, name=name, region=region, currency=ccy, fx=fx, income_med_usd=inc, bureau_cov=bur,
                country_risk=risk, card_pen=card, loan_pen=loan, bnpl_launch_m=bnpl_m, weight=w, apr_card=apr_card,
                apr_loan=apr_loan, mdr=mdr, bureau_name=bureau, cost_index=cost, base_infl=infl, base_unemp=unemp,
                base_rate=rate)


COUNTRIES = [
    _c("ID", "Indonesia", "Asia", "IDR", 15800, 450, .55, .05, .30, .40, 20, .16, .30, .24, .040, "SLIK (OJK)", .8, 3.0, 5.0, 6.0),
    _c("VN", "Vietnam", "Asia", "VND", 24500, 420, .45, .10, .22, .38, 24, .09, .28, .26, .035, "CIC", .7, 3.5, 2.3, 4.5),
    _c("PH", "Philippines", "Asia", "PHP", 57, 380, .35, .25, .18, .35, 18, .08, .36, .30, .040, "CIC (PH)", .6, 4.5, 4.5, 6.0),
    _c("IN", "India", "Asia", "INR", 83, 330, .60, .10, .20, .35, 8, .17, .36, .22, .030, "CIBIL", .6, 5.0, 7.0, 6.5),
    _c("MX", "Mexico", "Americas", "MXN", 18, 650, .70, .20, .35, .35, 16, .10, .55, .40, .045, "Buro de Credito", .9, 5.0, 3.0, 9.0),
    _c("BR", "Brazil", "Americas", "BRL", 5.2, 560, .85, .30, .45, .35, 10, .13, .70, .45, .045, "Serasa", .9, 5.5, 8.0, 11.0),
    _c("PL", "Poland", "Europe", "PLN", 4.0, 1250, .95, -.20, .40, .35, 12, .06, .20, .15, .030, "BIK", 1.3, 6.0, 3.0, 5.5),
    _c("CZ", "Czechia", "Europe", "CZK", 23, 1250, .95, -.25, .38, .33, 22, .05, .22, .14, .030, "CBCB", 1.3, 4.5, 2.8, 4.5),
    _c("NG", "Nigeria", "Africa", "NGN", 1500, 220, .25, .40, .12, .30, 28, .07, .55, .55, .050, "CRC", .5, 20.0, 5.0, 18.0),
    _c("US", "United States", "Americas", "USD", 1, 3300, .98, -.10, .70, .30, 14, .09, .22, .15, .050, "FICO", 2.0, 3.0, 4.0, 5.0),
]
CITIES = {  # city, lat, lon, weight, tier(1 = megacity .. 3 = smaller)
    "ID": [("Jakarta", -6.2088, 106.8456, .38, 1), ("Surabaya", -7.2575, 112.7521, .18, 1), ("Bandung", -6.9175, 107.6191, .16, 2),
           ("Medan", 3.5952, 98.6722, .14, 2), ("Batam", 1.0456, 104.0305, .08, 3), ("Makassar", -5.1477, 119.4327, .06, 2)],
    "VN": [("Ho Chi Minh City", 10.8231, 106.6297, .38, 1), ("Hanoi", 21.0285, 105.8542, .34, 1), ("Da Nang", 16.0544, 108.2022, .10, 2),
           ("Hai Phong", 20.8449, 106.6881, .09, 2), ("Can Tho", 10.0452, 105.7469, .09, 3)],
    "PH": [("Manila", 14.5995, 120.9842, .45, 1), ("Cebu", 10.3157, 123.8854, .15, 2), ("Davao", 7.1907, 125.4553, .15, 2),
           ("Quezon City", 14.6760, 121.0437, .15, 1), ("Cagayan de Oro", 8.4542, 124.6319, .10, 3)],
    "IN": [("Mumbai", 19.0760, 72.8777, .25, 1), ("Delhi", 28.7041, 77.1025, .25, 1), ("Bengaluru", 12.9716, 77.5946, .20, 1),
           ("Hyderabad", 17.3850, 78.4867, .15, 2), ("Chennai", 13.0827, 80.2707, .15, 2)],
    "MX": [("Mexico City", 19.4326, -99.1332, .40, 1), ("Guadalajara", 20.6597, -103.3496, .20, 2), ("Monterrey", 25.6866, -100.3161, .22, 1),
           ("Puebla", 19.0414, -98.2063, .10, 3), ("Tijuana", 32.5149, -117.0382, .08, 3)],
    "BR": [("Sao Paulo", -23.5505, -46.6333, .40, 1), ("Rio de Janeiro", -22.9068, -43.1729, .25, 1), ("Belo Horizonte", -19.9167, -43.9345, .12, 2),
           ("Brasilia", -15.7939, -47.8828, .12, 2), ("Recife", -8.0476, -34.8770, .11, 3)],
    "PL": [("Warsaw", 52.2297, 21.0122, .30, 1), ("Krakow", 50.0647, 19.9450, .20, 2), ("Lodz", 51.7592, 19.4560, .15, 3),
           ("Wroclaw", 51.1079, 17.0385, .18, 2), ("Gdansk", 54.3520, 18.6466, .17, 2)],
    "CZ": [("Prague", 50.0755, 14.4378, .40, 1), ("Brno", 49.1951, 16.6068, .25, 2), ("Ostrava", 49.8209, 18.2625, .18, 3),
           ("Plzen", 49.7384, 13.3736, .09, 3), ("Liberec", 50.7663, 15.0543, .08, 3)],
    "NG": [("Lagos", 6.5244, 3.3792, .50, 1), ("Abuja", 9.0765, 7.3986, .18, 1), ("Kano", 12.0022, 8.5920, .12, 2),
           ("Port Harcourt", 4.8156, 7.0498, .10, 2), ("Ibadan", 7.3775, 3.9470, .10, 3)],
    "US": [("New York", 40.7128, -74.0060, .28, 1), ("Los Angeles", 34.0522, -118.2437, .22, 1), ("Chicago", 41.8781, -87.6298, .17, 1),
           ("Houston", 29.7604, -95.3698, .17, 2), ("Atlanta", 33.7490, -84.3880, .16, 2)],
}
SHOCKS = {"PL": [(0, 10, 1.2)], "CZ": [(0, 9, 1.0)], "US": [(0, 7, 0.5)], "BR": [(6, 17, 0.8)], "NG": [(12, 26, 1.6)],
          "VN": [(6, 15, 0.9)], "PH": [(14, 22, 0.4)], "IN": [(0, 6, 0.3)]}

# ---- collections playbook used by the legacy policy (ground truth of what works) ----
ACTIONS = ["none", "sms_reminder", "push_reminder", "call", "call_payment_plan", "restructure_offer", "field_visit", "legal_notice"]
EFF = dict(none=0.0, sms_reminder=.06, push_reminder=.05, call=.20, call_payment_plan=.30, restructure_offer=.34,
           field_visit=.25, legal_notice=.10)
COST = dict(none=0.0, sms_reminder=.03, push_reminder=.01, call=1.8, call_payment_plan=2.5, restructure_offer=3.5,
            field_visit=14.0, legal_notice=22.0)
CONTACT_F = dict(none=0.0, sms_reminder=.90, push_reminder=.70, call=1.0, call_payment_plan=1.0, restructure_offer=.9,
                 field_visit=.6, legal_notice=.95)
ALLOWED = {1: ["none", "sms_reminder", "push_reminder", "call"], 2: ["none", "sms_reminder", "call", "restructure_offer"],
           3: ["none", "call", "call_payment_plan", "restructure_offer", "field_visit"],
           4: ["none", "call_payment_plan", "restructure_offer", "field_visit", "legal_notice"]}
P_ROLL_BASE = np.array([.42, .62, .80, .92])
TAU_CARD_SPEND = 0.22       # avg card-spend reduction once a card holder adopts BNPL (young: more)
LOAN_SUBSTITUTION = 0.30    # share of term-loan originations skipped by BNPL-wanters once BNPL is available
ATTRITION_MULT = 1.35       # card attrition multiplier after BNPL adoption
ADVERSE_SELECTION = 0.25    # lower-risk card holders adopt BNPL a bit more


def p_roll(mpd, risk, stress):
    base = P_ROLL_BASE[np.minimum(mpd, 4) - 1]
    return sigmoid(logit(base) + 0.55 * risk + 0.25 * stress)


def choose_actions(rng, mpd, balance):
    n = len(mpd)
    bucket = np.minimum(mpd, 4)
    act = np.where(bucket == 1, "sms_reminder", np.where(bucket == 2, "call", np.where(bucket == 3, "call_payment_plan",
                   np.where(balance > 500, "field_visit", "legal_notice")))).astype(object)
    rs = (bucket >= 3) & (rng.random(n) < 0.25)
    act[rs] = "restructure_offer"
    rnd = rng.random(n) < 0.20
    for b in (1, 2, 3, 4):
        sel = rnd & (bucket == b)
        if sel.any():
            act[sel] = rng.choice(ALLOWED[b], sel.sum())
    return act, np.where(rnd, "random", "rule").astype(object)


def delinquency_step(rng, ctx, cid, ci, m, mpd, p_first, balance, contact, col, acc_ids, product, mob):
    """Shared state machine: first miss / roll forward / cure, with collections actions affecting cure."""
    n = len(mpd)
    cur = mpd == 0
    miss = cur & (rng.random(n) < p_first)
    new = np.zeros(n, int)
    new[miss] = 1
    d = ~cur
    contacted = np.zeros(n, bool)
    if d.any():
        di = np.where(d)[0]
        act, by = choose_actions(rng, mpd[di], balance[di])
        cf = pd.Series(act).map(CONTACT_F).values
        cont = rng.random(len(di)) < (contact[cid[di]] * cf)
        eff = pd.Series(act).map(EFF).values * cont
        pr = p_roll(mpd[di], ctx["risk"][cid[di]], ctx["S"][ci[di], m]) * (1 - eff)
        roll = rng.random(len(di)) < pr
        new[di] = np.where(roll, mpd[di] + 1, 0)
        contacted[di] = cont
        cured = ~roll
        ptp = cont & (rng.random(len(di)) < np.where(cured, .9, .15))
        col.append(pd.DataFrame({
            "account_id": acc_ids[di], "product": product, "month_idx": m, "mpd_before": mpd[di],
            "bucket": np.minimum(mpd[di], 4), "balance_usd": balance[di].round(2), "action": act, "assigned_by": by,
            "contacted": cont, "promise_to_pay": ptp, "cost_usd": pd.Series(act).map(COST).values * ctx["cost_idx"][ci[di]],
            "cured": cured}))
    return new, miss, contacted


def dpd_days(rng, mpd):
    return np.where(mpd == 0, 0, (mpd - 1) * 30 + rng.integers(1, 31, len(mpd)))


def _status(mpd, wo=None, closed=None):
    s = np.where(mpd == 0, "active", np.where(mpd < 3, "delinquent", "npl")).astype(object)
    if wo is not None:
        s[wo] = "written_off"
    if closed is not None:
        s[closed] = "closed"
    return s


def _snap(ids, cust, product, m, mob, status, dpd, limit, bal, spend, n_orders, due, paid, interest, fee, rev, wo_amt):
    util = np.where(limit > 0, bal / np.maximum(limit, 1), 0.0)
    return pd.DataFrame({
        "account_id": ids, "customer_id": cust, "product": product, "month_idx": np.int8(m), "mob": mob.astype(np.int16),
        "status": status, "dpd": dpd.astype(np.int16), "credit_limit_usd": limit.round(0), "balance_usd": bal.round(2),
        "utilization": util.round(4), "spend_usd": spend.round(2), "n_orders": n_orders.astype(np.int16),
        "payment_due_usd": due.round(2), "payment_made_usd": paid.round(2), "interest_usd": interest.round(2),
        "fee_usd": fee.round(2), "revenue_usd": rev.round(2), "writeoff_usd": wo_amt.round(2)})


# =====================================================================================================
def sim_card(rng, ctx, acc, A_CARD):
    n = len(acc)
    cid, om = acc.cidx.values, acc.open_m.values
    ids, cust = acc.account_id.values, ctx["customer_id"][cid]
    limit = acc.limit_usd.values.astype(float).copy()
    ci, r, fr = ctx["ci"][cid], ctx["risk"][cid], ctx["fraud"][cid]
    apr = ctx["apr_card"][ci]
    s_rate = 0.11 * ctx["shop"][cid]
    rev = rng.random(n) < sigmoid(-0.35 + 0.75 * r)
    bal, mpd, st = np.zeros(n), np.zeros(n, int), np.zeros(n, int)
    util_prev = np.zeros(n)
    rows, ev, col, wo = [], [], [], []
    for m in range(N_MONTHS):
        idx = np.where((st == 0) & (om <= m))[0]
        if idx.size == 0:
            continue
        mob = m - om[idx]
        cc = cid[idx]
        c_ci = ci[idx]
        stress = ctx["S"][c_ci, m]
        bnpl_on = ctx["bnpl_open"][cc] <= m
        tau = ctx["tau"][cc] * bnpl_on
        seas = SEAS[MONTHS[m].month - 1]
        b0, lim, md = bal[idx], limit[idx], mpd[idx]
        spend = lim * s_rate[idx] * seas * (1 - tau) * rng.lognormal(0, 0.3, len(idx))
        spend = np.where(md > 0, 0, np.minimum(spend, np.maximum(lim - b0, 0)))
        interest = np.where(rev[idx] | (md > 0), b0 * apr[idx] / 12, 0.0)
        B = b0 + spend + interest
        min_due = np.minimum(B, np.maximum(25.0, 0.03 * B))
        due = np.where(mob >= 1, np.minimum(B, min_due * (1 + md)), 0.0)
        seas_hz = -0.5 + 0.9 * np.exp(-((mob - 9) / 7.0) ** 2)
        p_first = sigmoid(A_CARD + 0.95 * r[idx] + 0.40 * stress + 1.2 * (util_prev[idx] > 0.85) + seas_hz
                          + 4.0 * fr[idx] * (mob <= 2)) * (mob >= 1)
        new, miss, _ = delinquency_step(rng, ctx, cc, c_ci, m, md, p_first, b0, ctx["contact"], col, ids[idx], "CARD", mob)
        paid = np.zeros(len(idx))
        ok = (md == 0) & ~miss & (mob >= 1)
        full = ok & ~rev[idx]
        paid[full] = B[full]
        part = ok & rev[idx]
        paid[part] = min_due[part] + rng.random(part.sum()) * np.maximum(0.5 * B[part] - min_due[part], 0)
        cured = (md > 0) & (new == 0)
        paid[cured] = due[cured]
        nb = B - paid
        late_fee = np.where(miss, np.minimum(10.0, 0.05 * B), 0.0)
        nb = nb + late_fee
        woff = new >= 6
        wo_amt = np.where(woff, nb, 0.0)
        close_p = 0.004 * np.where(bnpl_on, ATTRITION_MULT, 1.0)
        closed = ok & (mob > 6) & (nb < 5) & (rng.random(len(idx)) < close_p * 6)
        interest_rec = np.where(new == 0, interest, 0.0)
        revenue = interest_rec + 0.015 * spend + late_fee
        bal_end = np.where(woff, 0.0, nb)
        dpd = np.where(woff, 180, dpd_days(rng, new))
        stat = _status(new, woff, closed)
        # --- quarterly limit review (legacy rule) ---
        new_lim = lim.copy()
        review = (mob >= 6) & (mob % 3 == 0) & ~woff & ~closed
        util_now = bal_end / np.maximum(lim, 1)
        inc = review & (new == 0) & (util_now > 0.5) & (r[idx] < 0.4) & (rng.random(len(idx)) < 0.55)
        pct = rng.choice([0.2, 0.3, 0.5], len(idx))
        new_lim[inc] = np.round(lim[inc] * (1 + pct[inc]) / 50) * 50
        dec = review & (new >= 2) & (rng.random(len(idx)) < 0.5)
        new_lim[dec] = np.maximum(np.round(lim[dec] * 0.7 / 50) * 50, 100)
        for mask, typ in ((inc, "increase"), (dec, "decrease")):
            if mask.any():
                ev.append(pd.DataFrame({"account_id": ids[idx][mask], "product": "CARD", "month_idx": m, "old_limit_usd": lim[mask],
                                        "new_limit_usd": new_lim[mask], "event_type": typ, "trigger": "quarterly_review"}))
        rows.append(_snap(ids[idx], cust[idx], "CARD", m, mob, stat, dpd, lim, bal_end, spend, np.zeros(len(idx)), due, paid,
                          interest_rec, late_fee + 0.015 * spend, revenue, wo_amt))
        bal[idx], mpd[idx], limit[idx] = bal_end, new, new_lim
        util_prev[idx] = util_now
        st[idx[woff]] = 1
        st[idx[closed]] = 2
        if woff.any():
            wo.append(pd.DataFrame({"account_id": ids[idx][woff], "product": "CARD", "month_idx": m, "exposure_usd": wo_amt[woff]}))
    return rows, ev, col, wo


def sim_loan(rng, ctx, acc, A_LOAN):
    n = len(acc)
    cid, om = acc.cidx.values, acc.open_m.values
    ids, cust = acc.account_id.values, ctx["customer_id"][cid]
    ci, r, fr = ctx["ci"][cid], ctx["risk"][cid], ctx["fraud"][cid]
    prin, apr, tenor = acc.principal_usd.values.astype(float), acc.apr.values, acc.tenor.values
    rm = apr / 12
    inst = prin * rm / (1 - (1 + rm) ** (-tenor))
    iti = inst / ctx["income"][cid]
    bal = prin.copy()
    mpd, st = np.zeros(n, int), np.zeros(n, int)
    rows, col, wo = [], [], []
    for m in range(N_MONTHS):
        idx = np.where((st == 0) & (om <= m))[0]
        if idx.size == 0:
            continue
        mob = m - om[idx]
        cc, c_ci = cid[idx], ci[idx]
        stress = ctx["S"][c_ci, m]
        b0, md = bal[idx], mpd[idx]
        interest = b0 * rm[idx]
        due = np.where(mob >= 1, np.minimum(b0 + interest, inst[idx] * (1 + md)), 0.0)
        seas_hz = -0.5 + 0.9 * np.exp(-((mob - 9) / 8.0) ** 2)
        p_first = sigmoid(A_LOAN + 1.0 * r[idx] + 0.40 * stress + 2.2 * (iti[idx] - 0.25) + seas_hz
                          + 4.0 * fr[idx] * (mob <= 2)) * (mob >= 1)
        new, miss, _ = delinquency_step(rng, ctx, cc, c_ci, m, md, p_first, b0, ctx["contact"], col, ids[idx], "LOAN", mob)
        ok = (md == 0) & ~miss & (mob >= 1)
        cured = (md > 0) & (new == 0)
        paid = np.where(ok, np.minimum(inst[idx], b0 + interest), np.where(cured, due, 0.0))
        nb = b0 + np.where(new == 0, 0.0, interest) - paid + np.where(ok | cured, interest, 0.0)
        nb = np.where((new > 0), b0 + interest, nb)
        late_fee = np.where(miss, np.minimum(15.0, 0.03 * inst[idx]), 0.0)
        prepay = ok & (mob >= 6) & (rng.random(len(idx)) < 0.004)
        paid = np.where(prepay, nb + paid, paid)
        nb = np.where(prepay, 0.0, nb)
        woff = new >= 6
        wo_amt = np.where(woff, nb, 0.0)
        matured = ok & (mob >= tenor[idx]) & (nb < 1)
        closed = prepay | matured
        interest_rec = np.where(new == 0, interest, 0.0)
        bal_end = np.where(woff, 0.0, np.maximum(nb, 0))
        dpd = np.where(woff, 180, dpd_days(rng, new))
        stat = _status(new, woff, closed)
        rows.append(_snap(ids[idx], cust[idx], "LOAN", m, mob, stat, dpd, prin[idx], bal_end, np.zeros(len(idx)), np.zeros(len(idx)),
                          due, paid, interest_rec, late_fee, interest_rec + late_fee, wo_amt))
        bal[idx], mpd[idx] = bal_end, new
        st[idx[woff]] = 1
        st[idx[closed]] = 2
        if woff.any():
            wo.append(pd.DataFrame({"account_id": ids[idx][woff], "product": "LOAN", "month_idx": m, "exposure_usd": wo_amt[woff]}))
    return rows, [], col, wo


def sim_bnpl(rng, ctx, acc, A_BNPL):
    n = len(acc)
    cid, om = acc.cidx.values, acc.open_m.values
    ids, cust = acc.account_id.values, ctx["customer_id"][cid]
    limit = acc.limit_usd.values.astype(float).copy()
    ci, r, fr = ctx["ci"][cid], ctx["risk"][cid], ctx["fraud"][cid]
    mdr = ctx["mdr"][ci]
    lam = 0.55 * ctx["shop"][cid]
    order_med = np.clip(ctx["income"][cid] * 0.08, 15, 250)
    o1, o2 = np.zeros(n), np.zeros(n)
    arrears = np.zeros(n)
    mpd, st, clean = np.zeros(n, int), np.zeros(n, int), np.zeros(n, int)
    util_prev = np.zeros(n)
    rows, ev, col, wo = [], [], [], []
    for m in range(N_MONTHS):
        idx = np.where((st == 0) & (om <= m))[0]
        if idx.size == 0:
            continue
        mob = m - om[idx]
        cc, c_ci = cid[idx], ci[idx]
        stress = ctx["S"][c_ci, m]
        md, lim = mpd[idx], limit[idx]
        seas = SEAS[MONTHS[m].month - 1]
        cnt = rng.poisson(lam[idx] * seas)
        cnt = np.where(md > 0, 0, cnt)
        amt = np.minimum(cnt * order_med[idx] * rng.lognormal(0, 0.35, len(idx)), np.maximum(lim - (o1[idx] * 2 / 3 + o2[idx] / 3 + arrears[idx]), 0))
        cnt = np.where(amt > 0, cnt, 0)
        due = np.where(mob >= 1, (o1[idx] + o2[idx]) / 3 + arrears[idx], 0.0)
        seas_hz = 0.5 * np.exp(-((mob - 3) / 3.0) ** 2)
        p_first = sigmoid(A_BNPL + 1.0 * r[idx] + 0.35 * stress + 0.6 * (util_prev[idx] > 0.8) + seas_hz
                          + 4.5 * fr[idx] * (mob <= 2)) * (mob >= 1)
        new, miss, _ = delinquency_step(rng, ctx, cc, c_ci, m, md, p_first, o1[idx] * 2 / 3 + arrears[idx], ctx["contact"], col, ids[idx], "BNPL", mob)
        first_inst = amt / 3
        ok = (md == 0) & ~miss
        cured = (md > 0) & (new == 0)
        paid = np.where(ok, np.where(mob >= 1, due, 0.0) + first_inst, np.where(cured, due + first_inst, first_inst * 0))
        arr_new = np.where(new > 0, np.where(md == 0, due, arrears[idx] + 0.0), 0.0)
        bal_end = amt * 2 / 3 + o1[idx] / 3 + arr_new
        late_fee = np.where(new > 0, 0.03 * arr_new, 0.0) * (new <= 3)
        woff = new >= 5
        wo_amt = np.where(woff, bal_end, 0.0)
        rev = mdr[idx] * amt + late_fee
        bal_out = np.where(woff, 0.0, bal_end)
        dpd = np.where(woff, 150, dpd_days(rng, new))
        stat = _status(new, woff)
        new_lim = lim.copy()
        step = (mob >= 3) & (mob % 3 == 0) & (new == 0) & (clean[idx] >= 3) & ~woff
        cap = np.maximum(ctx["income"][cc] * 0.8, 100)
        new_lim[step] = np.minimum(np.round(lim[step] * 1.25 / 10) * 10, cap[step])
        chg = step & (new_lim != lim)
        if chg.any():
            ev.append(pd.DataFrame({"account_id": ids[idx][chg], "product": "BNPL", "month_idx": m, "old_limit_usd": lim[chg],
                                    "new_limit_usd": new_lim[chg], "event_type": "increase", "trigger": "auto_step"}))
        rows.append(_snap(ids[idx], cust[idx], "BNPL", m, mob, stat, dpd, lim, bal_out, amt, cnt, due, paid,
                          np.zeros(len(idx)), late_fee + mdr[idx] * amt, rev, wo_amt))
        o2[idx], o1[idx] = o1[idx], amt
        arrears[idx] = np.where(woff, 0.0, arr_new)
        mpd[idx], limit[idx] = new, new_lim
        clean[idx] = np.where(new == 0, clean[idx] + 1, 0)
        util_prev[idx] = bal_out / np.maximum(new_lim, 1)
        st[idx[woff]] = 1
        if woff.any():
            wo.append(pd.DataFrame({"account_id": ids[idx][woff], "product": "BNPL", "month_idx": m, "exposure_usd": wo_amt[woff]}))
    return rows, ev, col, wo


# =====================================================================================================
def generate(n_customers: int = 100_000, seed: int = 7, out_dir: str = "data/world", verbose: bool = True,
             calib: dict | None = None) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(seed)
    calib = {"CARD": -3.7, "LOAN": -3.8, "BNPL": -4.4, **(calib or {})}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    C = pd.DataFrame(COUNTRIES)
    K = len(C)
    n = n_customers
    log = (lambda *a: print(f"[{time.time() - t0:5.1f}s]", *a)) if verbose else (lambda *a: None)

    # ------------------------------------------------------------------ macro
    S = np.zeros((K, N_MONTHS))
    macro_rows = []
    for k, cc in enumerate(C.code):
        e = np.zeros(N_MONTHS)
        for t in range(1, N_MONTHS):
            e[t] = 0.8 * e[t - 1] + rng.normal(0, 0.2)
        bump = np.zeros(N_MONTHS)
        for a, b, mag in SHOCKS.get(cc, []):
            w = np.zeros(N_MONTHS)
            half = (b - a) / 2
            for t in range(a, b + 1):
                w[t] = max(0, 1 - abs(t - (a + half)) / (half + 1))
            bump += mag * w
        S[k] = e + bump
        fxp = C.fx[k] * np.exp(np.cumsum(0.012 * np.maximum(S[k], 0)) * (3.0 if cc == "NG" else 0.5))
        for t in range(N_MONTHS):
            macro_rows.append((cc, t, MONTHS[t], C.base_infl[k] + 1.6 * S[k, t], C.base_unemp[k] + 0.6 * S[k, t],
                               C.base_rate[k] + 0.9 * S[k, t], fxp[t], S[k, t]))
    macro = pd.DataFrame(macro_rows, columns=["country", "month_idx", "month", "inflation_pct", "unemployment_pct",
                                              "policy_rate_pct", "fx_local_per_usd", "stress_index"])

    # ------------------------------------------------------------------ customers
    ci = rng.choice(K, n, p=C.weight / C.weight.sum())
    city, tier, lat, lon = np.empty(n, object), np.zeros(n, int), np.zeros(n), np.zeros(n)
    for k, cc in enumerate(C.code):
        sel = np.where(ci == k)[0]
        cl = CITIES[cc]
        w = np.array([c[3] for c in cl])
        pick = rng.choice(len(cl), len(sel), p=w / w.sum())
        city[sel] = np.array([c[0] for c in cl], object)[pick]
        tier[sel] = np.array([c[4] for c in cl])[pick]
    tier_mult = np.array([0, 1.3, 1.0, 0.8])[tier]
    inc_med = C.income_med_usd.values[ci] * tier_mult
    income = inc_med * np.exp(rng.normal(0, 0.6, n))
    z_inc = np.log(income / inc_med) / 0.6
    age = np.clip(rng.normal(36, 10, n), 21, 68).astype(int)
    emp = rng.choice(["salaried", "self_employed", "gig", "student", "retired", "unemployed"], n, p=[.55, .20, .12, .06, .04, .03]).astype(object)
    emp[(emp == "student") & (age > 28)] = "salaried"
    emp[(emp == "retired") & (age < 58)] = "salaried"
    edu = rng.choice(["secondary", "vocational", "bachelor", "postgraduate"], n, p=[.35, .20, .35, .10])
    years_job = np.minimum(np.round(rng.gamma(2, 3, n), 1), age - 19)
    resid = rng.choice(["own", "rent", "family"], n, p=[.4, .4, .2])
    hh = rng.poisson(2.2, n) + 1
    channel = rng.choice(["organic", "referral", "paid_social", "merchant_partner", "agent_branch", "aggregator"], n,
                         p=[.30, .15, .20, .20, .10, .05])
    os_ = rng.choice(["android", "ios", "other"], n, p=[.65, .30, .05])
    z, u = rng.normal(0, 1, n), rng.normal(0, 0.6, n)
    emp_r = pd.Series(emp).map({"salaried": -.2, "self_employed": .2, "gig": .45, "student": .3, "retired": -.1, "unemployed": .9}).values
    ch_r = pd.Series(channel).map({"organic": 0, "referral": -.1, "paid_social": .2, "merchant_partner": -.05, "agent_branch": .1, "aggregator": .15}).values
    fraud = rng.random(n) < 0.012
    risk = 0.85 * z + 0.30 * emp_r + 0.25 * ch_r - 0.25 * z_inc + u + C.country_risk.values[ci] + 3.0 * fraud
    has_b = rng.random(n) < (C.bureau_cov.values[ci] * np.where(age > 25, 1.0, 0.7))
    b_score = np.where(has_b, np.clip(650 - 70 * z + rng.normal(0, 35, n), 300, 850), np.nan).round()
    b_hist = np.where(has_b, np.clip(rng.gamma(2, 18, n) * (age - 18) / 20, 3, 240), np.nan).round()
    b_acct = np.where(has_b, rng.poisson(1.5 + 0.4 * np.clip(z_inc, 0, 2), n), np.nan)
    b_inq = np.where(has_b, rng.poisson(0.8 + 0.7 * np.maximum(z, 0), n), np.nan)
    b_dpd = np.where(has_b, np.where(rng.random(n) < sigmoid(-2.5 + 1.0 * z), rng.choice([15, 35, 65, 95], n), 0), np.nan)
    email_age = np.where(fraud, rng.exponential(40, n), rng.gamma(2, 750, n)).round()
    dev_apps = np.where(fraud, rng.poisson(4, n) + 2, rng.poisson(0.15, n))
    ip_match = np.where(fraud, rng.random(n) > 0.55, rng.random(n) > 0.02).astype(int)
    shop = np.exp(rng.normal(0, 0.45, n))
    contact = np.clip(sigmoid(0.55 + 0.7 * rng.normal(0, 1, n)), 0.15, 0.97)
    wts = 1 + 2 * np.arange(N_MONTHS) / (N_MONTHS - 1)
    join_m = rng.choice(N_MONTHS, n, p=wts / wts.sum())
    cust_id = np.array([f"CU{i:07d}" for i in range(1, n + 1)])
    customers = pd.DataFrame({
        "customer_id": cust_id, "country": C.code.values[ci], "city": city, "age": age, "employment_type": emp,
        "education": edu, "residence_type": resid, "household_size": hh, "years_at_job": years_job,
        "monthly_income_usd": income.round(0), "acquisition_channel": channel, "device_os": os_,
        "bureau_score": b_score, "bureau_history_months": b_hist, "bureau_active_accounts": b_acct,
        "bureau_inquiries_6m": b_inq, "bureau_max_dpd_12m": b_dpd, "email_age_days": email_age,
        "device_shared_apps_30d": dev_apps, "ip_country_match": ip_match, "join_month_idx": join_m,
        "join_date": MONTHS[join_m]})
    cities = pd.DataFrame([(cc, c[0], c[1], c[2], c[4]) for cc, cl in CITIES.items() for c in cl],
                          columns=["country", "city", "lat", "lon", "tier"])
    log(f"customers {n:,}")

    # ------------------------------------------------------------------ BNPL adoption (needed for cannibalization)
    launch = C.bnpl_launch_m.values[ci]
    aff = sigmoid(-0.45 + 0.9 * (age < 35) + 0.45 * (channel == "merchant_partner") + 0.3 * (os_ == "ios")
                  + 0.6 * np.log(shop) - ADVERSE_SELECTION * risk * 0.0)
    wants_bnpl = rng.random(n) < aff
    bnpl_m = np.maximum(join_m, launch) + np.floor(rng.exponential(5, n)).astype(int)
    tau = TAU_CARD_SPEND * np.where(age < 30, 1.4, np.where(age < 45, 1.0, 0.6))
    ctx = dict(customer_id=cust_id, ci=ci, risk=risk, fraud=fraud.astype(float), shop=shop, income=income, contact=contact,
               S=S, tau=tau, apr_card=C.apr_card.values, mdr=C.mdr.values, cost_idx=C.cost_index.values, bnpl_open=None)

    # ------------------------------------------------------------------ applications + legacy decisions
    lr = np.where(has_b, (650 - b_score) / 70, 0.35) + rng.normal(0, 0.8, n) + 0.3 * (dev_apps > 2)
    apps = []

    def make_apps(product, mask, month, req, approve_rate, extra=None):
        idx = np.where(mask & (month < N_MONTHS))[0]
        thr = np.quantile(lr[idx], approve_rate) if len(idx) else 0
        caught = fraud[idx] & (rng.random(len(idx)) < 0.35)
        appr = (lr[idx] <= thr) & ~caught
        reason = np.where(appr, "", np.where(caught, "fraud_check", np.where(rng.random(len(idx)) < 0.25, "affordability", "risk_score_low")))
        pending = month[idx] == N_MONTHS - 1
        dec = np.where(pending, "pending", np.where(appr, "approved", "declined"))
        df = pd.DataFrame({"customer_id": cust_id[idx], "cidx": idx, "product": product, "month_idx": month[idx],
                           "requested_amount_usd": req[idx].round(0), "decision": dec,
                           "decline_reason": np.where(pending, "", reason), "legacy_risk_score": lr[idx].round(3)})
        if extra:
            for k, v in extra.items():
                df[k] = v[idx]
        apps.append(df)
        return df

    def delay(p_now, mean):
        return np.where(rng.random(n) < p_now, 0, 1 + np.floor(rng.exponential(mean, n)).astype(int))

    pen = lambda col: np.clip(C[col].values[ci] * np.clip((income / inc_med) ** 0.4, 0.4, 1.8) * np.where((age > 24) & (age < 52), 1.1, 0.8), 0.02, 0.85)
    card_apply = rng.random(n) < pen("card_pen")
    card_m = join_m + delay(0.6, 8)
    card_req = np.clip(income * rng.uniform(1, 3, n), 100, 15000)
    loan_apply = rng.random(n) < pen("loan_pen")
    loan_m = join_m + delay(0.55, 9)
    # BNPL-wanters skip some term loans once BNPL is available in their country (substitution)
    skip = wants_bnpl & (launch <= loan_m) & (rng.random(n) < LOAN_SUBSTITUTION)
    loan_apply &= ~skip
    loan_prin = np.clip(income * rng.uniform(1, 6, n), 150, 25000)
    loan_ten = rng.choice([6, 12, 18, 24, 36, 48], n, p=[.10, .30, .15, .25, .15, .05])
    bn_req = np.clip(income * rng.uniform(0.2, 0.6, n), 50, 1500)
    ctx["bnpl_open"] = np.where(wants_bnpl & (rng.random(n) < 0.9), bnpl_m, 99)
    a_card = make_apps("CARD", card_apply, card_m, card_req, 0.66)
    a_loan = make_apps("LOAN", loan_apply, loan_m, loan_prin, 0.72, {"tenor": loan_ten})
    a_bn = make_apps("BNPL", wants_bnpl, bnpl_m, bn_req, 0.90)
    applications = pd.concat(apps, ignore_index=True)
    applications.insert(0, "application_id", [f"AP{i:08d}" for i in range(1, len(applications) + 1)])
    ctx["bnpl_open"] = np.where(np.isin(np.arange(n), a_bn.loc[a_bn.decision == "approved", "cidx"]), ctx["bnpl_open"], 99)
    log(f"applications {len(applications):,}")

    # ------------------------------------------------------------------ accounts
    def accounts_for(app, prefix, extra_cols):
        a = app[app.decision == "approved"].copy()
        a["account_id"] = [f"{prefix}-{i:07d}" for i in range(1, len(a) + 1)]
        a["open_m"] = a.month_idx
        ratio = np.clip(1.15 - 0.25 * np.maximum(a.legacy_risk_score.values, 0), 0.5, 1.0)
        a["granted_usd"] = (a.requested_amount_usd * ratio).round(-1).clip(lower=50)
        for k, v in extra_cols.items():
            a[k] = v(a)
        return a

    acc_card = accounts_for(a_card, "CC", {"limit_usd": lambda a: (a.granted_usd / 50).round() * 50})
    acc_loan = accounts_for(a_loan, "TL", {"principal_usd": lambda a: a.granted_usd,
                                           "apr": lambda a: (C.apr_loan.values[ci[a.cidx.values]] * np.exp(0.18 * risk[a.cidx.values])).clip(0.06, 0.9).round(3)})
    acc_bn = accounts_for(a_bn, "BN", {"limit_usd": lambda a: a.granted_usd})
    log(f"accounts card={len(acc_card):,} loan={len(acc_loan):,} bnpl={len(acc_bn):,}")

    R = {}
    for name, fn, acc, key in (("CARD", sim_card, acc_card, "CARD"), ("LOAN", sim_loan, acc_loan, "LOAN"), ("BNPL", sim_bnpl, acc_bn, "BNPL")):
        R[name] = fn(rng, ctx, acc, calib[key])
        log(f"simulated {name}: {sum(len(x) for x in R[name][0]):,} account-months")

    snaps = pd.concat([x for k in R for x in R[k][0]], ignore_index=True)
    snaps["month"] = pd.DatetimeIndex(MONTHS[snaps.month_idx.values.astype(int)])
    snaps["dpd_bucket"] = pd.cut(snaps.dpd, [-1, 0, 29, 59, 89, 10_000], labels=["Current", "1-29", "30-59", "60-89", "90+"]).astype(object)
    snaps.loc[snaps.status == "written_off", "dpd_bucket"] = "Written off"
    for c in ("account_id", "customer_id", "product", "status", "dpd_bucket"):
        snaps[c] = snaps[c].astype("category")

    events = pd.concat([x for k in R for x in R[k][1]], ignore_index=True) if any(R[k][1] for k in R) else pd.DataFrame()
    collections = pd.concat([x for k in R for x in R[k][2]], ignore_index=True)
    wo_list = pd.concat([x for k in R for x in R[k][3]], ignore_index=True) if any(R[k][3] for k in R) else pd.DataFrame(columns=["account_id", "product", "month_idx", "exposure_usd"])
    if len(wo_list):
        rec_mean = wo_list["product"].map({"CARD": .20, "LOAN": .25, "BNPL": .13}).values
        rr = rng.beta(2, 2 / rec_mean - 2)
        wo_list["recovery_rate"] = rr.round(3)
        wo_list["recovery_total_usd"] = (wo_list.exposure_usd * rr).round(2)
        el = np.clip((N_MONTHS - 1 - wo_list.month_idx.values) / 12, 0, 1)
        wo_list["recovered_to_date_usd"] = (wo_list.recovery_total_usd * el).round(2)
        wo_list["writeoff_month"] = MONTHS[wo_list.month_idx.values.astype(int)]

    accounts = pd.concat([
        acc_card.assign(product="CARD", limit_or_principal_usd=acc_card.limit_usd, tenor=np.nan, apr=C.apr_card.values[ci[acc_card.cidx.values]]),
        acc_loan.assign(product="LOAN", limit_or_principal_usd=acc_loan.principal_usd),
        acc_bn.assign(product="BNPL", limit_or_principal_usd=acc_bn.limit_usd, tenor=3, apr=0.0)], ignore_index=True)
    accounts = accounts[["account_id", "customer_id", "product", "open_m", "limit_or_principal_usd", "tenor", "apr"]].rename(
        columns={"open_m": "open_month_idx"})
    accounts["open_date"] = MONTHS[accounts.open_month_idx.values.astype(int)]
    accounts = accounts.merge(customers[["customer_id", "country", "city"]], on="customer_id")
    applications = applications.drop(columns=["cidx"]).merge(customers[["customer_id", "country", "city"]], on="customer_id")
    applications["app_date"] = MONTHS[applications.month_idx.values.astype(int)]
    acc_map = pd.concat([acc_card[["customer_id", "account_id"]].assign(product="CARD"), acc_loan[["customer_id", "account_id"]].assign(product="LOAN"),
                         acc_bn[["customer_id", "account_id"]].assign(product="BNPL")])
    applications = applications.merge(acc_map, on=["customer_id", "product"], how="left")
    if len(events):
        events["month"] = MONTHS[events.month_idx.values.astype(int)]
    collections["month"] = MONTHS[collections.month_idx.values.astype(int)]

    # ------------------------------------------------------------------ ground truth (kept apart from the app data)
    bn_adopt = np.where(ctx["bnpl_open"] < 99)[0]
    truth = {"tau_card_spend_young_mid_old": [TAU_CARD_SPEND * 1.4, TAU_CARD_SPEND, TAU_CARD_SPEND * 0.6],
             "tau_card_spend_avg_over_adopters_with_card": float(np.mean(tau[np.intersect1d(bn_adopt, acc_card.cidx.values)])),
             "loan_substitution_prob": LOAN_SUBSTITUTION, "card_attrition_multiplier": ATTRITION_MULT,
             "collections_effect_on_roll": EFF, "bnpl_launch_month_idx": dict(zip(C.code, map(int, C.bnpl_launch_m))),
             "n_customers": n, "seed": seed}
    (out / "ground_truth.json").write_text(json.dumps(truth, indent=2))

    for name, df in dict(customers=customers, cities=cities, countries=C, accounts=accounts, account_month=snaps,
                         applications=applications, limit_events=events, collections=collections, writeoffs=wo_list,
                         macro=macro).items():
        if len(df):
            df.to_parquet(out / f"{name}.parquet", index=False)
    meta = {"as_of": str(MONTHS[-1].date()), "n_months": N_MONTHS, "generated_seconds": round(time.time() - t0, 1),
            "rows": {k: int(len(v)) for k, v in dict(customers=customers, accounts=accounts, account_month=snaps,
                                                      applications=applications, collections=collections).items()}}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    log("written", out.resolve())
    return meta


if __name__ == "__main__":
    n_ = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    d_ = sys.argv[2] if len(sys.argv) > 2 else "data/world"
    print(generate(n_, out_dir=d_))
