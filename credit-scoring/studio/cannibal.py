"""Cannibalization analysis: how much of BNPL replaces card spend, term-loan origination and card retention.

Methods (card spend as the main example)
    naive level gap       adopters vs non-adopters after launch        -> biased (adopters shop more)
    before / after        adopters only, 6m before vs 6m after          -> biased (seasonality, trend)
    matched DiD           adopters vs matched controls with a pseudo adoption date in the same calendar
                          window (strata: country x age band x pre-period spend level) -> credible
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .generate_world import N_MONTHS, SEAS, MONTHS

WIN = 6


def _bins(x, qs=(1 / 3, 2 / 3)):
    cuts = np.quantile(x, qs)
    return np.searchsorted(cuts, x)


def analyse(world: Path, n_boot: int = 200, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    am = pd.read_parquet(world / "account_month.parquet", columns=["account_id", "customer_id", "product", "month_idx", "status",
                                                                  "spend_usd", "revenue_usd", "credit_limit_usd"])
    acc = pd.read_parquet(world / "accounts.parquet")
    cust = pd.read_parquet(world / "customers.parquet")
    am["customer_id"] = am["customer_id"].astype(str)
    am["status"] = am["status"].astype(str)
    cmap = {c: i for i, c in enumerate(cust.customer_id)}
    ncust = len(cust)
    country = cust.country.values
    age_band = np.digitize(cust.age.values, [30, 45])

    def mat(product, col):
        d = am[am["product"] == product]
        M = np.full((ncust, N_MONTHS), np.nan)
        M[d.customer_id.map(cmap).values, d.month_idx.values.astype(int)] = d[col].values
        return M

    open_m = {p: np.full(ncust, 999) for p in ("CARD", "LOAN", "BNPL")}
    for p in open_m:
        a = acc[acc["product"] == p]
        open_m[p][a.customer_id.map(cmap).values] = a.open_month_idx.values
    card_spend, card_rev = mat("CARD", "spend_usd"), mat("CARD", "revenue_usd")
    bn_rev = mat("BNPL", "revenue_usd")
    card_closed = np.full(ncust, 999)
    cl = am[(am["product"] == "CARD") & (am["status"] == "closed")]
    card_closed[cl.customer_id.map(cmap).values] = cl.month_idx.values
    bn_spend = mat("BNPL", "spend_usd")

    # ---------------- samples ----------------
    eligible = (open_m["CARD"] < 999) & (open_m["BNPL"] > open_m["CARD"] + WIN)
    adopt = eligible & (open_m["BNPL"] <= N_MONTHS - 1 - WIN)
    ctrl = (open_m["CARD"] < 999) & (open_m["BNPL"] == 999)
    A = np.where(adopt)[0]
    pseudo = np.full(ncust, -1)
    ev_by_country = {c: open_m["BNPL"][A][country[A] == c] for c in np.unique(country)}
    Cn = []
    for i in np.where(ctrl)[0]:
        pool = ev_by_country.get(country[i])
        if pool is None or len(pool) == 0:
            continue
        p = int(rng.choice(pool))
        if open_m["CARD"][i] + WIN <= p <= N_MONTHS - 1 - WIN:
            pseudo[i] = p
            Cn.append(i)
    Cn = np.array(Cn)
    ev = open_m["BNPL"].copy()
    ev[Cn] = pseudo[Cn]

    def window_sum(M, idx, lo, hi):
        out = np.zeros(len(idx))
        for k in range(lo, hi + 1):
            col = np.clip(ev[idx] + k, 0, N_MONTHS - 1)
            out += np.nan_to_num(M[idx, col])
        return out

    pre_s = {"A": window_sum(card_spend, A, -WIN, -1), "C": window_sum(card_spend, Cn, -WIN, -1)}
    post_s = {"A": window_sum(card_spend, A, 1, WIN), "C": window_sum(card_spend, Cn, 1, WIN)}
    pre_r = {"A": window_sum(card_rev, A, -WIN, -1), "C": window_sum(card_rev, Cn, -WIN, -1)}
    post_r = {"A": window_sum(card_rev, A, 1, WIN), "C": window_sum(card_rev, Cn, 1, WIN)}
    bn_gain = window_sum(bn_rev, A, 1, WIN)

    allpre = np.concatenate([pre_s["A"], pre_s["C"]])
    lvl = _bins(allpre)
    strata = {"A": (country_code := pd.factorize(country)[0])[A] * 100 + age_band[A] * 10 + lvl[:len(A)],
              "C": country_code[Cn] * 100 + age_band[Cn] * 10 + lvl[len(A):]}
    keys = np.unique(np.concatenate([strata["A"], strata["C"]]))
    sidx = {k: i for i, k in enumerate(keys)}
    sa, sc = np.array([sidx[k] for k in strata["A"]]), np.array([sidx[k] for k in strata["C"]])
    S = len(keys)

    def did(pre_a, post_a, pre_c, post_c, sa_, sc_, w_pre=None):
      with np.errstate(divide="ignore", invalid="ignore"):
            pa, qa = np.bincount(sa_, pre_a, S), np.bincount(sa_, post_a, S)
            pc, qc = np.bincount(sc_, pre_c, S), np.bincount(sc_, post_c, S)
            ok = (pa > 0) & (pc > 0) & (qc > 0)
            eff = np.where(ok, (qa / np.maximum(pa, 1e-9)) / (qc / np.maximum(pc, 1e-9)) - 1, 0.0)
            w = np.where(ok, pa, 0.0)
            return float((eff * w).sum() / max(w.sum(), 1e-9)), (eff, w, pa, qa, pc, qc, ok)

    est, parts = did(pre_s["A"], post_s["A"], pre_s["C"], post_s["C"], sa, sc)
    boots = []
    for _ in range(n_boot):
        ia, ic = rng.integers(0, len(A), len(A)), rng.integers(0, len(Cn), len(Cn))
        boots.append(did(pre_s["A"][ia], post_s["A"][ia], pre_s["C"][ic], post_s["C"][ic], sa[ia], sc[ic])[0])
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    # naive estimates
    all_card = (open_m["CARD"] < 999)
    never = np.where(all_card & (open_m["BNPL"] == 999))[0]
    naive_level = float(post_s["A"].mean() / np.mean([np.nansum(card_spend[i, max(0, N_MONTHS - 1 - WIN + 1):]) for i in never]) - 1)
    before_after = float(post_s["A"].sum() / max(pre_s["A"].sum(), 1e-9) - 1)

    # revenue view (USD over the 6 months after adoption, adopters only)
    est_r, parts_r = did(pre_r["A"], post_r["A"], pre_r["C"], post_r["C"], sa, sc)
    rev_lost = float(-(est_r * pre_r["A"].sum()))
    rev_gain = float(bn_gain.sum())
    # term-loan take-up: customers who joined after BNPL launch (BNPL available throughout), adopters vs matched never-adopters
    launch_map = json.loads((world / "ground_truth.json").read_text())["bnpl_launch_month_idx"]
    launch_c = np.array([launch_map[c] for c in country])
    join = cust.join_month_idx.values
    pool = join >= launch_c
    has_loan = (open_m["LOAN"] < 999).astype(float)
    ls = pd.factorize(country)[0] * 100 + age_band * 10 + np.minimum(join // 8, 5)
    adA, adC = pool & (open_m["BNPL"] < 999), pool & (open_m["BNPL"] == 999)
    ka = pd.DataFrame({"s": ls[adA], "l": has_loan[adA]}).groupby("s")["l"].agg(["mean", "size"])
    kc = pd.DataFrame({"s": ls[adC], "l": has_loan[adC]}).groupby("s")["l"].agg(["mean", "size"])
    jj = ka.join(kc, lsuffix="_a", rsuffix="_c", how="inner")
    loan_ratio = float((jj["mean_a"] * jj["size_a"]).sum() / max((jj["mean_c"] * jj["size_a"]).sum(), 1e-9))
    close_a = ((card_closed[A] > ev[A]) & (card_closed[A] <= ev[A] + 12)).mean()
    close_c = ((card_closed[Cn] > ev[Cn]) & (card_closed[Cn] <= ev[Cn] + 12)).mean()

    # event study: spend index around adoption (adopters vs controls), normalised to pre-period mean
    rows = []
    for grp, idx in (("Adopters", A), ("Matched controls", Cn)):
        base = window_sum(card_spend, idx, -WIN, -1).sum() / WIN
        for k in range(-WIN, WIN + 1):
            if k == 0:
                continue
            rows.append((grp, k, window_sum(card_spend, idx, k, k).sum() / base))
    es = pd.DataFrame(rows, columns=["group", "rel_month", "spend_index"])

    # staggered launch: card spend per card customer by months since launch (country average, deseasonalised)
    launch = launch_map
    cs_rows = []
    for c, L in launch.items():
        idx = np.where((country == c) & (open_m["CARD"] < 999))[0]
        tot = np.nansum(card_spend[idx], axis=0) / np.maximum((~np.isnan(card_spend[idx])).sum(axis=0), 1)
        ds = tot / SEAS[np.array([MONTHS[m].month - 1 for m in range(N_MONTHS)])]
        pre = ds[max(L - 6, 6):L].mean()
        for m in range(max(6, L - 6), N_MONTHS):
            cs_rows.append((c, m - L, ds[m] / pre))
    country_es = pd.DataFrame(cs_rows, columns=["country", "rel_month", "spend_index"])
    country_es = country_es[country_es.rel_month.between(-6, 18)]

    # adoption curve and risk stacking
    adopt_cum = (acc[acc["product"] == "BNPL"].groupby(["country", "open_month_idx"]).size().rename("n").reset_index())
    cur = pd.read_parquet(world / "scores_current.parquet", columns=["customer_id", "product", "balance_usd", "credit_limit_usd", "status", "risk_band"])
    cu = cur.groupby("customer_id").agg(n_products=("product", "nunique"), exposure=("balance_usd", "sum"), limits=("credit_limit_usd", "sum"),
                                        any_delq=("status", lambda s: (s != "active").any()))
    cu = cu.join(cust.set_index("customer_id")[["monthly_income_usd", "country"]])
    cu["limit_to_income"] = cu["limits"] / cu["monthly_income_usd"]
    stack = (cu.groupby("n_products").agg(customers=("exposure", "size"), avg_exposure_usd=("exposure", "mean"),
                                          avg_limit_to_income=("limit_to_income", "mean"), delinquent_share=("any_delq", "mean")).reset_index())

    out = {
        "n_adopters": int(len(A)), "n_controls": int(len(Cn)), "window_months": WIN,
        "card_spend_effect": {"matched_did": est, "ci95": ci, "naive_level_gap": naive_level, "before_after": before_after},
        "revenue_6m": {"card_revenue_lost_usd": rev_lost, "bnpl_revenue_gained_usd": rev_gain,
                       "cannibalization_rate": float(rev_lost / rev_gain) if rev_gain > 0 else None,
                       "net_incremental_usd": rev_gain - rev_lost},
        "loan_takeup_ratio": loan_ratio, "loan_takeup_n": [int(adA.sum()), int(adC.sum())],
        "card_closure_ratio_12m": float(close_a / max(close_c, 1e-9)),
        "by_country": [],
    }
    for c in np.unique(country):
        ma, mc = country[A] == c, country[Cn] == c
        if ma.sum() < 30 or mc.sum() < 30:
            continue
        ra = post_s["A"][ma].sum() / max(pre_s["A"][ma].sum(), 1e-9)
        rc = post_s["C"][mc].sum() / max(pre_s["C"][mc].sum(), 1e-9)
        out["by_country"].append({"country": c, "adopters": int(ma.sum()), "spend_effect": float(ra / rc - 1),
                                  "bnpl_revenue_6m": float(bn_gain[ma].sum()),
                                  "card_revenue_lost_6m": float(-((post_r["A"][ma].sum() / max(pre_r["A"][ma].sum(), 1e-9)) /
                                                                  (post_r["C"][mc].sum() / max(pre_r["C"][mc].sum(), 1e-9)) - 1) * pre_r["A"][ma].sum())})
    es.to_parquet(world / "cannibal_event_study.parquet", index=False)
    country_es.to_parquet(world / "cannibal_country_es.parquet", index=False)
    adopt_cum.to_parquet(world / "cannibal_adoption.parquet", index=False)
    stack.to_parquet(world / "cannibal_stacking.parquet", index=False)
    (world / "cannibal.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    from .data import WORLD
    print(json.dumps(analyse(WORLD), indent=2))
