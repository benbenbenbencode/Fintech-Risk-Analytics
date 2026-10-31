"""Model layer for the Studio: builds features from the monthly panel, trains every model and scores the book.

Models
    application_<product>   PD at origination (bad = 90+ DPD or write-off within 12 months on book)
    behavioral_<product>    PD for existing accounts (bad = reaches 90+ DPD / write-off in the next 6 months)
    collections_cure        probability that a delinquent account cures this month (+ treatment uplift table)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from crf.evaluate import metrics
from crf.modeling import model_cfg, train_scorecard

from . import text
from .generate_world import MONTHS, N_MONTHS

WORLD = Path(__file__).resolve().parents[1] / "data" / "world"
HORIZON = 6
LAST = N_MONTHS - 1
PRODUCTS = ["CARD", "LOAN", "BNPL"]
BEHAV_FEATURES = {
    "CARD": ["mob", "utilization", "util_chg_3m", "bal_growth_3m", "dpd_now", "max_dpd_3m", "max_dpd_6m", "late_months_6m",
             "pay_ratio_3m", "spend_trend", "limit_chg_6m", "balance_to_income", "limit_to_income", "income", "age",
             "bureau_score", "cust_n_active", "other_delinq", "macro_stress", "country", "employment_type", "acquisition_channel"],
    "LOAN": ["mob", "utilization", "bal_growth_3m", "dpd_now", "max_dpd_3m", "max_dpd_6m", "late_months_6m", "pay_ratio_3m",
             "balance_to_income", "income", "age", "bureau_score", "cust_n_active", "other_delinq", "macro_stress", "country",
             "employment_type", "acquisition_channel"],
    "BNPL": ["mob", "utilization", "util_chg_3m", "dpd_now", "max_dpd_3m", "max_dpd_6m", "late_months_6m", "pay_ratio_3m",
             "orders_3m", "spend_trend", "limit_chg_6m", "limit_to_income", "income", "age", "bureau_score", "cust_n_active",
             "other_delinq", "macro_stress", "country", "employment_type", "acquisition_channel"],
}
CATEGORICAL = {"country", "employment_type", "acquisition_channel", "education", "residence_type", "device_os", "product"}


def risk_band_vec(p: np.ndarray) -> np.ndarray:
    return np.select([p < .015, p < .035, p < .08, p < .18], ["A", "B", "C", "D"], "E")


def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(WORLD / f"{name}.parquet")


# ====================================================================================================
# panel features (behavioural)
# ====================================================================================================
def _group_helpers(acct: np.ndarray, mob: np.ndarray):
    n = len(acct)
    start = np.arange(n) - mob

    def lag(x, k):
        out = np.full(n, np.nan)
        if k < n:
            out[k:] = x[:-k]
        out[mob < k] = np.nan
        return out

    def rsum(x, w):
        cs = np.cumsum(np.nan_to_num(x))
        before = np.where(start > 0, cs[np.maximum(start - 1, 0)], 0.0)
        prev = np.where(mob >= w, np.roll(cs, w), before)
        return cs - prev

    def rmax(x, w):
        out = x.astype(float).copy()
        for k in range(1, w):
            out = np.fmax(out, lag(x, k))
        return out
    return lag, rsum, rmax


def build_panel(prod: str, am: pd.DataFrame, cust_by_code: pd.DataFrame, flags: pd.DataFrame, macro_stress: pd.DataFrame) -> pd.DataFrame:
    d = am[am["product"] == prod].sort_values(["acct_code", "month_idx"]).reset_index(drop=True)
    acct, mob = d["acct_code"].values, d["mob"].values.astype(int)
    lag, rsum, rmax = _group_helpers(acct, mob)
    dpd = d["dpd"].values.astype(float)
    bal, lim, util = d["balance_usd"].values, d["credit_limit_usd"].values, d["utilization"].values
    paid, due, spend, orders = d["payment_made_usd"].values, d["payment_due_usd"].values, d["spend_usd"].values, d["n_orders"].values.astype(float)
    bad_flag = (dpd >= 90) | (d["status"].values == "written_off")
    cc = d["cc"].values
    cu = cust_by_code.iloc[cc]
    income = cu["monthly_income_usd"].values
    P = pd.DataFrame({
        "account_id": d["account_id"].astype(str).values, "customer_id": d["customer_id"].astype(str).values, "month_idx": d["month_idx"].values.astype(int),
        "mob": mob, "status": d["status"].values, "dpd_now": dpd, "bad_flag": bad_flag,
        "utilization": util, "util_chg_3m": util - lag(util, 3),
        "bal_growth_3m": np.clip(bal / (lag(bal, 3) + 1.0) - 1, -1, 5),
        "max_dpd_3m": rmax(dpd, 3), "max_dpd_6m": rmax(dpd, 6), "late_months_6m": rsum((dpd > 0).astype(float), 6),
        "limit_chg_6m": np.clip(lim / np.maximum(lag(lim, 6), 1) - 1, -1, 5),
        "balance_to_income": bal / np.maximum(income, 1), "limit_to_income": lim / np.maximum(income, 1),
        "income": income, "age": cu["age"].values, "bureau_score": cu["bureau_score"].values,
        "country": cu["country"].values, "employment_type": cu["employment_type"].values,
        "acquisition_channel": cu["acquisition_channel"].values, "balance_usd": bal, "credit_limit_usd": lim,
        "spend_3m_usd": rsum(spend, 3), "payment_due_usd": due,
    })
    sum_due = rsum(due, 3)
    P["pay_ratio_3m"] = np.where(sum_due > 0, rsum(paid, 3) / np.maximum(sum_due, 1e-6), np.nan).clip(0, 1.5)
    s3, s6 = rsum(spend, 3), rsum(spend, 6)
    P["spend_trend"] = np.where(mob >= 6, np.clip(s3 / (s6 - s3 + 1.0), 0, 5), np.nan)
    P["orders_3m"] = rsum(orders, 3)
    f = flags.set_index(["cc", "month_idx"])
    idx = pd.MultiIndex.from_arrays([cc, d["month_idx"].values])
    ff = f.reindex(idx)
    P["cust_n_active"] = ff["cust_n_active"].values
    P["other_delinq"] = ((ff["cust_n_delq"].values - (dpd > 0)) > 0).astype(float)
    ci = macro_stress["index"].get_indexer(P["country"].values)
    P["macro_stress"] = macro_stress["values"][ci, P["month_idx"].values]
    # target: reaches 90+/written-off in the next HORIZON months
    n = len(P)
    fut = np.zeros(n, bool)
    for j in range(1, HORIZON + 1):
        sh, same = np.zeros(n, bool), np.zeros(n, bool)
        sh[:-j], same[:-j] = bad_flag[j:], acct[j:] == acct[:-j]
        fut |= sh & same
    P["future_bad"] = fut
    return P


def _types(feats):
    return {f: ("categorical" if f in CATEGORICAL else "numeric") for f in feats}


# ====================================================================================================
def _reasons_json(card, X: pd.DataFrame, top_n=3) -> list[str]:
    rc = card.reason_codes(X, top_n=top_n)
    return [json.dumps([{"f": r["feature"], "bin": r["bin"], "lost": r["points_lost"]} for r in items]) for items in rc]


def build_all(world: Path = WORLD, progress=print) -> dict:
    t0 = time.time()
    log = lambda *a: progress(f"[{time.time() - t0:5.1f}s] " + " ".join(str(x) for x in a))
    cust = pd.read_parquet(world / "customers.parquet")
    am = pd.read_parquet(world / "account_month.parquet")
    macro = pd.read_parquet(world / "macro.parquet")
    apps = pd.read_parquet(world / "applications.parquet")
    coll = pd.read_parquet(world / "collections.parquet")
    wo = pd.read_parquet(world / "writeoffs.parquet")
    accounts = pd.read_parquet(world / "accounts.parquet")
    log("loaded", len(am), "account-months")

    am["acct_code"] = am["account_id"].cat.codes
    am["cc"] = am["customer_id"].cat.codes
    cust_by_code = cust.set_index("customer_id").reindex(am["customer_id"].cat.categories)
    delq = (am["dpd"] > 0) & (am["status"] != "written_off")
    flags = (am.assign(delq=delq).groupby(["cc", "month_idx"]).agg(cust_n_active=("acct_code", "size"), cust_n_delq=("delq", "sum")))
    flags = flags.reset_index()
    piv = macro.pivot(index="country", columns="month_idx", values="stress_index")
    macro_stress = {"index": pd.Index(piv.index), "values": piv.values}

    cfg_b = model_cfg(split={"mode": "out_of_time", "test_fraction": 0.3, "seed": 1},
                      selection={"min_iv": 0.02, "max_corr": 0.7, "max_features": 12, "C": 1.0})
    results, registry, panels, current_rows, hist_rows = {}, [], {}, [], []

    # -------------------------------------------------------------- behavioural models
    for prod in PRODUCTS:
        P = build_panel(prod, am, cust_by_code, flags, macro_stress)
        feats = BEHAV_FEATURES[prod]
        elig = (~P["bad_flag"]) & P["status"].isin(["active", "delinquent"]) & (P["mob"] >= 3) & (P["month_idx"] <= LAST - HORIZON)
        tr = P[elig & (P["month_idx"] % 3 == 0)]
        X = tr[feats].reset_index(drop=True)
        y = tr["future_bad"].astype(int).reset_index(drop=True)
        dates = tr["month_idx"].reset_index(drop=True)
        res = train_scorecard(X, y, _types(feats), dates, cfg_b)
        res["population"] = {"rows": int(len(X)), "bad_rate": float(y.mean())}
        results[f"behavioral_{prod}"] = res
        m = res["metrics"].set_index("set")
        registry.append(dict(id=f"behavioral_{prod}", product=prod, kind="Behavioral PD (existing accounts)",
                             target="Reaches 90+ DPD or write-off within the next 6 months", rows=int(len(X)), bad_rate=float(y.mean()),
                             auc=float(m.loc["test", "auc"]), ks=float(m.loc["test", "ks"]), gini=float(m.loc["test", "gini"]),
                             challenger_auc=float(res["challenger"]["metrics"]["auc"]), psi=float(res["score_psi"]),
                             features=len(res["selected"])))
        log(f"behavioral {prod}: rows={len(X):,} bad={y.mean():.2%} AUC={m.loc['test','auc']:.3f} KS={m.loc['test','ks']:.3f} (GBM {res['challenger']['metrics']['auc']:.3f})")
        # score the whole panel (PD history + current book)
        sc_rows = (P["mob"] >= 3) & (~P["bad_flag"]) & P["status"].isin(["active", "delinquent"])
        S = res["card"].score(P.loc[sc_rows, feats])
        P["pd_6m"] = np.nan
        P.loc[sc_rows, "pd_6m"] = S["pd"].values
        P["score"] = np.nan
        P.loc[sc_rows, "score"] = S["score"].values
        hist_rows.append(P.loc[sc_rows & (P["month_idx"] >= 6), ["account_id", "month_idx", "pd_6m"]].assign(product=prod))
        cur = P[(P["month_idx"] == LAST) & P["status"].isin(["active", "delinquent", "npl"])].copy()
        okm = cur["pd_6m"].notna()
        cur["reasons"] = "[]"
        if okm.any():
            cur.loc[okm, "reasons"] = _reasons_json(res["card"], cur.loc[okm, feats])
        cur["score_source"] = np.where(okm, "behavioral", "pending")
        cur["product"] = prod
        current_rows.append(cur)
        panels[prod] = P[["account_id", "month_idx", "mob", "balance_usd", "utilization", "max_dpd_6m", "late_months_6m",
                          "pay_ratio_3m", "income", "age", "bureau_score", "country", "other_delinq", "macro_stress", "dpd_now"]]

    # -------------------------------------------------------------- application models
    bad12 = (am[am["mob"] <= 12].assign(b=lambda d: (d["dpd"] >= 90) | (d["status"] == "written_off")).groupby("account_id", observed=True)["b"].max())
    bad12.index = bad12.index.astype(str)
    mac_long = macro.set_index(["country", "month_idx"])["stress_index"]
    app_scores = []
    cstat = cust.set_index("customer_id")
    for prod in PRODUCTS:
        a = apps[apps["product"] == prod].merge(cust.drop(columns=["country", "city"]), on="customer_id", how="left")
        a["country"] = cstat.loc[a["customer_id"], "country"].values
        a["city_tier"] = a["city"].map(pd.read_parquet(world / "cities.parquet").set_index("city")["tier"])
        a["thin_file"] = a["bureau_score"].isna().astype(float)
        a["req_to_income"] = a["requested_amount_usd"] / a["monthly_income_usd"].clip(lower=1)
        a["macro_stress_app"] = mac_long.reindex(pd.MultiIndex.from_arrays([a["country"], a["month_idx"]])).values
        feats = ["country", "city_tier", "age", "employment_type", "education", "residence_type", "household_size", "years_at_job",
                 "monthly_income_usd", "acquisition_channel", "device_os", "bureau_score", "bureau_history_months",
                 "bureau_active_accounts", "bureau_inquiries_6m", "bureau_max_dpd_12m", "thin_file", "email_age_days",
                 "device_shared_apps_30d", "ip_country_match", "requested_amount_usd", "req_to_income", "macro_stress_app"]
        if prod == "LOAN":
            a["tenor"] = a["tenor"] if "tenor" in a else np.nan
            feats.append("tenor")
        a = a.reset_index(drop=True)
        train = a[(a["decision"] == "approved") & (a["month_idx"] <= LAST - 12)].copy()
        train["bad"] = bad12.reindex(train["account_id"]).fillna(False).astype(int).values
        Xa = train[feats].reset_index(drop=True)
        ya = train["bad"].reset_index(drop=True)
        da = train["month_idx"].reset_index(drop=True)
        types = _types(feats)
        types["city_tier"] = "numeric"
        res = train_scorecard(Xa, ya, types, da, model_cfg(split={"mode": "out_of_time", "test_fraction": 0.3, "seed": 1}))
        res["population"] = {"rows": int(len(Xa)), "bad_rate": float(ya.mean())}
        results[f"application_{prod}"] = res
        m = res["metrics"].set_index("set")
        registry.append(dict(id=f"application_{prod}", product=prod, kind="Application PD (onboarding)",
                             target="90+ DPD or write-off within 12 months of opening", rows=int(len(Xa)), bad_rate=float(ya.mean()),
                             auc=float(m.loc["test", "auc"]), ks=float(m.loc["test", "ks"]), gini=float(m.loc["test", "gini"]),
                             challenger_auc=float(res["challenger"]["metrics"]["auc"]), psi=float(res["score_psi"]),
                             features=len(res["selected"])))
        log(f"application {prod}: rows={len(Xa):,} bad={ya.mean():.2%} AUC={m.loc['test','auc']:.3f} KS={m.loc['test','ks']:.3f} (GBM {res['challenger']['metrics']['auc']:.3f})")
        S = res["card"].score(a[feats])
        out = a[["application_id", "customer_id", "product", "month_idx", "decision", "decline_reason", "requested_amount_usd",
                 "legacy_risk_score", "account_id"]].copy()
        out["pd_app"], out["score_app"] = S["pd"].values, S["score"].values
        out["reasons"] = _reasons_json(res["card"], a[feats])
        app_scores.append(out)
    app_scores = pd.concat(app_scores, ignore_index=True)
    app_scores["risk_band"] = risk_band_vec(app_scores["pd_app"].values)
    app_scores.to_parquet(world / "app_scores.parquet", index=False)

    # -------------------------------------------------------------- collections cure model
    P_all = pd.concat(panels.values(), ignore_index=True)
    P_all["m_next"] = P_all["month_idx"] + 1
    c = coll.merge(P_all.drop(columns=["month_idx"]), left_on=["account_id", "month_idx"], right_on=["account_id", "m_next"], how="left", suffixes=("", "_p"))
    c = c.sort_values(["account_id", "month_idx"]).reset_index(drop=True)
    g = c.groupby("account_id", observed=True)
    c["prior_contact_rate"] = (g["contacted"].cumsum() - c["contacted"]) / np.maximum(g.cumcount(), 1)
    c.loc[g.cumcount() == 0, "prior_contact_rate"] = np.nan
    cfeats = ["bucket", "balance_usd", "utilization", "max_dpd_6m", "late_months_6m", "pay_ratio_3m", "mob", "income", "age",
              "bureau_score", "other_delinq", "macro_stress", "prior_contact_rate", "country", "product"]
    c["product"] = c["product"].astype(str)
    c["country"] = c["country"].astype(str)
    cc_ = c.dropna(subset=["balance_usd"]).reset_index(drop=True)
    Xc = cc_[cfeats].copy()
    for col in ("country", "product"):
        Xc[col] = Xc[col].astype("category").cat.codes
    yc = cc_["cured"].astype(int)
    cut = cc_["month_idx"].quantile(0.7)
    tr, te = cc_["month_idx"] <= cut, cc_["month_idx"] > cut
    hgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=20, min_samples_leaf=50,
                                         categorical_features=[cfeats.index("country"), cfeats.index("product")], random_state=0)
    hgb.fit(Xc[tr], yc[tr])
    pc = hgb.predict_proba(Xc[te])[:, 1]
    cm = metrics(yc[te], pc)
    country_codes = dict(zip(cc_["country"].astype("category").cat.categories, range(100)))
    product_codes = dict(zip(cc_["product"].astype("category").cat.categories, range(10)))
    log(f"collections cure model: rows={len(cc_):,} cure rate={yc.mean():.2%} AUC={cm['auc']:.3f} KS={cm['ks']:.3f}")
    registry.append(dict(id="collections_cure", product="ALL", kind="Collections cure propensity",
                         target="Account returns to current within the month", rows=int(len(cc_)), bad_rate=float(1 - yc.mean()),
                         auc=cm["auc"], ks=cm["ks"], gini=cm["gini"], challenger_auc=float("nan"), psi=float("nan"), features=len(cfeats)))
    rnd = coll[coll["assigned_by"] == "random"]
    up = rnd.groupby(["bucket", "action"]).agg(n=("cured", "size"), cure_rate=("cured", "mean"), cost=("cost_usd", "mean")).reset_index()
    base = up[up.action == "none"].set_index("bucket")[["cure_rate", "n"]].rename(columns={"cure_rate": "p0", "n": "n0"})
    up = up.merge(base, on="bucket", how="left")
    up["uplift"] = up["cure_rate"] - up["p0"]
    up["se"] = np.sqrt(up.cure_rate * (1 - up.cure_rate) / up.n + up.p0 * (1 - up.p0) / up.n0)
    up.loc[up.action == "none", ["uplift", "se"]] = 0.0

    # -------------------------------------------------------------- current book
    cur = pd.concat(current_rows, ignore_index=True)
    cur["bucket"] = np.minimum(np.ceil(cur["dpd_now"] / 30), 4).astype(int)
    # cure probability for delinquent accounts
    dm = cur["status"].isin(["delinquent", "npl"])
    pcx = pd.Series(np.nan, index=cur.index)
    if dm.any():
        cx = cur.loc[dm].copy()
        cx["prior_contact_rate"] = np.nan
        last_contact = coll.sort_values("month_idx").groupby("account_id", observed=True)["contacted"].mean()
        cx["prior_contact_rate"] = last_contact.reindex(cx["account_id"]).values
        Z = cx[cfeats].copy()
        Z["country"] = Z["country"].map(country_codes).fillna(-1).astype(int)
        Z["product"] = Z["product"].map(product_codes).fillna(-1).astype(int)
        pcx.loc[dm] = hgb.predict_proba(Z)[:, 1]
    cur["cure_prob"] = pcx
    # fall back to application PD for accounts too new for behavioural scoring
    newm = cur["pd_6m"].isna() & ~cur["status"].isin(["npl"])
    ap = app_scores.dropna(subset=["account_id"]).set_index("account_id")
    cur.loc[newm, "pd_6m"] = ap["pd_app"].reindex(cur.loc[newm, "account_id"]).values * 0.55   # 12m PD -> rough 6m
    cur.loc[newm, "reasons"] = ap["reasons"].reindex(cur.loc[newm, "account_id"]).fillna("[]").values
    cur.loc[newm, "score_source"] = "application"
    cur.loc[cur["status"] == "npl", "pd_6m"] = 1.0
    cur["risk_band"] = np.where(cur["status"] == "npl", "NPL", risk_band_vec(cur["pd_6m"].fillna(0.02).values))
    # 3 months ago PD (watchlist)
    hist = pd.concat(hist_rows, ignore_index=True)
    hist.to_parquet(world / "pd_history.parquet", index=False)
    h3 = hist[hist["month_idx"] == LAST - 3].set_index("account_id")["pd_6m"]
    cur["pd_3m_ago"] = h3.reindex(cur["account_id"]).values
    cur["watchlist"] = ((cur["pd_6m"] >= 0.04) & (cur["pd_6m"] >= 1.5 * cur["pd_3m_ago"]) & (cur["status"] != "npl")).fillna(False)
    keep = ["account_id", "customer_id", "product", "mob", "status", "dpd_now", "bucket", "balance_usd", "credit_limit_usd", "utilization",
            "max_dpd_6m", "late_months_6m", "pay_ratio_3m", "spend_3m_usd", "payment_due_usd", "other_delinq", "pd_6m", "pd_3m_ago", "score", "risk_band", "reasons",
            "score_source", "cure_prob", "watchlist"]
    cur[keep].to_parquet(world / "scores_current.parquet", index=False)

    # -------------------------------------------------------------- loss rates (for expected-loss maths)
    lgd = wo.groupby("product").apply(lambda d: 1 - d["recovery_total_usd"].sum() / d["exposure_usd"].sum(), include_groups=False).to_dict()
    up.to_parquet(world / "uplift.parquet", index=False)
    json.dump({"lgd": lgd}, open(world / "loss_rates.json", "w"))
    for r in results.values():
        r.pop("scored", None)
        if "challenger" in r:
            r["challenger"].pop("model", None)
    joblib.dump({"results": results, "collections": {"model": hgb, "features": cfeats, "metrics": cm,
                                                    "country_codes": country_codes, "product_codes": product_codes},
                 "bands": text.BAND_NAMES}, world / "models.joblib", compress=3)
    pd.DataFrame(registry).to_json(world / "registry.json", orient="records", indent=1)
    log("models saved")
    return {"registry": registry}


if __name__ == "__main__":
    build_all()
