"""Cached portfolio statistics used by several pages."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import data

LAST = 47


def _where(products, countries, p="m.", c="a."):
    w, params = ["TRUE"], []
    if products:
        w.append(f"{p}product IN ({','.join('?' * len(products))})"); params += list(products)
    if countries:
        w.append(f"{c}country IN ({','.join('?' * len(countries))})"); params += list(countries)
    return " AND ".join(w), params


def snapshot(month_idx: int, products=None, countries=None) -> dict:
    w, p = _where(products, countries)
    r = data.q(f"""SELECT count(*) accounts, sum(m.balance_usd) exposure,
        sum(CASE WHEN m.status='npl' THEN m.balance_usd ELSE 0 END)/NULLIF(sum(m.balance_usd),0) npl_ratio,
        sum(CASE WHEN m.dpd>=30 AND m.status<>'written_off' THEN m.balance_usd ELSE 0 END)/NULLIF(sum(m.balance_usd),0) dpd30_ratio
        FROM account_month m JOIN accounts a USING (account_id)
        WHERE m.month_idx = ? AND m.status IN ('active','delinquent','npl') AND {w}""", [month_idx] + p)
    return r.iloc[0].to_dict()


def kpis(products=None, countries=None) -> dict:
    now, ago = snapshot(LAST, products, countries), snapshot(LAST - 3, products, countries)
    w, p = _where(products, countries, "b.", "b.")
    cur = data.q(f"""SELECT avg(b.pd_6m) FILTER (WHERE b.status<>'npl') avg_pd,
        sum(CASE WHEN b.watchlist THEN 1 ELSE 0 END) watchlist, sum(CASE WHEN b.watchlist THEN b.balance_usd ELSE 0 END) watchlist_exposure,
        sum(CASE WHEN b.status<>'active' THEN 1 ELSE 0 END) in_collections FROM book b WHERE {w}""", p).iloc[0].to_dict()
    return {**now, **cur, "npl_ago": ago["npl_ratio"], "dpd30_ago": ago["dpd30_ratio"], "accounts_ago": ago["accounts"],
            "exposure_ago": ago["exposure"]}


def trend(products=None, countries=None) -> pd.DataFrame:
    w, p = _where(products, countries)
    d = data.q(f"""SELECT m.month_idx, m.product,
        sum(CASE WHEN m.status='npl' THEN m.balance_usd ELSE 0 END)/NULLIF(sum(m.balance_usd),0) npl_ratio,
        sum(CASE WHEN m.dpd>=30 AND m.status<>'written_off' THEN m.balance_usd ELSE 0 END)/NULLIF(sum(m.balance_usd),0) dpd30_ratio,
        sum(m.balance_usd) exposure, count(*) accounts
        FROM account_month m JOIN accounts a USING (account_id) WHERE m.status IN ('active','delinquent','npl') AND {w}
        GROUP BY 1,2 ORDER BY 1,2""", p)
    d["month"] = d["month_idx"].map(data.month_label)
    return d


def geo(products=None, countries=None) -> pd.DataFrame:
    w, p = _where(products, countries, "b.", "b.")
    return data.q(f"""SELECT b.country, b.city, c.lat, c.lon, count(*) accounts, sum(b.balance_usd) exposure,
        sum(CASE WHEN b.status='npl' THEN b.balance_usd ELSE 0 END)/NULLIF(sum(b.balance_usd),0) npl_ratio, avg(b.pd_6m) avg_pd
        FROM book b JOIN cities c ON c.city=b.city WHERE {w} GROUP BY 1,2,3,4""", p)


def band_mix(products=None, countries=None) -> pd.DataFrame:
    w, p = _where(products, countries, "b.", "b.")
    return data.q(f"SELECT b.product, b.risk_band, count(*) accounts, sum(b.balance_usd) exposure FROM book b WHERE {w} GROUP BY 1,2", p)


def roll_rates(products=None, countries=None, months=12) -> pd.DataFrame:
    w, p = _where(products, countries)
    d = data.q(f"""WITH x AS (SELECT m.account_id, m.month_idx, m.dpd_bucket b FROM account_month m JOIN accounts a USING (account_id)
                   WHERE m.month_idx >= ? AND {w})
        SELECT a.b from_b, c.b to_b, count(*) n FROM x a JOIN x c ON a.account_id=c.account_id AND c.month_idx=a.month_idx+1
        GROUP BY 1,2""", [LAST - months] + p)
    d["share"] = d["n"] / d.groupby("from_b")["n"].transform("sum")
    return d


def vintage(products=None, countries=None) -> pd.DataFrame:
    w, p = _where(products, countries, "a.", "a.")
    return data.q(f"""WITH fb AS (SELECT account_id, min(mob) first_bad FROM account_month
                                  WHERE dpd >= 90 OR status = 'written_off' GROUP BY 1)
        SELECT a.product, (2022 + (a.open_month_idx + 9) // 12) AS cohort, k AS mob,
               avg(CASE WHEN fb.first_bad <= k THEN 1.0 ELSE 0.0 END) bad_rate, count(*) accounts
        FROM accounts a LEFT JOIN fb USING (account_id) CROSS JOIN range(3, 37, 3) t(k)
        WHERE a.open_month_idx + k <= {LAST} AND {w} GROUP BY 1,2,3 HAVING count(*) >= 150 ORDER BY 1,2,3""", p)


def briefing_inputs(products=None, countries=None) -> dict:
    k = kpis(products, countries)
    g = data.q("""SELECT country, sum(CASE WHEN status='npl' THEN balance_usd ELSE 0 END)/NULLIF(sum(balance_usd),0) r FROM book GROUP BY 1 ORDER BY r DESC""")
    pr = data.q("""SELECT product, sum(CASE WHEN status='npl' THEN balance_usd ELSE 0 END)/NULLIF(sum(balance_usd),0) r FROM book GROUP BY 1 ORDER BY r DESC""")
    return {"month": data.month_label(LAST), "accounts": int(k["accounts"]), "exposure": float(k["exposure"]), "npl_ratio": float(k["npl_ratio"]),
            "npl_delta": float(k["npl_ratio"] - k["npl_ago"]), "dpd30_ratio": float(k["dpd30_ratio"]), "worst_country": g.country[0],
            "worst_country_npl": float(g.r[0]), "worst_product": pr["product"][0], "worst_product_npl": float(pr.r[0]),
            "watchlist": int(k["watchlist"]), "watchlist_exposure": float(k["watchlist_exposure"]), "in_collections": int(k["in_collections"]),
            "top_action": "payment plans and restructuring offers"}
