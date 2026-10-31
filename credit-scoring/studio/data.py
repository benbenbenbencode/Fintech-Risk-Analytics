"""Data access: DuckDB over Parquet (fast list/search/aggregation for hundreds of thousands of accounts)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import duckdb
import joblib
import pandas as pd

WORLD = Path(__file__).resolve().parents[1] / "data" / "world"
TABLES = ["customers", "cities", "countries", "accounts", "account_month", "applications", "limit_events", "collections",
          "writeoffs", "macro", "app_scores", "scores_current", "pd_history", "uplift"]
_con = None


def ready() -> bool:
    return all((WORLD / f"{n}.parquet").exists() for n in ("account_month", "scores_current", "app_scores")) and \
        (WORLD / "models.joblib").exists() and (WORLD / "cannibal.json").exists()


def con():
    global _con
    if _con is None:
        _con = duckdb.connect()
        for t in TABLES:
            p = WORLD / f"{t}.parquet"
            if p.exists():
                _con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_parquet('{p.as_posix()}')")
        _con.execute("""CREATE VIEW book AS
            SELECT s.*, a.country, a.city, a.open_date, a.open_month_idx, a.limit_or_principal_usd AS original_amount_usd,
                   c.age, c.employment_type, c.monthly_income_usd, c.bureau_score, c.acquisition_channel, c.education
            FROM scores_current s JOIN accounts a USING (account_id) JOIN customers c ON c.customer_id = s.customer_id""")
    return _con


@lru_cache(maxsize=512)
def _q(sql: str, params: tuple) -> pd.DataFrame:
    return con().execute(sql, list(params)).df()


def q(sql: str, params: tuple | list = ()) -> pd.DataFrame:
    return _q(sql, tuple(params)).copy()


def reset():
    global _con
    _q.cache_clear()
    if _con is not None:
        _con.close()
    _con = None
    models.cache_clear()


@lru_cache(maxsize=1)
def meta() -> dict:
    return json.loads((WORLD / "meta.json").read_text())


@lru_cache(maxsize=1)
def models() -> dict:
    return joblib.load(WORLD / "models.joblib")


@lru_cache(maxsize=1)
def registry() -> pd.DataFrame:
    return pd.read_json(WORLD / "registry.json")


@lru_cache(maxsize=1)
def cannibal() -> dict:
    return json.loads((WORLD / "cannibal.json").read_text())


@lru_cache(maxsize=1)
def loss_rates() -> dict:
    return json.loads((WORLD / "loss_rates.json").read_text())["lgd"]


@lru_cache(maxsize=1)
def ground_truth() -> dict:
    p = WORLD / "ground_truth.json"
    return json.loads(p.read_text()) if p.exists() else {}


@lru_cache(maxsize=1)
def unit_economics() -> pd.DataFrame:
    """Revenue yield, exposure-at-default and LGD per product, measured from the data itself."""
    d = q("""SELECT product, sum(revenue_usd)*12.0/NULLIF(sum(credit_limit_usd),0) AS rev_yield_granted,
                    sum(revenue_usd)*12.0/NULLIF(sum(balance_usd),0) AS annual_yield
             FROM account_month WHERE status IN ('active','delinquent') GROUP BY product""").set_index("product")
    e = q("""SELECT w.product, avg(w.exposure_usd / a.limit_or_principal_usd) AS ead_default_ratio
             FROM writeoffs w JOIN accounts a USING (account_id) GROUP BY 1""").set_index("product")
    d = d.join(e)
    d["lgd"] = pd.Series(loss_rates())
    return d


def as_of_idx() -> int:
    return meta()["n_months"] - 1


def month_label(idx: int) -> str:
    return (pd.Timestamp("2022-10-31") + pd.offsets.MonthEnd(int(idx))).strftime("%b %Y")
