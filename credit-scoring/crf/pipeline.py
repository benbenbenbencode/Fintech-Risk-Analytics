"""End-to-end orchestration for the CSV-based lab. Everything is driven by the config dict."""
from __future__ import annotations

import pandas as pd

from .config import load_config
from .features import build_features
from .loader import build_base, load_tables
from .modeling import train_scorecard
from .target import build_target


def run_pipeline(cfg_src, tables: dict | None = None, base_dir=None) -> dict:
    cfg = load_config(cfg_src, base_dir)
    tables = tables or load_tables(cfg)
    base, rep = build_base(tables)

    tgt, waterfall = build_target(base, rep, cfg)
    X, types, warns = build_features(base, rep, cfg)
    elig = tgt.index[tgt["eligible"]]
    if len(elig) < 200:
        raise ValueError(f"Only {len(elig)} eligible loans after maturity filters - need more history "
                         f"or shorter observation/performance windows.")
    disb = base.set_index("loan_id")["disbursement_date"]
    ds, y_all, dates = X.loc[elig].copy(), tgt.loc[elig, "bad"].astype(int), disb.loc[elig]

    res = train_scorecard(ds, y_all, types, dates, cfg, with_challenger=False)
    scored = res["scored"].rename(columns={"date": "disbursement_date"})
    res["scored"] = scored
    vint = (scored.assign(month=scored["disbursement_date"].dt.to_period("M").dt.to_timestamp())
            .groupby("month").agg(loans=("bad", "size"), bad_rate=("bad", "mean")).reset_index())
    t = cfg["target"]
    res.update({
        "cfg": cfg, "waterfall": waterfall, "warnings": warns, "vintage": vint, "X": ds,
        "target_summary": {"bad_rate": float(y_all.mean()), "n_model": int(len(y_all)),
                           "n_train": int(len(res["train_ids"])), "n_test": int(len(res["test_ids"])),
                           "bad_dpd": t["bad_dpd"], "obs": t["observation_months"], "perf": t["performance_months"],
                           "type": cfg["scorecard_type"]},
        "tables_preview": {k: v.head(5) for k, v in tables.items()},
        "table_rows": {k: len(v) for k, v in tables.items()},
    })
    return res
