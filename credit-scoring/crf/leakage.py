"""Leakage checks: banned names and 'too good to be true' predictors."""
from __future__ import annotations

import pandas as pd


def check_leakage(iv_table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    pats = [p.lower() for p in cfg["leakage"]["banned_patterns"]]
    max_iv = cfg["leakage"]["max_iv"]
    rows = []
    for r in iv_table.itertuples():
        reasons = []
        hit = [p for p in pats if p in r.feature.lower()]
        if hit:
            reasons.append(f"name matches banned pattern '{hit[0]}'")
        if r.iv > max_iv:
            reasons.append(f"single-feature IV {r.iv:.2f} > {max_iv}")
        rows.append({"feature": r.feature, "iv": r.iv, "flagged": bool(reasons), "reason": "; ".join(reasons)})
    return pd.DataFrame(rows)
