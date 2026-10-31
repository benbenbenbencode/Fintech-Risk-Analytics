"""Feature builders: application-level, ratios and behavioural (repayment) features."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dpd import compute_dpd


def build_behavioral(base: pd.DataFrame, rep: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Repayment behaviour inside the observation window only (no leakage into the performance window)."""
    obs = cfg["target"]["observation_months"]
    disb = base.set_index("loan_id")["disbursement_date"]
    obs_end = disb + pd.DateOffset(months=obs)

    r = rep.copy()
    r["obs_end"] = r["loan_id"].map(obs_end)
    r = r[r["due_date"] <= r["obs_end"]].sort_values(["loan_id", "installment_no"]).copy()
    r["dpd"] = compute_dpd(r["due_date"], r["paid_date"], r["obs_end"])
    r["late"] = (r["dpd"] > 0).astype(int)
    r["late30"] = (r["dpd"] >= 30).astype(int)
    r["paid_flag"] = (r["paid_date"].notna() & (r["paid_date"] <= r["obs_end"]))
    r["paid_amt"] = r["due_amount"].where(r["paid_flag"], 0.0)

    g = r.groupby("loan_id")
    feats = pd.DataFrame({
        "obs_max_dpd": g["dpd"].max(),
        "obs_mean_dpd": g["dpd"].mean(),
        "obs_late_count": g["late"].sum(),
        "obs_late30_count": g["late30"].sum(),
        "obs_first_inst_dpd": g["dpd"].first(),
        "obs_last_inst_dpd": g["dpd"].last(),
        "obs_paid_ratio": g["paid_amt"].sum() / g["due_amount"].sum(),
    })
    feats["obs_dpd_trend"] = feats["obs_last_inst_dpd"] - feats["obs_first_inst_dpd"]

    for w in cfg["features"].get("rolling_windows", []):
        if 0 < w < obs:
            start = r["obs_end"] - pd.DateOffset(months=w)
            sub = r[r["due_date"] > start]
            feats[f"late_count_last_{w}m"] = sub.groupby("loan_id")["late"].sum()
            feats[f"max_dpd_last_{w}m"] = sub.groupby("loan_id")["dpd"].max()
    return feats


def build_features(base: pd.DataFrame, rep: pd.DataFrame, cfg: dict):
    """Returns (X indexed by loan_id, {feature: 'numeric'|'categorical'}, warnings)."""
    f = cfg["features"]
    b = base.set_index("loan_id")
    X = pd.DataFrame(index=b.index)
    types: dict[str, str] = {}
    warns: list[str] = []

    for c in f.get("numeric", []):
        if c in b.columns:
            X[c] = pd.to_numeric(b[c], errors="coerce")
            types[c] = "numeric"
        else:
            warns.append(f"numeric feature '{c}' not found - skipped")
    for c in f.get("categorical", []):
        if c in b.columns:
            X[c] = b[c].astype("object")
            types[c] = "categorical"
        else:
            warns.append(f"categorical feature '{c}' not found - skipped")
    for name, pair in (f.get("ratios") or {}).items():
        num, den = pair
        if num in b.columns and den in b.columns:
            X[name] = pd.to_numeric(b[num], errors="coerce") / pd.to_numeric(b[den], errors="coerce").replace(0, np.nan)
            types[name] = "numeric"
        else:
            warns.append(f"ratio '{name}' needs '{num}' and '{den}' - skipped")

    if cfg["scorecard_type"] == "behavioral" and cfg["target"]["observation_months"] > 0:
        B = build_behavioral(base, rep, cfg)
        for c in B.columns:
            X[c] = B[c].reindex(X.index)
            types[c] = "numeric"
    return X, types, warns
