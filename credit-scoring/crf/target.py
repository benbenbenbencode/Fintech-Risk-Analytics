"""Target (bad) definition driven entirely by config."""
from __future__ import annotations

import pandas as pd

from .dpd import compute_dpd


def build_target(base: pd.DataFrame, rep: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = cfg["target"]
    as_of = pd.Timestamp(cfg["as_of_date"])
    obs, perf, bad_dpd = t["observation_months"], t["performance_months"], t["bad_dpd"]

    disb = base.set_index("loan_id")["disbursement_date"]
    obs_end = disb + pd.DateOffset(months=obs)
    perf_end = disb + pd.DateOffset(months=obs + perf)
    measure = perf_end.clip(upper=as_of)
    matured = perf_end <= as_of

    r = rep.copy()
    r["obs_end"] = r["loan_id"].map(obs_end)
    r["perf_end"] = r["loan_id"].map(perf_end)
    r["measure"] = r["loan_id"].map(measure)
    r["dpd"] = compute_dpd(r["due_date"], r["paid_date"], r["measure"])

    in_win = (r["due_date"] > r["obs_end"]) & (r["due_date"] <= r["perf_end"])
    perf_agg = r[in_win].groupby("loan_id").agg(perf_max_dpd=("dpd", "max"), n_perf=("dpd", "size"))

    out = pd.DataFrame(index=disb.index).join(perf_agg)
    out["matured"] = matured
    out["n_perf"] = out["n_perf"].fillna(0).astype(int)
    out["bad"] = (out["perf_max_dpd"] >= bad_dpd).astype(float).where(out["perf_max_dpd"].notna())

    out["obs_max_dpd"] = float("nan")
    if obs > 0:
        ro = r[r["due_date"] <= r["obs_end"]]
        d = compute_dpd(ro["due_date"], ro["paid_date"], ro["obs_end"])
        out["obs_max_dpd"] = d.groupby(ro["loan_id"]).max()
    thr = t["exclude_if_obs_dpd_ge"]
    out["already_bad"] = (out["obs_max_dpd"] >= thr) if (obs > 0 and thr > 0) else False
    out["eligible"] = out["matured"] & (out["n_perf"] > 0) & ~out["already_bad"]

    n0 = len(out)
    n1 = int(out["matured"].sum())
    n2 = int((out["matured"] & (out["n_perf"] > 0)).sum())
    n3 = int(out["eligible"].sum())
    wf = pd.DataFrame({
        "step": ["All loans", "Performance window fully matured", "Has installments in window",
                 "Not already bad at observation end"],
        "remaining": [n0, n1, n2, n3],
    })
    wf["dropped"] = (-wf["remaining"].diff()).fillna(0).astype(int)
    return out, wf
