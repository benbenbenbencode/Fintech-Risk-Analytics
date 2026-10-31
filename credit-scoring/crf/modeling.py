"""Reusable modelling core: any feature table + binary target -> scorecard, evaluation, stability, challenger."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import DEFAULTS, _merge
from .evaluate import decile_table, ks_curve, metrics, roc_points
from .leakage import check_leakage
from .psi import psi_from_labels, psi_numeric, psi_status
from .scorecard import Scorecard
from .woe import FeatureBinner


def strength(iv: float) -> str:
    return ("useless" if iv < 0.02 else "weak" if iv < 0.1 else "medium" if iv < 0.3
            else "strong" if iv < 0.5 else "suspicious")


def model_cfg(**over) -> dict:
    cfg = _merge(DEFAULTS, {})
    return _merge(cfg, over)


def split_ids(dates: pd.Series, cfg: dict):
    sp = cfg["split"]
    n_test = int(round(len(dates) * sp["test_fraction"]))
    if sp["mode"] == "out_of_time":
        order = dates.sort_values(kind="stable").index
        return order[:-n_test], order[-n_test:]
    rng = np.random.default_rng(sp["seed"])
    perm = rng.permutation(dates.index.values)
    return perm[n_test:], perm[:n_test]


def challenger(Xtr, ytr, Xte, yte, types: dict) -> dict:
    """Gradient-boosting benchmark on the same raw features (to show what the scorecard gives up)."""
    def enc(df, cats=None):
        out = df.copy()
        for c, t in types.items():
            if c in out and t == "categorical":
                out[c] = out[c].astype("object").where(out[c].notna(), "NA").astype(str)
        return out
    A, B = enc(Xtr), enc(Xte)
    cats = [c for c in A.columns if types.get(c) == "categorical"]
    for c in cats:
        codes = {v: i for i, v in enumerate(sorted(set(A[c]) | set(B[c])))}
        A[c], B[c] = A[c].map(codes).astype(int), B[c].map(codes).astype(int)
    A = A.astype(float)
    B = B.astype(float)
    mask = [c in cats for c in A.columns]
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=24, min_samples_leaf=60,
                                       l2_regularization=1.0, categorical_features=mask, random_state=0)
    m.fit(A, ytr)
    p = m.predict_proba(B)[:, 1]
    return {"model": m, "metrics": metrics(yte, p), "p_test": pd.Series(p, index=Xte.index)}


def train_scorecard(X: pd.DataFrame, y: pd.Series, types: dict, dates: pd.Series, cfg: dict,
                    with_challenger: bool = True) -> dict:
    train_ids, test_ids = split_ids(dates, cfg)
    Xtr, Xte, ytr, yte = X.loc[train_ids], X.loc[test_ids], y.loc[train_ids], y.loc[test_ids]
    if ytr.nunique() < 2 or yte.nunique() < 2:
        raise ValueError("Train or test set has only one class - check the bad definition / windows.")
    bn = cfg["binning"]
    binners = {f: FeatureBinner(f, types[f], bn["max_bins"], bn["min_bin_pct"], bn["monotonic"]).fit(Xtr[f], ytr.values)
               for f in Xtr.columns}
    iv_table = (pd.DataFrame({"feature": list(binners), "type": [types[f] for f in binners],
                              "iv": [b.iv for b in binners.values()], "bins": [len(b.labels) for b in binners.values()]})
                .sort_values("iv", ascending=False).reset_index(drop=True))
    iv_table["strength"] = iv_table["iv"].map(strength)
    leak = check_leakage(iv_table, cfg)
    dropped = set(leak.loc[leak["flagged"], "feature"]) if cfg["leakage"]["auto_drop"] else set()
    sel = cfg["selection"]
    cand = iv_table[(iv_table["iv"] >= sel["min_iv"]) & ~iv_table["feature"].isin(dropped)]["feature"].tolist()
    woe_tr = pd.DataFrame({f: binners[f].transform(Xtr[f]) for f in cand})
    chosen: list[str] = []
    for f in cand:
        if len(chosen) >= sel["max_features"]:
            break
        if all(abs(np.nan_to_num(woe_tr[f].corr(woe_tr[s]))) <= sel["max_corr"] for s in chosen):
            chosen.append(f)
    if not chosen:
        raise ValueError("No feature passed selection - lower selection.min_iv or check the data.")
    iv_map = dict(zip(iv_table["feature"], iv_table["iv"]))
    card = Scorecard(binners, cfg).fit(woe_tr[chosen], ytr.values, iv_map)

    s_tr, s_te = card.score(Xtr), card.score(Xte)
    m_tr, m_te = metrics(ytr, s_tr["pd"]), metrics(yte, s_te["pd"])
    scored = pd.concat([s_tr.assign(split="train"), s_te.assign(split="test")])
    scored["bad"] = y.reindex(scored.index)
    scored["date"] = dates.reindex(scored.index)
    fpsi = []
    for f in card.features:
        v = psi_from_labels(binners[f].assign(Xtr[f]), binners[f].assign(Xte[f]))
        fpsi.append({"feature": f, "psi": v, "status": psi_status(v)})
    res = {
        "iv_table": iv_table, "leakage": leak, "selected": card.features, "coef_table": card.coef_table,
        "points_table": card.points_table,
        "metrics": pd.DataFrame([{"set": "train", **m_tr}, {"set": "test", **m_te}]),
        "roc": {"train": roc_points(ytr, s_tr["pd"]), "test": roc_points(yte, s_te["pd"])},
        "ks": {"train": ks_curve(ytr, s_tr["pd"]), "test": ks_curve(yte, s_te["pd"])},
        "deciles": {"train": decile_table(ytr, s_tr["pd"], s_tr["score"]), "test": decile_table(yte, s_te["pd"], s_te["score"])},
        "score_psi": psi_numeric(s_tr["score"], s_te["score"]), "feature_psi": pd.DataFrame(fpsi),
        "scored": scored, "binners": binners, "card": card, "train_ids": train_ids, "test_ids": test_ids,
        "types": types,
    }
    if with_challenger:
        ch = challenger(Xtr, ytr, Xte, yte, types)
        res["challenger"] = {"metrics": ch["metrics"], "model": ch["model"]}
    return res
