"""Model evaluation: AUC / Gini / KS, lift and calibration by decile."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def metrics(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    return {"n": int(len(y)), "bad_rate": float(y.mean()), "auc": float(auc),
            "gini": float(2 * auc - 1), "ks": float((tpr - fpr).max())}


def roc_points(y, p) -> pd.DataFrame:
    fpr, tpr, _ = roc_curve(np.asarray(y), np.asarray(p))
    return pd.DataFrame({"fpr": fpr, "tpr": tpr})


def ks_curve(y, p) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}).sort_values("p", ascending=False)
    df["pop"] = np.arange(1, len(df) + 1) / len(df)
    df["cum_bad"] = (df["y"] == 1).cumsum() / max((df["y"] == 1).sum(), 1)
    df["cum_good"] = (df["y"] == 0).cumsum() / max((df["y"] == 0).sum(), 1)
    return df[["pop", "cum_bad", "cum_good"]].iloc[:: max(len(df) // 200, 1)]


def decile_table(y, pd_, score, n: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "pd": np.asarray(pd_), "score": np.asarray(score)})
    df = df.sort_values("pd", ascending=False).reset_index(drop=True)
    df["decile"] = pd.qcut(np.arange(len(df)), n, labels=False) + 1     # 1 = riskiest
    g = df.groupby("decile").agg(count=("y", "size"), bad=("y", "sum"), avg_pd=("pd", "mean"),
                                 min_score=("score", "min"), max_score=("score", "max")).reset_index()
    g["bad_rate"] = g["bad"] / g["count"]
    g["cum_bad_capture"] = g["bad"].cumsum() / max(g["bad"].sum(), 1)
    g["lift"] = g["bad_rate"] / max(df["y"].mean(), 1e-9)
    return g
