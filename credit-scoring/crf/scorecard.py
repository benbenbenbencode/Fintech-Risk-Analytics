"""Logistic-regression scorecard on WoE features, with points table and reason codes."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class Scorecard:
    def __init__(self, binners: dict, cfg: dict):
        self.binners = binners
        sc = cfg["scorecard"]
        self.factor = sc["pdo"] / np.log(2)
        self.offset = sc["base_score"] - self.factor * np.log(sc["base_odds"])
        self.C = cfg["selection"]["C"]

    def fit(self, woe: pd.DataFrame, y_bad, iv: dict) -> "Scorecard":
        y_good = 1 - np.asarray(y_bad)
        feats = list(woe.columns)
        while feats:
            lr = LogisticRegression(C=self.C, max_iter=2000).fit(woe[feats], y_good)
            coefs = dict(zip(feats, lr.coef_[0]))
            neg = [f for f, c in coefs.items() if c <= 0]     # WoE = ln(good/bad) -> coef must be positive
            if not neg:
                break
            feats.remove(min(neg, key=lambda f: iv[f]))
        if not feats:
            raise ValueError("No feature kept a positive coefficient - check data and selection settings.")
        self.features, self.coef, self.intercept = feats, coefs, float(lr.intercept_[0])
        self.const = (self.offset + self.factor * self.intercept) / len(feats)
        rows = []
        for f in feats:
            t = self.binners[f].table[["feature", "bin", "count", "pct", "bad_rate", "woe"]].copy()
            t["points"] = self.factor * self.coef[f] * t["woe"] + self.const
            rows.append(t)
        self.points_table = pd.concat(rows, ignore_index=True)
        self.max_points = self.points_table.groupby("feature")["points"].max()
        self.coef_table = (pd.DataFrame({"feature": feats, "coefficient": [self.coef[f] for f in feats],
                                         "iv": [iv[f] for f in feats]}))
        return self

    def feature_points(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({f: self.factor * self.coef[f] * self.binners[f].transform(X[f]) + self.const
                             for f in self.features}, index=X.index)

    def score(self, X: pd.DataFrame) -> pd.DataFrame:
        P = self.feature_points(X)
        score = P.sum(axis=1)
        logit_good = (score - self.offset) / self.factor
        return pd.DataFrame({"score": score, "pd": 1 / (1 + np.exp(logit_good))}, index=X.index)

    def reason_codes(self, X: pd.DataFrame, top_n: int = 4) -> pd.Series:
        """Per row: the features costing the most points versus the best possible bin."""
        P = self.feature_points(X)
        lost = (self.max_points.reindex(self.features).values[None, :] - P.values)
        labels = {f: self.binners[f].assign(X[f]) for f in self.features}
        order = np.argsort(-lost, axis=1)[:, :top_n]
        out = []
        for r, idx in enumerate(order):
            items = []
            for j in idx:
                if lost[r, j] < 0.5:
                    continue
                f = self.features[j]
                items.append({"feature": f, "value": X[f].iloc[r], "bin": labels[f].iloc[r],
                              "points_lost": round(float(lost[r, j]), 1)})
            out.append(items)
        return pd.Series(out, index=X.index)
