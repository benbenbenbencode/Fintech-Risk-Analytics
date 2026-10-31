"""WoE / IV binning (numeric with monotonic merging + categorical)."""
from __future__ import annotations

import numpy as np
import pandas as pd

MISSING, OTHER = "Missing", "OTHER"


def woe_iv(good, bad, smoothing: float = 0.5):
    good, bad = np.asarray(good, float), np.asarray(bad, float)
    k = len(good)
    dg = (good + smoothing) / (good.sum() + smoothing * k)
    db = (bad + smoothing) / (bad.sum() + smoothing * k)
    w = np.log(dg / db)           # WoE = ln(%good / %bad): higher = safer
    return w, (dg - db) * w


def _fmt(v: float) -> str:
    return f"{v:.4g}"


def _merge(edges, good, bad, i):
    g = np.concatenate([good[:i], [good[i] + good[i + 1]], good[i + 2:]])
    b = np.concatenate([bad[:i], [bad[i] + bad[i + 1]], bad[i + 2:]])
    return edges[:i] + edges[i + 1:], g, b


class FeatureBinner:
    def __init__(self, name: str, kind: str, max_bins: int = 8, min_bin_pct: float = 0.05,
                 monotonic: bool = True, smoothing: float = 0.5):
        self.name, self.kind = name, kind
        self.max_bins, self.min_bin_pct = max_bins, min_bin_pct
        self.monotonic, self.smoothing = monotonic, smoothing

    # ------------------------------------------------------------------ fit
    def fit(self, x: pd.Series, y) -> "FeatureBinner":
        y = np.asarray(y, dtype=float)
        if self.kind == "numeric":
            labels, good, bad = self._fit_numeric(x, y)
        else:
            labels, good, bad = self._fit_categorical(x, y)
        woe, ivb = woe_iv(good, bad, self.smoothing)
        tot = good + bad
        self.labels = labels
        self.woe = woe
        self.iv = float(ivb.sum())
        self._woe_map = dict(zip(labels, woe))
        self.table = pd.DataFrame({
            "feature": self.name, "bin": labels, "count": tot.astype(int),
            "pct": tot / tot.sum(), "good": good.astype(int), "bad": bad.astype(int),
            "bad_rate": bad / np.maximum(tot, 1), "woe": woe, "iv": ivb,
        })
        return self

    def _fit_numeric(self, x, y):
        xv = pd.to_numeric(x, errors="coerce").values.astype(float)
        miss = np.isnan(xv)
        n_total = len(xv)
        valid, yv = xv[~miss], y[~miss]
        edges: list[float] = []
        if valid.size and np.unique(valid).size > 1:
            qs = np.linspace(0, 1, self.max_bins + 1)[1:-1]
            edges = [float(e) for e in np.unique(np.quantile(valid, qs)) if e < valid.max()]

        def counts(ed):
            idx = np.searchsorted(ed, valid, side="left")
            bad = np.bincount(idx, weights=yv, minlength=len(ed) + 1)
            tot = np.bincount(idx, minlength=len(ed) + 1).astype(float)
            return tot - bad, bad

        good, bad = counts(edges)
        while len(good) > 1:                       # merge tiny bins
            pct = (good + bad) / n_total
            i = int(np.argmin(pct))
            if pct[i] >= self.min_bin_pct:
                break
            if i == 0:
                j = 0
            elif i == len(good) - 1:
                j = i - 1
            else:
                j = i - 1 if pct[i - 1] <= pct[i + 1] else i
            edges, good, bad = _merge(edges, good, bad, j)

        if self.monotonic and len(good) > 2:        # enforce monotonic bad rate
            br = bad / np.maximum(good + bad, 1)
            c = np.corrcoef(np.arange(len(br)), br)[0, 1] if br.std() > 0 else 0.0
            direction = 1 if (np.nan_to_num(c) >= 0) else -1
            while len(good) > 1:
                br = bad / np.maximum(good + bad, 1)
                viol = np.where(np.diff(br) * direction < -1e-12)[0]
                if len(viol) == 0:
                    break
                edges, good, bad = _merge(edges, good, bad, int(viol[0]))

        self.edges = edges
        if not edges:
            self._num_labels = ["All"]
        else:
            self._num_labels = ([f"<= {_fmt(edges[0])}"]
                                + [f"({_fmt(edges[i - 1])}, {_fmt(edges[i])}]" for i in range(1, len(edges))]
                                + [f"> {_fmt(edges[-1])}"])
        labels = list(self._num_labels)
        if miss.any():
            labels.append(MISSING)
            good = np.append(good, float((y[miss] == 0).sum()))
            bad = np.append(bad, float((y[miss] == 1).sum()))
        return labels, good, bad

    def _fit_categorical(self, x, y):
        s = pd.Series(x.values, dtype="object")
        miss = s.isna().values
        n_total = len(s)
        s = s.where(~miss, None)
        s = s.map(lambda v: v if v is None else str(v))
        df = pd.DataFrame({"c": s[~miss].values, "y": y[~miss]})
        grp = df.groupby("c")["y"].agg(["size", "sum"])
        keep = grp[grp["size"] / n_total >= self.min_bin_pct]
        self.cats = list(keep.index)
        rest = grp.drop(index=self.cats)
        keep = keep.assign(rate=keep["sum"] / keep["size"]).sort_values("rate")
        self.cats = list(keep.index)
        labels = list(keep.index)
        bad = list(keep["sum"].astype(float))
        good = list((keep["size"] - keep["sum"]).astype(float))
        self.has_other = len(rest) > 0
        if self.has_other:
            labels.append(OTHER)
            bad.append(float(rest["sum"].sum()))
            good.append(float((rest["size"] - rest["sum"]).sum()))
        if miss.any():
            labels.append(MISSING)
            bad.append(float(y[miss].sum()))
            good.append(float((y[miss] == 0).sum()))
        return labels, np.array(good), np.array(bad)

    # ------------------------------------------------------------ transform
    def assign(self, x: pd.Series) -> pd.Series:
        if self.kind == "numeric":
            xv = pd.to_numeric(x, errors="coerce").values.astype(float)
            miss = np.isnan(xv)
            idx = np.searchsorted(self.edges, np.where(miss, 0.0, xv), side="left")
            lab = np.array(self._num_labels, dtype=object)[idx]
            lab[miss] = MISSING
            return pd.Series(lab, index=x.index)
        cats = set(self.cats)

        def f(v):
            if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
                return MISSING
            v = str(v)
            return v if v in cats else OTHER
        return x.map(f)

    def transform(self, x: pd.Series) -> pd.Series:
        return self.assign(x).map(self._woe_map).fillna(0.0).astype(float)
