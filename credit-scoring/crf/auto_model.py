"""Model tournament: fit several candidates on the same split, compare them on
the same metrics, and pick one by an explicit, auditable policy.

Candidates
    scorecard   WoE + logistic regression - the interpretable bank model
    logistic    logistic regression on encoded raw features (no binning)
    gbm         histogram gradient boosting
    gbm_cal     the same GBM with isotonic calibration
    rf          random forest

Why a tournament instead of "just use GBM": in credit risk the interpretable
scorecard is often preferred even at a small AUC cost, because adverse-action
reasons must be explainable. The policy makes that trade-off explicit instead of
hiding it in a default.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULTS, _merge
from .evaluate import decile_table, metrics, roc_points
from .leakage import check_leakage
from .modeling import split_ids, strength
from .scorecard import Scorecard
from .woe import FeatureBinner


# ---------------------------------------------------------------------------
# encoding
# ---------------------------------------------------------------------------


class FrameEncoder:
    """Turns a mixed frame into an all-numeric frame, remembering the mapping.

    Kept as a small explicit class (rather than a ColumnTransformer) so the
    fitted object pickles cleanly and can be reused by the explanation module.
    """

    def __init__(self):
        self.numeric: list[str] = []
        self.categorical: list[str] = []
        self.maps: dict[str, dict] = {}
        self.types: dict[str, str] = {}

    def fit(self, X: pd.DataFrame, types: dict[str, str]) -> "FrameEncoder":
        self.types = dict(types)
        self.numeric = [c for c in X.columns if types.get(c) == "numeric"]
        self.categorical = [c for c in X.columns if types.get(c) != "numeric"]
        for column in self.categorical:
            values = X[column].astype("object").where(X[column].notna(), "__NA__").astype(str)
            categories = sorted(values.unique())
            self.maps[column] = {value: index for index, value in enumerate(categories)}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for column in self.numeric:
            if column in X.columns:
                out[column] = pd.to_numeric(X[column], errors="coerce")
        for column in self.categorical:
            if column in X.columns:
                values = X[column].astype("object").where(X[column].notna(), "__NA__").astype(str)
                mapping = self.maps.get(column, {})
                out[column] = values.map(mapping).astype("float64")
        return out

    @property
    def feature_names(self) -> list[str]:
        return self.numeric + self.categorical


# ---------------------------------------------------------------------------
# fitted model container
# ---------------------------------------------------------------------------


@dataclass
class FittedModel:
    name: str
    label: str
    kind: str  # "scorecard" | "sklearn"
    features: list[str]
    types: dict[str, str]
    interpretable: bool = False
    encoder: FrameEncoder | None = None
    estimator: object | None = None
    binners: dict | None = None
    card: Scorecard | None = None

    def predict_pd(self, X: pd.DataFrame) -> np.ndarray:
        frame = X.reindex(columns=self.features)
        if self.kind == "scorecard":
            return self.card.score(frame)["pd"].to_numpy()
        encoded = self.encoder.transform(frame)
        return self.estimator.predict_proba(encoded)[:, 1]

    def score_points(self, X: pd.DataFrame) -> np.ndarray | None:
        """Human-scale score, only meaningful for the scorecard."""
        if self.kind != "scorecard":
            return None
        return self.card.score(X.reindex(columns=self.features))["score"].to_numpy()


@dataclass
class TournamentResult:
    leaderboard: pd.DataFrame
    models: dict[str, FittedModel]
    chosen: str
    reason: str
    roc: dict[str, pd.DataFrame]
    calibration: pd.DataFrame
    importance: pd.DataFrame
    test_frame: pd.DataFrame
    iv_table: pd.DataFrame
    leakage: pd.DataFrame
    selected_features: list[str]
    dropped_features: pd.DataFrame
    train_ids: pd.Index
    test_ids: pd.Index
    warnings: list[str] = field(default_factory=list)

    def chosen_model(self) -> FittedModel:
        return self.models[self.chosen]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def calibration_slope(y, p) -> float:
    """Slope of y ~ logit(p). 1.0 is perfect, <1 over-dispersed, >1 under."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    if y.min() == y.max():
        return float("nan")
    model = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(model.coef_[0][0])


def _lift(y, p) -> float:
    table = decile_table(y, p, p * 1000, n=10)
    return float(table["lift"].iloc[0]) if not table.empty else float("nan")


def _evaluate(y, p) -> dict:
    base = metrics(y, p)
    base["brier"] = float(brier_score_loss(y, p))
    base["calib_slope"] = calibration_slope(y, p)
    base["top_decile_lift"] = _lift(y, p)
    return base


def permutation_importance(
    predict: Callable[[pd.DataFrame], np.ndarray],
    X: pd.DataFrame,
    y,
    n_repeats: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """AUC drop when a feature is shuffled - model agnostic, no extra dependency."""
    y = np.asarray(y)
    baseline = roc_auc_score(y, predict(X))
    rng = np.random.default_rng(seed)
    rows = []
    for column in X.columns:
        drops = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            drops.append(baseline - roc_auc_score(y, predict(shuffled)))
        rows.append({"feature": column, "importance": float(np.mean(drops)),
                     "std": float(np.std(drops))})
    frame = pd.DataFrame(rows).sort_values("importance", ascending=False)
    frame["relative"] = frame["importance"] / max(frame["importance"].max(), 1e-9)
    return frame.reset_index(drop=True)


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------


def select_features(Xtr, ytr, types, cfg) -> tuple[dict, pd.DataFrame, list[str], pd.DataFrame]:
    """WoE bin every candidate, score it by IV, drop leakage and correlated twins."""
    bn = cfg["binning"]
    binners = {
        f: FeatureBinner(f, types[f], bn["max_bins"], bn["min_bin_pct"], bn["monotonic"]).fit(
            Xtr[f], ytr.to_numpy()
        )
        for f in Xtr.columns
    }
    iv_table = pd.DataFrame(
        {
            "feature": list(binners),
            "type": [types.get(f, "?") for f in binners],
            "iv": [b.iv for b in binners.values()],
            "bins": [len(b.labels) for b in binners.values()],
        }
    ).sort_values("iv", ascending=False).reset_index(drop=True)
    iv_table["strength"] = iv_table["iv"].map(strength)

    leak = check_leakage(iv_table, cfg)
    dropped = set(leak.loc[leak["flagged"], "feature"]) if cfg["leakage"]["auto_drop"] else set()

    sel = cfg["selection"]
    candidates = iv_table[
        (iv_table["iv"] >= sel["min_iv"]) & ~iv_table["feature"].isin(dropped)
    ]["feature"].tolist()

    woe = pd.DataFrame({f: binners[f].transform(Xtr[f]) for f in candidates})
    chosen: list[str] = []
    for feature in candidates:
        if len(chosen) >= sel["max_features"]:
            break
        if all(abs(np.nan_to_num(woe[feature].corr(woe[other]))) <= sel["max_corr"] for other in chosen):
            chosen.append(feature)

    return binners, iv_table, chosen, leak


def _make_estimators(seed: int = 0) -> dict[str, tuple[str, object, bool]]:
    """name -> (label, estimator, interpretable)"""
    return {
        "logistic": (
            "Logistic regression (raw)",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=3000)),
            ]),
            False,
        ),
        "gbm": (
            "Gradient boosting",
            HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.06, max_leaf_nodes=24,
                min_samples_leaf=40, l2_regularization=1.0, random_state=seed,
            ),
            False,
        ),
        "rf": (
            "Random forest",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("clf", RandomForestClassifier(
                    n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=seed,
                )),
            ]),
            False,
        ),
    }


def run_tournament(
    X: pd.DataFrame,
    y: pd.Series,
    types: dict[str, str],
    dates: pd.Series,
    cfg: dict,
    models: tuple[str, ...] = ("scorecard", "logistic", "gbm", "gbm_cal", "rf"),
    include_importance: bool = True,
) -> TournamentResult:
    warnings: list[str] = []
    train_ids, test_ids = split_ids(dates, cfg)
    Xtr, Xte = X.loc[train_ids], X.loc[test_ids]
    ytr, yte = y.loc[train_ids], y.loc[test_ids]

    if ytr.nunique() < 2 or yte.nunique() < 2:
        raise ValueError(
            "Train or test split contains a single class. Check the bad definition, "
            "the performance window, or use a different split."
        )

    binners, iv_table, chosen_features, leak = select_features(Xtr, ytr, types, cfg)
    if not chosen_features:
        raise ValueError(
            "No feature survived IV/correlation selection. Lower selection.min_iv, "
            "provide more history, or check that the bad definition is meaningful."
        )

    fitted: dict[str, FittedModel] = {}
    rows: list[dict] = []
    curves: dict[str, pd.DataFrame] = {}

    # ---- scorecard --------------------------------------------------------
    if "scorecard" in models:
        started = time.time()
        woe_tr = pd.DataFrame({f: binners[f].transform(Xtr[f]) for f in chosen_features})
        card = Scorecard(binners, cfg).fit(woe_tr, ytr.to_numpy(), dict(zip(iv_table["feature"], iv_table["iv"])))
        p_tr = card.score(Xtr)["pd"].to_numpy()
        p_te = card.score(Xte)["pd"].to_numpy()

        fitted["scorecard"] = FittedModel(
            name="scorecard", label="WoE scorecard (logistic)", kind="scorecard",
            features=list(card.features), types=types, interpretable=True,
            binners=binners, card=card,
        )
        rows.append({"model": "scorecard", "label": "WoE scorecard (logistic)", "interpretable": True,
                     **{f"test_{k}": v for k, v in _evaluate(yte, p_te).items()},
                     **{f"train_{k}": v for k, v in _evaluate(ytr, p_tr).items()},
                     "seconds": round(time.time() - started, 2)})
        curves["scorecard"] = roc_points(yte, p_te)

    # ---- encoded sklearn models ------------------------------------------
    # Drop collinear twins before fitting anything on raw features. With two
    # near-identical columns the coefficients are unidentifiable (the loss is
    # flat along their difference), which makes both predictions and per-row
    # explanations unstable. Keep the higher-IV member of each pair.
    order = iv_table["feature"].tolist()
    numeric_order = [f for f in order if types.get(f) == "numeric" and f in Xtr.columns]
    correlation = Xtr[numeric_order].corr().abs() if len(numeric_order) > 1 else None

    encoded_features: list[str] = []
    redundant: list[str] = []
    for feature in order:
        if feature not in Xtr.columns:
            continue
        if (
            correlation is not None
            and feature in numeric_order
            and any(
                abs(correlation.loc[feature, kept]) > 0.95
                for kept in encoded_features
                if kept in numeric_order
            )
        ):
            redundant.append(feature)
            continue
        encoded_features.append(feature)

    if redundant:
        warnings.append(
            f"{len(redundant)} collinear feature(s) dropped before modelling: {', '.join(redundant[:5])}"
            + (" ..." if len(redundant) > 5 else "")
        )

    encoder = FrameEncoder().fit(Xtr[encoded_features], types)
    Atr, Ate = encoder.transform(Xtr), encoder.transform(Xte)

    estimators = _make_estimators(cfg["split"].get("seed", 0))
    for name in models:
        if name in ("scorecard", "gbm_cal"):
            continue
        if name not in estimators:
            continue
        label, estimator, interpretable = estimators[name]
        started = time.time()
        estimator.fit(Atr, ytr)
        p_te = estimator.predict_proba(Ate)[:, 1]
        p_tr = estimator.predict_proba(Atr)[:, 1]

        fitted[name] = FittedModel(
            name=name, label=label, kind="sklearn", features=encoder.feature_names,
            types=types, interpretable=interpretable, encoder=encoder, estimator=estimator,
        )
        rows.append({"model": name, "label": label, "interpretable": interpretable,
                     **{f"test_{k}": v for k, v in _evaluate(yte, p_te).items()},
                     **{f"train_{k}": v for k, v in _evaluate(ytr, p_tr).items()},
                     "seconds": round(time.time() - started, 2)})
        curves[name] = roc_points(yte, p_te)

    # ---- calibrated GBM ---------------------------------------------------
    if "gbm_cal" in models and "gbm" in fitted:
        started = time.time()
        base = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_leaf_nodes=24,
            min_samples_leaf=40, l2_regularization=1.0, random_state=0,
        )
        calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
        calibrated.fit(Atr, ytr)
        p_te = calibrated.predict_proba(Ate)[:, 1]
        p_tr = calibrated.predict_proba(Atr)[:, 1]

        fitted["gbm_cal"] = FittedModel(
            name="gbm_cal", label="Gradient boosting (calibrated)", kind="sklearn",
            features=encoder.feature_names, types=types, interpretable=False,
            encoder=encoder, estimator=calibrated,
        )
        rows.append({"model": "gbm_cal", "label": "Gradient boosting (calibrated)",
                     "interpretable": False,
                     **{f"test_{k}": v for k, v in _evaluate(yte, p_te).items()},
                     **{f"train_{k}": v for k, v in _evaluate(ytr, p_tr).items()},
                     "seconds": round(time.time() - started, 2)})
        curves["gbm_cal"] = roc_points(yte, p_te)

    leaderboard = pd.DataFrame(rows).sort_values("test_auc", ascending=False).reset_index(drop=True)

    chosen, reason = choose_model(leaderboard, policy=cfg.get("selection", {}).get("policy", "balanced"))

    # ---- outputs ----------------------------------------------------------
    model = fitted[chosen]
    p_te = model.predict_pd(Xte)
    test_frame = pd.DataFrame({"y": yte, "pd": p_te}, index=Xte.index)
    score = model.score_points(Xte)
    if score is not None:
        test_frame["score"] = score

    importance = pd.DataFrame(columns=["feature", "importance", "std", "relative"])
    if include_importance:
        try:
            importance = permutation_importance(model.predict_pd, Xte, yte, n_repeats=5)
        except Exception as exc:  # never let an explanation break training
            warnings.append(f"Permutation importance failed: {exc}")

    calibration = pd.DataFrame(
        {
            "model": leaderboard["model"],
            "slope": leaderboard["test_calib_slope"],
            "brier": leaderboard["test_brier"],
        }
    )

    return TournamentResult(
        leaderboard=leaderboard,
        models=fitted,
        chosen=chosen,
        reason=reason,
        roc=curves,
        calibration=calibration,
        importance=importance,
        test_frame=test_frame,
        iv_table=iv_table,
        leakage=leak,
        selected_features=chosen_features,
        dropped_features=leak[leak["flagged"]][["feature", "reason"]].rename(columns={"reason": "leakage"}),
        train_ids=train_ids,
        test_ids=test_ids,
        warnings=warnings,
    )


def choose_model(leaderboard: pd.DataFrame, policy: str = "balanced") -> tuple[str, str]:
    """Pick a model and explain the pick in one sentence, with the numbers."""
    table = leaderboard.set_index("model")
    best = table["test_auc"].idxmax()
    best_auc = table.loc[best, "test_auc"]

    if policy == "regulatory":
        reference = "scorecard" if "scorecard" in table.index else best
        ref_auc = table.loc[reference, "test_auc"]
        lift = best_auc - ref_auc
        calibrated = 0.8 <= table.loc[best, "test_calib_slope"] <= 1.25
        if best != reference and lift >= 0.01 and calibrated:
            return best, (
                f"{best} selected over {reference}: +{lift:.3f} AUC "
                f"({best_auc:.3f} vs {ref_auc:.3f}) with an acceptable calibration slope "
                f"({table.loc[best, 'test_calib_slope']:.2f}). "
                "The gain exceeds the 0.01 threshold required to give up scorecard interpretability."
            )
        return reference, (
            f"{reference} retained: the best challenger ({best}, AUC {best_auc:.3f}) does not beat it "
            f"by the 0.01 margin required to justify losing interpretable reason codes "
            f"(scorecard AUC {ref_auc:.3f})."
        )

    if policy == "performance":
        return best, f"{best} has the highest out-of-time AUC ({best_auc:.3f}); interpretability not weighted."

    # balanced: the highest AUC among models whose probabilities can be trusted.
    # Falling back to "closest to slope 1" regardless of AUC would happily pick a
    # much worse ranker, which is the wrong trade for a cut-off decision.
    acceptable = table[(table["test_calib_slope"] >= 0.70) & (table["test_calib_slope"] <= 1.40)]

    if acceptable.empty:
        return best, (
            f"{best} has the best AUC ({best_auc:.3f}), but every candidate is miscalibrated "
            f"(slopes {table['test_calib_slope'].min():.2f} to {table['test_calib_slope'].max():.2f}). "
            "Treat the probabilities as ranking only, or recalibrate before using them for pricing "
            "or provisioning."
        )

    winner = acceptable["test_auc"].idxmax()
    winner_auc = float(table.loc[winner, "test_auc"])
    winner_slope = float(table.loc[winner, "test_calib_slope"])

    if winner != best:
        return winner, (
            f"{winner} selected over {best}: {best} ranks slightly better "
            f"(AUC {best_auc:.3f} vs {winner_auc:.3f}) but its probabilities are off "
            f"(slope {table.loc[best, 'test_calib_slope']:.2f}); {winner} is calibrated "
            f"(slope {winner_slope:.2f}) with AUC {winner_auc:.3f}."
        )

    return winner, (
        f"{winner} has the best out-of-time AUC ({winner_auc:.3f}) and acceptable calibration "
        f"(slope {winner_slope:.2f})."
    )
