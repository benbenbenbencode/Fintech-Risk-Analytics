"""One call from raw tables to a trained, explained, audited model.

    result = run_auto(config_path)          # or run_auto(cfg_dict, tables=...)
    result.tournament.leaderboard           # what was tried, what won, why
    result.explanations[0].text             # plain-language applicant brief
    result.fairness["flags"]                # adverse-impact warnings

The point is that nothing here requires a hand-written feature list: the schema
is profiled, child tables are aggregated point-in-time, and the model is chosen
by a declared policy rather than a hardcoded default.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .auto_model import FittedModel, TournamentResult, run_tournament
from .auto_profile import FeatureSet, discover_features, profile_tables, relationship_map
from .config import load_config
from .explain import Explanation, explain_applicant, risk_bands, safe_values
from .fairness import assess, detect_protected
from .loader import build_base, load_tables
from .target import build_target

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MIN_ELIGIBLE = 200
TARGET_COLUMN = "bad"


@dataclass
class AutoResult:
    cfg: dict
    profile: dict
    relationships: list
    featureset: FeatureSet
    tournament: TournamentResult
    target_summary: dict
    waterfall: pd.DataFrame
    deciles: pd.DataFrame
    explanations: list[Explanation]
    fairness: dict
    warnings: list[str] = field(default_factory=list)
    artifact_path: str | None = None
    seconds: float = 0.0

    # kept so the UI can explain any applicant on demand
    X: pd.DataFrame | None = None
    base: pd.DataFrame | None = None
    safe: dict = field(default_factory=dict)
    cuts: np.ndarray | None = None
    train_pd: np.ndarray | None = None

    # -- convenience for the UI -------------------------------------------

    @property
    def model(self) -> FittedModel:
        return self.tournament.chosen_model()

    def explain(self, row_id, language: str = "en") -> Explanation | None:
        if self.X is None or row_id not in self.X.index:
            return None
        row = self.X.loc[row_id]
        return explain_applicant(
            self.model, row, self.safe, self.train_pd, self.cuts,
            row_id=row_id, language=language,
        )

    def applicant_frame(self, ids) -> pd.DataFrame:
        """Raw (un-aggregated) applicant view for the UI, without model inputs."""
        if self.base is None:
            return pd.DataFrame()
        frame = self.base.set_index("loan_id")
        return frame.reindex(ids)

    def to_summary(self) -> dict:
        leaderboard = self.tournament.leaderboard
        best = leaderboard.iloc[0]
        chosen = leaderboard[leaderboard["model"] == self.tournament.chosen].iloc[0]
        return {
            "chosen": self.tournament.chosen,
            "chosen_label": chosen["label"],
            "reason": self.tournament.reason,
            "auc": float(chosen["test_auc"]),
            "ks": float(chosen["test_ks"]),
            "best_model": str(best["model"]),
            "best_auc": float(best["test_auc"]),
            "n_features": len(self.tournament.selected_features),
            "n_candidates": int(len(self.tournament.iv_table)),
            "bad_rate": self.target_summary.get("bad_rate"),
            "n_model": self.target_summary.get("n_model"),
            "approval_rate": self.fairness.get("approval_rate"),
            "fairness_flags": len(self.fairness.get("flags", [])),
        }


def _cutoff_series(base: pd.DataFrame, cfg: dict) -> pd.Series | None:
    """Per-loan date after which repayment rows must not be used."""
    if cfg["scorecard_type"] != "behavioral":
        return None
    obs = int(cfg["target"]["observation_months"])
    if obs <= 0:
        return None
    disb = base.set_index("loan_id")["disbursement_date"]
    return (disb + pd.DateOffset(months=obs)).to_dict()


def run_auto(
    cfg_source=None,
    tables: dict | None = None,
    *,
    explain_n: int = 5,
    approval_rate: float = 0.8,
    save_artifact: bool = True,
    models: tuple[str, ...] = ("scorecard", "logistic", "gbm", "gbm_cal", "rf"),
    language: str = "en",
) -> AutoResult:
    started = time.time()
    t0 = time.time()
    cfg = load_config(cfg_source or {"as_of_date": "2025-12-31",
                                     "data": {"loans": {}, "repayments": {}}})
    tables = tables or load_tables(cfg)
    if not tables:
        raise ValueError("No tables loaded - check data paths in the config.")

    base, repayments = build_base(tables)
    target_frame, waterfall = build_target(base, repayments, cfg)

    eligible = target_frame.index[target_frame["eligible"]]
    if len(eligible) < MIN_ELIGIBLE:
        raise ValueError(
            f"Only {len(eligible)} eligible loans after maturity filters; need at least "
            f"{MIN_ELIGIBLE}. Extend the history, shorten the performance window, "
            "or lower target.bad_dpd."
        )

    # ---- automatic schema + features -------------------------------------
    related = {}
    if cfg["scorecard_type"] == "behavioral" and "repayments" in tables:
        related["repayments"] = repayments
    for name, frame in tables.items():
        # customers and bureau are already merged into the base row by
        # build_base(); aggregating them again only produces empty columns.
        if name in ("loans", "repayments", "customers", "bureau") or frame is None or frame.empty:
            continue
        related[name] = frame

    profiles = profile_tables({"base": base, **related})
    relationships = relationship_map(profiles)

    featureset = discover_features(
        base,
        related,
        profiles,
        id_column="loan_id",
        date_column="disbursement_date",
        cutoff_by_loan=_cutoff_series(base, cfg),
        as_of=pd.Timestamp(cfg["as_of_date"]),
        banned_patterns=tuple(cfg["leakage"]["banned_patterns"]) + (
            "default_flag", "bad_flag", "is_bad", "target",
        ),
        # The target column itself must never become a feature. A column called
        # simply "bad" does not match any banned pattern, so name it explicitly.
        exclude=(TARGET_COLUMN,),
    )

    X = featureset.X.loc[eligible]
    y = target_frame.loc[eligible, "bad"].astype(int)
    dates = base.set_index("loan_id").loc[eligible, "disbursement_date"]

    if X.empty or not featureset.types:
        raise ValueError(
            "No usable features were discovered. Check that the base table has numeric "
            "or low-cardinality categorical columns and that they are not all identifiers."
        )

    # ---- tournament -------------------------------------------------------
    t1 = time.time()
    tournament = run_tournament(X, y, featureset.types, dates, cfg, models=models)
    train_ids, test_ids = tournament.train_ids, tournament.test_ids

    model = tournament.chosen_model()
    Xtr = X.loc[train_ids]
    train_pd = model.predict_pd(Xtr)
    safe = safe_values(model, Xtr)
    cuts, band_note = risk_bands(train_pd)

    # ---- explanations -----------------------------------------------------
    sample_ids = list(test_ids[: max(1, explain_n)])
    explanations = []
    for row_id in sample_ids:
        try:
            explanations.append(
                explain_applicant(model, X.loc[row_id], safe, train_pd, cuts,
                                  row_id=row_id, language=language)
            )
        except Exception as exc:  # an explanation must never break training
            explanations.append(None)
            tournament.warnings.append(f"explanation failed for {row_id}: {exc}")

    # ---- fairness ---------------------------------------------------------
    test_raw = base.set_index("loan_id").reindex(test_ids)
    test_pd = pd.Series(model.predict_pd(X.loc[test_ids]), index=test_ids)
    try:
        fairness = assess(test_raw, test_pd, approval_rate=approval_rate,
                          y=y.loc[test_ids], protected=detect_protected(test_raw))
    except Exception as exc:
        fairness = {"protected": {}, "results": {}, "flags": [], "cutoff": None,
                    "error": str(exc)}

    deciles = pd.DataFrame(
        {"y": y.loc[test_ids].to_numpy(), "pd": test_pd.to_numpy()}
    ).assign(score=lambda d: (1 - d["pd"]) * 1000)
    from .evaluate import decile_table  # local import keeps the module light

    deciles = decile_table(deciles["y"], deciles["pd"], deciles["score"])

    target_summary = {
        "bad_rate": float(y.mean()),
        "n_model": int(len(y)),
        "n_train": int(len(train_ids)),
        "n_test": int(len(test_ids)),
        "bad_dpd": cfg["target"]["bad_dpd"],
        "obs": cfg["target"]["observation_months"],
        "perf": cfg["target"]["performance_months"],
        "type": cfg["scorecard_type"],
        "band_note": band_note,
    }

    warnings = list(tournament.warnings)
    if not featureset.types:
        warnings.append("No features survived discovery.")

    result = AutoResult(
        cfg=cfg,
        profile=profiles,
        relationships=relationships,
        featureset=featureset,
        tournament=tournament,
        target_summary=target_summary,
        waterfall=waterfall,
        deciles=deciles,
        explanations=explanations,
        fairness=fairness,
        warnings=warnings,
        X=X,
        base=base,
        safe=safe,
        cuts=cuts,
        train_pd=train_pd,
        seconds=round(time.time() - started, 1),
    )

    if save_artifact:
        try:
            ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
            path = ARTIFACT_DIR / f"auto_{pd.Timestamp.now():%Y%m%d-%H%M%S}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "features": tournament.selected_features,
                    "types": featureset.types,
                    "provenance": featureset.provenance,
                    "safe": safe,
                    "cuts": cuts,
                    "train_pd": train_pd,
                    "leaderboard": tournament.leaderboard,
                    "cfg": {k: v for k, v in cfg.items() if not k.startswith("_")},
                    "target_summary": target_summary,
                },
                path,
            )
            result.artifact_path = str(path)
        except Exception as exc:
            result.warnings.append(f"Could not save artifact: {exc}")

    return result
