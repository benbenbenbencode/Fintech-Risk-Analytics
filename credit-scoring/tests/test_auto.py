"""Tests for the automatic (schema-agnostic) credit scoring layer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crf.auto_model import FrameEncoder, choose_model, run_tournament
from crf.auto_profile import classify_column, discover_features, profile_tables
from crf.config import DEFAULTS, _merge
from crf.explain import contributions, explain_applicant, risk_bands, safe_values
from crf.fairness import assess, choose_cutoff, disparate_impact, detect_protected


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def make_population(n: int = 900, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.lognormal(mean=10.0, sigma=0.45, size=n)
    dti = rng.uniform(0.05, 0.85, size=n)
    tenure = rng.integers(0, 120, size=n)
    region = rng.choice(["north", "south", "east"], size=n, p=[0.5, 0.3, 0.2])
    channel = rng.choice(["branch", "online"], size=n)
    age = rng.integers(21, 66, size=n)
    sex = rng.choice(["F", "M"], size=n)

    risk = -3.2 + 3.4 * dti - 0.9 * np.log(income / 20_000) - 0.006 * tenure + 0.45 * (region == "south")
    probability = 1 / (1 + np.exp(-risk))
    bad = rng.binomial(1, probability)

    dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 660, size=n), unit="D")

    return pd.DataFrame(
        {
            "loan_id": [f"L{i:05d}" for i in range(n)],
            "disbursement_date": dates,
            "monthly_income": income.round(2),
            "dti": dti.round(4),
            "tenure_months": tenure,
            "region": region,
            "channel": channel,
            "age": age,
            "sex": sex,
            "collection_status": rng.choice(["ok", "late"], size=n),  # deliberately leaky name
            "bad": bad,
        }
    )


@pytest.fixture(scope="module")
def population() -> pd.DataFrame:
    return make_population()


@pytest.fixture(scope="module")
def fitted(population):
    frame = population.set_index("loan_id")
    drop = ["bad", "disbursement_date", "age", "sex", "collection_status"]
    X = frame.drop(columns=[c for c in drop if c in frame.columns])
    types = {"monthly_income": "numeric", "dti": "numeric", "tenure_months": "numeric",
             "region": "categorical", "channel": "categorical"}
    y = frame["bad"].astype(int)
    cfg = _merge(DEFAULTS, {})
    result = run_tournament(X, y, types, frame["disbursement_date"], cfg,
                            models=("scorecard", "logistic", "gbm"))
    return result, X, y, types, cfg


# ---------------------------------------------------------------------------
# profiling
# ---------------------------------------------------------------------------


def test_classify_identifier_constant_and_types(population):
    rows = len(population)

    identifier = classify_column("loan_id", population["loan_id"], "base", rows)
    assert identifier.kind == "identifier" and not identifier.usable

    constant = classify_column("flag", pd.Series(["x"] * rows), "base", rows)
    assert constant.kind == "constant" and not constant.usable

    numeric = classify_column("monthly_income", population["monthly_income"], "base", rows)
    assert numeric.kind == "numeric" and numeric.usable

    categorical = classify_column("region", population["region"], "base", rows)
    assert categorical.kind == "categorical" and categorical.usable

    date = classify_column("disbursement_date", population["disbursement_date"], "base", rows)
    assert date.kind == "datetime"

    free_text = classify_column("notes", pd.Series([f"text {i}" for i in range(rows)]), "base", rows)
    assert free_text.kind == "in a" or free_text.kind in ("identifier", "text")


def test_discovery_drops_leakage_protected_and_builds_ratio(population):
    frame = population.set_index("loan_id")
    features = discover_features(
        frame.reset_index(),
        related={},
        profiles=profile_tables({"base": frame.reset_index()}),
        id_column="loan_id",
        date_column="disbursement_date",
        as_of=pd.Timestamp("2026-01-01"),
    )

    assert "collection_status" not in features.X.columns, "leaky name must be dropped"
    assert "sex" not in features.X.columns, "protected attribute must not be a feature"
    assert "age" not in features.X.columns, "age is treated as protected by default"
    assert "bad" not in features.X.columns, "target must never become a feature"
    assert "monthly_income" in features.X.columns

    reasons = " ".join(features.dropped["reason"].fillna("").tolist()).lower()
    assert "banned pattern" in reasons
    assert "protected" in reasons

    # A days-since feature should exist for the disbursement date.
    assert any(c.startswith("days_since_") for c in features.X.columns)


def test_encoder_handles_unseen_categories():
    encoder = FrameEncoder().fit(
        pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]}),
        {"a": "numeric", "b": "categorical"},
    )
    out = encoder.transform(pd.DataFrame({"a": [3.0], "b": ["never-seen"]}))
    assert out["a"].iloc[0] == 3.0
    assert np.isnan(out["b"].iloc[0]), "unseen category should encode as missing, not crash"


# ---------------------------------------------------------------------------
# tournament
# ---------------------------------------------------------------------------


def test_tournament_runs_and_selects(fitted):
    result, X, y, _, _ = fitted

    assert not result.leaderboard.empty
    assert set(result.leaderboard["model"]) >= {"scorecard", "logistic", "gbm"}
    assert result.chosen in set(result.leaderboard["model"])
    assert result.reason

    for column in ("test_auc", "test_ks", "test_brier", "test_calib_slope"):
        assert column in result.leaderboard.columns

    best = result.leaderboard["test_auc"].max()
    assert 0.5 <= best <= 1.0
    assert result.selected_features

    # The chosen model must actually score the test rows.
    p = result.chosen_model().predict_pd(X.loc[result.test_ids])
    assert len(p) == len(result.test_ids)
    assert ((p >= 0) & (p <= 1)).all()


def test_selection_policy_prefers_scorecard_in_regulatory_mode(fitted):
    result, _, _, _, _ = fitted
    chosen, reason = choose_model(result.leaderboard, policy="regulatory")
    assert chosen in result.leaderboard["model"].values
    assert "AUC" in reason or "calibrat" in reason


# ---------------------------------------------------------------------------
# explanation
# ---------------------------------------------------------------------------


def test_explanation_produces_reasons_and_text(fitted):
    result, X, _, _, _ = fitted
    model = result.chosen_model()
    train = X.loc[result.train_ids]

    train_pd = model.predict_pd(train)
    safe = safe_values(model, train)
    cuts, _ = risk_bands(train_pd)

    row_id = result.test_ids[0]
    explanation = explain_applicant(model, X.loc[row_id], safe, train_pd, cuts, row_id=row_id)

    assert explanation.row_id == str(row_id)
    assert 0 <= explanation.pd <= 1
    assert explanation.band in ("A", "B", "C", "D", "E")
    assert not explanation.contributions.empty
    assert explanation.text
    assert "band" in explanation.text.lower() or "skor" in explanation.text.lower()

    # Contributions must be finite and at least one factor must move risk.
    assert np.isfinite(explanation.contributions["delta_pd"]).all()
    assert explanation.reasons or explanation.offsets


def test_contributions_are_exact_and_bounded(fitted):
    result, X, _, _, _ = fitted
    model = result.chosen_model()
    train = X.loc[result.train_ids]

    train_pd = model.predict_pd(train)
    safe = safe_values(model, train)
    cuts, _ = risk_bands(train_pd)

    explanation = explain_applicant(model, X.loc[result.test_ids[0]], safe, train_pd, cuts)
    table = explanation.contributions

    assert not table.empty
    assert explanation.method.startswith("exact"), explanation.method
    assert np.isfinite(table["delta_pd"]).all()
    # A single factor must never claim more than the whole probability space.
    assert (table["delta_pd"].abs() <= 1.0).all()
    assert {"feature", "label", "delta_pp"}.issubset(table.columns)


def test_nonlinear_model_falls_back_to_approximate(fitted):
    result, X, _, _, _ = fitted
    if "gbm" not in result.models:
        pytest.skip("no gradient-boosting candidate in this fixture")

    model = result.models["gbm"]
    train = X.loc[result.train_ids]
    safe = safe_values(model, train)

    table = contributions(model, X.loc[result.test_ids[0]], safe)
    assert not table.empty
    assert table["method"].iloc[0].startswith("approximate")
    assert np.isfinite(table["delta_pd"]).all()


# ---------------------------------------------------------------------------
# fairness
# ---------------------------------------------------------------------------


def test_protected_detection_and_disparate_impact():
    frame = pd.DataFrame(
        {
            "gender": ["F"] * 150 + ["M"] * 150,
            "pd": [0.02] * 150 + [0.30] * 150,
        }
    )
    protected = detect_protected(frame)
    assert protected.get("sex") == "gender"

    report = assess(frame, frame["pd"], approval_rate=0.5)
    table = report["results"]["sex"]
    ratio = disparate_impact(table)

    assert ratio < 0.8, "a skewed group should fail the four-fifths rule"
    assert report["flags"], "the audit must raise a flag"
    assert "four-fifths" in " ".join(report["flags"]).lower() or "disparate impact" in " ".join(report["flags"]).lower()


def test_cutoff_tracks_requested_approval_rate():
    values = pd.Series(np.linspace(0, 1, 1000))
    for rate in (0.1, 0.5, 0.9):
        cutoff = choose_cutoff(values, rate)
        approved = (values <= cutoff).mean()
        assert abs(approved - rate) < 0.02
