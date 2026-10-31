"""Per-applicant explanations.

Three strategies, in order of exactness:

1. **Scorecard** - points are additive by construction, so an applicant's score
   is exactly the sum of per-feature points. The contribution of a feature is
   simply the points it contributes, expressed as a probability impact.

2. **Linear model** (logistic on encoded features) - the logit is exactly
   b + sum(w_i * z_i), so each feature's contribution is w_i * z_i. Stable,
   additive, no counterfactual extrapolation.

3. **Anything else** (trees, ensembles) - single-feature counterfactual: replace
   the value with its safest observed value and measure the PD change. This is
   approximate and can be unstable when features are correlated, so the result is
   labelled with its method.

The first two are exact decompositions. That matters: with correlated features a
counterfactual can move the prediction wildly (a 3% PD jumping to 67%), which
would make the reason shown to a customer nonsense.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

BANDS = ("A", "B", "C", "D", "E")

PRETTY = {
    "pd": "probability of default",
    "aov": "average order value",
    "dti": "debt-to-income",
    "lti": "loan-to-income",
    "dpd": "days past due",
    "apr": "interest rate",
    "nunique": "distinct count",
}

TEXT = {
    "en": {
        "headline": "Score {score} / 850 - band {band}",
        "no_score": "Probability of default {pd:.1%} - band {band}",
        "position": "This applicant sits at the {pct:.0f}th percentile of risk (higher = riskier).",
        "drivers": "Factors increasing the risk:",
        "offset": "Factors holding the risk down:",
        "none": "No single factor dominates; the decision rests on the overall pattern.",
        "compare": "Compared with the scored population, the model puts this applicant {relation} the median.",
        "higher": "above",
        "lower": "below",
    },
    "id": {
        "headline": "Skor {score} / 850 - band {band}",
        "no_score": "Probabilitas gagal bayar {pd:.1%} - band {band}",
        "position": "Pemohon ini berada di persentil {pct:.0f} (makin tinggi makin berisiko).",
        "drivers": "Faktor yang menaikkan risiko:",
        "offset": "Faktor yang menahan risiko:",
        "none": "Tidak ada satu faktor yang dominan; keputusan bergantung pada pola keseluruhan.",
        "compare": "Dibanding populasi yang dinilai, posisi pemohon ini {relation} median.",
        "higher": "di atas",
        "lower": "di bawah",
    },
}


def humanize(name: str) -> str:
    text = name
    if text.startswith("ratio_"):
        text = text[len("ratio_"):].replace("_to_", " / ")
    for key, label in PRETTY.items():
        text = text.replace(f"_{key}_", f" {label} ").replace(f"_{key}", f" {label}")
    return text.replace("_", " ").strip().capitalize()


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


# ---------------------------------------------------------------------------
# what counts as a "safe" value (used by the approximate path and the UI)
# ---------------------------------------------------------------------------


def safe_values(model, X_train: pd.DataFrame, q: int = 8, min_count: int = 30) -> dict:
    """For each feature, the observed value in the lowest-risk bucket."""
    base_pd = model.predict_pd(X_train)
    safe: dict = {}

    for feature in model.features:
        if feature not in X_train.columns:
            continue
        series = X_train[feature]

        if model.types.get(feature) == "numeric":
            values = pd.to_numeric(series, errors="coerce")
            try:
                buckets = pd.qcut(values, q, duplicates="drop")
            except (ValueError, TypeError):
                continue
            grouped = (
                pd.DataFrame({"bucket": buckets, "pd": base_pd})
                .groupby("bucket", observed=True)["pd"]
                .agg(["mean", "size"])
            )
            grouped = grouped[grouped["size"] >= min_count]
            if grouped.empty:
                continue
            best = grouped["mean"].idxmin()
            subset = values[buckets == best].dropna()
            if not subset.empty:
                safe[feature] = float(subset.median())
        else:
            text = series.astype("object").where(series.notna(), "__NA__").astype(str)
            grouped = (
                pd.DataFrame({"value": text, "pd": base_pd})
                .groupby("value")["pd"]
                .agg(["mean", "size"])
            )
            grouped = grouped[grouped["size"] >= min_count]
            if grouped.empty:
                continue
            best = grouped["mean"].idxmin()
            if best == "__NA__":
                safe[feature] = np.nan
            else:
                # Preserve the column dtype so a code column of ints stays 12,
                # not "12" or 12.0 (that would hit a different bin at scoring).
                try:
                    safe[feature] = pd.Series([best]).astype(series.dtype).iloc[0]
                except (ValueError, TypeError):
                    safe[feature] = best

    return safe


def risk_bands(train_pd: np.ndarray) -> tuple[np.ndarray, str]:
    cuts = np.quantile(np.asarray(train_pd), [0.2, 0.4, 0.6, 0.8])
    return cuts, "Bands are population quintiles of model PD (A = safest)."


def band_of(pd_value: float, cuts: np.ndarray) -> str:
    index = int(np.searchsorted(cuts, pd_value))
    return BANDS[min(index, len(BANDS) - 1)]


# ---------------------------------------------------------------------------
# exact decompositions
# ---------------------------------------------------------------------------


def _linear_pipeline(model) -> bool:
    estimator = model.estimator
    if model.kind != "sklearn" or not hasattr(estimator, "named_steps"):
        return False
    classifier = estimator.named_steps.get("clf")
    return classifier is not None and hasattr(classifier, "coef_")


def _scorecard_contributions(model, frame: pd.DataFrame) -> pd.DataFrame:
    """Exact, and centred on the average applicant.

    Points are additive, but subtracting a feature's points entirely also removes
    its share of the intercept, which makes every factor look protective. So the
    comparison is against the population-average points for that feature:
    positive = worse than a typical applicant, negative = better.
    """
    card = model.card
    points = card.feature_points(frame)
    score = float(points.sum(axis=1).iloc[0])

    def pd_from(total_points: float) -> float:
        logit_good = (total_points - card.offset) / card.factor
        return _sigmoid(-logit_good)

    base_pd = pd_from(score)

    # Population-average points per feature, weighted by bin frequency.
    table = card.points_table
    mean_points = {
        feature: float((group["pct"] * group["points"]).sum())
        for feature, group in table.groupby("feature")
    }

    rows = []
    for feature in points.columns:
        deviation = float(points[feature].iloc[0]) - mean_points.get(feature, 0.0)
        without = pd_from(score - deviation)
        rows.append(
            {
                "feature": feature,
                "label": humanize(feature),
                "value": frame[feature].iloc[0],
                "safe_value": np.nan,
                "deviation_points": deviation,
                "delta_pd": base_pd - without,
            }
        )
    return pd.DataFrame(rows)


def _linear_contributions(model, frame: pd.DataFrame) -> pd.DataFrame:
    """Exact: logit = b + sum(w_i * z_i)."""
    estimator = model.estimator
    encoded = model.encoder.transform(frame.reindex(columns=model.features))

    transformed = encoded
    for _, step in estimator.steps[:-1]:
        transformed = step.transform(transformed)

    classifier = estimator.named_steps["clf"]
    coefficients = np.asarray(classifier.coef_)[0]
    intercept = float(np.asarray(classifier.intercept_)[0])

    matrix = pd.DataFrame(np.asarray(transformed), columns=model.features, index=frame.index)
    logit_terms = matrix.mul(coefficients, axis=1)
    logit_total = float(logit_terms.iloc[0].sum() + intercept)
    base_pd = _sigmoid(logit_total)

    rows = []
    for feature in logit_terms.columns:
        term = float(logit_terms[feature].iloc[0])
        without = _sigmoid(logit_total - term)
        rows.append(
            {
                "feature": feature,
                "label": humanize(feature),
                "value": frame[feature].iloc[0],
                "safe_value": np.nan,
                "logit": term,
                "delta_pd": base_pd - without,
            }
        )
    return pd.DataFrame(rows)


def _counterfactual_contributions(model, row: pd.Series, safe: dict) -> pd.DataFrame:
    """Approximate: replace one feature at a time with its safest value."""
    features = [f for f in model.features if f in row.index and f in safe]
    if not features:
        return pd.DataFrame()

    template = row.reindex(model.features).astype(object)
    frame = pd.DataFrame([template] * len(features), index=features).astype(object)
    for feature in features:
        frame.loc[feature, feature] = safe[feature]

    baseline = float(model.predict_pd(row.to_frame().T.reindex(columns=model.features))[0])
    alternative = model.predict_pd(frame)

    rows = [
        {
            "feature": feature,
            "label": humanize(feature),
            "value": row.get(feature),
            "safe_value": safe[feature],
            "delta_pd": baseline - alternative[i],
        }
        for i, feature in enumerate(features)
    ]
    return pd.DataFrame(rows)


def contributions(model, row: pd.Series, safe: dict | None = None) -> pd.DataFrame:
    """Per-feature contribution, using the most exact method the model allows."""
    frame = row.to_frame().T.reindex(columns=model.features)

    if model.kind == "scorecard":
        table = _scorecard_contributions(model, frame)
        table["method"] = "exact (scorecard points)"
    elif _linear_pipeline(model):
        table = _linear_contributions(model, frame)
        table["method"] = "exact (logit decomposition)"
    else:
        table = _counterfactual_contributions(model, row, safe or {})
        table["method"] = "approximate (counterfactual)"

    if table.empty:
        return table

    table["delta_pp"] = table["delta_pd"] * 100
    table["abs_delta"] = table["delta_pd"].abs()
    return table.sort_values("delta_pd", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# single applicant
# ---------------------------------------------------------------------------


@dataclass
class Explanation:
    row_id: str
    pd: float
    score: float | None
    band: str
    percentile: float
    contributions: pd.DataFrame
    reasons: list[dict] = field(default_factory=list)
    offsets: list[dict] = field(default_factory=list)
    text: str = ""

    @property
    def method(self) -> str:
        if self.contributions.empty or "method" not in self.contributions:
            return "unknown"
        return str(self.contributions["method"].iloc[0])

    def to_dict(self) -> dict:
        return {
            "row_id": self.row_id,
            "pd": self.pd,
            "score": self.score,
            "band": self.band,
            "percentile": self.percentile,
            "contributions": self.contributions.to_dict(orient="records"),
            "reasons": self.reasons,
            "offsets": self.offsets,
            "method": self.method,
            "text": self.text,
        }


def explain_applicant(
    model,
    row: pd.Series,
    safe: dict,
    train_pd: np.ndarray,
    cuts: np.ndarray,
    *,
    row_id: str | None = None,
    top_k: int = 4,
    language: str = "en",
    min_effect_pp: float = 0.05,
) -> Explanation:
    table = contributions(model, row, safe)
    frame = row.to_frame().T.reindex(columns=model.features)
    baseline = float(model.predict_pd(frame)[0])
    score_values = model.score_points(frame)
    score = float(score_values[0]) if score_values is not None else None

    percentile = float((np.asarray(train_pd) < baseline).mean()) * 100
    band = band_of(baseline, cuts)

    positive = table[table["delta_pp"] >= min_effect_pp].head(top_k)
    negative = table[table["delta_pp"] <= -min_effect_pp].sort_values("delta_pd").head(top_k)

    reasons = [
        {"feature": r.feature, "label": r.label, "value": r.value,
         "delta_pp": round(float(r.delta_pp), 2), "direction": "increases risk"}
        for r in positive.itertuples()
    ]
    offsets = [
        {"feature": r.feature, "label": r.label, "value": r.value,
         "delta_pp": round(float(r.delta_pp), 2), "direction": "reduces risk"}
        for r in negative.itertuples()
    ]

    text = render_text(baseline, score, band, percentile, reasons, offsets, language=language)

    return Explanation(
        row_id=str(row_id if row_id is not None else row.name),
        pd=baseline,
        score=score,
        band=band,
        percentile=percentile,
        contributions=table,
        reasons=reasons,
        offsets=offsets,
        text=text,
    )


def render_text(pd_value, score, band, percentile, reasons, offsets, language: str = "en") -> str:
    words = TEXT.get(language, TEXT["en"])
    lines: list[str] = []

    if score is not None:
        lines.append(words["headline"].format(score=int(round(score)), band=band))
    else:
        lines.append(words["no_score"].format(pd=pd_value, band=band))

    lines.append(words["position"].format(pct=percentile))
    lines.append(words["compare"].format(relation=words["higher"] if percentile >= 50 else words["lower"]))

    if reasons:
        lines.append(words["drivers"])
        for item in reasons:
            lines.append(f"  - {item['label']}: +{item['delta_pp']:.1f} pp")

    if offsets:
        lines.append(words["offset"])
        for item in offsets:
            lines.append(f"  - {item['label']}: {item['delta_pp']:.1f} pp")

    if not reasons and not offsets:
        lines.append(words["none"])

    return "\n".join(lines)
