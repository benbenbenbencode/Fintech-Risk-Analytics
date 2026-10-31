"""Fairness and protected-attribute auditing.

Credit decisions are regulated in most jurisdictions: you must be able to show
that decisions are not disproportionately adverse to a protected group, and
that protected attributes are not model inputs.

This module detects sensitive columns, blocks them from the feature set (that
happens in `auto_profile.discover_features`), and measures the outcome gap with
the four-fifths rule - a widely used screening heuristic, not a legal opinion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .auto_profile import PROTECTED_TOKENS

FOUR_FIFTHS = 0.8


def detect_protected(df: pd.DataFrame) -> dict[str, str]:
    """Map a sensitivity group -> column name, when such a column exists."""
    found: dict[str, str] = {}
    for column in df.columns:
        lowered = column.lower()
        for group, tokens in PROTECTED_TOKENS.items():
            if any(token in lowered for token in tokens):
                found.setdefault(group, column)
                break
    return found


def choose_cutoff(pd_values, approval_rate: float = 0.8) -> float:
    """PD threshold that approves roughly `approval_rate` of the population."""
    values = np.asarray(pd_values, dtype=float)
    approval_rate = float(np.clip(approval_rate, 0.05, 0.95))
    return float(np.quantile(values, approval_rate))


def _bands(series: pd.Series, groups: int = 4) -> pd.Series:
    """Turn a continuous attribute (typically age) into ordered bands."""
    try:
        return pd.qcut(series.rank(method="first"), groups, labels=False, duplicates="drop").map(
            lambda v: f"band_{int(v) + 1}" if pd.notna(v) else "unknown"
        )
    except (ValueError, TypeError):
        return series.astype(str)


def protected_frame(df: pd.DataFrame, protected: dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for group, column in protected.items():
        series = df[column]
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 10:
            out[group] = _bands(series)  # e.g. age -> age bands
        else:
            out[group] = series.astype("object").where(series.notna(), "unknown").astype(str)
    return out


def group_table(
    groups: pd.DataFrame,
    group_name: str,
    approved: pd.Series,
    pd_values: pd.Series,
    y: pd.Series | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "group": groups[group_name].to_numpy(),
            "approved": np.asarray(approved, dtype=int),
            "pd": np.asarray(pd_values, dtype=float),
        }
    )
    if y is not None:
        frame["bad"] = np.asarray(y)

    aggregation = {
        "applicants": ("approved", "size"),
        "approval_rate": ("approved", "mean"),
        "mean_pd": ("pd", "mean"),
    }
    if y is not None:
        aggregation["bad_rate"] = ("bad", "mean")

    table = frame.groupby("group").agg(**aggregation).reset_index()
    table = table.sort_values("applicants", ascending=False).reset_index(drop=True)
    return table


def disparate_impact(table: pd.DataFrame) -> float:
    """Lowest / highest approval rate. Below 0.8 triggers the four-fifths rule."""
    if table.empty or table["approval_rate"].max() <= 0:
        return float("nan")
    return float(table["approval_rate"].min() / table["approval_rate"].max())


def assess(
    df: pd.DataFrame,
    pd_values: pd.Series,
    approval_rate: float = 0.8,
    y: pd.Series | None = None,
    protected: dict[str, str] | None = None,
) -> dict:
    """Full fairness report across every detected protected attribute."""
    protected = protected if protected is not None else detect_protected(df)
    if not protected:
        return {"protected": {}, "results": {}, "flags": [], "cutoff": None}

    cutoff = choose_cutoff(pd_values, approval_rate)
    approved = pd.Series(np.asarray(pd_values) <= cutoff, index=df.index)
    groups = protected_frame(df, protected)

    results: dict[str, pd.DataFrame] = {}
    flags: list[str] = []

    for group in protected:
        table = group_table(groups, group, approved, pd_values, y)
        ratio = disparate_impact(table)
        table["di_ratio"] = ratio
        results[group] = table

        if np.isfinite(ratio) and ratio < FOUR_FIFTHS:
            worst = table.sort_values("approval_rate").iloc[0]["group"]
            best = table.sort_values("approval_rate").iloc[-1]["group"]
            flags.append(
                f"{group}: approval rate for '{worst}' is {table['approval_rate'].min():.0%} vs "
                f"{table['approval_rate'].max():.0%} for '{best}' "
                f"(disparate impact ratio {ratio:.2f} < {FOUR_FIFTHS})."
            )

    return {
        "protected": protected,
        "results": results,
        "flags": flags,
        "cutoff": cutoff,
        "approval_rate": float(approved.mean()),
    }
