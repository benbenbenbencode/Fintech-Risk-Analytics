"""Automatic schema profiling and feature discovery.

The v1 framework required you to list features by hand in `config.yaml`. This
module removes that: point it at a dict of tables and it works out

  * what each column actually is (identifier / date / numeric / categorical / text)
  * which columns are unsafe or useless (constant, >90% missing, leaked outcome,
    protected attribute, free text)
  * how the tables relate (key columns appearing in more than one table)
  * which derived features are worth building (date recency, cross-table
    aggregates, amount ratios)

Everything it produces is traceable: every feature carries a provenance string,
and every rejected column carries a reason, both of which the UI shows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ID_TOKENS = ("_id", "id_", "uuid", "guid", "key", "code")
DATE_TOKENS = ("date", "dt", "_at", "time", "month", "year", "created", "updated")

# Columns that leak the outcome or are decided after default.
DEFAULT_BANNED = (
    "collection", "legal", "recovery", "writeoff", "charge_off",
    "default_flag", "bad_flag", "is_bad", "target",
)

# Sensitive attributes: detected so they can be excluded and audited, never used
# as model inputs by default.
PROTECTED_TOKENS: dict[str, tuple[str, ...]] = {
    "sex": ("sex", "gender", "jenis_kelamin"),
    "age": ("age", "umur", "birth", "dob", "tanggal_lahir"),
    "marital_status": ("marital", "perkawinan", "nikah"),
    "religion": ("religion", "agama"),
    "ethnicity": ("ethnic", "race", "suku", "ras"),
    "nationality": ("nationality", "citizenship", "kewarganegaraan"),
    "disability": ("disab", "difabel", "cacat"),
}

AMOUNT_TOKENS = ("amount", "income", "principal", "salary", "outstanding", "balance",
                 "limit", "price", "cost", "value", "revenue", "payment")
DENOMINATOR_TOKENS = ("income", "salary", "principal", "limit", "revenue")


@dataclass
class ColumnProfile:
    table: str
    column: str
    kind: str
    n_unique: int
    missing_pct: float
    detail: str = ""
    usable: bool = True
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "table": self.table,
            "column": self.column,
            "kind": self.kind,
            "n_unique": self.n_unique,
            "missing_pct": round(self.missing_pct * 100, 2),
            "usable": self.usable,
            "reason": self.reason or self.detail,
        }


@dataclass
class TableProfile:
    name: str
    rows: int
    columns: list[ColumnProfile] = field(default_factory=list)
    key_columns: list[str] = field(default_factory=list)
    date_columns: list[str] = field(default_factory=list)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.to_dict() for c in self.columns])


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------


def _looks_like_id(name: str, series: pd.Series, rows: int) -> bool:
    lowered = name.lower()
    if any(lowered.endswith(t) or lowered.startswith(t) for t in ID_TOKENS):
        return True
    if rows and series.nunique(dropna=True) >= rows * 0.98 and series.dtype != float:
        # Near-unique and not a float measure: almost certainly an identifier.
        return True
    return False


def _looks_protected(name: str) -> str | None:
    lowered = name.lower()
    for group, tokens in PROTECTED_TOKENS.items():
        if any(token in lowered for token in tokens):
            return group
    return None


def classify_column(name: str, series: pd.Series, table: str, rows: int) -> ColumnProfile:
    missing = float(series.isna().mean())
    n_unique = int(series.nunique(dropna=True))

    base = dict(table=table, column=name, n_unique=n_unique, missing_pct=missing)

    if missing >= 0.999:
        return ColumnProfile(kind="empty", usable=False, reason="no values", **base)

    if n_unique <= 1:
        return ColumnProfile(kind="constant", usable=False, reason="only one distinct value", **base)

    if _looks_like_id(name, series, rows):
        return ColumnProfile(kind="identifier", usable=False, reason="identifier, not a predictor", **base)

    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnProfile(kind="datetime", **base)

    if pd.api.types.is_bool_dtype(series):
        return ColumnProfile(kind="categorical", detail="boolean", **base)

    if pd.api.types.is_numeric_dtype(series):
        if n_unique <= 12 and float((series.dropna() % 1 == 0).mean()) > 0.99:
            return ColumnProfile(kind="categorical", detail="low-cardinality integer code", **base)
        return ColumnProfile(kind="numeric", **base)

    # object / string
    numeric_share = pd.to_numeric(series, errors="coerce").notna().mean()
    if numeric_share >= 0.95:
        return ColumnProfile(kind="numeric", detail="numeric stored as text", **base)

    if any(token in name.lower() for token in DATE_TOKENS):
        parsed = pd.to_datetime(series, errors="coerce")
        if parsed.notna().mean() >= 0.8:
            return ColumnProfile(kind="datetime", detail="date stored as text", **base)

    uniqueness = n_unique / max(int(series.notna().sum()), 1)
    if uniqueness > 0.5:
        return ColumnProfile(kind="text", usable=False, reason="high-cardinality free text",
                             detail=f"{uniqueness:.0%} unique", **base)
    if n_unique <= 50:
        return ColumnProfile(kind="categorical", **base)

    return ColumnProfile(kind="text", usable=False, reason="too many categories for a stable bin",
                         detail=f"{n_unique} distinct values", **base)


def profile_table(name: str, df: pd.DataFrame) -> TableProfile:
    rows = len(df)
    profile = TableProfile(name=name, rows=rows)
    for column in df.columns:
        profile.columns.append(classify_column(column, df[column], name, rows))
        kind = profile.columns[-1].kind
        if kind == "identifier":
            profile.key_columns.append(column)
        if kind == "datetime":
            profile.date_columns.append(column)
    return profile


def profile_tables(tables: dict[str, pd.DataFrame]) -> dict[str, TableProfile]:
    return {name: profile_table(name, df) for name, df in tables.items()}


def relationship_map(profiles: dict[str, TableProfile]) -> list[dict]:
    """Keys that appear in more than one table, i.e. candidate joins."""
    seen: dict[str, list[str]] = {}
    for name, profile in profiles.items():
        for key in profile.key_columns:
            seen.setdefault(key, []).append(name)
    return [
        {"key": key, "tables": tables}
        for key, tables in sorted(seen.items(), key=lambda kv: -len(kv[1]))
        if len(tables) > 1
    ]


# ---------------------------------------------------------------------------
# feature discovery
# ---------------------------------------------------------------------------


@dataclass
class FeatureSet:
    X: pd.DataFrame
    types: dict[str, str]
    provenance: dict[str, str]
    dropped: pd.DataFrame
    profile: dict[str, TableProfile]

    @property
    def features(self) -> list[str]:
        return list(self.X.columns)

    def summary(self) -> pd.DataFrame:
        rows = [
            {"feature": f, "type": self.types.get(f, "?"), "source": self.provenance.get(f, "")}
            for f in self.X.columns
        ]
        return pd.DataFrame(rows)


TARGET_NAMES = {"bad", "default", "target", "label", "outcome", "is_bad", "is_default", "y"}


def _is_banned(name: str, patterns: tuple[str, ...]) -> str | None:
    lowered = name.lower().strip()

    # Exact match on common target names. Substring matching would be wrong
    # here: it would also drop legitimate features such as "bad_rate".
    if lowered in TARGET_NAMES:
        return "target column"

    for pattern in patterns:
        if pattern in lowered:
            return pattern
    return None


def _numeric_columns(df: pd.DataFrame, profile: TableProfile, limit: int = 6) -> list[str]:
    usable = [c.column for c in profile.columns if c.kind == "numeric" and c.usable]
    # Prefer columns with fewer missings; keep the limit small so aggregation
    # does not explode into hundreds of near-duplicate features.
    scored = sorted(usable, key=lambda c: df[c].isna().mean())
    return scored[:limit]


def _aggregate_related(
    base: pd.DataFrame,
    related: pd.DataFrame,
    key: str,
    date_col: str | None,
    cutoff: pd.Series | None,
    profile: TableProfile,
    table: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Roll a child table up to one row per key, respecting a per-row cutoff date."""
    frame = related
    if date_col and cutoff is not None and date_col in frame.columns:
        if isinstance(cutoff, pd.Series):
            frame = frame.merge(cutoff.rename("_cutoff"), left_on=key, right_index=True, how="inner")
            frame = frame[pd.to_datetime(frame[date_col]) <= frame["_cutoff"]]
            frame = frame.drop(columns="_cutoff")
        else:
            frame = frame[pd.to_datetime(frame[date_col]) <= pd.Timestamp(cutoff)]

    if frame.empty:
        return pd.DataFrame(), {}

    grouped = frame.groupby(key)
    out = pd.DataFrame(index=grouped.size().index)
    out[f"{table}_rows"] = grouped.size()
    provenance = {f"{table}_rows": f"{table}: count of related rows"}

    for column in _numeric_columns(frame, profile):
        if column in (key, date_col):
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        out[f"{table}_{column}_mean"] = series.groupby(frame[key]).mean()
        out[f"{table}_{column}_max"] = series.groupby(frame[key]).max()
        provenance[f"{table}_{column}_mean"] = f"{table}.{column}: mean per key"
        provenance[f"{table}_{column}_max"] = f"{table}.{column}: max per key"

    categoricals = [c.column for c in profile.columns if c.kind == "categorical"][:3]
    for column in categoricals:
        out[f"{table}_{column}_nunique"] = frame.groupby(key)[column].nunique()
        provenance[f"{table}_{column}_nunique"] = f"{table}.{column}: distinct values per key"

    return out, provenance


def discover_features(
    base: pd.DataFrame,
    related: dict[str, pd.DataFrame] | None = None,
    profiles: dict[str, TableProfile] | None = None,
    *,
    id_column: str = "loan_id",
    date_column: str | None = "disbursement_date",
    cutoff_by_loan: pd.Series | None = None,
    as_of: pd.Timestamp | None = None,
    banned_patterns: tuple[str, ...] = DEFAULT_BANNED,
    exclude: tuple[str, ...] = (),
    add_ratios: bool = True,
    max_features: int = 200,
) -> FeatureSet:
    """Build a modelling frame from a base table plus optional child tables.

    `cutoff_by_loan` (loan_id -> timestamp) makes child-table aggregation
    point-in-time correct: only rows dated on or before the cutoff are counted,
    so repayment behaviour from the performance window cannot leak in.
    """
    related = related or {}
    profiles = profiles or {}

    base = base.drop_duplicates(id_column).set_index(id_column)
    X = pd.DataFrame(index=base.index)
    types: dict[str, str] = {}
    provenance: dict[str, str] = {}
    dropped: list[dict] = []

    base_profile = profiles.get("base") or profile_table("base", base.reset_index())

    for column in base.columns:
        profile = next((c for c in base_profile.columns if c.column == column), None)
        if profile is None:
            continue

        name = column.lower()
        if any(token in name for token in exclude):
            dropped.append({"column": column, "table": "base", "reason": "excluded by request"})
            continue

        banned = _is_banned(column, banned_patterns)
        if banned:
            dropped.append({"column": column, "table": "base",
                            "reason": f"name matches banned pattern '{banned}'"})
            continue

        group = _looks_protected(column)
        if group:
            dropped.append({"column": column, "table": "base",
                            "reason": f"protected attribute ({group}) - excluded from scoring"})
            continue

        if not profile.usable:
            dropped.append({"column": column, "table": "base", "reason": profile.reason or profile.kind})
            continue

        if profile.kind == "datetime":
            anchor = as_of or (pd.Timestamp(base[column].max()) if date_column == column else pd.Timestamp("now"))
            days = (anchor - pd.to_datetime(base[column])).dt.days
            feature = f"days_since_{column}"
            X[feature] = days.astype(float)
            types[feature] = "numeric"
            provenance[feature] = f"base.{column}: days before the snapshot date"
            continue

        if profile.kind in ("numeric", "categorical"):
            if column == date_column:
                continue
            value = base[column]
            if profile.kind == "numeric":
                value = pd.to_numeric(value, errors="coerce")
            X[column] = value
            types[column] = profile.kind
            provenance[column] = f"base.{column}"

    # ---- child tables -----------------------------------------------------
    for table, df in related.items():
        table_profile = profiles.get(table)
        if table_profile is None or df is None or df.empty:
            continue

        key = next((k for k in table_profile.key_columns if k in base.index.names or k == id_column), None)
        if key is None:
            key = next((k for k in table_profile.key_columns if k in base.columns), None)
        if key is None:
            dropped.append({"column": table, "table": table, "reason": "no join key to the base table"})
            continue

        date_col = table_profile.date_columns[0] if table_profile.date_columns else None
        cutoff = None
        if key == id_column and cutoff_by_loan is not None:
            cutoff = (cutoff_by_loan if isinstance(cutoff_by_loan, pd.Series)
                      else pd.Series(cutoff_by_loan))
        elif date_col and as_of is not None:
            # Customer-level table: a single snapshot date, not a per-loan window.
            cutoff = pd.Timestamp(as_of)

        agg, agg_provenance = _aggregate_related(base.reset_index(), df, key, date_col, cutoff,
                                                 table_profile, table)
        if agg.empty:
            continue

        if key == id_column:
            aligned = agg.reindex(base.index)
        else:
            # The child table is keyed by e.g. customer_id; map it onto loans.
            aligned = agg.reindex(base[key].to_numpy())
            aligned.index = base.index
        for column in agg.columns:
            X[column] = agg[column]
            types[column] = "numeric"
            provenance[column] = agg_provenance.get(column, f"{table}.{column}")

    # ---- ratios -----------------------------------------------------------
    if add_ratios:
        amounts = [
            f for f in X.columns
            if types.get(f) == "numeric"
            and any(token in f.lower() for token in AMOUNT_TOKENS)
            # A ratio of a count to an amount (e.g. "rows / income") is noise,
            # not a feature, and it demoralises readers of the explanation.
            and not any(bad in f.lower() for bad in ("rows", "count", "nunique", "installment_no"))
        ]
        denominators = [f for f in amounts
                        if any(token in f.lower() for token in DENOMINATOR_TOKENS)]
        made = 0
        for numerator in amounts:
            for denominator in denominators:
                if numerator == denominator or made >= 12:
                    continue
                feature = f"ratio_{numerator}_to_{denominator}"
                if feature in X.columns:
                    continue
                denominator_values = pd.to_numeric(X[denominator], errors="coerce").replace(0, np.nan)
                if denominator_values.notna().mean() < 0.3:
                    continue
                X[feature] = pd.to_numeric(X[numerator], errors="coerce") / denominator_values
                types[feature] = "numeric"
                provenance[feature] = f"ratio: {numerator} / {denominator}"
                made += 1

    # ---- prune ------------------------------------------------------------
    keep: list[str] = []
    for column in X.columns:
        series = X[column]
        missing = float(series.isna().mean())
        if missing >= 0.9:
            dropped.append({"column": column, "table": "derived", "reason": f"{missing:.0%} missing"})
            continue
        if series.nunique(dropna=True) <= 1:
            dropped.append({"column": column, "table": "derived", "reason": "constant after joins"})
            continue
        keep.append(column)

    X = X[keep]

    # Drop near-duplicate numeric columns (|corr| > 0.995), keeping the first.
    numeric = [c for c in X.columns if types.get(c) == "numeric"]
    if len(numeric) > 1:
        corr = X[numeric].corr(numeric_only=True).abs()
        redundant = set()
        for i, left in enumerate(numeric):
            for right in numeric[i + 1:]:
                value = corr.loc[left, right]
                if pd.notna(value) and value > 0.995 and right not in redundant:
                    redundant.add(right)
                    dropped.append({"column": right, "table": "derived",
                                    "reason": f"duplicate of {left} (corr {value:.3f})"})
        X = X.drop(columns=[c for c in redundant if c in X.columns])

    if X.shape[1] > max_features:
        # Keep the most complete numeric pair set; selection happens later.
        ranked = sorted(X.columns, key=lambda c: -float(X[c].notna().mean()))[:max_features]
        dropped.extend({"column": c, "table": "derived", "reason": "over max_features cap"}
                       for c in X.columns if c not in ranked)
        X = X[ranked]

    types = {c: types[c] for c in X.columns if c in types}
    provenance = {c: provenance.get(c, "") for c in X.columns}

    return FeatureSet(
        X=X,
        types=types,
        provenance=provenance,
        dropped=pd.DataFrame(dropped, columns=["column", "table", "reason"]),
        profile=profiles,
    )
