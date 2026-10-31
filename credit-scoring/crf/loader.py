"""Load raw tables and map them onto the canonical schema."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED = {
    "customers": ["customer_id"],
    "loans": ["loan_id", "customer_id", "disbursement_date", "principal", "tenor_months"],
    "repayments": ["loan_id", "installment_no", "due_date", "paid_date", "due_amount"],
    "bureau": ["customer_id"],
}
DATE_COLS = {
    "loans": ["disbursement_date"],
    "repayments": ["due_date", "paid_date"],
    "bureau": ["snapshot_date"],
}


def _resolve(path: str, base_dir: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(base_dir) / p


def read_table(name: str, spec: dict, base_dir: str) -> pd.DataFrame:
    df = pd.read_csv(_resolve(spec["path"], base_dir))
    mapping = spec.get("columns", {}) or {}
    df = df.rename(columns={raw: canon for canon, raw in mapping.items() if raw != canon and raw in df.columns})
    missing = [c for c in REQUIRED.get(name, []) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Table '{name}' is missing canonical columns {missing}. "
            f"Map them under data.{name}.columns as  canonical_name: raw_column_name."
        )
    for c in DATE_COLS.get(name, []):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def load_tables(cfg: dict) -> dict[str, pd.DataFrame]:
    return {name: read_table(name, spec, cfg["_base_dir"]) for name, spec in cfg["data"].items() if spec}


def build_base(tables: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per loan (customer + loan + point-in-time bureau) and the repayment schedule."""
    base = tables["loans"].drop_duplicates("loan_id").copy()
    rep = tables["repayments"].copy()

    cust = tables.get("customers")
    if cust is not None:
        base = base.merge(cust.drop_duplicates("customer_id"), on="customer_id", how="left", suffixes=("", "_cust"))

    bur = tables.get("bureau")
    if bur is not None:
        bur = bur.rename(columns={c: f"bureau_{c}" for c in bur.columns if c not in ("customer_id", "snapshot_date")})
        if "snapshot_date" in bur.columns:
            base = base.sort_values("disbursement_date")
            bur = bur.dropna(subset=["snapshot_date"]).sort_values("snapshot_date")
            base = pd.merge_asof(base, bur, left_on="disbursement_date", right_on="snapshot_date",
                                 by="customer_id", direction="backward").drop(columns="snapshot_date")
        else:
            base = base.merge(bur.drop_duplicates("customer_id"), on="customer_id", how="left")

    first = (rep.sort_values("installment_no").drop_duplicates("loan_id")[["loan_id", "due_amount"]]
             .rename(columns={"due_amount": "installment_amount"}))
    base = base.merge(first, on="loan_id", how="left")
    return base.sort_values("disbursement_date").reset_index(drop=True), rep
