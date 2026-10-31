"""Days-past-due computed as of an arbitrary (per-row) date: no look-ahead."""
from __future__ import annotations

import pandas as pd


def compute_dpd(due: pd.Series, paid: pd.Series, asof: pd.Series) -> pd.Series:
    """DPD of each installment as it looked on `asof`.
    Paid on/before asof -> days late at payment. Otherwise -> days overdue on asof.
    Installments not yet due on asof -> NaN."""
    paid_in = paid.notna() & (paid <= asof)
    end = paid.where(paid_in, asof)
    dpd = (end - due).dt.days.clip(lower=0)
    return dpd.where(due <= asof)
