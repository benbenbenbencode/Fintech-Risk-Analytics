"""CSV ingestion: the required format, a validator, and a repayment-only path.

Two ways to bring data in:

**A. Repayment history only** - one file with one row per instalment. This is the
minimum viable input for credit scoring: the target (how late did they get) and
the behavioural features both come from it. The loan-level table is derived:
disbursement date is approximated one month before the first due date, and
principal is the sum of scheduled instalments.

**B. Loans + repayments** - one row per loan plus one row per instalment. Always
prefer this: disbursement date, principal and tenor are then facts rather than
estimates.

`validate_tables` produces a pass/warn/fail checklist so a business user sees
exactly what is wrong before a model is trained on bad input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

COLUMN_SPEC: dict[str, dict[str, dict[str, tuple[str, str]]]] = {
    "loans": {
        "required": {
            "loan_id": ("text", "Unique loan identifier"),
            "customer_id": ("text", "Borrower identifier (may repeat across loans)"),
            "disbursement_date": ("date", "Date the money was paid out, YYYY-MM-DD"),
            "principal": ("number", "Amount lent"),
            "tenor_months": ("integer", "Number of instalments"),
        },
        "optional": {
            "interest_rate_apr": ("number", "Annual interest rate, percent"),
            "purpose": ("text", "Loan purpose"),
            "product": ("text", "Product name"),
            "channel": ("text", "Branch / online / partner"),
        },
    },
    "repayments": {
        "required": {
            "loan_id": ("text", "Must exist in the loan table"),
            "installment_no": ("integer", "1, 2, 3, ... within each loan"),
            "due_date": ("date", "Scheduled payment date"),
            "paid_date": ("date", "Actual payment date; leave empty if unpaid"),
            "due_amount": ("number", "Amount that was due"),
        },
        "optional": {
            "paid_amount": ("number", "Amount actually paid"),
        },
    },
    "customers": {
        "required": {"customer_id": ("text", "Borrower identifier")},
        "optional": {
            "monthly_income": ("number", "Declared or verified income"),
            "age": ("integer", "Detected and excluded from scoring, audited separately"),
            "employment_type": ("text", "Permanent / contract / outsourced"),
            "region": ("text", "Geographic region"),
            "marital_status": ("text", "Detected and excluded from scoring"),
        },
    },
    "bureau": {
        "required": {"customer_id": ("text", "Borrower identifier")},
        "optional": {
            "snapshot_date": ("date", "Date of the bureau pull"),
            "active_loans": ("number", "Number of open facilities"),
            "total_outstanding": ("number", "Total outstanding balance"),
            "inquiries_3m": ("number", "Credit enquiries in the last 3 months"),
        },
    },
}

DATE_COLUMNS = {"disbursement_date", "due_date", "paid_date", "snapshot_date"}
MAX_UNPARSEABLE = 0.02


@dataclass
class Check:
    name: str
    status: str  # "pass" | "warn" | "fail"
    message: str
    fix: str = ""

    def to_dict(self) -> dict:
        return {"check": self.name, "status": self.status, "message": self.message, "fix": self.fix}


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(c.status == "fail" for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == "fail"]

    @property
    def warnings(self) -> list[Check]:
        return [c for c in self.checks if c.status == "warn"]

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame([c.to_dict() for c in self.checks])

    def summary(self) -> str:
        fails, warns = len(self.failures), len(self.warnings)
        if fails:
            return f"{fails} blocking problem(s), {warns} warning(s). Fix the blockers before running."
        if warns:
            return f"Format is usable, with {warns} warning(s) worth reading."
        return "Format looks good."


def required_columns(table: str) -> list[str]:
    return list(COLUMN_SPEC.get(table, {}).get("required", {}))


def optional_columns(table: str) -> list[str]:
    return list(COLUMN_SPEC.get(table, {}).get("optional", {}))


def template_frame(table: str) -> pd.DataFrame:
    """Header plus two illustrative rows, for the download button."""
    examples = {
        "loans": {
            "loan_id": ["L0001", "L0002"],
            "customer_id": ["C001", "C002"],
            "disbursement_date": ["2024-03-15", "2024-04-02"],
            "principal": [5000, 12000],
            "tenor_months": [12, 24],
            "interest_rate_apr": [18.5, 15.0],
            "purpose": ["working capital", "vehicle"],
        },
        "repayments": {
            "loan_id": ["L0001", "L0001", "L0002"],
            "installment_no": [1, 2, 1],
            "due_date": ["2024-04-15", "2024-05-15", "2024-05-02"],
            "paid_date": ["2024-04-15", "", "2024-05-09"],
            "due_amount": [520, 520, 640],
            "paid_amount": [520, 0, 640],
        },
        "customers": {
            "customer_id": ["C001", "C002"],
            "monthly_income": [800, 1500],
            "age": [34, 41],
            "employment_type": ["permanent", "contract"],
            "region": ["Jakarta", "Bandung"],
        },
        "bureau": {
            "customer_id": ["C001", "C002"],
            "snapshot_date": ["2024-03-01", "2024-04-01"],
            "active_loans": [2, 0],
            "total_outstanding": [1500, 0],
            "inquiries_3m": [1, 0],
        },
    }
    return pd.DataFrame(examples.get(table, {c: [] for c in required_columns(table)}))


def template_csv(table: str) -> str:
    return template_frame(table).to_csv(index=False)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def _missing(df: pd.DataFrame, columns: list[str]) -> list[str]:
    normalised = {"".join(ch for ch in c.lower() if ch.isalnum()): c for c in df.columns}
    return [c for c in columns if "".join(ch for ch in c.lower() if ch.isalnum()) not in normalised]


def _parse_dates(df: pd.DataFrame, column: str) -> tuple[pd.Series, float]:
    parsed = pd.to_datetime(df[column], errors="coerce")
    present = df[column].notna().sum()
    bad = 1 - (parsed.notna().sum() / present) if present else 0.0
    return parsed, float(bad)


def validate_tables(tables: dict[str, pd.DataFrame]) -> ValidationReport:
    report = ValidationReport()

    loans = tables.get("loans")
    repayments = tables.get("repayments")

    if loans is None and repayments is None:
        report.checks.append(
            Check("input", "fail", "No loan or repayment table supplied.",
                  "Upload at least a repayment history file.")
        )
        return report

    for name, df in tables.items():
        if df is None or df.empty:
            report.checks.append(Check(f"{name}: non-empty", "fail", f"The {name} table is empty.", ""))
            continue

        required = required_columns(name)
        missing = _missing(df, required)
        report.checks.append(
            Check(
                f"{name}: required columns", "fail" if missing else "pass",
                f"Missing: {', '.join(missing)}" if missing else f"All {len(required)} required columns found.",
                "Rename the columns to the canonical names, or map them in the app." if missing else "",
            )
        )

    if loans is not None and not loans.empty:
        if "loan_id" in loans.columns:
            duplicates = int(loans["loan_id"].duplicated().sum())
            report.checks.append(
                Check(
                    "loans: loan_id unique", "fail" if duplicates else "pass",
                    f"{duplicates:,} duplicate loan_id value(s)." if duplicates else "Loan IDs are unique.",
                    "One row per loan. Aggregate or drop duplicates." if duplicates else "",
                )
            )

        if "disbursement_date" in loans.columns:
            parsed, bad = _parse_dates(loans, "disbursement_date")
            report.checks.append(
                Check(
                    "loans: disbursement_date is a date", "fail" if bad > MAX_UNPARSEABLE else "pass",
                    f"{bad:.1%} unparseable." if bad else
                    f"Parsed; range {parsed.min():%Y-%m-%d} to {parsed.max():%Y-%m-%d}.",
                    "Use YYYY-MM-DD (2024-03-15)." if bad else "",
                )
            )
            report.stats["date_min"] = str(parsed.min().date()) if parsed.notna().any() else None
            report.stats["date_max"] = str(parsed.max().date()) if parsed.notna().any() else None

        if "principal" in loans.columns:
            amount = pd.to_numeric(loans["principal"], errors="coerce")
            bad = int((amount.isna() | (amount <= 0)).sum())
            report.checks.append(
                Check(
                    "loans: principal is a positive number", "warn" if bad else "pass",
                    f"{bad:,} row(s) missing or not positive." if bad else "All amounts positive.",
                    "Remove zero/negative lent amounts." if bad else "",
                )
            )

    if repayments is not None and not repayments.empty:
        if "due_date" in repayments.columns:
            parsed, bad = _parse_dates(repayments, "due_date")
            report.checks.append(
                Check(
                    "repayments: due_date is a date", "fail" if bad > MAX_UNPARSEABLE else "pass",
                    f"{bad:.1%} unparseable." if bad else
                    f"Parsed; instalments through {parsed.max():%Y-%m-%d}.",
                    "Use YYYY-MM-DD." if bad else "",
                )
            )
            report.stats["last_due_date"] = str(parsed.max().date()) if parsed.notna().any() else None

        if "paid_date" in repayments.columns:
            _, bad = _parse_dates(repayments, "paid_date")
            blank = repayments["paid_date"].isna().sum()
            report.checks.append(
                Check(
                    "repayments: paid_date parses (blank means unpaid)", "warn" if bad > MAX_UNPARSEABLE else "pass",
                    f"{bad:.1%} of filled values unparseable." if bad
                    else f"{blank:,} blank value(s) treated as unpaid.",
                    "Leave unpaid instalments empty rather than guessing." if bad else "",
                )
            )

        if "loan_id" in repayments.columns and loans is not None and "loan_id" in loans.columns:
            known = set(loans["loan_id"].astype(str))
            present = set(repayments["loan_id"].astype(str))
            orphans = len(present - known)
            share = 1 - (len(present & known) / len(present)) if present else 0
            report.checks.append(
                Check(
                    "repayments: every loan_id exists in loans",
                    "fail" if share > 0.05 else "warn" if orphans else "pass",
                    f"{orphans:,} loan_id(s) in repayments are not in the loan table.",
                    "Check the ID format (leading zeros, numeric vs text)." if orphans else "",
                )
            )

            if "loan_id" in loans.columns:
                with_repayments = set(repayments["loan_id"].astype(str))
                missing_history = len(known - with_repayments)
                report.checks.append(
                    Check(
                        "loans: every loan has repayment history", "warn" if missing_history else "pass",
                        f"{missing_history:,} loan(s) have no repayment rows.",
                        "Those loans cannot be scored on behaviour and will be dropped." if missing_history else "",
                    )
                )

        if "due_amount" in repayments.columns:
            amount = pd.to_numeric(repayments["due_amount"], errors="coerce")
            bad = int((amount.isna() | (amount <= 0)).sum())
            report.checks.append(
                Check(
                    "repayments: due_amount is positive", "warn" if bad else "pass",
                    f"{bad:,} row(s) missing or not positive." if bad else "All due amounts positive.",
                    "Remove or fix zero-amount instalments." if bad else "",
                )
            )

        per_loan = repayments.groupby("loan_id").size()
        report.stats["median_installments_per_loan"] = int(np.median(per_loan)) if len(per_loan) else 0
        report.stats["loans_in_repayments"] = int(len(per_loan))
        report.checks.append(
            Check(
                "repayments: enough history per loan", "warn" if len(per_loan) and np.median(per_loan) < 3 else "pass",
                f"Median {int(np.median(per_loan)) if len(per_loan) else 0} instalment(s) per loan.",
                "Very short histories make behavioural features noisy." if len(per_loan) and np.median(per_loan) < 3 else "",
            )
        )

    return report


# ---------------------------------------------------------------------------
# repayment-history-only mode
# ---------------------------------------------------------------------------


def derive_loans(repayments: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Build a loan table from a repayment history alone.

    Approximations, and why:
      * disbursement_date = first due_date minus one month. Scheduled instalments
        usually start a month after payout. It is an estimate, so the derived
        column is reported to the user.
      * principal = sum of scheduled instalments (ignores interest breakdown).
      * tenor_months = highest instalment number, or the row count.
    """
    frame = repayments.copy()
    frame["due_date"] = pd.to_datetime(frame["due_date"], errors="coerce")
    frame = frame.dropna(subset=["due_date"])
    if "due_amount" in frame.columns:
        frame["due_amount"] = pd.to_numeric(frame["due_amount"], errors="coerce")

    grouped = frame.groupby("loan_id")
    loans = pd.DataFrame(index=grouped.size().index)
    loans.index.name = "loan_id"

    loans["customer_id"] = loans.index.astype(str)
    loans["disbursement_date"] = grouped["due_date"].min() - pd.DateOffset(months=1)

    if "due_amount" in frame.columns:
        loans["principal"] = grouped["due_amount"].sum().round(2)
    else:
        loans["principal"] = 1.0

    if "installment_no" in frame.columns:
        loans["tenor_months"] = grouped["installment_no"].max().astype(int)
    else:
        loans["tenor_months"] = grouped.size().astype(int)

    loans = loans.reset_index()
    loans["day_of_month"] = pd.to_datetime(loans["disbursement_date"]).dt.day

    info = {
        "mode": "repayment_history_only",
        "approximated": ["disbursement_date", "principal", "tenor_months"],
        "notes": [
            "disbursement_date was estimated as one month before the first due date.",
            "principal was estimated as the sum of scheduled instalments.",
            "tenor_months was estimated from the highest instalment number.",
            "Columns such as income, purpose or region are unavailable in this mode; "
            "add a loans or customers file to use them.",
        ],
        "loans": int(len(loans)),
    }
    return loans, info


def suggest_as_of(repayments: pd.DataFrame) -> pd.Timestamp | None:
    if repayments is None or repayments.empty or "due_date" not in repayments.columns:
        return None
    dates = pd.to_datetime(repayments["due_date"], errors="coerce")
    if dates.notna().sum() == 0:
        return None
    return dates.max().normalize()
