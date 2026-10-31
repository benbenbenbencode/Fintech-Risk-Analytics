"""Tests for CSV ingestion: format validation and the repayment-only path."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from crf.ingest import (
    COLUMN_SPEC,
    derive_loans,
    required_columns,
    suggest_as_of,
    template_csv,
    template_frame,
    validate_tables,
)

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def make_loans(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_id": [f"L{i}" for i in range(n)],
            "customer_id": [f"C{i}" for i in range(n)],
            "disbursement_date": pd.date_range("2024-01-01", periods=n, freq="30D"),
            "principal": [1000.0 + 100 * i for i in range(n)],
            "tenor_months": [12] * n,
        }
    )


def make_repayments(loans: pd.DataFrame, months: int = 6) -> pd.DataFrame:
    rows = []
    for loan in loans.itertuples():
        for number in range(1, months + 1):
            due = loan.disbursement_date + pd.DateOffset(months=number)
            rows.append(
                {
                    "loan_id": loan.loan_id,
                    "installment_no": number,
                    "due_date": due,
                    "paid_date": due if number % 3 else pd.NaT,
                    "due_amount": 100.0,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# format specification
# ---------------------------------------------------------------------------


def test_template_contains_every_required_column():
    for table in ("loans", "repayments", "customers", "bureau"):
        frame = template_frame(table)
        for column in required_columns(table):
            assert column in frame.columns, f"{table} template missing {column}"
        # The downloadable text must round-trip through pandas.
        parsed = pd.read_csv(pd.io.common.StringIO(template_csv(table)))
        assert list(parsed.columns) == list(frame.columns)


def test_column_spec_documents_types():
    for table, spec in COLUMN_SPEC.items():
        for group in ("required", "optional"):
            for column, (dtype, description) in spec.get(group, {}).items():
                assert dtype in {"text", "number", "integer", "date"}
                assert description, f"{table}.{column} has no description"


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def test_valid_tables_pass():
    loans = make_loans()
    repayments = make_repayments(loans)
    report = validate_tables({"loans": loans, "repayments": repayments})

    assert report.ok, [c.message for c in report.failures]
    assert not report.failures


def test_missing_column_is_a_blocking_failure():
    loans = make_loans().drop(columns=["principal"])
    report = validate_tables({"loans": loans, "repayments": make_repayments(make_loans())})

    assert not report.ok
    assert any("required columns" in c.name and "principal" in c.message for c in report.failures)


def test_duplicate_loan_id_is_a_blocking_failure():
    loans = make_loans()
    duplicated = pd.concat([loans, loans.head(1)], ignore_index=True)
    report = validate_tables({"loans": duplicated, "repayments": make_repayments(loans)})

    assert not report.ok
    assert any("unique" in c.name for c in report.failures)


def test_orphan_repayments_are_flagged():
    loans = make_loans()
    repayments = make_repayments(loans)
    repayments.loc[0, "loan_id"] = "NOT_A_LOAN"

    report = validate_tables({"loans": loans, "repayments": repayments})
    check = next(c for c in report.checks if "exists in loans" in c.name)
    assert check.status in ("warn", "fail")


def test_bad_dates_are_reported_not_crashed():
    loans = make_loans()
    loans["disbursement_date"] = ["not-a-date"] * len(loans)
    report = validate_tables({"loans": loans, "repayments": make_repayments(make_loans())})

    assert not report.ok
    assert any("date" in c.name for c in report.failures)


# ---------------------------------------------------------------------------
# repayment-only mode
# ---------------------------------------------------------------------------


def test_derive_loans_builds_a_usable_loan_table():
    source = make_loans()
    repayments = make_repayments(source, months=6)

    loans, info = derive_loans(repayments)

    assert len(loans) == len(source)
    for column in required_columns("loans"):
        assert column in loans.columns

    assert info["mode"] == "repayment_history_only"
    assert "principal" in info["approximated"]
    # One month before the first instalment, by construction.
    first_due = repayments[repayments["loan_id"] == "L0"]["due_date"].min()
    derived = pd.Timestamp(loans.loc[loans["loan_id"] == "L0", "disbursement_date"].iloc[0])
    assert derived == pd.Timestamp(first_due) - pd.DateOffset(months=1)


def test_derived_loans_validate():
    source = make_loans()
    repayments = make_repayments(source)
    loans, _ = derive_loans(repayments)

    report = validate_tables({"loans": loans, "repayments": repayments})
    assert report.ok, [c.message for c in report.failures]


def test_suggest_as_of_uses_the_last_due_date():
    source = make_loans()
    repayments = make_repayments(source, months=3)
    suggestion = suggest_as_of(repayments)

    assert suggestion is not None
    assert suggestion == pd.to_datetime(repayments["due_date"]).max().normalize()


# ---------------------------------------------------------------------------
# the shipped example dataset
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not EXAMPLES.exists(), reason="examples/ not generated")
def test_shipped_examples_are_valid():
    tables = {name: pd.read_csv(EXAMPLES / f"{name}.csv")
              for name in ("loans", "repayments", "customers", "bureau")}
    report = validate_tables(tables)

    assert report.ok, [f"{c.name}: {c.message}" for c in report.failures]
    assert len(tables["loans"]) >= 100
    assert tables["repayments"]["loan_id"].nunique() == len(tables["loans"])
