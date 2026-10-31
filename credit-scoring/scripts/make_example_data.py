"""Generate the example dataset shipped in examples/.

Deterministic, so the numbers in the docs stay stable. The data is synthetic but
structured the way a real loan book is: instalments fall due monthly, some are
paid late, and lateness is driven by affordability rather than at random.

    python scripts/make_example_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "examples"

N_CUSTOMERS = 320
N_LOANS = 480
SEED = 20240131

REGIONS = ["Jakarta", "Bandung", "Surabaya", "Medan", "Makassar"]
EMPLOYMENT = ["permanent", "contract", "outsourced"]
PURPOSES = ["working_capital", "vehicle", "education", "home_improvement", "debt_consolidation"]


def main() -> None:
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- customers -------------------------------------------------------
    income = np.round(rng.lognormal(mean=7.9, sigma=0.55, size=N_CUSTOMERS), 2)
    customers = pd.DataFrame(
        {
            "customer_id": [f"C{i:04d}" for i in range(1, N_CUSTOMERS + 1)],
            "monthly_income": income,
            "age": rng.integers(21, 61, size=N_CUSTOMERS),
            "employment_type": rng.choice(EMPLOYMENT, size=N_CUSTOMERS, p=[0.55, 0.30, 0.15]),
            "region": rng.choice(REGIONS, size=N_CUSTOMERS, p=[0.35, 0.2, 0.2, 0.15, 0.1]),
            "marital_status": rng.choice(["single", "married", "divorced"], size=N_CUSTOMERS, p=[0.45, 0.45, 0.10]),
        }
    )

    # ---- bureau snapshot -------------------------------------------------
    bureau = pd.DataFrame(
        {
            "customer_id": customers["customer_id"],
            "snapshot_date": pd.Timestamp("2024-12-31"),
            "active_loans": rng.poisson(1.2, size=N_CUSTOMERS),
            "total_outstanding": np.round(rng.gamma(2.0, 900, size=N_CUSTOMERS), 2),
            "inquiries_3m": rng.poisson(0.7, size=N_CUSTOMERS),
        }
    )

    # ---- loans -----------------------------------------------------------
    borrower = rng.choice(customers["customer_id"].to_numpy(), size=N_LOANS, replace=True)
    frame = customers.set_index("customer_id").loc[borrower].reset_index()
    frame = frame.merge(bureau.drop(columns="snapshot_date"), on="customer_id", how="left")

    principal = np.round(
        np.clip(frame["monthly_income"].to_numpy() * rng.uniform(2.0, 11.0, N_LOANS), 500, 40000), -2
    )
    tenor = rng.choice([6, 12, 18, 24], size=N_LOANS, p=[0.2, 0.45, 0.2, 0.15])
    apr = np.round(rng.uniform(12.0, 26.0, size=N_LOANS), 2)

    start = pd.Timestamp("2022-01-01")
    offset = rng.integers(0, (pd.Timestamp("2024-06-30") - start).days, size=N_LOANS)

    loans = pd.DataFrame(
        {
            "loan_id": [f"L{i:05d}" for i in range(1, N_LOANS + 1)],
            "customer_id": frame["customer_id"],
            "disbursement_date": (start + pd.to_timedelta(offset, unit="D")).normalize(),
            "principal": principal,
            "tenor_months": tenor,
            "interest_rate_apr": apr,
            "purpose": rng.choice(PURPOSES, size=N_LOANS),
            "monthly_income": frame["monthly_income"].to_numpy(),
            "employment_type": frame["employment_type"].to_numpy(),
            "region": frame["region"].to_numpy(),
            "active_loans": frame["active_loans"].to_numpy(),
            "total_outstanding": frame["total_outstanding"].to_numpy(),
            "inquiries_3m": frame["inquiries_3m"].to_numpy(),
        }
    )

    # ---- repayment schedule ---------------------------------------------
    rows = []
    for loan in loans.itertuples(index=False):
        monthly_rate = loan.interest_rate_apr / 100 / 12
        instalment = loan.principal * (1 + monthly_rate * loan.tenor_months) / loan.tenor_months
        instalment = round(instalment, 2)

        # Affordability drives lateness: an instalment that eats a large share of
        # income makes a late payment much more likely (and later still).
        dti = instalment / max(loan.monthly_income, 1)
        burden = (
            2.6 * dti
            + 0.35 * (loan.employment_type == "outsourced")
            + 0.18 * (loan.employment_type == "contract")
            + 0.10 * loan.inquiries_3m
            + 0.04 * loan.active_loans
            - 0.25
        )

        for number in range(1, loan.tenor_months + 1):
            due = loan.disbursement_date + pd.DateOffset(months=number)
            shock = rng.normal(0, 0.6)
            late_probability = 1 / (1 + np.exp(-(burden + shock - 1.4)))

            days_late = 0
            if rng.random() < late_probability:
                days_late = int(rng.choice([3, 7, 15, 30, 45, 60, 95, 120], p=[0.3, 0.2, 0.2, 0.12, 0.08, 0.05, 0.03, 0.02]))
                paid = due + pd.Timedelta(days=days_late)
                paid_amount = instalment
            else:
                paid = due + pd.Timedelta(days=int(rng.integers(-3, 2)))
                paid_amount = instalment

            # A small share never pays: leave paid_date empty.
            if days_late >= 95 and rng.random() < 0.45:
                paid = pd.NaT
                paid_amount = 0.0

            rows.append(
                {
                    "loan_id": loan.loan_id,
                    "installment_no": number,
                    "due_date": due,
                    "paid_date": paid,
                    "due_amount": instalment,
                    "paid_amount": paid_amount,
                }
            )

    repayments = pd.DataFrame(rows)

    loans_out = loans.drop(columns=["monthly_income", "employment_type", "region",
                                    "active_loans", "total_outstanding", "inquiries_3m"])
    loans_out.to_csv(OUT / "loans.csv", index=False)
    repayments.to_csv(OUT / "repayments.csv", index=False, date_format="%Y-%m-%d")
    customers.to_csv(OUT / "customers.csv", index=False)
    bureau.to_csv(OUT / "bureau.csv", index=False)

    print(f"customers : {len(customers):>7,} rows")
    print(f"loans     : {len(loans_out):>7,} rows")
    print(f"repayments: {len(repayments):>7,} rows")
    print(f"bureau    : {len(bureau):>7,} rows")
    print(f"date range: {loans_out['disbursement_date'].min():%Y-%m-%d} "
          f"to {loans_out['disbursement_date'].max():%Y-%m-%d}")


if __name__ == "__main__":
    main()
