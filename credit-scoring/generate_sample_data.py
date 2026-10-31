"""Generate a synthetic payroll-loan dataset (English column names).

Four tables are written to ./data:
    customers.csv, loans.csv, repayments.csv, bureau.csv

The data has a hidden risk driver, so the features carry realistic signal:
high installment-to-income, weak bureau history, short employer tenure and
early late payments all push the probability of default up.

`loans.collection_status` is deliberately leaky (set AFTER default) so you can
see the leakage checker work.

Usage:  python generate_sample_data.py [n_loans] [out_dir]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n: int = 6000, seed: int = 42, out_dir: str = "data", as_of: str = "2025-12-31") -> dict:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    as_of_ts = pd.Timestamp(as_of)

    # ---------------- customers ----------------
    customer_id = np.array([f"C{100000 + i}" for i in range(n)])
    loan_id = np.array([f"L{500000 + i}" for i in range(n)])
    age = rng.integers(21, 56, n)
    education = rng.choice(["high_school", "diploma", "bachelor", "master"], n, p=[0.35, 0.20, 0.38, 0.07])
    marital = rng.choice(["single", "married", "divorced"], n, p=[0.45, 0.48, 0.07])
    employment = rng.choice(["permanent", "contract", "outsourced"], n, p=[0.55, 0.28, 0.17])
    region = rng.choice(["north", "south", "east", "west", "central"], n)
    tenure = np.clip(rng.gamma(2.0, 14, n), 1, 240).round().astype(int)
    income = np.round(np.exp(rng.normal(6.8, 0.45, n)), 0)

    # ---------------- loans ----------------
    start, end = pd.Timestamp("2023-01-01"), pd.Timestamp("2025-06-30")
    span = (end - start).days
    disb = start + pd.to_timedelta(rng.integers(0, span, n), unit="D")
    principal = np.round(income * rng.uniform(0.3, 1.6, n), 0)
    tenor = rng.choice([6, 9, 12], n, p=[0.35, 0.25, 0.40])
    apr = rng.uniform(0.18, 0.36, n).round(3)
    purpose = rng.choice(["medical", "education", "household", "debt_consolidation", "emergency"], n,
                         p=[0.15, 0.15, 0.30, 0.20, 0.20])
    repeat = rng.binomial(1, 0.35, n)
    installment = principal * (1 + apr * tenor / 12) / tenor
    ito = installment / income

    # ---------------- bureau ----------------
    active = rng.poisson(1.6, n)
    outstanding = np.round(active * income * rng.uniform(0.2, 2.0, n))
    max_coll = rng.choice([1, 2, 3, 4, 5], n, p=[0.82, 0.10, 0.04, 0.02, 0.02])
    inquiries = rng.poisson(1.2, n)
    writeoff = rng.binomial(1, 0.03, n)

    # ---------------- hidden risk driver ----------------
    z = (-2.9
         + 4.0 * (ito - 0.2)
         + 0.45 * (max_coll - 1)
         + 0.20 * inquiries
         + 1.0 * writeoff
         + 0.25 * (active >= 4)
         + 0.15 * (outstanding / income)
         - 0.012 * np.minimum(tenure, 60)
         - 0.6 * repeat
         + 0.3 * (employment == "contract")
         + 0.6 * (employment == "outsourced")
         - 0.012 * (age - 35)
         + rng.normal(0, 0.5, n))
    pd_true = sigmoid(z)
    default = rng.random(n) < pd_true

    # ---------------- repayments ----------------
    rows = []
    default_start = np.zeros(n, dtype=int)
    for i in range(n):
        t = int(tenor[i])
        s = 0
        if default[i]:
            s = int(rng.choice([1, 2, 3])) if rng.random() < 0.25 else int(rng.integers(4, t + 1))
            default_start[i] = s
        p_mild = 0.30 if default[i] else 0.06 + 0.20 * pd_true[i]
        for k in range(1, t + 1):
            due = disb[i] + pd.DateOffset(months=k)
            if s and k >= s:
                paid, amt = pd.NaT, 0.0
            else:
                if rng.random() < p_mild:
                    delay = int(rng.integers(1, 12))
                elif (not default[i]) and rng.random() < 0.015:
                    delay = int(rng.integers(31, 75))
                else:
                    delay = -int(rng.integers(0, 4))
                paid = due + pd.Timedelta(days=delay)
                amt = round(float(installment[i]), 2)
                if paid > as_of_ts:        # not paid yet at snapshot
                    paid, amt = pd.NaT, 0.0
            rows.append((loan_id[i], k, due, paid, round(float(installment[i]), 2), amt))
    repayments = pd.DataFrame(rows, columns=["loan_id", "installment_no", "due_date", "paid_date",
                                             "due_amount", "paid_amount"])

    # ---------------- leaky field ----------------
    status = np.where(rng.random(n) < 0.03, "reminder", "none").astype(object)
    for i in range(n):
        if default[i]:
            first_missed = disb[i] + pd.DateOffset(months=int(default_start[i]))
            if first_missed + pd.Timedelta(days=30) <= as_of_ts:
                status[i] = "legal" if rng.random() < 0.3 else "collection"

    customers = pd.DataFrame({
        "customer_id": customer_id, "age": age, "education": education, "marital_status": marital,
        "employment_type": employment, "region": region, "employer_tenure_months": tenure,
        "monthly_income": income,
    })
    loans = pd.DataFrame({
        "loan_id": loan_id, "customer_id": customer_id, "disbursement_date": disb.date if hasattr(disb, "date") else disb,
        "principal": principal, "tenor_months": tenor, "interest_rate_apr": apr, "purpose": purpose,
        "is_repeat_borrower": repeat, "collection_status": status,
    })
    loans["disbursement_date"] = pd.to_datetime(disb).strftime("%Y-%m-%d")
    bureau = pd.DataFrame({
        "customer_id": customer_id,
        "snapshot_date": (pd.to_datetime(disb) - pd.to_timedelta(rng.integers(1, 15, n), unit="D")).strftime("%Y-%m-%d"),
        "active_loans": active, "total_outstanding": outstanding, "max_collectibility": max_coll,
        "inquiries_3m": inquiries, "has_writeoff": writeoff,
    })
    repayments["due_date"] = repayments["due_date"].dt.strftime("%Y-%m-%d")
    repayments["paid_date"] = repayments["paid_date"].dt.strftime("%Y-%m-%d")

    customers.to_csv(out / "customers.csv", index=False)
    loans.to_csv(out / "loans.csv", index=False)
    bureau.to_csv(out / "bureau.csv", index=False)
    repayments.to_csv(out / "repayments.csv", index=False)
    return {"customers": customers, "loans": loans, "bureau": bureau, "repayments": repayments,
            "default_rate": float(default.mean())}


if __name__ == "__main__":
    n_loans = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    folder = sys.argv[2] if len(sys.argv) > 2 else "data"
    res = generate(n_loans, out_dir=folder)
    print(f"Wrote {n_loans} loans, {len(res['repayments'])} installments to {folder}/ "
          f"(true default rate ~{res['default_rate']:.1%})")
