"""
InsightCopilot - Synthetic Dataset Generator
Location: scripts/generate_pinjol_data.py
Creates 10 relational CSV files for Fintech P2P Lending.
"""
import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def generate_pinjol_dataset(output_dir="pinjol_csvs"):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)
    random.seed(42)

    cities = ["Jakarta Selatan", "Jakarta Barat", "Surabaya", "Bandung", "Medan", "Semarang", "Tangerang", "Bekasi", "Depok", "Palembang"]
    risk_grades = ["A", "B", "C", "D", "E"]

    n_borrowers = 500
    borrower_ids = [f"BW-{1000 + i}" for i in range(n_borrowers)]
    df_borrowers = pd.DataFrame({
        "borrower_id": borrower_ids,
        "name": [f"Borrower_{i}" for i in range(n_borrowers)],
        "age": np.random.randint(20, 55, size=n_borrowers),
        "monthly_income": np.random.choice([3500000, 5000000, 7500000, 10000000, 15000000, 25000000], size=n_borrowers, p=[0.2, 0.3, 0.25, 0.15, 0.07, 0.03]),
        "city": np.random.choice(cities, size=n_borrowers),
        "credit_score": np.random.randint(450, 850, size=n_borrowers),
        "risk_grade": np.random.choice(risk_grades, size=n_borrowers, p=[0.15, 0.30, 0.35, 0.12, 0.08]),
        "created_at": [datetime(2025, 1, 1) + timedelta(days=int(i)) for i in np.random.randint(0, 300, size=n_borrowers)]
    })
    df_borrowers.to_csv(f"{output_dir}/dim_borrowers.csv", index=False)

    partners = [
        {"partner_id": "PRT-01", "partner_name": "Tokopedia Merchant Partner", "channel_type": "E-Commerce", "commission_rate": 0.02},
        {"partner_id": "PRT-02", "partner_name": "Gojek Driver Agent", "channel_type": "Gig Economy", "commission_rate": 0.015},
        {"partner_id": "PRT-03", "partner_name": "Direct Mobile App", "channel_type": "Direct Organic", "commission_rate": 0.0},
        {"partner_id": "PRT-04", "partner_name": "PayLater Checkout Widget", "channel_type": "POS Checkout", "commission_rate": 0.025},
        {"partner_id": "PRT-05", "partner_name": "Offline Agent Network Medan", "channel_type": "Offline Agent", "commission_rate": 0.03}
    ]
    pd.DataFrame(partners).to_csv(f"{output_dir}/dim_partners.csv", index=False)

    n_apps = 1200
    app_ids = [f"APP-{5000 + i}" for i in range(n_apps)]
    df_apps = pd.DataFrame({
        "application_id": app_ids,
        "borrower_id": np.random.choice(borrower_ids, size=n_apps),
        "partner_id": np.random.choice([p["partner_id"] for p in partners], size=n_apps),
        "applied_amount": np.random.choice([1000000, 2000000, 3500000, 5000000, 10000000, 15000000], size=n_apps),
        "status": np.random.choice(["APPROVED", "REJECTED", "CANCELLED"], size=n_apps, p=[0.65, 0.30, 0.05]),
        "applied_at": [datetime(2026, 1, 1) + timedelta(days=float(i/4)) for i in range(n_apps)]
    })
    df_apps.to_csv(f"{output_dir}/fct_loan_applications.csv", index=False)

    approved_apps = df_apps[df_apps["status"] == "APPROVED"].copy()
    n_loans = len(approved_apps)
    approved_apps["loan_id"] = [f"LN-{8000 + i}" for i in range(n_loans)]
    approved_apps["principal_amount"] = approved_apps["applied_amount"]
    approved_apps["interest_rate_monthly"] = 0.025
    approved_apps["tenor_months"] = np.random.choice([1, 3, 6, 12], size=n_loans, p=[0.4, 0.3, 0.2, 0.1])
    approved_apps["disbursed_at"] = approved_apps["applied_at"] + pd.to_timedelta(np.random.randint(1, 24, size=n_loans), unit='h')
    approved_apps["loan_status"] = np.random.choice(["CURRENT", "PAID_OFF", "DEFAULTED_TWP90", "LATE_1_30"], size=n_loans, p=[0.45, 0.35, 0.12, 0.08])
    
    df_loans = approved_apps[["loan_id", "application_id", "borrower_id", "partner_id", "principal_amount", "interest_rate_monthly", "tenor_months", "disbursed_at", "loan_status"]]
    df_loans.to_csv(f"{output_dir}/fct_loans.csv", index=False)

    repayments = []
    rep_id_counter = 10000
    for _, row in df_loans.iterrows():
        tenor = int(row["tenor_months"])
        total_due = (row["principal_amount"] / tenor) + (row["principal_amount"] * row["interest_rate_monthly"])
        disbursed = row["disbursed_at"]
        for m in range(1, tenor + 1):
            due_date = disbursed + timedelta(days=30 * m)
            is_default = row["loan_status"] == "DEFAULTED_TWP90" and m >= 2
            is_late = row["loan_status"] == "LATE_1_30" and m == tenor
            status = "UNPAID" if is_default else ("LATE" if is_late else "PAID_ON_TIME")
            paid_amt = 0.0 if is_default else total_due
            dpd = 95 if is_default else (18 if is_late else 0)
            repayments.append({
                "repayment_id": f"RP-{rep_id_counter}",
                "loan_id": row["loan_id"],
                "borrower_id": row["borrower_id"],
                "installment_no": m,
                "due_date": due_date,
                "paid_date": due_date + timedelta(days=dpd) if dpd > 0 else due_date - timedelta(days=1),
                "due_amount": round(total_due, 2),
                "paid_amount": round(paid_amt, 2),
                "status": status,
                "dpd": dpd
            })
            rep_id_counter += 1
    df_repayments = pd.DataFrame(repayments)
    df_repayments.to_csv(f"{output_dir}/fct_repayments.csv", index=False)

    assessments = [{
        "assessment_id": f"AST-{row['application_id']}",
        "application_id": row["application_id"],
        "borrower_id": row["borrower_id"],
        "p2p_score": random.randint(500, 850),
        "bank_statement_score": random.randint(400, 900),
        "device_risk_score": random.choice(["LOW", "MEDIUM", "HIGH"]),
        "dscr_ratio": round(random.uniform(1.2, 3.5), 2),
        "approved_limit": row["applied_amount"] if row["status"] == "APPROVED" else 0
    } for _, row in df_apps.iterrows()]
    pd.DataFrame(assessments).to_csv(f"{output_dir}/fct_credit_assessments.csv", index=False)

    collections = [{
        "collection_id": f"COL-{7000 + i}",
        "repayment_id": row["repayment_id"],
        "loan_id": row["loan_id"],
        "borrower_id": row["borrower_id"],
        "agent_id": f"AGT-0{random.randint(1, 5)}",
        "dpd_bucket": "DPD 90+" if row["dpd"] >= 90 else "DPD 1-30",
        "contact_outcome": random.choice(["PTP_PROMISED_TO_PAY", "NO_ANSWER", "REFUSED_TO_PAY"]),
        "recovered_amount": row["paid_amount"],
        "updated_at": datetime(2026, 3, 1) + timedelta(days=random.randint(1, 20))
    } for i, (_, row) in enumerate(df_repayments[df_repayments["status"].isin(["UNPAID", "LATE"])].iterrows())]
    pd.DataFrame(collections).to_csv(f"{output_dir}/fct_collections.csv", index=False)

    fraud = [{
        "fraud_id": f"FRD-{300 + i}",
        "application_id": random.choice(app_ids),
        "borrower_id": random.choice(borrower_ids),
        "fraud_type": random.choice(["IDENTITY_THEFT", "FAKE_INCOME_DOC", "DEVICE_FARMING"]),
        "risk_flag": random.choice(["CRITICAL", "HIGH", "MEDIUM"]),
        "action_taken": random.choice(["AUTO_REJECT", "FLAGGED_FOR_MANUAL_REVIEW"]),
        "detected_at": datetime(2026, 2, 1) + timedelta(days=random.randint(1, 40))
    } for i in range(45)]
    pd.DataFrame(fraud).to_csv(f"{output_dir}/fct_fraud_logs.csv", index=False)

    investors = [
        {"investor_id": "INV-01", "investor_name": "Bank Central Asia (Institutional Lender)", "investor_type": "Institutional Bank", "commitment_amount": 50000000000},
        {"investor_id": "INV-02", "investor_name": "Alpha Fintech Fund V", "investor_type": "Venture Debt", "commitment_amount": 25000000000},
        {"investor_id": "INV-03", "investor_name": "Retail Lender Syndicate", "investor_type": "Retail P2P", "commitment_amount": 10000000000},
        {"investor_id": "INV-04", "investor_name": "Mandiri Capital Indonesia", "investor_type": "Corporate CVC", "commitment_amount": 30000000000}
    ]
    pd.DataFrame(investors).to_csv(f"{output_dir}/dim_investors.csv", index=False)

    funding = [{
        "funding_id": f"FND-{9000 + i}",
        "loan_id": row["loan_id"],
        "investor_id": random.choice([inv["investor_id"] for inv in investors]),
        "funded_amount": row["principal_amount"],
        "funded_at": row["disbursed_at"] - timedelta(minutes=random.randint(5, 60))
    } for i, (_, row) in enumerate(df_loans.iterrows())]
    pd.DataFrame(funding).to_csv(f"{output_dir}/fct_loan_funding.csv", index=False)

    print(f"Generated 10 Fintech tables in {output_dir}")

if __name__ == "__main__":
    generate_pinjol_dataset()
