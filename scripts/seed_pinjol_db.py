"""
InsightCopilot - Database Seeder
Location: scripts/seed_pinjol_db.py
Seeds generated CSV datasets into SQLite `pinjol_fintech.db`.
"""
import os
import sys
import sqlite3
import pandas as pd

# Allow importing from scripts directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate_pinjol_data import generate_pinjol_dataset

def seed_database(db_path="pinjol_fintech.db", csv_dir="pinjol_csvs"):
    print("🚀 Generating synthetic datasets...")
    generate_pinjol_dataset(csv_dir)

    print(f"📦 Ingesting CSVs into SQLite `{db_path}`...")
    conn = sqlite3.connect(db_path)

    tables = [
        "dim_borrowers", "dim_partners", "fct_loan_applications",
        "fct_loans", "fct_repayments", "fct_credit_assessments",
        "fct_collections", "fct_fraud_logs", "dim_investors", "fct_loan_funding"
    ]

    for t in tables:
        csv_file = os.path.join(csv_dir, f"{t}.csv")
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            df.to_sql(t, conn, if_exists="replace", index=False)
            print(f"  ✓ Seeded `{t}` ({len(df)} rows)")

    conn.close()
    print("✅ Database seeding complete!")

if __name__ == "__main__":
    seed_database()
