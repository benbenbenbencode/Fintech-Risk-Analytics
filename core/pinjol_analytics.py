"""
InsightCopilot - Fintech Analytics & OJK KPI Engine
Location: core/pinjol_analytics.py
Computes TWP90, TKB90, NPL, and anomaly insights from SQLite database.
"""
import sqlite3
import pandas as pd
from typing import Dict, Any

class PinjolAnalyticsEngine:
    def __init__(self, db_path: str = "pinjol_fintech.db"):
        self.db_path = db_path

    def get_ojk_kpis(self) -> Dict[str, Any]:
        conn = sqlite3.connect(self.db_path)
        df_loans = pd.read_sql_query("SELECT loan_status, principal_amount FROM fct_loans", conn)
        total_disbursed = df_loans["principal_amount"].sum()
        
        defaulted_df = df_loans[df_loans["loan_status"] == "DEFAULTED_TWP90"]
        defaulted_amt = defaulted_df["principal_amount"].sum()
        
        twp90_rate = (defaulted_amt / total_disbursed * 100) if total_disbursed > 0 else 0.0
        tkb90_rate = 100.0 - twp90_rate

        conn.close()
        return {
            "total_disbursed_idr": total_disbursed,
            "twp90_pct": round(twp90_rate, 2),
            "tkb90_pct": round(tkb90_rate, 2),
            "total_loans": len(df_loans),
            "defaulted_loans": len(defaulted_df)
        }
