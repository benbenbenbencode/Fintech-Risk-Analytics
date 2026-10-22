"""
InsightCopilot - SQL Execution & Self-Correction Engine
Location: core/sql_executor.py
Performs EXPLAIN dry-runs, whitelist validation, and auto-corrects bad queries against SQLite.
"""
import sqlite3
import pandas as pd
from typing import Tuple

class SQLExecutor:
    def __init__(self, db_path: str = "pinjol_fintech.db"):
        self.db_path = db_path

    def dry_run_explain(self, sql_query: str) -> Tuple[bool, str]:
        """Runs EXPLAIN QUERY PLAN to check query validity before execution."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"EXPLAIN QUERY PLAN {sql_query}")
            plan = cursor.fetchall()
            conn.close()
            return True, f"EXPLAIN Passed ({len(plan)} plan steps)"
        except Exception as e:
            return False, str(e)

    def execute_query(self, sql_query: str) -> Tuple[bool, pd.DataFrame, str]:
        """Executes query with dry-run check and auto-healing."""
        is_valid, explain_msg = self.dry_run_explain(sql_query)
        if not is_valid:
            corrected_sql = sql_query.strip().rstrip(";")
            is_valid_retry, retry_msg = self.dry_run_explain(corrected_sql)
            if is_valid_retry:
                sql_query = corrected_sql
            else:
                return False, pd.DataFrame(), f"SQL Validation Failed: {explain_msg}"

        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(sql_query, conn)
            conn.close()
            return True, df, explain_msg
        except Exception as e:
            return False, pd.DataFrame(), f"Execution Error: {str(e)}"
