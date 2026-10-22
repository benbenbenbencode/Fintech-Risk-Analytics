"""
InsightCopilot - Graph-Based Auto-Join Engine
Location: core/graph_join_engine.py
Uses NetworkX to find shortest schema paths between tables based on Foreign Key topology.
"""
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional

class GraphJoinEngine:
    def __init__(self):
        self.graph = nx.Graph()
        self._build_pinjol_schema_graph()

    def _build_pinjol_schema_graph(self):
        edges = [
            ("dim_borrowers", "fct_loan_applications", {"on": "dim_borrowers.borrower_id = fct_loan_applications.borrower_id", "left": "borrower_id", "right": "borrower_id"}),
            ("dim_partners", "fct_loan_applications", {"on": "dim_partners.partner_id = fct_loan_applications.partner_id", "left": "partner_id", "right": "partner_id"}),
            ("fct_loan_applications", "fct_loans", {"on": "fct_loan_applications.application_id = fct_loans.application_id", "left": "application_id", "right": "application_id"}),
            ("fct_loans", "fct_repayments", {"on": "fct_loans.loan_id = fct_repayments.loan_id", "left": "loan_id", "right": "loan_id"}),
            ("fct_loans", "fct_collections", {"on": "fct_loans.loan_id = fct_collections.loan_id", "left": "loan_id", "right": "loan_id"}),
            ("fct_loans", "fct_loan_funding", {"on": "fct_loans.loan_id = fct_loan_funding.loan_id", "left": "loan_id", "right": "loan_id"}),
            ("dim_investors", "fct_loan_funding", {"on": "dim_investors.investor_id = fct_loan_funding.investor_id", "left": "investor_id", "right": "investor_id"}),
            ("fct_loan_applications", "fct_credit_assessments", {"on": "fct_loan_applications.application_id = fct_credit_assessments.application_id", "left": "application_id", "right": "application_id"}),
            ("fct_loan_applications", "fct_fraud_logs", {"on": "fct_loan_applications.application_id = fct_fraud_logs.application_id", "left": "application_id", "right": "application_id"}),
            ("dim_borrowers", "fct_loans", {"on": "dim_borrowers.borrower_id = fct_loans.borrower_id", "left": "borrower_id", "right": "borrower_id"}),
        ]
        for u, v, data in edges:
            self.graph.add_edge(u, v, **data)

    def generate_join_clause(self, selected_tables: List[str]) -> Tuple[str, List[str]]:
        if not selected_tables:
            return "", []
        if len(selected_tables) == 1:
            return f"FROM {selected_tables[0]}", selected_tables

        primary_table = selected_tables[0]
        visited_nodes = {primary_table}
        join_clauses = [f"FROM {primary_table}"]

        for target in selected_tables[1:]:
            if target in visited_nodes:
                continue
            shortest_path = None
            min_len = float("inf")

            for start_node in list(visited_nodes):
                if nx.has_path(self.graph, start_node, target):
                    path = nx.shortest_path(self.graph, start_node, target)
                    if len(path) < min_len:
                        min_len = len(path)
                        shortest_path = path

            if shortest_path:
                for i in range(len(shortest_path) - 1):
                    u = shortest_path[i]
                    v = shortest_path[i+1]
                    if v not in visited_nodes:
                        edge_data = self.graph.get_edge_data(u, v)
                        join_clauses.append(f"LEFT JOIN {v} ON {edge_data['on']}")
                        visited_nodes.add(v)

        return " ".join(join_clauses), list(visited_nodes)
