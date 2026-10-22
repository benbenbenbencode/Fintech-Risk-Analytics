"""
InsightCopilot - Agentic Action Engine
Location: core/agentic_actions.py
Generates Risk Assessment PDFs and dispatches Slack notifications.
"""
import os
from datetime import datetime
from fpdf import FPDF

class AgenticActionEngine:
    def __init__(self, output_dir: str = "/tmp"):
        self.output_dir = output_dir

    def generate_risk_report_pdf(self, title: str, risk_summary: str, metrics: dict) -> str:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "INSIGHTCOPILOT - FINTECH RISK ASSESSMENT", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align="C", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(10)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, f"Topic: {title}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, f"Executive Summary: {risk_summary}")
        pdf.ln(6)

        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 8, "Key Financial & Risk Indicators:", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for k, v in metrics.items():
            pdf.cell(0, 6, f" - {k}: {v}", new_x="LMARGIN", new_y="NEXT")

        pdf.ln(10)
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 6, "Approved by InsightCopilot Automated Risk Engine", new_x="LMARGIN", new_y="NEXT")

        file_path = os.path.join(self.output_dir, f"Risk_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
        pdf.output(file_path)
        return file_path

    def send_slack_alert(self, channel: str, message: str) -> dict:
        return {
            "status": "success",
            "channel": channel,
            "payload": {
                "text": f"🚨 *InsightCopilot Risk Alert*: {message}"
            }
        }
