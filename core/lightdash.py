"""
InsightCopilot - Lightdash Dashboard Generator
Location: core/lightdash.py
Auto-generates Lightdash YAML configurations from dbt models & query insights.
"""
import yaml
from typing import Dict, Any, List

class LightdashGenerator:
    def __init__(self, project_name: str = "pinjol-dbt-analytics"):
        self.project_name = project_name

    def generate_yaml(self, dashboard_name: str, summary: str, charts: List[Dict[str, Any]]) -> str:
        tiles = [
            {
                "title": "📌 AI Risk Summary",
                "type": "markdown",
                "properties": {"content": f"### {dashboard_name}\n**Context**: {summary}"},
                "x": 0, "y": 0, "w": 12, "h": 2
            }
        ]
        for idx, chart in enumerate(charts):
            tiles.append({
                "title": chart.get("title", f"Chart {idx+1}"),
                "type": "saved_chart",
                "properties": {
                    "chartConfig": {"type": chart.get("type", "column"), "fieldConfig": {"x": chart["x"], "y": [chart["y"]]}},
                    "metricQuery": {"exploreName": chart["table"], "dimensions": [chart["x"]], "metrics": [chart["y"]]}
                },
                "x": (idx % 2) * 6,
                "y": 2 + (idx // 2) * 4,
                "w": 6,
                "h": 4
            })

        dashboard_config = {
            "name": dashboard_name,
            "description": summary,
            "spaceUuid": "pinjol-risk-space",
            "tiles": tiles
        }
        return yaml.dump(dashboard_config, sort_keys=False, default_flow_style=False)
