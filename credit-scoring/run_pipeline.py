"""CLI:  python run_pipeline.py --config config.yaml --out outputs"""
import argparse
from pathlib import Path

from crf.pipeline import run_pipeline

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--out", default="outputs")
args = ap.parse_args()

res = run_pipeline(args.config)
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
res["iv_table"].to_csv(out / "iv_table.csv", index=False)
res["leakage"].to_csv(out / "leakage_report.csv", index=False)
res["points_table"].to_csv(out / "scorecard_points.csv", index=False)
res["metrics"].to_csv(out / "metrics.csv", index=False)
res["deciles"]["test"].to_csv(out / "lift_test.csv", index=False)
res["feature_psi"].to_csv(out / "feature_psi.csv", index=False)
res["scored"].to_csv(out / "scored_loans.csv")
print(res["metrics"].round(3).to_string(index=False))
print("selected:", res["selected"])
print("score PSI (train vs test): %.3f" % res["score_psi"])
print("written to", out.resolve())
