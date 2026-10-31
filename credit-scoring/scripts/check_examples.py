"""Validate the shipped example dataset end to end.

    python scripts/check_examples.py           # format + pipeline
    python scripts/check_examples.py --format  # format only, fast

Exits non-zero if the example data would not run, so CI can rely on it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from crf.ingest import validate_tables

EXAMPLES = ROOT / "examples"
TABLES = ("loans", "repayments", "customers", "bureau")


def load() -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(EXAMPLES / f"{name}.csv") for name in TABLES}


def show_report(tables: dict[str, pd.DataFrame]) -> bool:
    report = validate_tables(tables)
    width = max(len(c.name) for c in report.checks) + 2
    for check in report.checks:
        mark = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[check.status]
        print(f"[{mark}] {check.name:<{width}} {check.message}")
        if check.fix and check.status != "pass":
            print(f"       -> {check.fix}")
    print(f"\n{report.summary()}")
    return report.ok


def main() -> int:
    tables = load()
    print("rows:", {k: len(v) for k, v in tables.items()}, "\n")

    if not show_report(tables):
        return 1

    if "--format" in sys.argv:
        return 0

    from crf.auto_pipeline import run_auto

    print("\nrunning the full pipeline on examples/config.yaml ...")
    result = run_auto(EXAMPLES / "config.yaml", save_artifact=False)
    summary = result.to_summary()

    print(f"\nbad rate      : {summary['bad_rate']:.2%}")
    print(f"loans modelled: {summary['n_model']:,}")
    print(f"features kept : {summary['n_features']} (of {summary['n_candidates']} candidates)")
    print(f"chosen model  : {summary['chosen']} (AUC {summary['auc']:.3f})")
    print(f"reason        : {summary['reason']}")
    print("\nleaderboard:")
    print(result.tournament.leaderboard[["model", "test_auc", "test_ks", "test_calib_slope"]]
          .round(3).to_string(index=False))

    example = next((e for e in result.explanations if e is not None), None)
    if example:
        print("\nexample explanation:")
        print(example.text)

    print(f"\nwarnings: {result.warnings or 'none'}")
    print(f"total {result.seconds}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
