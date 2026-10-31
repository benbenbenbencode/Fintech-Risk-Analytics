"""Smoke test for the automatic pipeline.

Generates the demo loan book if needed, runs the full auto pipeline, and prints
what the system discovered, what it chose, and why.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    sys.path.insert(0, str(ROOT))

    if not (ROOT / "data" / "loans.csv").exists():
        from generate_sample_data import generate

        print("generating demo data ...")
        generate(6000, out_dir=str(ROOT / "data"))

    from crf.auto_pipeline import run_auto

    result = run_auto(ROOT / "config.yaml", save_artifact=False)
    summary = result.to_summary()

    print("\n=== DISCOVERED ===")
    print(f"tables: {[p.name for p in result.profile.values()]}")
    print(f"relationships: {result.relationships}")
    print(f"features kept: {len(result.featureset.X.columns)}")
    for feature in list(result.featureset.X.columns)[:12]:
        print(f"  {feature:42} {result.featureset.types.get(feature):12} {result.featureset.provenance.get(feature)}")
    print(f"columns rejected: {len(result.featureset.dropped)}")
    for row in result.featureset.dropped.head(8).itertuples():
        print(f"  - {row.column:36} {row.reason}")

    print("\n=== TARGET ===")
    ts = result.target_summary
    print(f"bad rate {ts['bad_rate']:.2%} | model rows {ts['n_model']} | "
          f"train {ts['n_train']} / test {ts['n_test']} | DPD>={ts['bad_dpd']} in {ts['perf']}m")

    print("\n=== TOURNAMENT ===")
    cols = ["model", "label", "test_auc", "test_ks", "test_brier", "test_calib_slope",
            "test_top_decile_lift", "seconds"]
    print(result.tournament.leaderboard[cols].to_string(index=False))
    print(f"\nCHOSEN: {summary['chosen']} — {summary['reason']}")

    print("\n=== TOP FEATURES (permutation importance) ===")
    importance = result.tournament.importance
    print(importance.head(10)[["feature", "importance", "relative"]].to_string(index=False))

    print("\n=== EXPLANATION SAMPLE ===")
    example = next((e for e in result.explanations if e is not None), None)
    if example:
        print(example.text)
    else:
        print("no explanation produced")

    print("\n=== FAIRNESS ===")
    fairness = result.fairness
    print(f"protected attributes detected: {fairness.get('protected')}")
    print(f"approval cut-off PD: {fairness.get('cutoff')}, approval rate {fairness.get('approval_rate')}")
    for group, table in (fairness.get("results") or {}).items():
        print(f"\n[{group}] DI ratio {table['di_ratio'].iloc[0]:.3f}")
        print(table[["group", "applicants", "approval_rate", "mean_pd"]].to_string(index=False))
    for flag in fairness.get("flags", []):
        print("FLAG:", flag)
    if not fairness.get("flags"):
        print("no adverse-impact flags")

    print("\n=== ON-DEMAND EXPLAIN ===")
    test_id = result.tournament.test_ids[0]
    one = result.explain(test_id, language="id")
    print(f"loan {test_id}: PD {one.pd:.3%} score {one.score} band {one.band}")
    print(one.text)

    print("\nwarnings:", result.warnings or "none")
    print(f"total {result.seconds}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
