#!/usr/bin/env python3
"""
who-gets-funded — orchestrator for the tuned, full-cohort stage (runs R010-R018).

WHAT IT RUNS, in order
  03d  convert the feature matrices to Parquet          (once, ~2 min)
  R010 tune the fractional model, 40 draws              |
  R011 tune the binary model, 40 draws                  |  the long part
  R012 final fractional fit, full cohort, tuned         |
  R013 final binary fit, full cohort, tuned             |
  R014 SHAP on the tuned model, 50,000 rows
  R015 counterfactual search, base = loans that actually fell short,
       target p_funded 0.975 (the 25th percentile of loans that did fund)
  R016 fairness of recourse, both readings
  R017 group calibration and error rates by group
  R018 final comparison folder

  Every step is runnable on its own; this only saves typing and guarantees the
  order. Steps are skipped, not repeated, when their outputs already exist,
  unless --force is given, so an interrupted run can simply be restarted.

WHAT IT NEVER TOUCHES
  The database. The 2019 test fold during tuning. The R001 naive floors. The
  existing R001-R009 outputs, which keep their own filenames throughout.

USAGE   python3 src/run_full_pipeline.py
        python3 src/run_full_pipeline.py --n-draws 20 --skip 03d
        python3 src/run_full_pipeline.py --only R012,R013
        python3 src/run_full_pipeline.py --dry-run

Code written with the assistance of Claude (Anthropic).
"""
import argparse
import os
import subprocess
import sys
import time

FULL = "outputs/tables/feat_v1_full.csv"


def steps(a):
    f = ["--features", a.features]
    return [
        ("03d", "feature matrices to Parquet",
         [sys.executable, "src/features/03d_to_parquet.py"],
         "outputs/tables/feat_v1_full.parquet"),

        ("R010", "tune the fractional model",
         [sys.executable, "src/models/13_tune_gbm.py", "--kind", "frac",
          "--n-draws", str(a.n_draws), "--resume"] + f,
         "outputs/models/R010_best_params_frac.json"),

        ("R011", "tune the binary model",
         [sys.executable, "src/models/13_tune_gbm.py", "--kind", "binary",
          "--n-draws", str(a.n_draws), "--resume"] + f,
         "outputs/models/R011_best_params_binary.json"),

        ("R012", "final fractional fit, full cohort",
         [sys.executable, "src/models/14_final_fits.py", "--kind", "frac"] + f,
         "outputs/models/R012_gbm_frac.pkl"),

        ("R013", "final binary fit, full cohort",
         [sys.executable, "src/models/14_final_fits.py", "--kind", "binary"] + f,
         "outputs/models/R013_gbm_binary.pkl"),

        ("R014", "SHAP on the tuned model",
         [sys.executable, "src/explain/09_shap_analysis.py",
          "--model-run", "R012", "--out-id", "R014",
          "--sample-n", str(a.shap_n)] + f,
         "outputs/tables/R014_shap_global.csv"),

        ("R015", "counterfactual search, actual-short base",
         [sys.executable, "src/explain/10_counterfactuals.py",
          "--model-run", "R013", "--out-id", "R015",
          "--base", "actual-short", "--threshold", str(a.cf_threshold),
          "--max-changes", str(a.cf_max_changes),
          "--n-instances", str(a.cf_n)] + f,
         "outputs/tables/R015_counterfactuals.csv"),

        ("R016", "fairness of recourse",
         [sys.executable, "src/equity/11_recourse_audit.py",
          "--cf-id", "R015", "--out-id", "R016"],
         "outputs/tables/R016_recourse_by_gender.csv"),

        ("R017", "group calibration and error rates",
         [sys.executable, "src/equity/15_group_calibration.py",
          "--pred-id", "R012", "--score-id", "R013", "--out-id", "R017"],
         "outputs/tables/R017_calibration_by_group.csv"),

        ("R018", "final comparison folder",
         [sys.executable, "src/models/16_compare_final.py"],
         "outputs/final_comparison/C2_headline.csv"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=FULL)
    ap.add_argument("--n-draws", type=int, default=40)
    ap.add_argument("--shap-n", type=int, default=50000)
    ap.add_argument("--cf-n", type=int, default=0, help="0 = every loan in the base")
    ap.add_argument("--cf-threshold", type=float, default=0.975,
                    help="target p_funded that counts as recourse achieved. "
                         "0.975 is the lower quartile of loans that actually "
                         "funded, so the target is anchored in the data rather "
                         "than set at an arbitrary cutoff. The 0.5 default is "
                         "unusable here: 93.5%% of loans that actually fell "
                         "short already score above it.")
    ap.add_argument("--cf-max-changes", type=int, default=3)
    ap.add_argument("--only", default=None, help="comma-separated step ids")
    ap.add_argument("--skip", default="", help="comma-separated step ids")
    ap.add_argument("--force", action="store_true", help="re-run completed steps")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not os.path.exists("src/pipeline_common.py"):
        sys.exit("run this from the repository root")

    only = set(a.only.split(",")) if a.only else None
    skip = set(x for x in a.skip.split(",") if x)
    plan = steps(a)

    print("=" * 90)
    print("who-gets-funded — tuned full-cohort pipeline")
    print("=" * 90)
    for sid, name, cmd, sentinel in plan:
        state = "skip" if sid in skip or (only and sid not in only) else (
            "done" if os.path.exists(sentinel) and not a.force else "run")
        print(f"  {sid:<5} {name:<45} {state}")
    print()

    if a.dry_run:
        for sid, name, cmd, _ in plan:
            print(f"  {sid}: {' '.join(cmd)}")
        return

    t_all = time.time()
    for sid, name, cmd, sentinel in plan:
        if sid in skip or (only and sid not in only):
            continue
        if os.path.exists(sentinel) and not a.force:
            print(f"\n>>> {sid} already complete ({sentinel}) — skipping. "
                  f"Use --force to re-run.")
            continue
        print("\n" + "=" * 90)
        print(f">>> {sid}  {name}")
        print(f"    {' '.join(cmd)}")
        print("=" * 90, flush=True)
        t0 = time.time()
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"\n!!! {sid} failed with exit code {r.returncode}. "
                  f"Nothing after it has run. Fix, then restart this script — "
                  f"completed steps are skipped automatically.")
            sys.exit(r.returncode)
        print(f"\n<<< {sid} finished in {time.time()-t0:,.0f}s")

    print("\n" + "=" * 90)
    print(f"ALL STEPS COMPLETE in {(time.time()-t_all)/60:,.1f} min")
    print("  outputs/final_comparison/  holds the assembled comparison")
    print("=" * 90)


if __name__ == "__main__":
    main()
