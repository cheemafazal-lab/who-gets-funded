#!/usr/bin/env python3
# Code written with the assistance of Claude (Anthropic).
"""
who-gets-funded — pipeline orchestrator: R003 → R009 in one command.

Runs each step as its own process (every step remains independently runnable and
independently readable for the report), streams all output live, mirrors it to a
timestamped log, and shows an overall step bar. A step failure stops the run and
says exactly which script to rerun. The out-of-regime step skips itself politely
if its feature file has not been built yet.

USAGE   python3 src/run_pipeline.py                # dev sample, full defaults
        python3 src/run_pipeline.py --quick        # smoke test (small settings)
        python3 src/run_pipeline.py --features outputs/tables/feat_v1_full.csv
        python3 src/run_pipeline.py --only 09 10 11
        python3 src/run_pipeline.py --skip 08
"""
import argparse, datetime, os, subprocess, sys, time

BAR_W = 34
STEPS = [
    ("06", "src/models/06_gbm_temporal_cv.py",  "R003 GBM fractional, temporal CV"),
    ("07", "src/models/07_gbm_binary_aux.py",   "R004 binary aux model"),
    ("08", "src/models/08_oot_regime_test.py",  "R005 out-of-regime test"),
    ("09", "src/explain/09_shap_analysis.py",   "R006 SHAP"),
    ("10", "src/explain/10_counterfactuals.py", "R007 counterfactuals"),
    ("11", "src/equity/11_recourse_audit.py",   "R008 recourse audit"),
    ("12", "src/models/12_compare_models.py",   "R009 comparison folder"),
]


def hms(s):
    s = int(s)
    return f"{s}s" if s < 60 else (f"{s//60}m {s%60:02d}s" if s < 3600
                                   else f"{s//3600}h {(s%3600)//60:02d}m")


def master_bar(done, total, t0):
    fill = int(BAR_W * done / total)
    sys.stdout.write(f"\n  PIPELINE [{'#'*fill}{'.'*(BAR_W-fill)}] "
                     f"step {done}/{total} complete  ({hms(time.time()-t0)} total)\n")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/tables/feat_v1_dev200k.csv")
    ap.add_argument("--oot-features", default="outputs/tables/feat_oot_2020plus.csv")
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: 300 trees, 5k SHAP rows, 300 CF instances")
    ap.add_argument("--only", nargs="*", default=None, help="run only these step ids")
    ap.add_argument("--skip", nargs="*", default=[], help="skip these step ids")
    a = ap.parse_args()

    if not os.path.exists("src/run_pipeline.py"):
        sys.exit("run this from the repo root:  cd path/to/who-gets-funded")
    if not os.path.exists(a.features):
        sys.exit(f"missing {a.features} — run src/features/03b_make_dev_sample.py first")

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    logp = f"outputs/tables/pipeline_log_{stamp}.txt"
    os.makedirs("outputs/tables", exist_ok=True)

    steps = [s for s in STEPS if (a.only is None or s[0] in a.only)
             and s[0] not in a.skip]
    print("=" * 78)
    print("who-gets-funded  model pipeline")
    print("=" * 78)
    print(f"  features : {a.features}")
    print(f"  log      : {logp}")
    print(f"  steps    : {', '.join(s[0] for s in steps)}"
          + ("   [--quick smoke-test settings]" if a.quick else ""))
    if not os.path.exists(a.oot_features) and any(s[0] == "08" for s in steps):
        print(f"  note     : {a.oot_features} not found — step 08 will skip itself;")
        print("             build it with: python3 src/features/03c_build_oot_features.py")

    t0 = time.time()
    with open(logp, "w", encoding="utf8") as log:
        for k, (sid, script, label) in enumerate(steps, 1):
            hdr = (f"\n{'#'*78}\n# STEP {sid}  {label}\n"
                   f"# {script}\n{'#'*78}\n")
            print(hdr, end="")
            log.write(hdr)
            cmd = [sys.executable, script]
            if sid in ("06", "07", "09", "10"):
                cmd += ["--features", a.features]
            if sid == "08":
                cmd += ["--oot-features", a.oot_features]
            if a.quick:
                cmd += {"06": ["--n-estimators", "300"],
                        "07": ["--n-estimators", "300"],
                        "09": ["--sample-n", "5000"],
                        "10": ["--n-instances", "300"]}.get(sid, [])
            t1 = time.time()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
            while True:
                chunk = proc.stdout.read(1024)
                if not chunk:
                    break
                sys.stdout.buffer.write(chunk)
                sys.stdout.flush()
                log.write(chunk.decode("utf8", errors="replace"))
            rc = proc.wait()
            log.write(f"\n[step {sid} exit {rc}, {hms(time.time()-t1)}]\n")
            if rc != 0:
                print(f"\n!! STEP {sid} FAILED (exit {rc}). Fix and rerun just it:\n"
                      f"   python3 {script}\n"
                      f"   then continue with: python3 src/run_pipeline.py --only "
                      + " ".join(s[0] for s in steps[k:]))
                sys.exit(rc)
            master_bar(k, len(steps), t0)

    print(f"\nPIPELINE COMPLETE in {hms(time.time()-t0)}")
    print(f"  everything is in outputs/model_comparison/   (log: {logp})")


if __name__ == "__main__":
    main()
