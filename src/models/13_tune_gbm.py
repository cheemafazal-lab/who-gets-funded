#!/usr/bin/env python3
"""
who-gets-funded — R010 / R011: hyperparameter search for the gradient boosting models.

WHAT
  R003 and R004 were fitted on fixed, deliberately untuned defaults. This step
  chooses their hyperparameters by random search on the full cohort.

VALIDATION DESIGN — the 2019 test fold is never loaded
  Only posting years 2013-2018 are read from disk. Two inner folds, both
  expanding and both chronological:

      inner fold 1:  train 2013-2016   validate 2017
      inner fold 2:  train 2013-2017   validate 2018

  The validation year doubles as the early-stopping set, which is what keeps the
  boundary clean: tree count and hyperparameters are both chosen using training
  years only. The 2019 fold and the R001 per-fold naive floors in RESULTS_LOG
  4.1 are untouched by this step and remain a genuine held-out test.

SELECTION METRIC
  Fractional model : the mean, across both inner folds, of the composite score —
                     half the percentage improvement over the B0 naive floor on
                     RMSE plus half the percentage improvement on shortfall-MAE.
                     Lower is better. Raw MAE is excluded by design: the median
                     of share_raised is exactly 1.0, so B0 wins raw MAE by
                     construction and selecting on it rewards a model that
                     ignores shortfall (RESULTS_LOG 4.1).
  Binary model     : mean PR-AUC on the shortfall class, the rare and
                     interesting one. Never accuracy — B3 reaches 93-96%
                     accuracy at 0% recall.

  The B0 floor for an inner validation year is recomputed on that year's slice,
  because the R001 table covers the 2016-2019 outer folds only.

BUDGET AND RESUMABILITY
  Candidate 0 is always the untuned R003/R004 default, so the sweep reports
  whether tuning earned anything at all. Every candidate is appended to the log
  CSV the moment it finishes, so an interrupted run loses at most one candidate
  and --resume picks up where it stopped.

INPUT   outputs/tables/feat_v1_full.parquet  (falls back to the .csv)
OUTPUT  outputs/tables/R010_tuning_frac.csv      every candidate, fractional
        outputs/tables/R011_tuning_binary.csv    every candidate, binary
        outputs/models/R010_best_params_frac.json
        outputs/models/R011_best_params_binary.json
        outputs/figures/F25_tuning_traces.png

USAGE   python3 src/models/13_tune_gbm.py --kind frac   --n-draws 40
        python3 src/models/13_tune_gbm.py --kind binary --n-draws 40
        python3 src/models/13_tune_gbm.py --kind both   --n-draws 40 --resume

Code written with the assistance of Claude (Anthropic).
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pipeline_common import (SEED, INNER_VAL_YEARS, TARGET_FRAC, TARGET_BIN,
                             CAT_COLS, GBM_PARAMS, feature_cols, load_features,
                             build_categories, apply_categories, detect_engine,
                             fit_gbm, sample_params, frac_metrics, composite_score,
                             b0_floor_from, mem_mb, hms, rule, mpl, tidy,
                             save_fig, S1, S2, DEEMPH, BASELINE)
from sklearn.metrics import average_precision_score, roc_auc_score

LOG = {"frac": "R010_tuning_frac.csv", "binary": "R011_tuning_binary.csv"}
BEST = {"frac": "R010_best_params_frac.json", "binary": "R011_best_params_binary.json"}
RUN = {"frac": "R010", "binary": "R011"}


# --------------------------------------------------------------- fold prep ---
def build_folds(df, feats, engine):
    """Materialise each inner fold once. Category levels and the design matrices
    do not depend on the hyperparameters, so building them here rather than
    inside the candidate loop saves the bulk of the wall-clock."""
    folds = []
    for vy in INNER_VAL_YEARS:
        tr = df[df.posting_year < vy]
        va = df[df.posting_year == vy]
        if len(tr) == 0 or len(va) == 0:
            sys.exit(f"inner fold for validation year {vy} is empty — check the cohort")
        cats = build_categories(tr, CAT_COLS, engine)
        f = {"val_year": int(vy),
             "Xtr": apply_categories(tr[feats], cats),
             "Xva": apply_categories(va[feats], cats),
             "y_frac_tr": tr[TARGET_FRAC].to_numpy(dtype=float),
             "y_frac_va": va[TARGET_FRAC].to_numpy(dtype=float),
             "y_bin_tr": tr[TARGET_BIN].to_numpy(dtype=int),
             "y_bin_va": va[TARGET_BIN].to_numpy(dtype=int),
             "n_train": len(tr), "n_val": len(va)}
        f["short_va"] = f["y_bin_va"] == 0
        f["floor"] = b0_floor_from(f["y_frac_va"])
        folds.append(f)
        print(f"  inner fold: train 2013-{vy-1} ({len(tr):,})  validate {vy} "
              f"({len(va):,}, {int(f['short_va'].sum()):,} short, "
              f"B0 RMSE {f['floor'][1]:.6f} / shortfall-MAE {f['floor'][2]:.6f})")
    return folds


# ------------------------------------------------------------- evaluation ----
def evaluate(params, folds, kind, engine, seed=SEED):
    """Fit one candidate on both inner folds. Returns a flat record."""
    rec, per_fold, iters, secs = {}, [], [], []
    for f in folds:
        t0 = time.time()
        ytr = f["y_frac_tr"] if kind == "frac" else f["y_bin_tr"]
        yva = f["y_frac_va"] if kind == "frac" else f["y_bin_va"]
        m, best = fit_gbm(f["Xtr"], ytr, kind, engine, params=params,
                          valid=(f["Xva"], yva), seed=seed)
        secs.append(time.time() - t0)
        iters.append(best)

        if kind == "frac":
            p = np.clip(np.asarray(m.predict(f["Xva"]), dtype=float), 0, 1)
            mets = frac_metrics(f["y_frac_va"], p, f["short_va"])
            score = composite_score(mets, f["floor"])
            per_fold.append(score)
            rec[f"f{f['val_year']}_mae"] = round(mets["mae"], 6)
            rec[f"f{f['val_year']}_rmse"] = round(mets["rmse"], 6)
            rec[f"f{f['val_year']}_mae_short"] = round(mets["mae_short"], 6)
            rec[f"f{f['val_year']}_score"] = round(score, 4)
        else:
            p_funded = (m.predict_proba(f["Xva"])[:, 1] if hasattr(m, "predict_proba")
                        else np.clip(m.predict(f["Xva"]), 0, 1))
            p_short = 1.0 - np.asarray(p_funded, dtype=float)
            y_short = (f["y_bin_va"] == 0).astype(int)
            pr = average_precision_score(y_short, p_short)
            per_fold.append(pr)
            rec[f"f{f['val_year']}_pr_auc"] = round(pr, 6)
            rec[f"f{f['val_year']}_auc"] = round(roc_auc_score(y_short, p_short), 6)
            rec[f"f{f['val_year']}_prevalence"] = round(float(y_short.mean()), 6)
        del m

    rec["score"] = round(float(np.mean(per_fold)), 6)
    rec["best_iterations"] = "|".join(str(i) for i in iters)
    rec["fit_seconds"] = round(sum(secs), 1)
    return rec


def better(a, b, kind):
    """Is score a better than score b? Composite is lower-is-better, PR-AUC is
    higher-is-better."""
    if b is None:
        return True
    return a < b if kind == "frac" else a > b


# ------------------------------------------------------------------ sweep ----
def sweep(kind, df, feats, engine, a):
    rule(f"who-gets-funded  {RUN[kind]}  hyperparameter search — {kind} model")
    print(f"  inner validation years : {INNER_VAL_YEARS} (2019 never loaded)")
    print(f"  selection metric       : "
          + ("composite, mean of RMSE and shortfall-MAE improvement vs B0 "
             "(lower is better)" if kind == "frac"
             else "mean PR-AUC on the shortfall class (higher is better)"))
    folds = build_folds(df, feats, engine)
    print(f"  fold matrices in memory: "
          f"{sum(mem_mb(f['Xtr']) + mem_mb(f['Xva']) for f in folds):,.0f} MB")

    log_path = os.path.join(a.out, LOG[kind])
    done, best_score, best_rec = [], None, None
    if a.resume and os.path.exists(log_path):
        prev = pd.read_csv(log_path)
        done = prev.candidate.tolist()
        for _, r in prev.iterrows():
            if better(r["score"], best_score, kind):
                best_score, best_rec = r["score"], r.to_dict()
        print(f"  resuming: {len(done)} candidate(s) already logged, "
              f"best score so far {best_score}")

    rng = np.random.default_rng(a.seed)
    # Draw the whole plan up front so --resume reproduces the identical sequence.
    plan = [("untuned default", dict(GBM_PARAMS))]
    plan[0][1].pop("early_stopping_rounds", None)
    plan[0][1]["n_estimators"] = 3000
    for i in range(1, a.n_draws + 1):
        plan.append((f"random {i}", sample_params(rng, kind)))

    t_start = time.time()
    for cand, (label, params) in enumerate(plan):
        if cand in done:
            continue
        if a.time_budget and (time.time() - t_start) / 60 > a.time_budget:
            print(f"\n  time budget of {a.time_budget} min reached — stopping at "
                  f"candidate {cand}. Re-run with --resume to continue.")
            break

        rec = {"candidate": cand, "label": label, "kind": kind, "engine": engine}
        rec.update(evaluate(params, folds, kind, engine))
        rec["params_json"] = json.dumps(params, sort_keys=True)

        row = pd.DataFrame([rec])
        row.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)

        flag = ""
        if better(rec["score"], best_score, kind):
            best_score, best_rec = rec["score"], rec
            flag = "  <-- best so far"
        el = time.time() - t_start
        eta = (len(plan) - cand - 1) * el / max(1, cand + 1 - len(done))
        print(f"  [{cand:>3}/{len(plan)-1}] {label:<16} score {rec['score']:>10.4f}"
              f"  iters {rec['best_iterations']:<10} {rec['fit_seconds']:>6.1f}s"
              f"  elapsed {hms(el)} ETA {hms(eta)}{flag}")

    if best_rec is None:
        sys.exit("  no candidate completed — nothing to write")

    # ------------------------------------------------------------ winner ----
    params = json.loads(best_rec["params_json"])
    iters = [int(x) for x in str(best_rec["best_iterations"]).split("|")]
    # Refit on the full training span uses the largest inner-fold tree count as
    # the cap, with early stopping still active, so the final model is never
    # forced to the sweep's tree count.
    payload = {"run": RUN[kind], "kind": kind, "engine": engine,
               "selected_on": ("composite RMSE + shortfall-MAE vs B0"
                               if kind == "frac" else "PR-AUC on shortfall"),
               "inner_val_years": INNER_VAL_YEARS,
               "score": float(best_rec["score"]),
               "candidate": int(best_rec["candidate"]),
               "label": best_rec["label"],
               "best_iterations_inner": iters,
               "n_estimators_cap": int(max(iters) * 2),
               "seed": a.seed, "params": params}
    bp = os.path.join(a.modeldir, BEST[kind])
    os.makedirs(a.modeldir, exist_ok=True)
    with open(bp, "w") as fh:
        json.dump(payload, fh, indent=2)

    untuned = pd.read_csv(log_path)
    u = untuned[untuned.candidate == 0]
    print(f"\n  WINNER  candidate {best_rec['candidate']} ({best_rec['label']})"
          f"  score {best_rec['score']:.4f}")
    if len(u):
        d = float(u.iloc[0]["score"])
        verdict = ("tuning improves on the untuned default"
                   if better(best_rec["score"], d, kind)
                   else "TUNING DID NOT BEAT THE UNTUNED DEFAULT — keep R003/R004 params")
        print(f"  untuned default score {d:.4f}  ->  {verdict}")
    print(f"  parameters -> {bp}")
    return log_path


# ----------------------------------------------------------------- figure ----
def trace_figure(a):
    plt = mpl()
    have = [(k, os.path.join(a.out, LOG[k])) for k in ("frac", "binary")
            if os.path.exists(os.path.join(a.out, LOG[k]))]
    if not have:
        return
    fig, axes = plt.subplots(1, len(have), figsize=(3.9 * len(have), 3.2),
                             squeeze=False)
    for ax, (kind, path) in zip(axes[0], have):
        d = pd.read_csv(path).sort_values("candidate")
        run = np.minimum.accumulate(d.score) if kind == "frac" \
            else np.maximum.accumulate(d.score)
        ax.plot(d.candidate, d.score, marker="o", ms=3, lw=0, color=DEEMPH,
                label="candidate", zorder=3)
        ax.plot(d.candidate, run, lw=2, color=S1 if kind == "frac" else S2,
                label="best so far", zorder=4)
        u = d[d.candidate == 0]
        if len(u):
            ax.axhline(float(u.iloc[0]["score"]), color=BASELINE, lw=1.2,
                       ls="--", zorder=2)
            ax.text(0.02, 0.04, "dashed = untuned default", transform=ax.transAxes,
                    fontsize=7.2, color=BASELINE)
        ax.set_xlabel("Candidate")
        ax.set_ylabel("Composite score (lower better)" if kind == "frac"
                      else "PR-AUC, shortfall (higher better)")
        tidy(ax)
        ax.legend(fontsize=7.5)
        ax.set_title(f"{RUN[kind]} — {'fractional' if kind=='frac' else 'binary aux'}",
                     fontsize=9, loc="left", pad=6)
    fig.suptitle("F25  Hyperparameter search, validated inside the training years only",
                 x=0.005, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, a.figdir, "F25_tuning_traces.png",
             "Random search traces for the fractional and binary models")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/tables/feat_v1_full.csv")
    ap.add_argument("--kind", choices=["frac", "binary", "both"], default="both")
    ap.add_argument("--n-draws", type=int, default=40)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--time-budget", type=float, default=None,
                    help="minutes; stop cleanly and allow --resume")
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--figdir", default="outputs/figures")
    ap.add_argument("--modeldir", default="outputs/models")
    a = ap.parse_args()

    engine = detect_engine()
    if engine != "lightgbm":
        print("  !! engine is not lightgbm — the search space is written for it. "
              "Install lightgbm before tuning (pip3 install lightgbm).")

    rule("who-gets-funded  tuning stage — loading training years only")
    train_years = list(range(2013, max(INNER_VAL_YEARS) + 1))
    print(f"  features : {a.features}")
    print(f"  years    : {train_years[0]}-{train_years[-1]}  "
          f"(2019 is the held-out test fold and is not read)")
    t0 = time.time()
    df = load_features(a.features, years=train_years)
    feats = feature_cols(df)
    print(f"  {len(df):,} rows, {len(feats)} features, "
          f"{mem_mb(df):,.0f} MB in memory, loaded in {hms(time.time()-t0)}")
    assert df.posting_year.max() < 2019, "2019 leaked into the tuning frame"

    os.makedirs(a.out, exist_ok=True)
    for kind in (["frac", "binary"] if a.kind == "both" else [a.kind]):
        sweep(kind, df, feats, engine, a)
    trace_figure(a)

    rule("tuning complete")
    print("  next: python3 src/models/14_final_fits.py --features "
          "outputs/tables/feat_v1_full.csv")


if __name__ == "__main__":
    main()
