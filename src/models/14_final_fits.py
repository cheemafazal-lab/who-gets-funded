#!/usr/bin/env python3
"""
who-gets-funded — R012 / R013: final fits, tuned parameters, full cohort.

WHAT
  Refits both gradient boosting models on the complete 2013-2019 cohort
  (1,344,542 loans) using the hyperparameters chosen by R010/R011, and re-runs
  the same expanding-window temporal validation used for R001-R004 so that
  every number is directly comparable to the naive floors and to the untuned
  results.

  R012 replaces R003 (fractional target, cross-entropy objective).
  R013 replaces R004 (auxiliary binary model, positive class = shortfall).

WHAT DOES NOT CHANGE
  The folds, the objective, the feature set and the evaluation metrics are
  identical to R003/R004. Only the training data volume and the hyperparameters
  differ, so the comparison isolates exactly those two things.

  The B0 floors in RESULTS_LOG 4.1 were computed on the full cohort. R003/R004
  were scored on the 200k development sample, so their percentages carried a
  0.1-1.2% denominator mismatch. R012/R013 are scored on the same full cohort
  the floors came from, so that mismatch disappears here.

EARLY STOPPING
  The most recent 10% of each fold's training rows, chronologically. The 2019
  test fold is never used to stop, tune or select.

INPUT   outputs/tables/feat_v1_full.parquet  (falls back to the .csv)
        outputs/models/R010_best_params_frac.json     (optional)
        outputs/models/R011_best_params_binary.json   (optional)
OUTPUT  outputs/tables/R012_gbm_folds.csv
        outputs/tables/R012_gain_importance.csv
        outputs/tables/R012_test2019_predictions.csv
        outputs/tables/R013_binary_folds.csv
        outputs/tables/R013_test2019_scores.csv
        outputs/models/R012_gbm_frac.pkl
        outputs/models/R013_gbm_binary.pkl
        outputs/figures/F26_tuned_gain_importance.png
        outputs/figures/F27_tuned_binary_roc_pr.png

USAGE   python3 src/models/14_final_fits.py
        python3 src/models/14_final_fits.py --kind frac --untuned

Code written with the assistance of Claude (Anthropic).
"""
import argparse
import gc
import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pipeline_common import (SEED, B0, B3_ACC, FOLD_TEST_YEARS, CANON_TEST,
                             TARGET_FRAC, TARGET_BIN, CAT_COLS, feature_cols,
                             load_features, build_categories, apply_categories,
                             detect_engine, fit_gbm, gain_importance, save_bundle,
                             load_tuned, TUNED_FRAC_PATH, TUNED_BIN_PATH,
                             mem_mb, hms, rule, mae, rmse, mpl, tidy, save_fig,
                             S1, S2, DEEMPH)
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, log_loss, roc_curve,
                             precision_recall_curve)


def tuned_for(kind, untuned, modeldir):
    if untuned:
        print("  parameters: UNTUNED defaults (--untuned given)")
        return None, None
    path = os.path.join(modeldir, os.path.basename(
        TUNED_FRAC_PATH if kind == "frac" else TUNED_BIN_PATH))
    if not os.path.exists(path):
        print(f"  parameters: no tuning file at {path} — falling back to untuned "
              "defaults. Run src/models/13_tune_gbm.py first.")
        return None, None
    with open(path) as fh:
        payload = json.load(fh)
    p = dict(payload["params"])
    cap = payload.get("n_estimators_cap")
    if cap:
        p["n_estimators"] = int(cap)
    print(f"  parameters: tuned, {payload['run']} candidate "
          f"{payload['candidate']} ({payload['label']}), "
          f"selected on {payload['selected_on']}")
    print(f"              {json.dumps(payload['params'], sort_keys=True)}")
    return p, payload


# ------------------------------------------------------------- fractional ----
def run_frac(df, feats, engine, params, payload, a):
    rule("R012  fractional target, tuned, full cohort")
    rows, t0 = [], time.time()
    for ty in FOLD_TEST_YEARS:
        tr = df[df.posting_year < ty]
        te = df[df.posting_year == ty]
        cats = build_categories(tr, CAT_COLS, engine)
        Xtr, Xte = apply_categories(tr[feats], cats), apply_categories(te[feats], cats)
        t1 = time.time()
        m, best = fit_gbm(Xtr, tr[TARGET_FRAC].to_numpy(dtype=float), "frac",
                          engine, params=params)
        p = np.clip(np.asarray(m.predict(Xte), dtype=float), 0, 1)
        y = te[TARGET_FRAC].to_numpy(dtype=float)
        short = te[TARGET_BIN].to_numpy() == 0
        b0m, b0r, b0s = B0[int(ty)]
        rows.append({
            "test_year": int(ty), "train_years": f"2013-{int(ty)-1}",
            "n_train": len(tr), "n_test": len(te), "engine": engine,
            "tuned": int(params is not None), "best_iteration": best,
            "fit_seconds": round(time.time() - t1, 1),
            "mae": round(mae(y, p), 6), "b0_mae": b0m,
            "mae_vs_b0_pct": round(100 * (mae(y, p) / b0m - 1), 3),
            "rmse": round(rmse(y, p), 6), "b0_rmse": b0r,
            "rmse_vs_b0_pct": round(100 * (rmse(y, p) / b0r - 1), 3),
            "n_short": int(short.sum()),
            "mae_short": round(mae(y[short], p[short]), 6), "b0_mae_short": b0s,
            "mae_short_vs_b0_pct": round(100 * (mae(y[short], p[short]) / b0s - 1), 3),
            "mean_pred": round(float(p.mean()), 6),
            "mean_actual": round(float(y.mean()), 6)})
        print(f"  {ty}: train {len(tr):>9,}  test {len(te):>8,}  iters {best:>4}  "
              f"RMSE {rows[-1]['rmse_vs_b0_pct']:+7.2f}%  "
              f"shortfall-MAE {rows[-1]['mae_short_vs_b0_pct']:+7.2f}%  "
              f"({hms(time.time()-t1)})")
        del Xtr, Xte, m
        gc.collect()

    folds = pd.DataFrame(rows)
    folds.to_csv(os.path.join(a.out, "R012_gbm_folds.csv"), index=False)
    print(f"\n  fold means: RMSE {folds.rmse_vs_b0_pct.mean():+.2f}%  "
          f"shortfall-MAE {folds.mae_short_vs_b0_pct.mean():+.2f}%  vs B0")

    rule("R012  canonical model (train 2013-2018) for SHAP and counterfactuals")
    tr = df[df.posting_year < CANON_TEST]
    te = df[df.posting_year == CANON_TEST]
    cats = build_categories(tr, CAT_COLS, engine)
    t1 = time.time()
    m, best = fit_gbm(apply_categories(tr[feats], cats),
                      tr[TARGET_FRAC].to_numpy(dtype=float), "frac", engine,
                      params=params)
    print(f"  fitted on {len(tr):,} rows in {hms(time.time()-t1)} "
          f"(best iteration {best})")
    bundle = {"run": "R012", "kind": "frac", "engine": engine, "feat_cols": feats,
              "cat_cols": CAT_COLS, "categories": cats, "train_span": "2013-2018",
              "best_iteration": best, "seed": SEED, "tuned": params is not None,
              "tuning": payload, "model": m}
    save_bundle(bundle, os.path.join(a.modeldir, "R012_gbm_frac.pkl"))

    p = np.clip(np.asarray(m.predict(apply_categories(te[feats], cats)),
                           dtype=float), 0, 1)
    pred = te[["loan_id", "posting_year", "gender", "country_iso", "sector"]].copy()
    pred["y_share"] = te[TARGET_FRAC].to_numpy(dtype=float)
    pred["y_short"] = (te[TARGET_BIN].to_numpy() == 0).astype(int)
    pred["pred_share"] = np.round(p, 6)
    pred.to_csv(os.path.join(a.out, "R012_test2019_predictions.csv"), index=False)
    print(f"  test-2019 predictions -> R012_test2019_predictions.csv ({len(pred):,} rows)")

    gi = gain_importance(bundle).sort_values("gain", ascending=False)
    gi["gain_share_pct"] = (100 * gi.gain / gi.gain.sum()).round(3)
    gi.to_csv(os.path.join(a.out, "R012_gain_importance.csv"), index=False)

    plt = mpl()
    top = gi.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.barh(range(len(top)), top.gain_share_pct, height=0.5, color=S1, zorder=3)
    ax.set_yticks(range(len(top)), top.feature, fontsize=8)
    ax.set_xlabel("Share of total split gain (%)")
    tidy(ax, xgrid=True)
    ax.set_title("F26  Gain importance, tuned model on the full cohort, top 15\n"
                 f"Train 2013-2018 ({len(tr):,} loans), engine {engine}",
                 fontsize=9.5, loc="left", pad=8)
    save_fig(fig, a.figdir, "F26_tuned_gain_importance.png",
             "Gain importance, tuned full-cohort fractional model")
    del m, bundle
    gc.collect()


# ----------------------------------------------------------------- binary ----
def run_binary(df, feats, engine, params, payload, a):
    rule("R013  auxiliary binary model, tuned, full cohort")
    rows, t0 = [], time.time()
    for ty in FOLD_TEST_YEARS:
        tr = df[df.posting_year < ty]
        te = df[df.posting_year == ty]
        cats = build_categories(tr, CAT_COLS, engine)
        Xtr, Xte = apply_categories(tr[feats], cats), apply_categories(te[feats], cats)
        t1 = time.time()
        m, best = fit_gbm(Xtr, tr[TARGET_BIN].to_numpy(dtype=int), "binary",
                          engine, params=params)
        p_funded = (m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba")
                    else np.clip(m.predict(Xte), 0, 1))
        p_short = 1.0 - np.asarray(p_funded, dtype=float)
        y_short = (te[TARGET_BIN].to_numpy() == 0).astype(int)
        pred_short = (p_short >= 0.5).astype(int)
        tp = int(((pred_short == 1) & (y_short == 1)).sum())
        fp = int(((pred_short == 1) & (y_short == 0)).sum())
        rows.append({
            "test_year": int(ty), "train_years": f"2013-{int(ty)-1}",
            "n_train": len(tr), "n_test": len(te), "engine": engine,
            "tuned": int(params is not None), "best_iteration": best,
            "fit_seconds": round(time.time() - t1, 1),
            "auc": round(roc_auc_score(y_short, p_short), 6),
            "pr_auc_short": round(average_precision_score(y_short, p_short), 6),
            "pr_auc_floor_prevalence": round(float(y_short.mean()), 6),
            "brier": round(brier_score_loss(y_short, p_short), 6),
            "log_loss": round(log_loss(y_short, np.clip(p_short, 1e-15, 1-1e-15)), 6),
            "recall_short_at_0.5": round(tp / max(1, int(y_short.sum())), 6),
            "precision_short_at_0.5": round(tp / max(1, tp + fp), 6),
            "accuracy_pct": round(100 * float((pred_short == y_short).mean()), 4),
            "b3_accuracy_pct": B3_ACC[int(ty)],
            "n_short": int(y_short.sum())})
        print(f"  {ty}: train {len(tr):>9,}  AUC {rows[-1]['auc']:.4f}  "
              f"PR-AUC {rows[-1]['pr_auc_short']:.4f} "
              f"(floor {rows[-1]['pr_auc_floor_prevalence']:.4f})  "
              f"({hms(time.time()-t1)})")
        del Xtr, Xte, m
        gc.collect()

    folds = pd.DataFrame(rows)
    folds.to_csv(os.path.join(a.out, "R013_binary_folds.csv"), index=False)
    print(f"\n  fold means: AUC {folds.auc.mean():.4f}  "
          f"PR-AUC {folds.pr_auc_short.mean():.4f}")

    rule("R013  canonical model (train 2013-2018) for the counterfactual step")
    tr = df[df.posting_year < CANON_TEST]
    te = df[df.posting_year == CANON_TEST]
    cats = build_categories(tr, CAT_COLS, engine)
    t1 = time.time()
    m, best = fit_gbm(apply_categories(tr[feats], cats),
                      tr[TARGET_BIN].to_numpy(dtype=int), "binary", engine,
                      params=params)
    print(f"  fitted on {len(tr):,} rows in {hms(time.time()-t1)} "
          f"(best iteration {best})")
    bundle = {"run": "R013", "kind": "binary", "engine": engine, "feat_cols": feats,
              "cat_cols": CAT_COLS, "categories": cats, "train_span": "2013-2018",
              "best_iteration": best, "seed": SEED, "tuned": params is not None,
              "tuning": payload, "model": m}
    save_bundle(bundle, os.path.join(a.modeldir, "R013_gbm_binary.pkl"))

    Xte = apply_categories(te[feats], cats)
    p_funded = (m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba")
                else np.clip(m.predict(Xte), 0, 1))
    sc = te[["loan_id", "posting_year", "gender", "country_iso", "sector"]].copy()
    sc["y_short"] = (te[TARGET_BIN].to_numpy() == 0).astype(int)
    sc["p_funded"] = np.round(np.asarray(p_funded, dtype=float), 6)
    sc["p_short"] = np.round(1 - np.asarray(p_funded, dtype=float), 6)
    sc.to_csv(os.path.join(a.out, "R013_test2019_scores.csv"), index=False)
    print(f"  test-2019 scores -> R013_test2019_scores.csv ({len(sc):,} rows)")

    y_short, p_short = sc.y_short.to_numpy(), sc.p_short.to_numpy()
    fpr, tpr, _ = roc_curve(y_short, p_short)
    prec, rec, _ = precision_recall_curve(y_short, p_short)
    plt = mpl()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    ax = axes[0]
    ax.plot(fpr, tpr, color=S1, lw=2, zorder=4)
    ax.plot([0, 1], [0, 1], color=DEEMPH, lw=1.2, zorder=2)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    tidy(ax)
    ax.set_title(f"ROC — AUC {roc_auc_score(y_short, p_short):.4f}\n"
                 "grey diagonal = the naive floor, 0.5", fontsize=9, loc="left", pad=6)
    ax = axes[1]
    ax.plot(rec, prec, color=S2, lw=2, zorder=4)
    prev = float(y_short.mean())
    ax.axhline(prev, color=DEEMPH, lw=1.2, zorder=2)
    ax.text(0.02, prev + 0.015, f"prevalence floor {prev:.3f}", fontsize=7.5)
    ax.set_xlabel("Recall (shortfall)"); ax.set_ylabel("Precision (shortfall)")
    ax.set_ylim(0, 1.02)
    tidy(ax)
    ax.set_title(f"Precision-recall — PR-AUC {average_precision_score(y_short, p_short):.4f}\n"
                 "positive class = shortfall", fontsize=9, loc="left", pad=6)
    fig.suptitle("F27  Tuned auxiliary binary model, full cohort, test 2019",
                 x=0.005, ha="left", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, a.figdir, "F27_tuned_binary_roc_pr.png",
             "Tuned binary aux model ROC and PR curves, full cohort, test 2019")
    del m, bundle
    gc.collect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/tables/feat_v1_full.csv")
    ap.add_argument("--kind", choices=["frac", "binary", "both"], default="both")
    ap.add_argument("--untuned", action="store_true",
                    help="ignore the tuning files and use the R003/R004 defaults")
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--figdir", default="outputs/figures")
    ap.add_argument("--modeldir", default="outputs/models")
    a = ap.parse_args()

    engine = detect_engine()
    rule("who-gets-funded  R012 / R013  final fits on the full cohort")
    print(f"  engine   : {engine}")
    print(f"  features : {a.features}")
    t0 = time.time()
    df = load_features(a.features)
    feats = feature_cols(df)
    print(f"  {len(df):,} rows, {len(feats)} features, {mem_mb(df):,.0f} MB "
          f"in memory, loaded in {hms(time.time()-t0)}")
    yrs = sorted(df.posting_year.unique().tolist())
    print(f"  posting years present: {yrs[0]}-{yrs[-1]}")
    if len(df) < 1_000_000:
        print("  !! this looks like the development sample, not the full cohort. "
              "Pass --features outputs/tables/feat_v1_full.csv for the final fits.")

    os.makedirs(a.out, exist_ok=True)
    if a.kind in ("frac", "both"):
        p, pl = tuned_for("frac", a.untuned, a.modeldir)
        run_frac(df, feats, engine, p, pl, a)
    if a.kind in ("binary", "both"):
        p, pl = tuned_for("binary", a.untuned, a.modeldir)
        run_binary(df, feats, engine, p, pl, a)

    rule("R012 / R013 complete")
    print("  next: steps 09 (SHAP), 10 (counterfactuals), 11 (recourse), "
          "15 (group calibration), 12 (comparison)")


if __name__ == "__main__":
    main()
