#!/usr/bin/env python3
# Code written with the assistance of Claude (Anthropic).
"""
who-gets-funded — R004: auxiliary binary model (fully funded vs not), temporal CV.

WHAT
  Exists solely to feed the counterfactual tooling, which is classification-
  oriented (supervisor feedback point 3). It does not replace the fractional
  model. Positive class throughout = SHORTFALL (is_fully_funded == 0): the rare,
  interesting class, so PR-AUC is meaningful and its naive floor is the
  shortfall prevalence. Accuracy is reported only to show why it must never be
  quoted alone (B3 reaches 93-96% accuracy with 0% recall).

INPUT   outputs/tables/feat_v1_dev200k.csv   (--features .../feat_v1_full.csv)
OUTPUT  outputs/tables/R004_binary_folds.csv
        outputs/tables/R004_test2019_scores.csv    per-loan p(funded), for step 10
        outputs/models/R004_gbm_binary.pkl         canonical model, train<=2018
        outputs/figures/F14_binary_roc_pr.png

USAGE   python3 src/models/07_gbm_binary_aux.py [--features PATH] [--n-estimators N]
"""
import argparse, os, sys, time, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pipeline_common import (SEED, B3_ACC, FOLD_TEST_YEARS, CANON_TEST,
                             TARGET_BIN, CAT_COLS, feature_cols, load_features,
                             build_categories, apply_categories, detect_engine,
                             fit_gbm, save_bundle, bar, hms, rule, mpl, tidy,
                             save_fig, S1, S2, DEEMPH, BASELINE)
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss,
                             log_loss, roc_curve, precision_recall_curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/tables/feat_v1_dev200k.csv")
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--figdir", default="outputs/figures")
    ap.add_argument("--modeldir", default="outputs/models")
    ap.add_argument("--n-estimators", type=int, default=None)
    a = ap.parse_args()

    engine = detect_engine()
    rule("who-gets-funded  R004 auxiliary binary model (for counterfactuals only)")
    print(f"  engine   : {engine}\n  features : {a.features}")
    print("  positive class = SHORTFALL (is_fully_funded == 0)")

    df = load_features(a.features)
    feats = feature_cols(df)
    print(f"  {len(df):,} rows, {len(feats)} features")

    rows, t0 = [], time.time()
    rule(f"Step 1 of 2  temporal cross-validation ({len(FOLD_TEST_YEARS)} folds)")
    for i, ty in enumerate(FOLD_TEST_YEARS, 1):
        tr = df[df.posting_year < ty]
        te = df[df.posting_year == ty]
        bar(i - 1, len(FOLD_TEST_YEARS), t0, f"{ty}: fitting on {len(tr):,}")
        cats = build_categories(tr, CAT_COLS, engine)
        t1 = time.time()
        m, best = fit_gbm(apply_categories(tr[feats], cats),
                          tr[TARGET_BIN].values, "binary", engine, a.n_estimators)
        Xte = apply_categories(te[feats], cats)
        p_funded = (m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba")
                    else np.clip(m.predict(Xte), 0, 1))
        y_short = (te[TARGET_BIN].values == 0).astype(int)
        p_short = 1.0 - p_funded
        prev = float(y_short.mean())
        pred_short = (p_short >= 0.5).astype(int)
        tp = int(((pred_short == 1) & (y_short == 1)).sum())
        fp = int(((pred_short == 1) & (y_short == 0)).sum())
        rows.append({
            "test_year": int(ty), "train_years": f"2013-{int(ty)-1}",
            "n_train": len(tr), "n_test": len(te), "engine": engine,
            "best_iteration": best, "fit_seconds": round(time.time() - t1, 1),
            "auc": round(roc_auc_score(y_short, p_short), 6),
            "pr_auc_short": round(average_precision_score(y_short, p_short), 6),
            "pr_auc_floor_prevalence": round(prev, 6),
            "brier": round(brier_score_loss(y_short, p_short), 6),
            "log_loss": round(log_loss(y_short, np.clip(p_short, 1e-15, 1 - 1e-15)), 6),
            "recall_short_at_0.5": round(tp / max(1, int(y_short.sum())), 6),
            "precision_short_at_0.5": round(tp / max(1, tp + fp), 6),
            "accuracy_pct": round(100 * float((pred_short == y_short).mean()), 4),
            "b3_accuracy_pct": B3_ACC[int(ty)],
            "n_short": int(y_short.sum())})
        bar(i, len(FOLD_TEST_YEARS), t0, f"{ty}: done")
    print()
    folds = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    folds.to_csv(os.path.join(a.out, "R004_binary_folds.csv"), index=False)
    show = ["test_year", "auc", "pr_auc_short", "pr_auc_floor_prevalence", "brier",
            "recall_short_at_0.5", "precision_short_at_0.5", "accuracy_pct",
            "b3_accuracy_pct"]
    print(folds[show].to_string(index=False))
    print("\n  floors: AUC 0.5, PR-AUC = prevalence, B3 accuracy with 0% recall.")

    rule("Step 2 of 2  canonical model (train<=2018) for the counterfactual step")
    tr = df[df.posting_year < CANON_TEST]
    te = df[df.posting_year == CANON_TEST]
    cats = build_categories(tr, CAT_COLS, engine)
    t1 = time.time()
    m, best = fit_gbm(apply_categories(tr[feats], cats), tr[TARGET_BIN].values,
                      "binary", engine, a.n_estimators)
    print(f"  fitted in {hms(time.time()-t1)}  (best iteration {best})")
    bundle = {"run": "R004", "kind": "binary", "engine": engine, "feat_cols": feats,
              "cat_cols": CAT_COLS, "categories": cats, "train_span": "2013-2018",
              "best_iteration": best, "seed": SEED, "model": m}
    save_bundle(bundle, os.path.join(a.modeldir, "R004_gbm_binary.pkl"))

    Xte = apply_categories(te[feats], cats)
    p_funded = (m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba")
                else np.clip(m.predict(Xte), 0, 1))
    sc = te[["loan_id", "posting_year", "gender", "country_iso", "sector"]].copy()
    sc["y_short"] = (te[TARGET_BIN].values == 0).astype(int)
    sc["p_funded"] = np.round(p_funded, 6)
    sc["p_short"] = np.round(1 - p_funded, 6)
    sc.to_csv(os.path.join(a.out, "R004_test2019_scores.csv"), index=False)
    print(f"  test-2019 scores -> R004_test2019_scores.csv ({len(sc):,} rows)")

    y_short = sc.y_short.values
    p_short = sc.p_short.values
    fpr, tpr, _ = roc_curve(y_short, p_short)
    prec, rec, _ = precision_recall_curve(y_short, p_short)
    plt = mpl()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2))
    ax = axes[0]
    ax.plot(fpr, tpr, color=S1, lw=2, zorder=4)
    ax.plot([0, 1], [0, 1], color=DEEMPH, lw=1.2, zorder=2)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    tidy(ax)
    ax.set_title(f"ROC — AUC {roc_auc_score(y_short, p_short):.4f}\n"
                 "(grey diagonal = the naive floor, 0.5)",
                 fontsize=9, loc="left", pad=6)
    ax = axes[1]
    ax.plot(rec, prec, color=S2, lw=2, zorder=4)
    prev = float(y_short.mean())
    ax.axhline(prev, color=DEEMPH, lw=1.2, zorder=2)
    ax.text(0.02, prev + 0.015, f"prevalence floor {prev:.3f}", fontsize=7.5)
    ax.set_xlabel("Recall (shortfall)")
    ax.set_ylabel("Precision (shortfall)")
    ax.set_ylim(0, 1.02)
    tidy(ax)
    ax.set_title(f"Precision–recall — PR-AUC {average_precision_score(y_short, p_short):.4f}\n"
                 "positive class = shortfall",
                 fontsize=9, loc="left", pad=6)
    fig.suptitle("F14  Auxiliary binary model, test 2019", x=0.005, ha="left",
                 fontsize=9.5, color="#0b0b0b")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, a.figdir, "F14_binary_roc_pr.png",
             "Binary aux model ROC and PR curves, test 2019")

    rule("R004 complete")


if __name__ == "__main__":
    main()
