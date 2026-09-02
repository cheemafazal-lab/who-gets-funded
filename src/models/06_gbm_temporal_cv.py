#!/usr/bin/env python3
# Code written with the assistance of Claude (Anthropic).
"""
who-gets-funded — R003: gradient boosting on the fractional target, temporal CV.

WHAT
  The main predictive model. Gradient boosting with a cross-entropy objective on
  share_raised in [0,1] — the GBM analogue of the fractional logit's QMLE. Same
  expanding-window folds as R001/R002 (train 2013..N-1, test N for N=2016..2019),
  so every number is directly comparable to the naive floors and the logit.

DEFAULTS (deliberately untuned — tuning is a later, separate stage)
  2,000 trees, learning rate 0.05, 63 leaves, min 100 rows per leaf, 0.9 row and
  column subsampling, early stopping after 100 rounds against a chronological
  validation slice (the most recent 10% of the training rows).

INPUT   outputs/tables/feat_v1_dev200k.csv   (--features .../feat_v1_full.csv)
OUTPUT  outputs/tables/R003_gbm_folds.csv          fold metrics vs B0 and R002
        outputs/tables/R003_gain_importance.csv    gain importance, final model
        outputs/tables/R003_test2019_predictions.csv
        outputs/models/R003_gbm_frac.pkl           canonical model, train<=2018
        outputs/figures/F13_gbm_gain_importance.png

USAGE   python3 src/models/06_gbm_temporal_cv.py [--features PATH] [--n-estimators N]
"""
import argparse, os, sys, time, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pipeline_common import (SEED, B0, FOLD_TEST_YEARS, CANON_TEST, TARGET_FRAC,
                             TARGET_BIN, CAT_COLS, feature_cols, load_features,
                             build_categories, apply_categories, detect_engine,
                             fit_gbm, gain_importance, save_bundle, bar, hms,
                             rule, mae, rmse, mpl, tidy, save_fig, S1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/tables/feat_v1_dev200k.csv")
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--figdir", default="outputs/figures")
    ap.add_argument("--modeldir", default="outputs/models")
    ap.add_argument("--n-estimators", type=int, default=None)
    a = ap.parse_args()

    engine = detect_engine()
    rule("who-gets-funded  R003 gradient boosting, fractional target, temporal CV")
    print(f"  engine   : {engine}"
          + ("   !! sklearn fallback uses squared error, not cross-entropy — "
             "prefer `pip3 install lightgbm`" if engine == "sklearn" else ""))
    print(f"  features : {a.features}")

    df = load_features(a.features)
    feats = feature_cols(df)
    print(f"  {len(df):,} rows, {len(feats)} features "
          f"({len(CAT_COLS)} categorical incl. partner_id and activity)")

    # comparison column from R002, if its folds file exists
    r2 = {}
    r2_path = os.path.join(a.out, "R002_logit_folds.csv")
    if os.path.exists(r2_path):
        for _, r in pd.read_csv(r2_path).iterrows():
            r2[int(r.test_year)] = float(r.mae)

    rows, t0 = [], time.time()
    rule(f"Step 1 of 2  temporal cross-validation ({len(FOLD_TEST_YEARS)} folds)")
    for i, ty in enumerate(FOLD_TEST_YEARS, 1):
        tr = df[df.posting_year < ty]
        te = df[df.posting_year == ty]
        bar(i - 1, len(FOLD_TEST_YEARS), t0, f"{ty}: fitting on {len(tr):,}")
        cats = build_categories(tr, CAT_COLS, engine)
        Xtr = apply_categories(tr[feats], cats)
        Xte = apply_categories(te[feats], cats)
        t1 = time.time()
        m, best = fit_gbm(Xtr, tr[TARGET_FRAC].values, "frac", engine,
                          a.n_estimators)
        p = np.clip(np.asarray(m.predict(Xte), dtype=float), 0, 1)
        y = te[TARGET_FRAC].values
        short = te[TARGET_BIN].values == 0
        b0m, b0r, b0s = B0[int(ty)]
        rows.append({
            "test_year": int(ty), "train_years": f"2013-{int(ty)-1}",
            "n_train": len(tr), "n_test": len(te), "engine": engine,
            "best_iteration": best, "fit_seconds": round(time.time() - t1, 1),
            "mae": round(mae(y, p), 6), "b0_mae": b0m,
            "mae_vs_b0_pct": round(100 * (mae(y, p) / b0m - 1), 3),
            "r002_mae": r2.get(int(ty), np.nan),
            "rmse": round(rmse(y, p), 6), "b0_rmse": b0r,
            "rmse_vs_b0_pct": round(100 * (rmse(y, p) / b0r - 1), 3),
            "n_short": int(short.sum()),
            "mae_short": round(mae(y[short], p[short]), 6), "b0_mae_short": b0s,
            "mae_short_vs_b0_pct": round(100 * (mae(y[short], p[short]) / b0s - 1), 3),
            "mean_pred": round(float(p.mean()), 6),
            "mean_actual": round(float(y.mean()), 6)})
        bar(i, len(FOLD_TEST_YEARS), t0, f"{ty}: done")
    print()
    folds = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    folds.to_csv(os.path.join(a.out, "R003_gbm_folds.csv"), index=False)
    show = ["test_year", "n_train", "best_iteration", "mae", "b0_mae",
            "mae_vs_b0_pct", "r002_mae", "rmse", "rmse_vs_b0_pct",
            "mae_short", "b0_mae_short", "mae_short_vs_b0_pct"]
    print(folds[show].to_string(index=False))
    print("\n  negative *_vs_b0_pct beats the naive floor; r002_mae is the logit.")

    rule("Step 2 of 2  canonical model (train<=2018) for SHAP / counterfactuals")
    tr = df[df.posting_year < CANON_TEST]
    te = df[df.posting_year == CANON_TEST]
    cats = build_categories(tr, CAT_COLS, engine)
    t1 = time.time()
    m, best = fit_gbm(apply_categories(tr[feats], cats), tr[TARGET_FRAC].values,
                      "frac", engine, a.n_estimators)
    print(f"  fitted in {hms(time.time()-t1)}  (best iteration {best})")
    bundle = {"run": "R003", "kind": "frac", "engine": engine, "feat_cols": feats,
              "cat_cols": CAT_COLS, "categories": cats, "train_span": "2013-2018",
              "best_iteration": best, "seed": SEED}
    bundle["model"] = m
    save_bundle(bundle, os.path.join(a.modeldir, "R003_gbm_frac.pkl"))

    p = np.clip(np.asarray(m.predict(apply_categories(te[feats], cats)),
                           dtype=float), 0, 1)
    pred = te[["loan_id", "posting_year", "gender", "country_iso", "sector"]].copy()
    pred["y_share"] = te[TARGET_FRAC].values
    pred["y_short"] = (te[TARGET_BIN].values == 0).astype(int)
    pred["pred_share"] = np.round(p, 6)
    pred.to_csv(os.path.join(a.out, "R003_test2019_predictions.csv"), index=False)
    print(f"  test-2019 predictions -> R003_test2019_predictions.csv ({len(pred):,} rows)")

    gi = gain_importance(bundle).sort_values("gain", ascending=False)
    gi["gain_share_pct"] = (100 * gi.gain / gi.gain.sum()).round(3)
    gi.to_csv(os.path.join(a.out, "R003_gain_importance.csv"), index=False)

    plt = mpl()
    top = gi.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.barh(range(len(top)), top.gain_share_pct, height=0.5, color=S1, zorder=3)
    ax.set_yticks(range(len(top)), top.feature, fontsize=8)
    ax.set_xlabel("Share of total split gain (%)")
    tidy(ax, xgrid=True)
    ax.set_title("F13  What the gradient boosting model uses — gain importance, top 15\n"
                 f"Canonical model, train 2013–2018, engine {engine} (untuned defaults)",
                 fontsize=9.5, loc="left", pad=8)
    save_fig(fig, a.figdir, "F13_gbm_gain_importance.png",
             "GBM gain importance, top 15, canonical model")

    rule("R003 complete")
    print("  record in RESULTS_LOG §4; comparison figures come from step 12")


if __name__ == "__main__":
    main()
