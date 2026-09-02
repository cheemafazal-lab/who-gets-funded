#!/usr/bin/env python3
# Code written with the assistance of Claude (Anthropic).
"""
who-gets-funded — R005: out-of-regime test on 2020-2026 loans.

WHAT
  Applies the FROZEN canonical models (trained 2013-2018, never refitted) to the
  post-flexible-funding era. This is a transportability finding, not a validation
  failure: Kiva changed the funding rule on 31 July 2019 (RESULTS_LOG §16), so
  degradation here measures how far a pre-change model transports across a
  documented platform regime change. share_raised may also have changed meaning
  after 2019 (open question 10) — printed as a caveat on every run.

INPUT   outputs/tables/feat_oot_2020plus.csv  (built by src/features/03c_build_oot_features.py,
        which needs the database — run it in the console first)
        outputs/models/R003_gbm_frac.pkl, R004_gbm_binary.pkl
OUTPUT  outputs/tables/R005_oot_by_year.csv
        outputs/figures/F15_oot_degradation.png

If the OOT feature file is missing this step SKIPS politely (exit 0) so the rest
of the pipeline can run without it.

USAGE   python3 src/models/08_oot_regime_test.py [--oot-features PATH]
"""
import argparse, os, sys, time, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pipeline_common import (TARGET_FRAC, TARGET_BIN, load_bundle, predict_frac,
                             predict_p_funded, bar, hms, rule, mae, rmse, mpl,
                             tidy, save_fig, S1, S2, DEEMPH)
from sklearn.metrics import roc_auc_score, brier_score_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oot-features", default="outputs/tables/feat_oot_2020plus.csv")
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--figdir", default="outputs/figures")
    ap.add_argument("--modeldir", default="outputs/models")
    a = ap.parse_args()

    rule("who-gets-funded  R005 out-of-regime test (2020+, frozen 2013-2018 models)")
    if not os.path.exists(a.oot_features):
        print(f"  SKIPPED — {a.oot_features} not found.")
        print("  Build it first (needs the database, run in your console):")
        print("    python3 src/features/03c_build_oot_features.py")
        return

    print("  ⚠ caveat (RESULTS_LOG §16 / open question 10): after Kiva's flexible-")
    print("    funding change of 31 Jul 2019, share_raised may record partner-topped")
    print("    amounts, so the target may not mean what it meant in the cohort.")

    df = pd.read_csv(a.oot_features, low_memory=False)
    print(f"  {len(df):,} out-of-regime rows, years "
          f"{int(df.posting_year.min())}-{int(df.posting_year.max())}")
    bf = load_bundle(os.path.join(a.modeldir, "R003_gbm_frac.pkl"))
    bb = load_bundle(os.path.join(a.modeldir, "R004_gbm_binary.pkl"))
    print(f"  models: R003 ({bf['engine']}), R004 ({bb['engine']}), both train 2013-2018, frozen")

    years = sorted(df.posting_year.unique())
    rows, t0 = [], time.time()
    for i, yr in enumerate(years, 1):
        bar(i - 1, len(years), t0, f"scoring {yr}")
        d = df[df.posting_year == yr]
        y = d[TARGET_FRAC].values
        y_short = (d[TARGET_BIN].values == 0).astype(int)
        p = predict_frac(bf, d)
        p_short = 1.0 - predict_p_funded(bb, d)
        naive = 1.0 - float(y.mean())
        short = y_short == 1
        row = {"posting_year": int(yr), "n": len(d), "n_short": int(short.sum()),
               "share_mean": round(float(y.mean()), 6),
               "naive_b0_mae": round(naive, 6),
               "model_mae": round(mae(y, p), 6),
               "mae_vs_b0_pct": round(100 * (mae(y, p) / max(naive, 1e-9) - 1), 2),
               "model_rmse": round(rmse(y, p), 6),
               "mae_short": round(mae(y[short], p[short]), 6) if short.any() else np.nan,
               "mean_pred_share": round(float(p.mean()), 6),
               "mean_pred_p_short": round(float(p_short.mean()), 6),
               "actual_short_rate": round(float(y_short.mean()), 6),
               "auc": (round(roc_auc_score(y_short, p_short), 6)
                       if 0 < y_short.sum() < len(y_short) else np.nan),
               "brier": round(brier_score_loss(y_short, p_short), 6)}
        rows.append(row)
        bar(i, len(years), t0, f"{yr} done")
    print()
    out = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    out.to_csv(os.path.join(a.out, "R005_oot_by_year.csv"), index=False)
    print(out.to_string(index=False))
    print("\n  read mean_pred_p_short vs actual_short_rate: the frozen model keeps")
    print("  predicting cohort-era shortfall into an era where the category has")
    print("  almost disappeared — that over-prediction IS the regime finding.")

    plt = mpl()
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.0), sharex=True)
    ax = axes[0]
    ax.plot(out.posting_year, out.model_mae, "-o", color=S1, lw=2, ms=7,
            markeredgecolor="#fcfcfb", markeredgewidth=1.5, label="Frozen model MAE", zorder=4)
    ax.plot(out.posting_year, out.naive_b0_mae, "-o", color=DEEMPH, lw=2, ms=7,
            markeredgecolor="#fcfcfb", markeredgewidth=1.5,
            label="Naive B0 floor (that year)", zorder=3)
    ax.set_ylabel("MAE on share_raised")
    tidy(ax)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("F15  Out-of-regime: a 2013–2018 model applied to the flexible-funding era\n"
                 "Top: fractional MAE vs the per-year naive floor. Bottom: predicted vs actual shortfall rate.",
                 fontsize=9.5, loc="left", pad=8)
    ax = axes[1]
    ax.plot(out.posting_year, 100 * out.mean_pred_p_short, "-o", color=S2, lw=2,
            ms=7, markeredgecolor="#fcfcfb", markeredgewidth=1.5,
            label="Mean predicted P(shortfall)", zorder=4)
    ax.plot(out.posting_year, 100 * out.actual_short_rate, "-o", color=S1, lw=2,
            ms=7, markeredgecolor="#fcfcfb", markeredgewidth=1.5,
            label="Actual shortfall rate", zorder=4)
    ax.set_ylabel("Per cent of loans")
    ax.set_xlabel("Posting year")
    ax.set_xticks(out.posting_year)
    tidy(ax)
    ax.legend(loc="upper right", fontsize=8)
    save_fig(fig, a.figdir, "F15_oot_degradation.png",
             "Out-of-regime degradation by year, frozen 2013-2018 models")

    rule("R005 complete")


if __name__ == "__main__":
    main()
