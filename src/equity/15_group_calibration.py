#!/usr/bin/env python3
"""
who-gets-funded — R017: group calibration and error rates by borrower group.

WHY THIS STEP EXISTS
  The equity audit had two of its three layers. Layer 1, market outcomes, was
  settled by the descriptive and composition work (RESULTS_LOG 5.1): the gender
  gap survives standardisation at 2.5-3.3x. Layer 3, fairness of recourse, is
  the novel contribution. Layer 2 — whether the MODEL treats groups differently,
  as opposed to whether the market does — was never computed. This step closes
  it, and with it the amber status on the equity audit.

  The distinction matters. A model can reproduce a real disparity faithfully and
  still be well calibrated within every group; or it can be well calibrated
  overall while systematically over-predicting failure for one group. Only the
  second is a property of the model rather than of the platform.

WHAT IS COMPUTED, per group and for the cohort as a whole
  Calibration   mean predicted shortfall probability against the observed
                shortfall rate, in deciles of predicted risk; expected
                calibration error; and the calibration intercept and slope from
                a logistic regression of the outcome on the predicted logit. A
                slope below 1 means the model's probabilities are too spread
                out for that group, above 1 too compressed.
  Error rates   at two operating points — the default 0.5 cutoff, and a
                prevalence-matched top-K% cut, which is the decision an analyst
                triaging a queue would actually make. False positive rate, false
                negative rate, recall and precision on the shortfall class, so
                the comparison is an equalised-odds style reading rather than an
                accuracy one.
  Fractional    MAE, RMSE, shortfall-MAE and the mean SIGNED error by group. The
                signed error is the one that shows directional bias: a positive
                value means the model over-predicts the share a group will
                raise.

  Groups: gender, region, sector, and the largest countries. Gender is the
  headline because it is where the disparity was established; the others are
  reported so the claim is not resting on a single split.

INPUT   outputs/tables/R012_test2019_predictions.csv   (fractional)
        outputs/tables/R013_test2019_scores.csv        (binary)
OUTPUT  outputs/tables/R017_calibration_by_group.csv
        outputs/tables/R017_calibration_bins.csv
        outputs/tables/R017_error_rates_by_group.csv
        outputs/tables/R017_fractional_error_by_group.csv
        outputs/figures/F30_calibration_by_gender.png
        outputs/figures/F31_error_rates_by_group.png

USAGE   python3 src/equity/15_group_calibration.py
        python3 src/equity/15_group_calibration.py --pred-id R003 --score-id R004

Code written with the assistance of Claude (Anthropic).

Note on statistical caution: these are point estimates. Group differences are
reported as measured, without significance tests, in the same spirit as the
composition check in RESULTS_LOG 5.1. Confidence intervals remain open
question 8.
"""
import argparse
import os
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pipeline_common import (rule, mae, rmse, mpl, tidy, save_fig,
                             S1, S2, DEEMPH, BASELINE, INK, MUTED)

EPS = 1e-9


# ------------------------------------------------------------- calibration ---
def calibration(y, p, n_bins=10):
    """Decile-of-risk calibration plus ECE, intercept and slope."""
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    # quantile bins on the predicted risk; duplicates collapse when the model
    # concentrates mass, which it does here because most loans are safe
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    rows, ece = [], 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        obs, pred = float(y[m].mean()), float(p[m].mean())
        w = m.sum() / len(y)
        ece += w * abs(obs - pred)
        rows.append({"bin": b, "n": int(m.sum()), "mean_predicted": round(pred, 6),
                     "observed_rate": round(obs, 6),
                     "gap": round(obs - pred, 6)})

    # calibration intercept and slope, logistic on the predicted logit
    slope = intercept = np.nan
    if 0 < y.mean() < 1:
        try:
            from sklearn.linear_model import LogisticRegression
            z = np.log(p / (1 - p)).reshape(-1, 1)
            lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
            lr.fit(z, y.astype(int))
            slope = float(lr.coef_[0][0])
            intercept = float(lr.intercept_[0])
        except Exception as e:                       # pragma: no cover
            print(f"    note: calibration slope unavailable ({type(e).__name__})")
    return rows, ece, intercept, slope


def rates_at(y_short, p_short, cut):
    """Confusion-derived rates at a probability cutoff."""
    pred = (p_short >= cut).astype(int)
    tp = int(((pred == 1) & (y_short == 1)).sum())
    fp = int(((pred == 1) & (y_short == 0)).sum())
    fn = int(((pred == 0) & (y_short == 1)).sum())
    tn = int(((pred == 0) & (y_short == 0)).sum())
    return {"n_flagged": tp + fp,
            "tpr_recall": round(tp / max(1, tp + fn), 6),
            "fpr": round(fp / max(1, fp + tn), 6),
            "fnr": round(fn / max(1, tp + fn), 6),
            "precision": round(tp / max(1, tp + fp), 6),
            "selection_rate": round((tp + fp) / max(1, len(y_short)), 6)}


# -------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--figdir", default="outputs/figures")
    ap.add_argument("--pred-id", default="R012", help="fractional predictions run id")
    ap.add_argument("--score-id", default="R013", help="binary scores run id")
    ap.add_argument("--out-id", default="R017")
    ap.add_argument("--min-group", type=int, default=200,
                    help="skip groups smaller than this")
    ap.add_argument("--top-countries", type=int, default=12)
    a = ap.parse_args()

    rule(f"who-gets-funded  {a.out_id}  group calibration and error rates")
    sp = os.path.join(a.out, f"{a.score_id}_test2019_scores.csv")
    pp = os.path.join(a.out, f"{a.pred_id}_test2019_predictions.csv")
    if not os.path.exists(sp):
        sys.exit(f"missing {sp} — run the binary model step first")
    sc = pd.read_csv(sp)
    fr = pd.read_csv(pp) if os.path.exists(pp) else None
    if fr is None:
        print(f"  note: {pp} not found — fractional error table will be skipped")

    y = sc.y_short.to_numpy()
    p = sc.p_short.to_numpy()
    prev = float(y.mean())
    k_cut = float(np.quantile(p, 1 - prev))   # prevalence-matched top-K% threshold
    print(f"  test-2019: {len(sc):,} loans, {int(y.sum()):,} short "
          f"({100*prev:.3f}%)")
    print(f"  operating points: p_short >= 0.500 (default) and "
          f">= {k_cut:.4f} (top {100*prev:.2f}%, prevalence-matched)")

    # ---- which group splits to report ---------------------------------
    splits = {"gender": sorted(sc.gender.dropna().unique().tolist()),
              "region": None, "sector": None, "country_iso": None}
    if "region" in sc.columns:
        splits["region"] = sorted(sc.region.dropna().unique().tolist())
    else:
        splits.pop("region")
    splits["sector"] = sorted(sc.sector.dropna().unique().tolist())
    splits["country_iso"] = (sc.country_iso.value_counts()
                             .head(a.top_countries).index.tolist())

    cal_rows, bin_rows, rate_rows = [], [], []

    def do_group(dim, name, mask):
        n = int(mask.sum())
        if n < a.min_group:
            return
        yy, pp_ = y[mask], p[mask]
        if yy.sum() == 0:
            print(f"    skip {dim}={name}: no shortfall loans in group")
            return
        bins, ece, icpt, slope = calibration(yy, pp_)
        cal_rows.append({"dimension": dim, "group": name, "n": n,
                         "observed_short_rate": round(float(yy.mean()), 6),
                         "mean_predicted": round(float(pp_.mean()), 6),
                         "bias_pred_minus_obs": round(float(pp_.mean() - yy.mean()), 6),
                         "ece": round(ece, 6),
                         "calib_intercept": (round(icpt, 4) if icpt == icpt else np.nan),
                         "calib_slope": (round(slope, 4) if slope == slope else np.nan)})
        for b in bins:
            bin_rows.append({"dimension": dim, "group": name, **b})
        for label, cut in (("p>=0.5", 0.5), ("top-K prevalence", k_cut)):
            rate_rows.append({"dimension": dim, "group": name, "n": n,
                              "operating_point": label, "cutoff": round(cut, 6),
                              "prevalence": round(float(yy.mean()), 6),
                              **rates_at(yy, pp_, cut)})

    do_group("overall", "all", np.ones(len(sc), bool))
    for dim, groups in splits.items():
        for g in groups:
            do_group(dim, g, (sc[dim] == g).to_numpy())

    cal = pd.DataFrame(cal_rows)
    rates = pd.DataFrame(rate_rows)
    os.makedirs(a.out, exist_ok=True)
    cal.to_csv(os.path.join(a.out, f"{a.out_id}_calibration_by_group.csv"), index=False)
    pd.DataFrame(bin_rows).to_csv(
        os.path.join(a.out, f"{a.out_id}_calibration_bins.csv"), index=False)
    rates.to_csv(os.path.join(a.out, f"{a.out_id}_error_rates_by_group.csv"), index=False)

    print("\n  CALIBRATION — gender and overall\n")
    print(cal[cal.dimension.isin(["overall", "gender"])].to_string(index=False))
    print("\n  ERROR RATES — gender and overall\n")
    print(rates[rates.dimension.isin(["overall", "gender"])].to_string(index=False))

    # ---- fractional errors by group -----------------------------------
    if fr is not None:
        rows = []
        for dim in ["overall"] + list(splits):
            groups = ["all"] if dim == "overall" else splits[dim]
            for g in groups:
                m = (np.ones(len(fr), bool) if dim == "overall"
                     else (fr[dim] == g).to_numpy())
                if m.sum() < a.min_group:
                    continue
                yv = fr.y_share.to_numpy()[m]
                pv = fr.pred_share.to_numpy()[m]
                sh = fr.y_short.to_numpy()[m] == 1
                rows.append({
                    "dimension": dim, "group": g, "n": int(m.sum()),
                    "mae": round(mae(yv, pv), 6), "rmse": round(rmse(yv, pv), 6),
                    "mae_short": (round(mae(yv[sh], pv[sh]), 6) if sh.any() else np.nan),
                    "n_short": int(sh.sum()),
                    "mean_signed_error": round(float(np.mean(pv - yv)), 6),
                    "mean_actual": round(float(yv.mean()), 6),
                    "mean_pred": round(float(pv.mean()), 6)})
        fe = pd.DataFrame(rows)
        fe.to_csv(os.path.join(a.out, f"{a.out_id}_fractional_error_by_group.csv"),
                  index=False)
        print("\n  FRACTIONAL ERROR — gender and overall  "
              "(positive signed error = model over-predicts the share raised)\n")
        print(fe[fe.dimension.isin(["overall", "gender"])].to_string(index=False))

    # ---- figures ------------------------------------------------------
    plt = mpl()
    bins_df = pd.DataFrame(bin_rows)
    gb = bins_df[bins_df.dimension == "gender"]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=BASELINE, lw=1.2, ls="--", zorder=2)
    for g, col in (("female", S1), ("male", S2)):
        d = gb[gb.group == g]
        if len(d):
            ax.plot(d.mean_predicted, d.observed_rate, marker="o", ms=4, lw=1.8,
                    color=col, label=f"{g.capitalize()} (n={int(d.n.sum()):,})",
                    zorder=4)
    ax.set_xlabel("Mean predicted probability of shortfall")
    ax.set_ylabel("Observed shortfall rate")
    tidy(ax); ax.legend(fontsize=8)
    ax.set_title("Calibration within group\ndashed = perfect calibration",
                 fontsize=9, loc="left", pad=6)

    ax = axes[1]
    rg = rates[(rates.dimension == "gender") &
               (rates.operating_point == "top-K prevalence")]
    metrics = ["tpr_recall", "fpr", "precision"]
    labels = ["Recall\n(shortfall)", "False positive\nrate", "Precision\n(shortfall)"]
    xs = np.arange(len(metrics))
    for i, (g, col) in enumerate((("female", S1), ("male", S2))):
        d = rg[rg.group == g]
        if not len(d):
            continue
        vals = [float(d.iloc[0][m]) for m in metrics]
        ax.bar(xs + (i - 0.5) * 0.34, vals, width=0.32, color=col,
               label=g.capitalize(), zorder=3)
        for x, v in zip(xs + (i - 0.5) * 0.34, vals):
            ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.2, color=INK)
    ax.set_xticks(xs, labels, fontsize=8)
    ax.set_ylim(0, max(0.35, rg[metrics].to_numpy().max() * 1.25))
    tidy(ax); ax.legend(fontsize=8)
    ax.set_title("Error rates at the prevalence-matched cut\n"
                 "equalised-odds reading, not accuracy", fontsize=9, loc="left", pad=6)

    fig.suptitle("F30  Does the model treat borrower groups differently? "
                 "Test 2019, tuned full-cohort model",
                 x=0.005, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, a.figdir, "F30_calibration_by_gender.png",
             "Within-group calibration and error rates by gender")

    # error rates across the wider splits
    wide = rates[(rates.operating_point == "top-K prevalence")
                 & (rates.dimension.isin(["sector", "region", "country_iso"]))]
    if len(wide):
        fig, ax = plt.subplots(figsize=(7.4, max(3.2, 0.22 * len(wide))))
        d = wide.sort_values("fnr")
        ypos = np.arange(len(d))
        ax.barh(ypos, d.fnr, height=0.55, color=S2, zorder=3)
        ax.set_yticks(ypos, [f"{r.dimension[:3]}: {r.group}" for r in d.itertuples()],
                      fontsize=7)
        ax.axvline(float(rates[(rates.dimension == "overall") &
                               (rates.operating_point == "top-K prevalence")].iloc[0].fnr),
                   color=BASELINE, lw=1.4, ls="--", zorder=2)
        ax.set_xlabel("False negative rate on shortfall (missed failures)")
        tidy(ax, xgrid=True)
        ax.set_title("F31  Who does the model miss? False negative rate by group\n"
                     "dashed = cohort-wide rate at the same operating point",
                     fontsize=9.5, loc="left", pad=8)
        save_fig(fig, a.figdir, "F31_error_rates_by_group.png",
                 "False negative rate by sector, region and country")

    rule(f"{a.out_id} complete")
    print("  this closes equity-audit layer 2; the status board entry can move "
          "from amber to complete once the numbers are filed in RESULTS_LOG.")


if __name__ == "__main__":
    main()
