#!/usr/bin/env python3
"""
who-gets-funded — R018: final comparison across every model in the project.

WHAT
  Assembles one self-contained folder holding the whole modelling story: the
  naive floors, the fractional logit, the untuned gradient boosting run on the
  development sample, and the tuned gradient boosting run on the full cohort.
  Everything is read from the CSVs the earlier steps wrote, so this step
  computes no model and can be re-run at any time.

READING THE COMPARISON
  Two things change between R003/R004 and R012/R013 — the amount of training
  data and the hyperparameters — so the tuned-versus-untuned columns measure
  both together. The sweep log (R010/R011) isolates the hyperparameter effect on
  its own, because both its arms use the identical full-cohort folds.

  One denominator note carries into every percentage. The B0 floors were
  computed on the full cohort. R003/R004 were scored on the 200k development
  sample, whose implied floors differ from the logged ones by 0.1-1.2%, worst on
  2017. R012/R013 are scored on the full cohort itself, so their percentages are
  exact. The comparison is still sound — the gap is far smaller than any effect
  reported — but the tuned rows are the ones measured against their own floor.

INPUT   outputs/tables/R001_naive_benchmark.csv   R002_logit_folds.csv
        R003_gbm_folds.csv   R004_binary_folds.csv
        R010_tuning_frac.csv R011_tuning_binary.csv
        R012_gbm_folds.csv   R013_binary_folds.csv
        R015_cf_summary.csv  R016_recourse_by_gender.csv
        R017_calibration_by_group.csv  R017_error_rates_by_group.csv
OUTPUT  outputs/final_comparison/  tables C1-C7, figures F35-F37, README.txt

USAGE   python3 src/models/16_compare_final.py

Code written with the assistance of Claude (Anthropic).
"""
import argparse
import json
import os
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pipeline_common import (B0, rule, mpl, tidy, save_fig, S1, S2, DEEMPH,
                             BASELINE, INK, MUTED)

FOLDS = [2016, 2017, 2018, 2019]


def maybe(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--dest", default="outputs/final_comparison")
    ap.add_argument("--figdir", default="outputs/figures")
    ap.add_argument("--modeldir", default="outputs/models")
    a = ap.parse_args()

    rule("who-gets-funded  R018  final comparison")
    os.makedirs(a.dest, exist_ok=True)
    P = lambda f: os.path.join(a.out, f)                      # noqa: E731
    missing = []

    r002 = maybe(P("R002_logit_folds.csv"))
    r003 = maybe(P("R003_gbm_folds.csv"))
    r004 = maybe(P("R004_binary_folds.csv"))
    r012 = maybe(P("R012_gbm_folds.csv"))
    r013 = maybe(P("R013_binary_folds.csv"))
    for name, d in (("R002", r002), ("R003", r003), ("R004", r004),
                    ("R012", r012), ("R013", r013)):
        if d is None:
            missing.append(name)

    # ---------------------------------------------------- C1 fractional ----
    rows = []
    for ty in FOLDS:
        b0m, b0r, b0s = B0[ty]
        rows.append({"model": "B0 naive (constant 1.0)", "sample": "full cohort",
                     "test_year": ty, "mae": b0m, "rmse": b0r, "mae_short": b0s})
    def sample_label(d):
        """Derive the training sample from the folds file rather than assuming
        it, so a run on the development sample is never labelled full cohort."""
        n = int(d.n_train.max()) + int(d[d.n_train == d.n_train.max()].n_test.iloc[0])
        return "full cohort" if n > 1_000_000 else f"dev {n//1000}k"

    for label, d, _ in (("R002 fractional logit", r002, None),
                        ("R003 gradient boosting (untuned)", r003, None),
                        ("R012 gradient boosting (tuned)", r012, None)):
        if d is None:
            continue
        samp = sample_label(d)
        for _, r in d.iterrows():
            rows.append({"model": label, "sample": samp,
                         "test_year": int(r.test_year), "mae": r.mae,
                         "rmse": r.rmse, "mae_short": r.mae_short})
    c1 = pd.DataFrame(rows)
    c1.to_csv(os.path.join(a.dest, "C1_fractional_by_fold.csv"), index=False)

    # ---------------------------------------------------- C2 headline ------
    head = []
    b0 = c1[c1.model.str.startswith("B0")]
    bm, br, bs = b0.mae.mean(), b0.rmse.mean(), b0.mae_short.mean()
    for label in c1.model.unique():
        d = c1[c1.model == label]
        head.append({
            "model": label, "sample": d["sample"].iloc[0],
            "mean_mae": round(d.mae.mean(), 6),
            "mean_rmse": round(d.rmse.mean(), 6),
            "mean_mae_short": round(d.mae_short.mean(), 6),
            "mae_vs_b0_pct": round(100 * (d.mae.mean() / bm - 1), 2),
            "rmse_vs_b0_pct": round(100 * (d.rmse.mean() / br - 1), 2),
            "mae_short_vs_b0_pct": round(100 * (d.mae_short.mean() / bs - 1), 2)})
    c2 = pd.DataFrame(head)
    c2.to_csv(os.path.join(a.dest, "C2_headline.csv"), index=False)
    print("\n  C2  FRACTIONAL HEADLINE (fold means, negative beats the floor)\n")
    print(c2.to_string(index=False))

    # ---------------------------------------------------- C3 binary --------
    rows = []
    for label, d, _ in (("R004 binary aux (untuned)", r004, None),
                        ("R013 binary aux (tuned)", r013, None)):
        if d is None:
            continue
        samp = sample_label(d)
        for _, r in d.iterrows():
            rows.append({"model": label, "sample": samp,
                         "test_year": int(r.test_year), "auc": r.auc,
                         "pr_auc_short": r.pr_auc_short,
                         "pr_auc_floor": r.pr_auc_floor_prevalence,
                         "pr_lift_x": round(r.pr_auc_short / r.pr_auc_floor_prevalence, 2),
                         "brier": r.brier,
                         "recall_at_0.5": r["recall_short_at_0.5"],
                         "precision_at_0.5": r["precision_short_at_0.5"],
                         "accuracy_pct": r.accuracy_pct,
                         "b3_accuracy_pct": r.b3_accuracy_pct})
    c3 = pd.DataFrame(rows)
    if len(c3):
        c3.to_csv(os.path.join(a.dest, "C3_binary_by_fold.csv"), index=False)
        print("\n  C3  BINARY AUXILIARY MODEL\n")
        print(c3.groupby(["model", "sample"])[
            ["auc", "pr_auc_short", "pr_auc_floor", "brier"]].mean()
            .round(4).to_string())

    # ---------------------------------------------------- C4 tuning --------
    trows = []
    for kind, run, log in (("frac", "R010", "R010_tuning_frac.csv"),
                           ("binary", "R011", "R011_tuning_binary.csv")):
        d = maybe(P(log))
        if d is None:
            missing.append(run)
            continue
        untuned = d[d.candidate == 0]
        best = (d.loc[d.score.idxmin()] if kind == "frac" else d.loc[d.score.idxmax()])
        trows.append({
            "run": run, "kind": kind, "candidates_evaluated": len(d),
            "metric": ("composite RMSE + shortfall-MAE vs B0 (lower better)"
                       if kind == "frac" else "PR-AUC shortfall (higher better)"),
            "untuned_score": (round(float(untuned.iloc[0].score), 6)
                              if len(untuned) else np.nan),
            "best_score": round(float(best.score), 6),
            "best_candidate": int(best.candidate), "best_label": best.label,
            "improvement": (round(float(untuned.iloc[0].score) - float(best.score), 6)
                            if len(untuned) else np.nan),
            "best_params": best.params_json})
    if trows:
        c4 = pd.DataFrame(trows)
        c4.to_csv(os.path.join(a.dest, "C4_tuning_summary.csv"), index=False)
        print("\n  C4  TUNING\n")
        print(c4[["run", "kind", "candidates_evaluated", "untuned_score",
                  "best_score", "best_candidate"]].to_string(index=False))

    # ---------------------------------------------------- C5-C7 downstream --
    for src, dst in (("R015_cf_summary.csv", "C5_counterfactual_summary.csv"),
                     ("R016_recourse_by_gender.csv", "C6_recourse_by_gender.csv"),
                     ("R017_calibration_by_group.csv", "C7_calibration_by_group.csv"),
                     ("R017_error_rates_by_group.csv", "C8_error_rates_by_group.csv"),
                     ("R017_fractional_error_by_group.csv",
                      "C9_fractional_error_by_group.csv")):
        if os.path.exists(P(src)):
            shutil.copy(P(src), os.path.join(a.dest, dst))
        else:
            missing.append(src)

    # ---------------------------------------------------- figures ----------
    plt = mpl()
    metrics = [("rmse", "RMSE", "b0_rmse"),
               ("mae_short", "Shortfall-MAE", "b0_mae_short"),
               ("mae", "MAE", "b0_mae")]
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.4))
    for ax, (m, title, _) in zip(axes, metrics):
        for label, col, mk in (("B0 naive (constant 1.0)", BASELINE, "s"),
                               ("R002 fractional logit", DEEMPH, "^"),
                               ("R003 gradient boosting (untuned)", S2, "o"),
                               ("R012 gradient boosting (tuned)", S1, "D")):
            d = c1[c1.model == label].sort_values("test_year")
            if not len(d):
                continue
            ax.plot(d.test_year, d[m], marker=mk, ms=4.5, lw=1.8, color=col,
                    label=label.split(" (")[0], zorder=4)
        ax.set_xticks(FOLDS)
        ax.set_xlabel("Test year")
        ax.set_ylabel(title)
        tidy(ax)
        ax.set_title(title, fontsize=9, loc="left", pad=6)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("F35  Every model against the naive floor, by test year — "
                 "lower is better throughout",
                 x=0.005, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, a.figdir, "F35_all_models_by_fold.png",
             "All fractional models against B0 by test year")
    shutil.copy(os.path.join(a.figdir, "F35_all_models_by_fold.png"), a.dest)

    if len(c3):
        fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3))
        for ax, m, title in ((axes[0], "auc", "AUC"),
                             (axes[1], "pr_auc_short", "PR-AUC (shortfall)")):
            for label, col in (("R004 binary aux (untuned)", S2),
                               ("R013 binary aux (tuned)", S1)):
                d = c3[c3.model == label].sort_values("test_year")
                if len(d):
                    ax.plot(d.test_year, d[m], marker="o", ms=4.5, lw=1.8,
                            color=col, label=label.split(" (")[0], zorder=4)
            if m == "pr_auc_short":
                d = c3[c3.model.str.contains("tuned")].sort_values("test_year")
                if len(d):
                    ax.plot(d.test_year, d.pr_auc_floor, lw=1.4, ls="--",
                            color=BASELINE, label="prevalence floor", zorder=3)
            ax.set_xticks(FOLDS); ax.set_xlabel("Test year"); ax.set_ylabel(title)
            tidy(ax); ax.legend(fontsize=7.5)
            ax.set_title(title, fontsize=9, loc="left", pad=6)
        fig.suptitle("F36  Auxiliary binary model, untuned versus tuned — "
                     "higher is better",
                     x=0.005, ha="left", fontsize=9.5, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        save_fig(fig, a.figdir, "F36_binary_untuned_vs_tuned.png",
                 "Binary aux model, untuned versus tuned")
        shutil.copy(os.path.join(a.figdir, "F36_binary_untuned_vs_tuned.png"), a.dest)

    # copy the standing figures across so the folder is self-contained
    for f in ("F25_tuning_traces.png", "F26_tuned_gain_importance.png",
              "F27_tuned_binary_roc_pr.png", "F28_cf_actions.png",
              "F29_recourse_gap.png", "F30_calibration_by_gender.png",
              "F31_error_rates_by_group.png", "F32_shap_global_bar.png",
              "F33_shap_beeswarm.png", "F34_shap_gender_compare.png"):
        src = os.path.join(a.figdir, f)
        if os.path.exists(src):
            shutil.copy(src, a.dest)

    with open(os.path.join(a.dest, "README.txt"), "w") as fh:
        fh.write(
            "who-gets-funded — final comparison folder (assembled by "
            "src/models/16_compare_final.py)\n\n"
            "TABLES\n"
            "  C1_fractional_by_fold.csv      MAE / RMSE / shortfall-MAE, every "
            "model, every fold\n"
            "  C2_headline.csv                fold means and % against the B0 floor\n"
            "  C3_binary_by_fold.csv          binary aux, untuned vs tuned\n"
            "  C4_tuning_summary.csv          search budget, winner, gain over "
            "the untuned default\n"
            "  C5_counterfactual_summary.csv  recourse feasibility, both readings\n"
            "  C6_recourse_by_gender.csv      the fairness-of-recourse result\n"
            "  C7_calibration_by_group.csv    within-group calibration\n"
            "  C8_error_rates_by_group.csv    equalised-odds style error rates\n"
            "  C9_fractional_error_by_group.csv  signed error by group\n\n"
            "FIGURES  F35-F36 built here; F25-F34 copied from the step scripts.\n\n"
            "SAMPLE NOTE\n"
            "  R002/R003/R004 were fitted on the 200,003-row development sample.\n"
            "  R012/R013 are fitted on the full 1,344,542-row cohort. The B0 "
            "floors\n"
            "  come from the full cohort throughout, so the tuned rows are "
            "measured\n"
            "  against their own denominator and the untuned rows against one "
            "that\n"
            "  differs by 0.1-1.2%.\n\n"
            "  All numbers are raw model output. Nothing here is interpreted.\n")

    if missing:
        print(f"\n  inputs not found (steps not yet run): {sorted(set(missing))}")
    rule("R018 complete")
    print(f"  {a.dest}/")


if __name__ == "__main__":
    main()
