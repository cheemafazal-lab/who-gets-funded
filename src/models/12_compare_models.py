#!/usr/bin/env python3
# Code written with the assistance of Claude (Anthropic).
"""
who-gets-funded — R009: assemble every result into one comparison folder.

WHAT
  Collects R001 (naive), R002 (fractional logit), R003 (GBM), R004 (binary aux),
  R005 (out-of-regime), R007/R008 (counterfactuals and recourse) into
  outputs/model_comparison/ — raw tables, cross-model figures, and copies of the
  step figures — so the whole model stage can be reviewed from a single folder.
  Missing inputs are skipped and listed, never fatal.

OUTPUT  outputs/model_comparison/T1_fractional_by_fold.csv
        outputs/model_comparison/T2_headline.csv
        outputs/model_comparison/T3_binary.csv
        outputs/model_comparison/T4_oot_by_year.csv        (if R005 ran)
        outputs/model_comparison/F21..F23_*.png + copies of F13-F20
        outputs/model_comparison/README.txt

USAGE   python3 src/models/12_compare_models.py
"""
import glob, os, shutil, sys, pathlib, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pipeline_common import (B0, FOLD_TEST_YEARS, rule, mpl, tidy, save_fig,
                             S1, S2, DEEMPH, INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--figdir", default="outputs/figures")
    ap.add_argument("--compdir", default="outputs/model_comparison")
    a = ap.parse_args()

    rule("who-gets-funded  R009 model comparison assembly")
    os.makedirs(a.compdir, exist_ok=True)
    missing = []

    def read(name):
        p = os.path.join(a.out, name)
        if os.path.exists(p):
            return pd.read_csv(p)
        missing.append(name)
        return None

    r2, r3 = read("R002_logit_folds.csv"), read("R003_gbm_folds.csv")
    r4, r5 = read("R004_binary_folds.csv"), read("R005_oot_by_year.csv")
    r7s, r8 = read("R007_cf_summary.csv"), read("R008_recourse_by_gender.csv")

    # ---- T1 fractional by fold -------------------------------------------
    rows = []
    for ty in FOLD_TEST_YEARS:
        b0m, b0r, b0s = B0[ty]
        rows.append({"model": "B0 naive (constant 1.0)", "test_year": ty,
                     "mae": b0m, "rmse": b0r, "mae_short": b0s})
    for tag, d in (("R002 fractional logit", r2), ("R003 gradient boosting", r3)):
        if d is not None:
            for _, r in d.iterrows():
                rows.append({"model": tag, "test_year": int(r.test_year),
                             "mae": float(r.mae), "rmse": float(r.rmse),
                             "mae_short": float(r.mae_short)})
    t1 = pd.DataFrame(rows)
    t1.to_csv(os.path.join(a.compdir, "T1_fractional_by_fold.csv"), index=False)
    print("\n  T1 — fractional MAE by fold\n")
    print(t1.pivot_table(index="model", columns="test_year", values="mae")
            .round(6).to_string())
    print("\n  T1 — MAE restricted to shortfall loans\n")
    print(t1.pivot_table(index="model", columns="test_year", values="mae_short")
            .round(4).to_string())

    # ---- T2 headline ------------------------------------------------------
    t2 = (t1.groupby("model")
            .agg(mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"),
                 mean_mae_short=("mae_short", "mean"))
            .round(6).reset_index())
    b0row = t2[t2.model.str.startswith("B0")].iloc[0]
    t2["mae_vs_b0_pct"] = (100 * (t2.mean_mae / b0row.mean_mae - 1)).round(2)
    t2["rmse_vs_b0_pct"] = (100 * (t2.mean_rmse / b0row.mean_rmse - 1)).round(2)
    t2["mae_short_vs_b0_pct"] = (100 * (t2.mean_mae_short / b0row.mean_mae_short - 1)).round(2)
    t2.to_csv(os.path.join(a.compdir, "T2_headline.csv"), index=False)
    print("\n  T2 — headline (means over the four folds; negative % beats B0)\n")
    print(t2.to_string(index=False))

    # ---- T3 binary --------------------------------------------------------
    if r4 is not None:
        keep = ["test_year", "auc", "pr_auc_short", "pr_auc_floor_prevalence",
                "brier", "recall_short_at_0.5", "precision_short_at_0.5",
                "accuracy_pct", "b3_accuracy_pct"]
        r4[keep].to_csv(os.path.join(a.compdir, "T3_binary.csv"), index=False)
        print("\n  T3 — binary aux vs floors\n")
        print(r4[keep].to_string(index=False))
    if r5 is not None:
        r5.to_csv(os.path.join(a.compdir, "T4_oot_by_year.csv"), index=False)
    if r7s is not None:
        r7s.to_csv(os.path.join(a.compdir, "T5_cf_summary.csv"), index=False)
    if r8 is not None:
        r8.to_csv(os.path.join(a.compdir, "T6_recourse_by_gender.csv"), index=False)

    # ---- figures ----------------------------------------------------------
    plt = mpl()

    def grouped(ax, value, title, ylab):
        models = [m for m in ["B0 naive (constant 1.0)", "R002 fractional logit",
                              "R003 gradient boosting"] if m in set(t1.model)]
        colors = {"B0 naive (constant 1.0)": DEEMPH,
                  "R002 fractional logit": S1, "R003 gradient boosting": S2}
        w = 0.8 / len(models)
        for k, mdl in enumerate(models):
            d = t1[t1.model == mdl].set_index("test_year")[value]
            xs = [i + (k - (len(models) - 1) / 2) * w for i in range(len(FOLD_TEST_YEARS))]
            ax.bar(xs, [d.get(ty, np.nan) for ty in FOLD_TEST_YEARS], width=w * 0.92,
                   color=colors[mdl], zorder=3,
                   label=mdl.replace(" (constant 1.0)", ""))
        ax.set_xticks(range(len(FOLD_TEST_YEARS)), [str(t) for t in FOLD_TEST_YEARS])
        ax.set_xlabel("Test year")
        ax.set_ylabel(ylab)
        tidy(ax)
        ax.legend(fontsize=7.5)
        ax.set_title(title, fontsize=9.5, loc="left", pad=8)

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    grouped(ax, "mae", "F21  Overall MAE by fold — every model vs the naive floor\n"
            "Lower is better; grey is B0 (predict 1.0 for every loan)", "MAE")
    save_fig(fig, a.compdir, "F21_mae_by_fold.png", "Overall MAE by fold, all models")

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    grouped(ax, "mae_short", "F22  MAE on the loans that actually fell short\n"
            "The metric overall MAE hides — B0 scores 0.52-0.69 here", "MAE on shortfall loans")
    save_fig(fig, a.compdir, "F22_mae_short_by_fold.png", "Shortfall MAE by fold, all models")

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    grouped(ax, "rmse", "F23  RMSE by fold\n"
            "Squared error penalises ignoring large shortfalls", "RMSE")
    save_fig(fig, a.compdir, "F23_rmse_by_fold.png", "RMSE by fold, all models")

    # copies of step figures so the folder is self-contained
    copied = []
    for pat in ("F13_*", "F14_*", "F15_*", "F16_*", "F17_*", "F18_*", "F19_*", "F20_*"):
        for src in glob.glob(os.path.join(a.figdir, pat)):
            shutil.copy2(src, a.compdir)
            copied.append(os.path.basename(src))

    manifest = [
        "who-gets-funded — model comparison folder (assembled by src/models/12_compare_models.py)",
        "",
        "TABLES",
        "  T1_fractional_by_fold.csv   MAE / RMSE / shortfall-MAE per fold per model",
        "  T2_headline.csv             fold means, % vs the B0 naive floor",
        "  T3_binary.csv               binary aux vs AUC/PR/accuracy floors",
        "  T4_oot_by_year.csv          out-of-regime (if run)",
        "  T5_cf_summary.csv           counterfactual feasibility (if run)",
        "  T6_recourse_by_gender.csv   recourse audit (if run)",
        "",
        "FIGURES  F21-F23 built here; F13-F20 copied from the step scripts",
        "  " + ", ".join(sorted(copied)) if copied else "  (no step figures found)",
        "",
        "MISSING INPUTS AT ASSEMBLY TIME: " + (", ".join(missing) if missing else "none"),
        "",
        "All numbers are raw model outputs. Nothing here is tuned and nothing is",
        "interpreted; analysis happens in the report stage.",
    ]
    with open(os.path.join(a.compdir, "README.txt"), "w", encoding="utf8") as fh:
        fh.write("\n".join(manifest) + "\n")

    rule("R009 complete")
    print(f"  everything -> {a.compdir}/")
    if missing:
        print(f"  skipped (not yet run): {', '.join(missing)}")


if __name__ == "__main__":
    main()
