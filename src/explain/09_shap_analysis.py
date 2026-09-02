#!/usr/bin/env python3
"""
who-gets-funded — R006: SHAP explanation of the canonical fractional model.

WHAT
  Exact TreeSHAP contributions, computed with the booster's native
  pred_contrib — no shap package required. The sample defaults to 50,000
  test-2019 rows, the lower bound of the 50-100k range set as a working rule
  when the pipeline was designed: exact TreeSHAP over the full population runs
  for days and buys nothing, because the attribution means stabilise long before
  that. Outputs are raw tables first, figures second, so every figure can be
  rebuilt from CSV without recomputing anything.

INPUT   outputs/models/<run>_gbm_frac.pkl + the features file
OUTPUT  outputs/tables/R006_shap_global.csv          mean |SHAP| and signed mean
        outputs/tables/R006_shap_by_gender.csv       female vs male mean |SHAP|
        outputs/tables/R006_shap_dependence_top8.csv long format for custom plots
        outputs/figures/F16_shap_global_bar.png
        outputs/figures/F17_shap_beeswarm.png        hand-rolled, no shap package
        outputs/figures/F18_shap_gender_compare.png

USAGE   python3 src/explain/09_shap_analysis.py
        python3 src/explain/09_shap_analysis.py --model-run R003 --out-id R006

Code written with the assistance of Claude (Anthropic).
"""
import argparse, os, sys, time, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pipeline_common import (SEED, CANON_TEST, load_features, load_bundle,
                             shap_contribs, bar, hms, rule, mpl, tidy, save_fig,
                             S1, S2, DEEMPH, BLUE_RAMP, MUTED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/tables/feat_v1_full.csv")
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--figdir", default="outputs/figures")
    ap.add_argument("--modeldir", default="outputs/models")
    ap.add_argument("--sample-n", type=int, default=50000)
    ap.add_argument("--model-run", default="R012",
                    help="R012 = tuned full-cohort model; R003 = untuned")
    ap.add_argument("--out-id", default="R014")
    a = ap.parse_args()

    rule(f"who-gets-funded  {a.out_id} SHAP (exact TreeSHAP via native pred_contrib)")
    FIG = ({"g": "F16_shap_global_bar.png", "b": "F17_shap_beeswarm.png",
            "x": "F18_shap_gender_compare.png"} if a.out_id == "R006" else
           {"g": "F32_shap_global_bar.png", "b": "F33_shap_beeswarm.png",
            "x": "F34_shap_gender_compare.png"})
    bundle = load_bundle(os.path.join(a.modeldir, f"{a.model_run}_gbm_frac.pkl"))
    print(f"  model: {a.model_run} {bundle['engine']}, train {bundle['train_span']}"
          + ("  [tuned]" if bundle.get("tuned") else "  [untuned defaults]"))

    df = load_features(a.features)
    te = df[df.posting_year == CANON_TEST]
    n = min(a.sample_n, len(te))
    samp = te.sample(n=n, random_state=SEED).reset_index(drop=True)
    print(f"  explaining {n:,} of {len(te):,} test-{CANON_TEST} rows (seed {SEED})")

    t0 = time.time()
    M, names = shap_contribs(bundle, samp)
    if M is None:
        print(f"  SKIPPED — {names}")
        return
    print(f"  contributions computed in {hms(time.time()-t0)}  "
          f"({M.shape[0]:,} x {M.shape[1]})")
    feats = names[:-1]
    C = M[:, :-1]                      # per-feature contributions, log-odds scale
    base = float(M[:, -1].mean())

    glob = pd.DataFrame({
        "feature": feats,
        "mean_abs_shap": np.abs(C).mean(0),
        "mean_shap_signed": C.mean(0),
        "share_of_total_pct": 100 * np.abs(C).mean(0) / np.abs(C).mean(0).sum(),
    }).sort_values("mean_abs_shap", ascending=False)
    glob["rank"] = range(1, len(glob) + 1)
    os.makedirs(a.out, exist_ok=True)
    glob.round(6).to_csv(os.path.join(a.out, f"{a.out_id}_shap_global.csv"), index=False)
    print(f"\n  top 10 by mean |SHAP| (log-odds scale; base value {base:.4f}):")
    print(glob.head(10)[["rank", "feature", "mean_abs_shap", "mean_shap_signed",
                         "share_of_total_pct"]].round(5).to_string(index=False))

    # by gender — the equity read of the explanation
    g = samp.gender.values
    rows = []
    for grp, mask in (("female", g == "female"), ("male", g == "male")):
        if mask.sum():
            rows.append(pd.DataFrame({
                "feature": feats, "group": grp, "n": int(mask.sum()),
                "mean_abs_shap": np.abs(C[mask]).mean(0),
                "mean_shap_signed": C[mask].mean(0)}))
    bg = pd.concat(rows, ignore_index=True)
    bg.round(6).to_csv(os.path.join(a.out, f"{a.out_id}_shap_by_gender.csv"), index=False)

    # dependence data for the top 8 (long format, re-plottable later)
    top8 = glob.feature.head(8).tolist()
    dep = []
    for f in top8:
        j = feats.index(f)
        dep.append(pd.DataFrame({
            "feature": f,
            "value": (samp[f] if pd.api.types.is_numeric_dtype(samp[f])
                      else samp[f].astype(str)),
            "shap": C[:, j], "gender": samp.gender.values}))
    pd.concat(dep, ignore_index=True).to_csv(
        os.path.join(a.out, f"{a.out_id}_shap_dependence_top8.csv"), index=False)
    print(f"  dependence data for {top8[:4]}... -> {a.out_id}_shap_dependence_top8.csv")

    # ---- figures ----------------------------------------------------------
    plt = mpl()
    top = glob.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.barh(range(len(top)), top.mean_abs_shap, height=0.5, color=S1, zorder=3)
    ax.set_yticks(range(len(top)), top.feature, fontsize=8)
    ax.set_xlabel("Mean |SHAP| (log-odds contribution to predicted share)")
    tidy(ax, xgrid=True)
    ax.set_title(f"{FIG['g'].split('_')[0]}  Global feature importance — mean |SHAP|, top 15\n"
                 f"{n:,} test-{CANON_TEST} loans, canonical model",
                 fontsize=9.5, loc="left", pad=8)
    save_fig(fig, a.figdir, FIG["g"], "SHAP global importance top 15")

    # hand-rolled beeswarm: x = SHAP value, colour = feature value (blue ramp)
    top10 = glob.feature.head(10).tolist()
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(7.0, 5.4))
    for row_i, f in enumerate(reversed(top10)):
        j = feats.index(f)
        take = rng.choice(len(samp), size=min(2500, len(samp)), replace=False)
        sv = C[take, j]
        v = samp[f].iloc[take]
        if pd.api.types.is_numeric_dtype(v):
            codes = v.astype(float).values
        else:
            codes = pd.Categorical(v.astype(str)).codes.astype(float)
            codes[codes < 0] = np.nan          # missing category
        ok = np.isfinite(codes)
        q = np.full(len(codes), np.nan)
        if ok.sum() > 1:
            order = codes[ok].argsort().argsort()
            q[ok] = order / max(1, ok.sum() - 1)
        cols = [BLUE_RAMP[min(6, int(x * 6.999))] if np.isfinite(x) else MUTED
                for x in q]
        ax.scatter(sv, row_i + rng.uniform(-0.27, 0.27, len(sv)), s=5, c=cols,
                   linewidths=0, alpha=0.75, zorder=3)
    ax.axvline(0, color=DEEMPH, lw=1.0, zorder=2)
    ax.set_yticks(range(len(top10)), list(reversed(top10)), fontsize=8)
    ax.set_xlabel("SHAP value (pushes predicted share down ← | → up)")
    tidy(ax, xgrid=True)
    ax.set_title(f"{FIG['b'].split('_')[0]}  SHAP beeswarm, top 10 features\n"
                 "Colour = feature value percentile (light blue low → dark blue high; grey = missing)",
                 fontsize=9.5, loc="left", pad=8)
    save_fig(fig, a.figdir, FIG["b"], "SHAP beeswarm top 10")

    # gender comparison
    piv = bg.pivot_table(index="feature", columns="group", values="mean_abs_shap")
    piv = piv.loc[[f for f in top10 if f in piv.index]].iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ys = np.arange(len(piv))
    h = 0.36
    ax.barh(ys + h / 2 + 0.01, piv.get("female", 0), height=h, color=S1,
            zorder=3, label="Female-tagged")
    ax.barh(ys - h / 2 - 0.01, piv.get("male", 0), height=h, color=S2,
            zorder=3, label="Male-tagged")
    ax.set_yticks(ys, piv.index, fontsize=8)
    ax.set_xlabel("Mean |SHAP| within group")
    tidy(ax, xgrid=True)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(f"{FIG['x'].split('_')[0]}  Do the same features drive predictions for both genders?\n"
                 "Mean |SHAP| computed separately within female- and male-tagged loans",
                 fontsize=9.5, loc="left", pad=8)
    save_fig(fig, a.figdir, FIG["x"],
             "SHAP importance by gender, top 10")

    rule(f"{a.out_id} complete")


if __name__ == "__main__":
    main()
