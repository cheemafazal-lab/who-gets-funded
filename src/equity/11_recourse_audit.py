#!/usr/bin/env python3
"""
who-gets-funded — R008: fairness of recourse — the novel contribution.

QUESTION
  Among comparable borrowers the model predicts will not fully fund, do some
  groups have to change MORE before the model predicts success?

METHOD (raw numbers only; interpretation happens later)
  From R007: by gender — attempts, feasibility rate, median recourse distance,
  median number of changes, median amount cut. Tests: two-proportion z on
  feasibility, Mann-Whitney U on distance and on n_changes. Comparability
  control: the same statistics recomputed WITHIN country × sector cells that
  contain at least --min-cell of each gender (composition held fixed, exactly as
  in the §5.1 standardisation), then summarised across cells.

BOTH READINGS
  The counterfactual step keeps loans that already scored at or above the target
  before any change was made — cases where the loan fell short but the model did
  not think it would. Including them measures recourse cost and model accuracy
  together; excluding them measures recourse cost alone on a base that the model
  has partly reselected. Neither is obviously right, so both are computed and
  filed, and the report states which one it quotes.

INPUT   outputs/tables/<cf-id>_counterfactuals.csv
OUTPUT  outputs/tables/<out-id>_recourse_by_gender.csv
        outputs/tables/<out-id>_recourse_within_strata.csv
        outputs/tables/<out-id>_recourse_tests.csv
        outputs/figures/F29_recourse_gap.png

USAGE   python3 src/equity/11_recourse_audit.py [--min-cell 10]

Code written with the assistance of Claude (Anthropic).
"""
import argparse, os, sys, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pipeline_common import rule, mpl, tidy, save_fig, S1, S2, DEEMPH, INK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--figdir", default="outputs/figures")
    ap.add_argument("--min-cell", type=int, default=10)
    ap.add_argument("--cf-id", default="R015")
    ap.add_argument("--out-id", default="R016")
    ap.add_argument("--figure", default="F29_recourse_gap.png")
    a = ap.parse_args()

    rule(f"who-gets-funded  {a.out_id} fairness of recourse")
    p = os.path.join(a.out, f"{a.cf_id}_counterfactuals.csv")
    if not os.path.exists(p):
        sys.exit(f"missing {p} — run the counterfactual step first")
    cf_all = pd.read_csv(p)
    cf_all = cf_all[cf_all.gender.isin(["female", "male"])].copy()
    if "already_at_target" not in cf_all.columns:
        cf_all["already_at_target"] = 0
    n_alr = int(cf_all.already_at_target.sum())
    print(f"  {len(cf_all):,} instances (female "
          f"{int((cf_all.gender=='female').sum()):,}, "
          f"male {int((cf_all.gender=='male').sum()):,})")
    print(f"  {n_alr:,} already at target before any change "
          f"({100*n_alr/max(1,len(cf_all)):.1f}%)")

    def block(d):
        ok = d[d.success == 1]
        return {"attempted": len(d), "feasible": len(ok),
                "feasibility_pct": round(100 * len(ok) / max(1, len(d)), 2),
                "median_distance": round(float(ok.distance.median()), 4) if len(ok) else np.nan,
                "mean_distance": round(float(ok.distance.mean()), 4) if len(ok) else np.nan,
                "median_changes": float(ok.n_changes.median()) if len(ok) else np.nan,
                "median_amount_cut_pct": (round(float(
                    ok.loc[ok.amount_cut_pct > 0, "amount_cut_pct"].median()), 1)
                    if (ok.amount_cut_pct > 0).any() else 0.0)}

    readings = [("all in base", cf_all),
                ("excluding already-at-target",
                 cf_all[cf_all.already_at_target == 0].copy())]

    by_rows = []
    for name, d in readings:
        by_rows += [{"reading": name, "group": g, **block(dd)}
                    for g, dd in d.groupby("gender")]
        by_rows.append({"reading": name, "group": "all", **block(d)})
    by = pd.DataFrame(by_rows)
    os.makedirs(a.out, exist_ok=True)
    by.to_csv(os.path.join(a.out, f"{a.out_id}_recourse_by_gender.csv"), index=False)
    print("\n  BY GENDER (raw, both readings)\n")
    print(by.to_string(index=False))

    # the headline reading, used for the tests and the figure
    cf = readings[1][1]
    print(f"\n  tests and figure use: {readings[1][0]} (n={len(cf):,})")

    # ---- tests ------------------------------------------------------------
    from scipy.stats import mannwhitneyu, norm
    f, m = cf[cf.gender == "female"], cf[cf.gender == "male"]
    fs, ms = f[f.success == 1], m[m.success == 1]
    tests = []
    p1, p2 = f.success.mean(), m.success.mean()
    n1, n2 = len(f), len(m)
    pp = (f.success.sum() + m.success.sum()) / (n1 + n2)
    se = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2)) or 1e-12
    z = (p1 - p2) / se
    tests.append({"test": "feasibility two-proportion z (female - male)",
                  "stat": round(float(z), 4),
                  "p_value": float(2 * norm.sf(abs(z))),
                  "effect": f"{100*(p1-p2):+.2f}pp"})
    for col in ("distance", "n_changes"):
        if len(fs) and len(ms):
            u, pv = mannwhitneyu(fs[col], ms[col], alternative="two-sided")
            tests.append({"test": f"Mann-Whitney U on {col} (successful CFs)",
                          "stat": round(float(u), 1), "p_value": float(pv),
                          "effect": f"median F {fs[col].median():.3f} vs "
                                    f"M {ms[col].median():.3f}"})
    tests = pd.DataFrame(tests)
    tests.insert(0, "reading", "excluding already-at-target")
    tests.to_csv(os.path.join(a.out, f"{a.out_id}_recourse_tests.csv"), index=False)
    print("\n  TESTS (no interpretation here)\n")
    print(tests.to_string(index=False))

    # ---- within country x sector ------------------------------------------
    cells = []
    for (cty, sec), d in cf.groupby(["country_iso", "sector"]):
        nf = int((d.gender == "female").sum())
        nm = int((d.gender == "male").sum())
        if nf < a.min_cell or nm < a.min_cell:
            continue
        bf, bm = block(d[d.gender == "female"]), block(d[d.gender == "male"])
        cells.append({"country_iso": cty, "sector": sec, "n_female": nf,
                      "n_male": nm,
                      "feas_female_pct": bf["feasibility_pct"],
                      "feas_male_pct": bm["feasibility_pct"],
                      "feas_gap_pp": round(bf["feasibility_pct"] - bm["feasibility_pct"], 2),
                      "med_dist_female": bf["median_distance"],
                      "med_dist_male": bm["median_distance"],
                      "dist_gap_f_minus_m": (round(bf["median_distance"] - bm["median_distance"], 4)
                                             if pd.notna(bf["median_distance"])
                                             and pd.notna(bm["median_distance"]) else np.nan)})
    strata = pd.DataFrame(cells)
    strata.to_csv(os.path.join(a.out, f"{a.out_id}_recourse_within_strata.csv"),
                  index=False)
    if len(strata):
        g = strata.dist_gap_f_minus_m.dropna()
        print(f"\n  WITHIN COUNTRY x SECTOR ({len(strata)} cells with >= "
              f"{a.min_cell} of each gender)")
        print(f"    median within-cell distance gap (F - M): "
              f"{g.median():+.4f}   cells where women need more: "
              f"{int((g > 0).sum())}/{len(g)}")
    else:
        print(f"\n  no country x sector cell reaches {a.min_cell} of each gender "
              f"— overall comparison only")

    # ---- figure -----------------------------------------------------------
    plt = mpl()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    ax = axes[0]
    bins = np.linspace(0, max(1e-9, float(cf[cf.success == 1].distance.max())), 24)
    for grp, col in (("female", S1), ("male", S2)):
        d = cf[(cf.gender == grp) & (cf.success == 1)].distance
        if len(d):
            ax.hist(d, bins=bins, histtype="step", lw=2, color=col,
                    label=f"{grp.capitalize()} (n={len(d):,})", zorder=3)
            ax.axvline(float(d.median()), color=col, lw=1.2, ls=":", zorder=2)
    ax.set_xlabel("Recourse distance (MAD units; dotted = median)")
    ax.set_ylabel("Successful counterfactuals")
    tidy(ax)
    ax.legend(fontsize=8)
    ax.set_title("How much must change", fontsize=9, loc="left", pad=6)
    ax = axes[1]
    sub = by[(by.group.isin(["female", "male"]))
             & (by.reading == "excluding already-at-target")]
    cols = [S1 if g == "female" else S2 for g in sub.group]
    ax.bar(range(len(sub)), sub.feasibility_pct, width=0.5, color=cols, zorder=3)
    for i, v in enumerate(sub.feasibility_pct):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(sub)), [g.capitalize() for g in sub.group])
    ax.set_ylabel("Feasible counterfactual found (%)")
    ax.set_ylim(0, 105)
    tidy(ax)
    ax.set_title("Whether recourse exists at all", fontsize=9, loc="left", pad=6)
    fig.suptitle(f"{a.figure.split('_')[0]}  Fairness of recourse — raw comparison, "
                 "interpretation deferred",
                 x=0.005, ha="left", fontsize=9.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, a.figdir, a.figure,
             "Recourse distance and feasibility by gender")

    rule(f"{a.out_id} complete")


if __name__ == "__main__":
    main()
