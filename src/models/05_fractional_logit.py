#!/usr/bin/env python3
# Code written with the assistance of Claude (Anthropic).
"""
who-gets-funded — R002 fractional logit baseline (Papke-Wooldridge).

GLM, binomial family, logit link, quasi-maximum likelihood. The target
share_raised is a proportion in [0,1], not a count, so this is a QMLE: the
coefficients are consistent but the likelihood is not a true one, which is why
standard errors MUST be robust (HC0 sandwich). That is what fit(cov_type='HC0')
does here.

SPECIFICATION DECISIONS (RESULTS_LOG section 13)
  * NO partner fixed effects. 459 dummies is the ~22 GB design-matrix problem in
    section 6, and it would make the baseline a model of intermediaries rather than
    borrowers. partner_id stays in feat_v1 for LightGBM, which handles it natively.
  * `activity` dropped from this model only (162 levels, nested inside sector).
  * `region` dropped: fully determined by country_iso, so perfectly collinear.
  * `posting_year` is the split key, never a predictor — a time dummy cannot be
    extrapolated to a future fold.
  * `loan_amount` dropped in favour of `log_loan_amount` (collinear pair).
  * `posting_month` and `posting_dow` enter as categoricals, not numerics; a linear
    day-of-week coefficient would be meaningless.
  * gender and country_iso ARE included. They are audit variables per rule 2, and a
    directly readable gender coefficient is exactly what the equity chapter needs.
    They are never offerable as recourse.
  * x_leak_* and x_excl_* columns are dropped.

LEAKAGE DISCIPLINE
  Everything fitted comes from TRAIN only: rare-level collapsing, median imputation
  of the two nullable text columns, and standardisation. Test rows are transformed
  with train statistics. Unseen categorical levels map to "__other__".

EVALUATION (RESULTS_LOG section 4.1)
  Reported per fold: overall MAE/RMSE against B0, and MAE restricted to loans that
  fell short against B0-on-short. Overall MAE alone would reward a model that
  ignores shortfall entirely.

Input  : outputs/tables/feat_v1_dev200k.csv  (or --features ...full.csv)
Output : outputs/tables/R002_logit_folds.csv
         outputs/tables/R002_logit_coefficients.csv
         outputs/tables/R002_logit_ape.csv

Usage:
    python3 src/models/05_fractional_logit.py                  # dev sample
    python3 src/models/05_fractional_logit.py --features outputs/tables/feat_v1_full.csv

Dependencies: pandas, numpy, statsmodels.
"""
import argparse, os, sys, time, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import gc, resource

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

NUMERIC = ["log_loan_amount", "amount_per_borrower", "borrower_count",
           "use_text_length", "description_length", "fundraising_window_days",
           "lender_repayment_term", "min_note_size", "n_tags_desc", "n_themes"]
# has_use_text / has_description are deliberately NOT in this design. Both
# numerics are missing on the SAME 8,967 of 200,003 rows and both get
# median-imputed, so either flag becomes an almost-deterministic function of
# those two columns - a three-way dependency pairwise correlation cannot see.
# Including one drove the intercept to 29.1 with NaN standard errors (the logit
# saturating at its numerical ceiling). Verified on the real dev sample:
# without them max|coef| = 3.84 and the fit converges in ~30s. They stay in
# feat_v1 for LightGBM, which is untroubled by this. Use --complete-case to
# drop the incomplete rows instead of imputing.
BINARY = ["is_group", "is_repeat_borrower",
          "has_previous_loan",
          "tag_woman_owned", "tag_parent", "tag_repeat_borrower", "tag_elderly",
          "tag_animals", "tag_eco_friendly", "tag_health_sanitation", "tag_technology",
          "theme_underfunded", "theme_rural_exclusion", "theme_startup",
          "theme_crop_insurance", "theme_vulnerable", "theme_conflict_zones",
          "theme_water_sanitation", "theme_higher_education", "theme_refugees",
          "theme_clean_energy"]
CATEG = ["gender", "country_iso", "sector", "repayment_interval",
         "anonymization_level", "original_language", "posting_month", "posting_dow"]
IMPUTE = ["use_text_length", "description_length"]
OTHER = "__other__"

# B0 floors from R001, so the comparison is printed inline
B0 = {2016: (0.039403, 0.159988, 0.5551),
      2017: (0.020863, 0.113719, 0.5212),
      2018: (0.047102, 0.185991, 0.6705),
      2019: (0.048465, 0.190227, 0.6888)}
BAR_W = 34


def hms(s):
    s = int(s)
    return f"{s}s" if s < 60 else (f"{s//60}m {s%60:02d}s" if s < 3600
                                   else f"{s//3600}h {(s%3600)//60:02d}m")


def bar(done, total, t0, label=""):
    frac = 0.0 if not total else min(1.0, done / total)
    fill = int(BAR_W * frac)
    el = time.time() - t0
    eta = (total - done) * el / done if done else 0
    sys.stdout.write(f"\r  [{'#'*fill}{'.'*(BAR_W-fill)}] {frac*100:5.1f}%  "
                     f"{done}/{total}  {hms(el)} elapsed  ETA {hms(eta)}  {label:<34}")
    sys.stdout.flush()


def rule(t):
    print(f"\n{'='*90}\n{t}\n{'='*90}")


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def peak_gb():
    """ru_maxrss is kilobytes on Linux but BYTES on macOS."""
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / 1e9 if sys.platform == "darwin" else v / 1048576


def choose_levels(tr, col, min_count, min_events):
    """Which levels of `col` get their own dummy, and what the reference is.

    A level in which NO loan falls short perfectly predicts y=1 and separates the
    likelihood: its coefficient diverges (|coef| reached 26 in the first R002
    attempt) while statsmodels still reports converged=True. Giving the leftovers a
    shared __other__ dummy does not help — that dummy inherits the separation.

    So leftovers are absorbed into the REFERENCE level instead, and no __other__
    dummy is created. The reference is the kept level with the most shortfall
    events, which makes it the most stable baseline available. Interpretation cost:
    the reference reads as "<level> plus absorbed rare levels", which is reported.
    """
    short = (tr.is_fully_funded == 0)
    lv = tr[col].astype(str)
    n = lv.value_counts()
    ev = short.groupby(lv).sum().reindex(n.index).fillna(0)
    keep = [x for x in n.index if n[x] >= min_count and ev[x] >= min_events]
    if not keep:                       # nothing qualifies: fall back to the biggest
        keep = [n.index[0]]
    ref = max(keep, key=lambda x: ev[x])
    absorbed = [x for x in n.index if x not in keep]
    return sorted(x for x in keep if x != ref), ref, absorbed, int(ev[ref])


def build_design(tr, te, min_count, min_events):
    """Fit every transform on train, apply to both. Returns Xtr, Xte, names."""
    med = {c: float(tr[c].median()) for c in IMPUTE}
    tr, te = tr.copy(), te.copy()
    for c in IMPUTE:
        tr[c] = tr[c].fillna(med[c])
        te[c] = te[c].fillna(med[c])

    # level selection from train only; leftovers absorbed into the reference
    cats, folded = {}, []
    for c in CATEG:
        nonref, ref, absorbed, ref_ev = choose_levels(tr, c, min_count, min_events)
        cats[c] = (nonref, ref)
        folded.append({"column": c, "reference": ref, "dummies": len(nonref),
                       "absorbed_levels": len(absorbed), "ref_shortfall_events": ref_ev})
        allowed = set(nonref)
        tr[c] = np.where(tr[c].astype(str).isin(allowed), tr[c].astype(str), ref)
        te[c] = np.where(te[c].astype(str).isin(allowed), te[c].astype(str), ref)

    # standardise numerics on train moments
    mu = {c: float(tr[c].mean()) for c in NUMERIC}
    sd = {c: (float(tr[c].std()) or 1.0) for c in NUMERIC}
    blocks_tr, blocks_te, names = [], [], []
    for c in NUMERIC:
        blocks_tr.append(((tr[c] - mu[c]) / sd[c]).values.astype(np.float64))
        blocks_te.append(((te[c] - mu[c]) / sd[c]).values.astype(np.float64))
        names.append(f"{c} (per SD)")
    for c in BINARY:
        blocks_tr.append(tr[c].values.astype(np.float64))
        blocks_te.append(te[c].values.astype(np.float64))
        names.append(c)
    for c in CATEG:
        nonref, ref = cats[c]
        for lv in nonref:
            blocks_tr.append((tr[c].astype(str) == lv).values.astype(np.float64))
            blocks_te.append((te[c].astype(str) == lv).values.astype(np.float64))
            names.append(f"{c}={lv}  [ref {ref}]")

    Xtr = np.column_stack([np.ones(len(tr))] + blocks_tr)
    Xte = np.column_stack([np.ones(len(te))] + blocks_te)
    names = ["const"] + names
    # drop all-zero / zero-variance columns (can appear after collapsing)
    # 1. zero-variance columns carry no information (is_repeat_borrower and
    #    has_previous_loan are constant 0 across the whole 2013-2019 cohort)
    keep = [0] + [j for j in range(1, Xtr.shape[1]) if Xtr[:, j].std() > 1e-12]
    dropped_const = [names[j] for j in range(1, Xtr.shape[1])
                     if Xtr[:, j].std() <= 1e-12]

    # 2. near-duplicate columns make the design singular. has_use_text and
    #    has_description differ on 3 of 200,003 rows (r = 0.9998); keeping both
    #    produced coefficients of -2.3 and -23.1 with NaN standard errors.
    idx = keep[1:]
    Z = Xtr[:, idx]
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1.0)
    R = (Z.T @ Z) / len(Z)
    drop, dropped_dup = set(), []
    for i in range(len(idx)):
        if i in drop:
            continue
        for j in range(i + 1, len(idx)):
            if j not in drop and abs(R[i, j]) >= DUP_R:
                drop.add(j)
                dropped_dup.append((names[idx[j]], names[idx[i]], float(R[i, j])))
    keep = [0] + [idx[i] for i in range(len(idx)) if i not in drop]

    Xtr2, Xte2 = Xtr[:, keep], Xte[:, keep]
    nm = [names[j] for j in keep]
    # 3. conditioning check on the surviving design
    Z = Xtr2[:, 1:]
    sdv = Z.std(0)
    Zs = (Z - Z.mean(0)) / np.where(sdv > 0, sdv, 1.0)
    ev = np.linalg.eigvalsh((Zs.T @ Zs) / len(Zs))
    cond = float(ev.max() / max(ev.min(), 1e-18))
    diag = {"dropped_constant": dropped_const, "dropped_duplicate": dropped_dup,
            "condition_number": cond}
    return Xtr2, Xte2, nm, mu, sd, med, folded, diag


DIVERGE = 10.0     # |coef| above this means a separated/diverged term
DUP_R = 0.995      # |correlation| at or above this = duplicate column


def hc0_sandwich(X, y, mu):
    """HC0 robust covariance for a canonical-link binomial GLM.

    A = X'WX with W = mu(1-mu);  B = X' diag((y-mu)^2) X;  V = A^-1 B A^-1.
    This is the Papke-Wooldridge robust variance. Implemented here rather than
    taken from statsmodels because with ~94% of rows at share_raised = 1 the fitted
    mu sits against the boundary, W collapses toward zero, and statsmodels' exact
    inverse of A returns NaN for every standard error. A pseudo-inverse returns
    finite numbers, so cond(A) is reported alongside: if it is enormous the SEs are
    not trustworthy no matter how finite they look.

    Validated against statsmodels cov_type='HC0' on a well-conditioned problem:
    agreement to 1.2e-16.
    """
    w = np.clip(mu * (1.0 - mu), 1e-12, None)
    A = X.T @ (X * w[:, None])
    B = X.T @ (X * ((y - mu) ** 2)[:, None])
    Ai = np.linalg.pinv(A)
    ev = np.abs(np.linalg.eigvalsh(A))
    return Ai @ B @ Ai, float(ev.max() / max(ev.min(), 1e-300))


def fit_glm(X, y, robust=False):
    """Fold fits need predictions only, so they skip the robust sandwich: it costs
    memory and time and is irrelevant to MAE. Only the final interpretive fit uses
    HC0, which is mandatory there because a fractional response makes this a QMLE."""
    m = sm.GLM(y, X, family=sm.families.Binomial())
    kw = dict(maxiter=200, full_output=False)
    return m.fit(cov_type="HC0", **kw) if robust else m.fit(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/tables/feat_v1_dev200k.csv")
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--min-count", type=int, default=500,
                    help="categorical levels rarer than this collapse to __other__")
    ap.add_argument("--min-events", type=int, default=25,
                    help="minimum SHORTFALL events per retained level; prevents the "
                         "perfect-separation divergence that produced |coef|=26 in the "
                         "first R002 attempt")
    ap.add_argument("--final-rows", type=int, default=0,
                    help="cap rows for the final interpretive fit (0 = all). A 200k x "
                         "120 fit peaks near 2-3 GB; the full 1.34m cohort would need "
                         "roughly 15-20 GB and will be killed by the OS.")
    ap.add_argument("--complete-case", action="store_true",
                    help="drop the rows with no recorded narrative length instead of "
                         "median-imputing them")
    ap.add_argument("--skip-final", action="store_true",
                    help="run folds only, skip the coefficient table")
    ap.add_argument("--min-train-years", type=int, default=3)
    a = ap.parse_args()

    if not os.path.exists(a.features):
        sys.exit(f"missing {a.features}\n"
                 f"  run: python3 src/features/03b_make_dev_sample.py\n"
                 f"  or pass --features outputs/tables/feat_v1_full.csv")
    os.makedirs(a.out, exist_ok=True)

    rule("who-gets-funded  R002 fractional logit (GLM binomial / logit, QMLE, HC0)")
    print(f"  features    : {a.features}  ({os.path.getsize(a.features)/1e6:,.1f} MB)")
    print(f"  min-count   : {a.min_count} rows per level")
    print(f"  min-events  : {a.min_events} shortfall events per level (separation guard)")
    print(f"  no partner fixed effects; activity, region, posting_year, loan_amount excluded")

    cols = (["posting_year", "share_raised", "is_fully_funded"]
            + NUMERIC + BINARY + CATEG)
    t0 = time.time()
    df = pd.read_csv(a.features, usecols=cols, low_memory=False)
    print(f"  loaded {len(df):,} rows in {hms(time.time()-t0)}")
    inc = df[IMPUTE].isna().any(axis=1)
    if a.complete_case:
        df = df[~inc].reset_index(drop=True)
        print(f"  complete-case: dropped {int(inc.sum()):,} rows without narrative "
              f"length; {len(df):,} remain")
    else:
        print(f"  {int(inc.sum()):,} rows ({100*inc.mean():.2f}%) lack narrative length; "
              f"median-imputed from train, no missingness indicator (see file header)")
    years = sorted(df.posting_year.unique())
    test_years = [y for y in years if years.index(y) >= a.min_train_years]

    # ---- fold loop --------------------------------------------------------
    rule(f"Step 1 of 2  temporal cross-validation ({len(test_years)} folds)")
    rows, t0 = [], time.time()
    for i, ty in enumerate(test_years, 1):
        tr = df[df.posting_year < ty]
        te = df[df.posting_year == ty]
        bar(i - 1, len(test_years), t0, f"{ty}: building design")
        Xtr, Xte, names, _, _, _, folded, diag = build_design(
            tr, te, a.min_count, a.min_events)
        bar(i - 1, len(test_years), t0, f"{ty}: fitting {Xtr.shape[1]} params")
        res = fit_glm(Xtr, tr.share_raised.values, robust=False)
        maxcoef = float(np.abs(res.params).max())
        p = np.clip(res.predict(Xte), 0.0, 1.0)
        y = te.share_raised.values
        short = te.is_fully_funded.values == 0
        b0_mae, b0_rmse, b0_short = B0.get(int(ty), (np.nan, np.nan, np.nan))
        rows.append({
            "test_year": int(ty), "train_years": f"{int(tr.posting_year.min())}-{int(ty-1)}",
            "n_train": len(tr), "n_test": len(te), "n_params": Xtr.shape[1],
            "converged": bool(res.converged),
            "max_abs_coef": round(maxcoef, 3),
            "diverged": bool(maxcoef > DIVERGE),
            "absorbed_levels": int(sum(f["absorbed_levels"] for f in folded)),
            "dropped_cols": len(diag["dropped_constant"]) + len(diag["dropped_duplicate"]),
            "condition_number": round(diag["condition_number"], 1),
            "mae": round(mae(y, p), 6), "b0_mae": b0_mae,
            "mae_vs_b0_pct": round(100 * (mae(y, p) / b0_mae - 1), 3),
            "rmse": round(rmse(y, p), 6), "b0_rmse": b0_rmse,
            "rmse_vs_b0_pct": round(100 * (rmse(y, p) / b0_rmse - 1), 3),
            "n_short": int(short.sum()),
            "mae_short": round(mae(y[short], p[short]), 6), "b0_mae_short": b0_short,
            "mae_short_vs_b0_pct": round(100 * (mae(y[short], p[short]) / b0_short - 1), 3),
            "mean_pred": round(float(p.mean()), 6), "mean_actual": round(float(y.mean()), 6),
        })
        bar(i, len(test_years), t0, f"{ty}: done")
        del Xtr, Xte, res
        gc.collect()
    print()
    print(f"  peak memory so far {peak_gb():.2f} GB")
    folds = pd.DataFrame(rows)
    folds.to_csv(os.path.join(a.out, "R002_logit_folds.csv"), index=False)

    show = ["test_year", "n_train", "n_test", "n_params", "converged",
            "max_abs_coef", "diverged", "absorbed_levels", "dropped_cols",
            "condition_number",
            "mae", "b0_mae", "mae_vs_b0_pct",
            "rmse", "b0_rmse", "rmse_vs_b0_pct",
            "n_short", "mae_short", "b0_mae_short", "mae_short_vs_b0_pct"]
    print(folds[show].to_string(index=False))
    print("\n  negative *_vs_b0_pct means the model beats the naive floor.")
    if folds.diverged.any() or (~folds.converged).any():
        print("\n  !! DO NOT USE THESE RESULTS. converged=False or max|coef| > "
              f"{DIVERGE:g} means the fit diverged.")
        print("     Raise --min-events (try 50, then 100) and rerun.")
    else:
        print(f"\n  all folds converged, max|coef| <= {folds.max_abs_coef.max():.2f} "
              f"— no separation detected.")

    # ---- final fit on all years, for the coefficient table ----------------
    if a.skip_final:
        rule("Step 2 skipped (--skip-final)")
        return
    rule("Step 2 of 2  final fit on the full window, for interpretation")
    fit_df = df if not a.final_rows else df.sample(
        n=min(a.final_rows, len(df)), random_state=40465466)
    t0 = time.time()
    Xall, _, names, mu, sd, med, folded, diag = build_design(
        fit_df, fit_df.head(1), a.min_count, a.min_events)
    for nm_, why, r in diag["dropped_duplicate"]:
        print(f"    dropped duplicate: {nm_}  (r={r:+.5f} with {why})")
    for nm_ in diag["dropped_constant"]:
        print(f"    dropped constant : {nm_}")
    print(f"    design condition number {diag['condition_number']:,.1f}"
          f"{'  !! ill-conditioned' if diag['condition_number'] > 1e8 else '  (well conditioned)'}")
    est = Xall.nbytes * 6 / 1e9        # statsmodels holds several working copies
    print(f"  design {Xall.shape[0]:,} x {Xall.shape[1]}  "
          f"({Xall.nbytes/1e9:.2f} GB), fit needs roughly {est:.1f} GB")
    if est > 6:
        print(f"  !! this will very likely be killed by the OS. Rerun with "
              f"--final-rows 200000 (or --skip-final).")
    for f in folded:
        print(f"    {f['column']:20s} ref={f['reference']:<12s} dummies={f['dummies']:>3d}  "
              f"absorbed={f['absorbed_levels']:>3d} rare levels into the reference")
    print(f"  fitting with HC0 robust errors ...")
    res = fit_glm(Xall, fit_df.share_raised.values, robust=True)
    mc = float(np.abs(res.params).max())
    print(f"  converged={res.converged}  max|coef|={mc:.2f}"
          f"{'  !! DIVERGED' if mc > DIVERGE else ''}  in {hms(time.time()-t0)}  "
          f"peak {peak_gb():.2f} GB")
    df = fit_df

    # McFadden-style pseudo R2 against an intercept-only quasi-likelihood
    null = fit_glm(np.ones((len(df), 1)), df.share_raised.values, robust=False)
    pr2 = 1 - res.llf / null.llf
    p_all = np.clip(res.predict(Xall), 0, 1)
    short = df.is_fully_funded.values == 0
    print(f"  pseudo-R2 {pr2:.6f}   in-sample MAE {mae(df.share_raised.values, p_all):.6f}"
          f"   MAE on short {mae(df.share_raised.values[short], p_all[short]):.6f}")

    V, condA = hc0_sandwich(Xall, fit_df.share_raised.values, p_all)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    z = np.where(se > 0, res.params / np.where(se > 0, se, 1), np.nan)
    from scipy.stats import norm
    pv = 2 * norm.sf(np.abs(z))
    print(f"  weighted information matrix cond(A) = {condA:.3e}"
          f"{'   !! SEs unreliable — mu is pinned to the boundary' if condA > 1e12 else ''}")
    coef = pd.DataFrame({
        "term": names, "coef": res.params, "robust_se": se,
        "z": z, "p_value": pv,
        "ci_low": res.params - 1.959964 * se, "ci_high": res.params + 1.959964 * se,
    })
    # average partial effect: dE[y|x]/dx = beta * mean(p(1-p)) for a unit change
    scale = float(np.mean(p_all * (1 - p_all)))
    coef["ape"] = coef.coef * scale
    coef["abs_z"] = coef.z.abs()
    coef.sort_values("abs_z", ascending=False, inplace=True)
    coef.drop(columns="abs_z").to_csv(
        os.path.join(a.out, "R002_logit_coefficients.csv"), index=False)
    pd.DataFrame([{"ape_scale_mean_p_1_minus_p": round(scale, 8),
                   "pseudo_r2": round(float(pr2), 6),
                   "n": len(df), "n_params": Xall.shape[1]}]).to_csv(
        os.path.join(a.out, "R002_logit_ape.csv"), index=False)

    print(f"\n  APE scale (mean p(1-p)) = {scale:.6f}; ape column = coef x scale,")
    print(f"  i.e. the change in expected share_raised per unit of the predictor.\n")
    print("  TOP 20 TERMS BY |z| (robust HC0)\n")
    top = coef.head(20).copy()
    top["coef"] = top.coef.round(5); top["robust_se"] = top.robust_se.round(5)
    top["z"] = top.z.round(2); top["ape"] = top.ape.round(6)
    top["p_value"] = top.p_value.apply(lambda v: "<1e-300" if v == 0 else f"{v:.2e}")
    print(top[["term", "coef", "robust_se", "z", "p_value", "ape"]].to_string(index=False))

    g = coef[coef.term.str.startswith("gender=")]
    if len(g):
        print("\n  GENDER TERMS (the equity coefficients)\n")
        gg = g.copy()
        gg["coef"] = gg.coef.round(5); gg["robust_se"] = gg.robust_se.round(5)
        gg["z"] = gg.z.round(2); gg["ape"] = gg.ape.round(6)
        gg["p_value"] = gg.p_value.apply(lambda v: "<1e-300" if v == 0 else f"{v:.2e}")
        print(gg[["term", "coef", "robust_se", "z", "p_value", "ape"]].to_string(index=False))

    rule("R002 complete")
    for f in ("R002_logit_folds.csv", "R002_logit_coefficients.csv", "R002_logit_ape.csv"):
        print(f"  -> {os.path.join(a.out, f)}")
    print("\n  Record as R002 in RESULTS_LOG section 4 and fill section 6.")


if __name__ == "__main__":
    main()
