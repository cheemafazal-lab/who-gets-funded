#!/usr/bin/env python3
# Code written with the assistance of Claude (Anthropic).
"""
who-gets-funded — R001 naive benchmark.

The floor every later model must beat. No features are used by B0/B1/B3; B2 uses
only a group lookup. Reads the feature matrix, never the database.

WHY FOUR BENCHMARKS
  B0  constant 1.0            The MAE-optimal constant: the median of share_raised
                              is exactly 1.0 because ~94% of loans fund fully. No
                              constant predictor can beat it on MAE, so this is the
                              hardest naive floor, not a strawman.
  B1  constant = train mean   The RMSE-optimal constant, and the conventional
                              R^2 = 0 reference point.
  B2  group mean lookup       "Simple cohort prediction" — train mean by country,
                              by partner, and by country x sector. Tests whether a
                              model adds anything beyond knowing where a loan came
                              from. Given that composition explains 14-28% of the
                              gender gap, this is the benchmark that actually bites.
  B3  binary majority class   For the auxiliary classifier. Always predicting
                              "fully funded" scores ~94% accuracy with 0% recall on
                              the loans that fall short. Reported so accuracy can
                              never be quoted alone.

FOLDS
  Expanding window: train on every earlier year, test on one year. The naive floor
  is derived from TRAIN and scored on TEST, exactly as a real model would be, so
  the comparison is valid. Per-fold floors are essential: the floor moves from
  0.0110 to 0.0485 across 2013-2019, so one global figure would flatter models
  tested on early years and punish those tested on late ones.

Input  : outputs/tables/feat_v1_full.csv
Output : outputs/tables/R001_naive_benchmark.csv   (fold x benchmark x metrics)
         outputs/tables/R001_naive_by_year.csv     (per-year floors, feeds F12)

Usage:
    cd path/to/who-gets-funded
    python3 src/models/04_naive_benchmark.py

Dependencies: pandas, numpy only. No sklearn, no lightgbm.
"""
import argparse, csv, math, os, sys, time
import numpy as np
import pandas as pd

NEEDED = ["posting_year", "share_raised", "is_fully_funded",
          "country_iso", "partner_id", "sector"]
BAR_W = 34


def hms(s):
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s//60}m {s%60:02d}s"
    return f"{s//3600}h {(s%3600)//60:02d}m"


def bar(done, total, t0, label="", width=BAR_W):
    frac = 0.0 if not total else min(1.0, done / total)
    fill = int(width * frac)
    el = time.time() - t0
    rate = done / el if el > 0 else 0
    eta = (total - done) / rate if rate > 0 and total else 0
    sys.stdout.write(f"\r  [{'#'*fill}{'.'*(width-fill)}] {frac*100:5.1f}%  "
                     f"{done:>9,} / {total:,}  {hms(el)} elapsed  ETA {hms(eta)}  {label:<28}")
    sys.stdout.flush()


def rule(t):
    print(f"\n{'='*86}\n{t}\n{'='*86}")


def table(rows, cols, fmts=None):
    fmts = fmts or {}
    txt = [[fmts.get(c, "{}").format(r[c]) if r[c] is not None else "" for c in cols]
           for r in rows]
    w = [max(len(cols[i]), *(len(t[i]) for t in txt)) if txt else len(cols[i])
         for i in range(len(cols))]
    print("  " + "  ".join(cols[i].ljust(w[i]) for i in range(len(cols))))
    print("  " + "  ".join("-" * w[i] for i in range(len(cols))))
    for t in txt:
        print("  " + "  ".join(t[i].ljust(w[i]) for i in range(len(cols))))


def count_rows(path):
    """Line count minus header, read in binary blocks."""
    n, size = 0, os.path.getsize(path)
    t0 = time.time()
    with open(path, "rb") as fh:
        read = 0
        while True:
            blk = fh.read(8 << 20)
            if not blk:
                break
            n += blk.count(b"\n")
            read += len(blk)
            bar(read, size, t0, "scanning file")
    print()
    return max(0, n - 1)


# ---- metrics (numpy only) ---------------------------------------------------
def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def brier(y_bin, p):
    return float(np.mean((p - y_bin) ** 2))


def logloss(y_bin, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y_bin * np.log(p) + (1 - y_bin) * np.log(1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/tables/feat_v1_full.csv")
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--min-train-years", type=int, default=3,
                    help="first test year needs at least this many earlier years")
    a = ap.parse_args()

    if not os.path.exists(a.features):
        sys.exit(f"missing {a.features} — run 03_build_features.py first")
    os.makedirs(a.out, exist_ok=True)

    rule("who-gets-funded  R001 naive benchmark")
    print(f"  features : {a.features}  ({os.path.getsize(a.features)/1e6:,.1f} MB)")
    print(f"  metrics  : MAE and RMSE (fractional), accuracy/Brier/log-loss (binary)")

    # ---- 1. load ----------------------------------------------------------
    rule("Step 1 of 4  loading the columns the benchmark needs")
    total = count_rows(a.features)
    print(f"  {total:,} data rows in file")
    t0 = time.time()
    parts, read = [], 0
    for chunk in pd.read_csv(a.features, usecols=NEEDED, chunksize=250_000,
                             low_memory=False):
        parts.append(chunk)
        read += len(chunk)
        bar(read, total, t0, f"{len(NEEDED)} of 54 columns")
    df = pd.concat(parts, ignore_index=True)
    del parts
    print()
    print(f"  loaded {len(df):,} rows x {df.shape[1]} columns in {hms(time.time()-t0)}")
    if len(df) != total:
        print(f"  !! WARNING loaded {len(df):,} but counted {total:,}")

    years = sorted(df.posting_year.unique())
    print(f"  posting years: {years[0]}-{years[-1]}")

    # ---- 2. per-year floors ----------------------------------------------
    rule("Step 2 of 4  per-year naive floors (constant 1.0)")
    by_year = []
    for y in years:
        d = df[df.posting_year == y]
        by_year.append({"posting_year": int(y), "n": len(d),
                        "mean_share": round(float(d.share_raised.mean()), 6),
                        "expiry_pct": round(100 * float(1 - d.is_fully_funded.mean()), 4),
                        "naive_mae_at_1": round(mae(d.share_raised.values,
                                                    np.ones(len(d))), 6),
                        "naive_rmse_at_1": round(rmse(d.share_raised.values,
                                                      np.ones(len(d))), 6)})
    table(by_year, ["posting_year", "n", "mean_share", "expiry_pct",
                    "naive_mae_at_1", "naive_rmse_at_1"],
          {"n": "{:,}", "mean_share": "{:.6f}", "expiry_pct": "{:.4f}",
           "naive_mae_at_1": "{:.6f}", "naive_rmse_at_1": "{:.6f}"})
    lo = min(r["naive_mae_at_1"] for r in by_year)
    hi = max(r["naive_mae_at_1"] for r in by_year)
    print(f"\n  floor ranges {lo:.6f} to {hi:.6f} — a {hi/lo:.1f}x spread. "
          f"Per-fold benchmarks are mandatory.")
    pd.DataFrame(by_year).to_csv(os.path.join(a.out, "R001_naive_by_year.csv"),
                                 index=False)

    # ---- 3. fold loop ----------------------------------------------------
    test_years = [y for y in years if years.index(y) >= a.min_train_years]
    cells = [("B2a_country", ["country_iso"]),
             ("B2b_partner", ["partner_id"]),
             ("B2c_country_sector", ["country_iso", "sector"])]
    n_units = len(test_years) * (2 + len(cells) + 1)
    rule(f"Step 3 of 4  expanding-window folds  ({len(test_years)} folds, "
         f"{n_units} benchmark evaluations)")
    results = []
    t0, unit = time.time(), 0
    for ty in test_years:
        tr = df[df.posting_year < ty]
        te = df[df.posting_year == ty]
        y_te = te.share_raised.values
        b_te = te.is_fully_funded.values.astype(float)
        base = dict(test_year=int(ty), train_years=f"{int(min(tr.posting_year))}-{int(ty-1)}",
                    n_train=len(tr), n_test=len(te))

        def rec(bid, name, pred, note=""):
            results.append({**base, "benchmark": bid, "description": name,
                            "mae": round(mae(y_te, pred), 6),
                            "rmse": round(rmse(y_te, pred), 6),
                            "coverage_pct": 100.0, "note": note})

        # B0 constant 1.0
        rec("B0", "constant 1.0 (MAE-optimal constant)", np.ones(len(te)))
        unit += 1; bar(unit, n_units, t0, f"{ty} B0")

        # B1 constant train mean
        m = float(tr.share_raised.mean())
        rec("B1", f"constant train mean ({m:.6f})", np.full(len(te), m))
        unit += 1; bar(unit, n_units, t0, f"{ty} B1")

        # B2 group-mean lookups, global train mean as fallback
        for bid, keys in cells:
            gm = tr.groupby(keys, observed=True).share_raised.mean()
            idx = (te[keys[0]] if len(keys) == 1
                   else pd.MultiIndex.from_frame(te[keys]))
            pred = pd.Series(idx).map(gm).values.astype(float) if len(keys) == 1 \
                else gm.reindex(idx).values.astype(float)
            seen = ~np.isnan(pred)
            pred = np.where(seen, pred, m)
            results.append({**base, "benchmark": bid,
                            "description": f"train mean by {' x '.join(keys)}",
                            "mae": round(mae(y_te, pred), 6),
                            "rmse": round(rmse(y_te, pred), 6),
                            "coverage_pct": round(100 * seen.mean(), 4),
                            "note": f"{(~seen).sum():,} test rows fell back to the global mean"})
            unit += 1; bar(unit, n_units, t0, f"{ty} {bid}")

        # B3 binary majority class
        p_funded = float(tr.is_fully_funded.mean())
        acc = float((b_te == 1).mean())
        n_short = int((b_te == 0).sum())
        results.append({**base, "benchmark": "B3",
                        "description": "binary: always predict fully funded",
                        "mae": round(mae(b_te, np.ones(len(te))), 6),
                        "rmse": round(rmse(b_te, np.ones(len(te))), 6),
                        "coverage_pct": 100.0,
                        "note": (f"accuracy {100*acc:.4f}% | AUC 0.500 by construction | "
                                 f"recall on the {n_short:,} short loans 0.0% | "
                                 f"Brier {brier(b_te, p_funded):.6f} | "
                                 f"log-loss {logloss(b_te, p_funded):.6f} "
                                 f"(constant p={p_funded:.6f})")})
        unit += 1; bar(unit, n_units, t0, f"{ty} B3")
    print()

    res = pd.DataFrame(results)
    res.to_csv(os.path.join(a.out, "R001_naive_benchmark.csv"), index=False)

    # ---- 4. report --------------------------------------------------------
    rule("Step 4 of 4  results")
    frac = res[res.benchmark != "B3"]
    print("  FRACTIONAL TARGET — MAE by fold (lower is better)\n")
    piv = frac.pivot_table(index="benchmark", columns="test_year", values="mae")
    print(piv.round(6).to_string())
    print("\n  best (lowest) naive MAE per fold — this is the number to beat:\n")
    for ty in piv.columns:
        col = piv[ty].dropna()
        print(f"    test {ty}:  {col.min():.6f}  ({col.idxmin()})   "
              f"[B0 = {piv.loc['B0', ty]:.6f}]")
    print("\n  GROUP-LOOKUP COVERAGE (unseen cells fall back to the global train mean)\n")
    cov = frac[frac.benchmark.str.startswith("B2")].pivot_table(
        index="benchmark", columns="test_year", values="coverage_pct")
    print(cov.round(3).to_string())
    print("\n  BINARY BASELINE (B3)\n")
    for _, r in res[res.benchmark == "B3"].iterrows():
        print(f"    test {r.test_year}: {r.note}")

    rule("R001 complete")
    print(f"  -> {os.path.join(a.out, 'R001_naive_benchmark.csv')}")
    print(f"  -> {os.path.join(a.out, 'R001_naive_by_year.csv')}")
    print("\n  Record as R001 in RESULTS_LOG section 4. Every subsequent model run")
    print("  must report its fold MAE against the matching row above.")


if __name__ == "__main__":
    main()
