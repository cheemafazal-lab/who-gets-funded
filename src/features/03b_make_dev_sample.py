#!/usr/bin/env python3
# Code written with the assistance of Claude (Anthropic).
"""
who-gets-funded — draw the ~200k stratified development sample from feat_v1_full.csv.

Why: HANDOVER rule 4 — build the pipeline on ~200k rows, run final fits on the
full cohort. Sampling is PROPORTIONAL (not balanced) so class prevalence, the
naive benchmark and calibration all stay comparable to the full cohort. A
balanced sample would silently invalidate every MAE comparison.

Strata: posting_year x is_fully_funded x gender.

Input  : outputs/tables/feat_v1_full.csv
Output : outputs/tables/feat_v1_dev200k.csv
         outputs/tables/feat_v1_sample_check.csv   (full vs sample marginals)

Memory-safe: reads the strata columns only to pick row positions, then streams
the full file in chunks keeping just the selected rows. Never holds the whole
matrix in memory.

Usage: python3 src/features/03b_make_dev_sample.py [target_n]
"""
import sys, os
import numpy as np
import pandas as pd

SRC    = "outputs/tables/feat_v1_full.csv"
OUT    = "outputs/tables/feat_v1_dev200k.csv"
CHECK  = "outputs/tables/feat_v1_sample_check.csv"
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
SEED   = 40465466        # student number: the draw is reproducible and documented
STRATA = ["posting_year", "is_fully_funded", "gender"]

if not os.path.exists(SRC):
    sys.exit(f"missing {SRC} — run 03_build_features.sql first")

print(f"reading strata keys from {SRC} ...")
keys = pd.read_csv(SRC, usecols=STRATA)
N = len(keys)
frac = TARGET / N
print(f"  {N:,} rows -> target {TARGET:,}, sampling fraction {frac:.6f}")

rng = np.random.default_rng(SEED)
keys["_g"] = keys[STRATA].astype(str).agg("|".join, axis=1)

# proportional allocation, floor of 1 per stratum so no cell disappears entirely
take = []
for _, pos in keys.groupby("_g", sort=True).indices.items():
    pos = np.asarray(pos)
    k = min(max(1, int(round(len(pos) * frac))), len(pos))
    take.append(rng.choice(pos, size=k, replace=False))
sel = np.sort(np.concatenate(take))
print(f"  selected {len(sel):,} rows across {len(take)} strata")

# boolean mask over row positions — chunk indices from read_csv are global and
# contiguous, so position i in the file is mask[i]
mask = np.zeros(N, dtype=bool)
mask[sel] = True

print(f"writing {OUT} ...")
first, written = True, 0
for chunk in pd.read_csv(SRC, chunksize=200_000, low_memory=False):
    keep = chunk[mask[chunk.index.values]]
    if len(keep):
        keep.to_csv(OUT, mode="w" if first else "a", header=first, index=False)
        first, written = False, written + len(keep)
print(f"  wrote {written:,} rows")
assert written == len(sel), f"wrote {written} but selected {len(sel)}"

# ---- verification: do the sample marginals match the full cohort? -----------
print("verifying marginals ...")
vcols = STRATA + ["share_raised", "sector", "country_iso"]
full = pd.read_csv(SRC, usecols=vcols, low_memory=False)
samp = pd.read_csv(OUT, usecols=vcols, low_memory=False)

rows = [
    {"dimension": "overall", "level": "expiry rate (%)",
     "full": round(100 * (1 - full.is_fully_funded.mean()), 4),
     "sample": round(100 * (1 - samp.is_fully_funded.mean()), 4)},
    {"dimension": "overall", "level": "mean share_raised",
     "full": round(full.share_raised.mean(), 6),
     "sample": round(samp.share_raised.mean(), 6)},
    {"dimension": "overall", "level": "naive MAE",
     "full": round(1 - full.share_raised.mean(), 6),
     "sample": round(1 - samp.share_raised.mean(), 6)},
    {"dimension": "overall", "level": "rows",
     "full": len(full), "sample": len(samp)},
]
for r in rows:
    r["diff"] = round(r["sample"] - r["full"], 6)


def share_rows(dim, levels=None):
    f = full[dim].value_counts(normalize=True)
    s = samp[dim].value_counts(normalize=True)
    for lvl in (levels if levels is not None else f.index):
        fv, sv = 100 * f.get(lvl, 0.0), 100 * s.get(lvl, 0.0)
        rows.append({"dimension": f"{dim} (% of rows)", "level": str(lvl),
                     "full": round(fv, 4), "sample": round(sv, 4),
                     "diff": round(sv - fv, 4)})


share_rows("posting_year")
share_rows("gender")
share_rows("sector")
share_rows("country_iso", full.country_iso.value_counts(normalize=True).head(15).index)

chk = pd.DataFrame(rows)
chk.to_csv(CHECK, index=False)
print(f"  wrote {CHECK}")

pct = chk[chk.dimension.str.contains("% of rows")].copy()
worst = pct.reindex(pct["diff"].abs().sort_values(ascending=False).index).head(5)
print(f"\n  full expiry {100*(1-full.is_fully_funded.mean()):.4f}%    "
      f"sample expiry {100*(1-samp.is_fully_funded.mean()):.4f}%")
print(f"  full naive MAE {1-full.share_raised.mean():.6f}    "
      f"sample naive MAE {1-samp.share_raised.mean():.6f}")
print("\n  largest marginal drifts (percentage points):")
for _, r in worst.iterrows():
    print(f"    {r.dimension:26s} {r.level:16s} {r['diff']:+.4f}")
print(f"\n  seed {SEED} — rerunning reproduces this exact sample")
