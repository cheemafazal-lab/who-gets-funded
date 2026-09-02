#!/usr/bin/env python3
"""
who-gets-funded — 03d: convert the feature matrices from CSV to Parquet.

WHY
  The full cohort is 1,344,542 rows by 54 columns. Read from CSV without a dtype
  schema, the nine categorical columns are materialised as Python objects and
  peak memory runs to several GB, which is what killed the first attempt at a
  full-cohort fit. Written to Parquet with the DTYPES schema in
  src/pipeline_common.py, the same data loads in a few hundred MB, in seconds,
  with the column types already correct and with year filters pushed down into
  the reader so unwanted years are never materialised.

  This is a storage change only. No row is added, dropped or altered, and the
  conversion verifies that before it writes.

INPUT   outputs/tables/feat_v1_full.csv
        outputs/tables/feat_v1_dev200k.csv
        outputs/tables/feat_oot_2020plus.csv
OUTPUT  the same three paths with a .parquet extension

USAGE   python3 src/features/03d_to_parquet.py
        python3 src/features/03d_to_parquet.py --only outputs/tables/feat_v1_full.csv

Code written with the assistance of Claude (Anthropic).
"""
import argparse
import os
import pathlib
import sys
import time

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pipeline_common import DTYPES, hms, mem_mb, rule

DEFAULT = ["outputs/tables/feat_v1_full.csv",
           "outputs/tables/feat_v1_dev200k.csv",
           "outputs/tables/feat_oot_2020plus.csv"]


def convert(csv_path, chunksize):
    out = csv_path[:-4] + ".parquet"
    if not os.path.exists(csv_path):
        print(f"  skip (not found): {csv_path}")
        return None

    head = pd.read_csv(csv_path, nrows=0)
    dt = {c: t for c, t in DTYPES.items() if c in head.columns}
    unknown = [c for c in head.columns if c not in DTYPES]
    if unknown:
        print(f"  note: no schema entry for {unknown} — pandas will infer these")

    t0 = time.time()
    src_mb = os.path.getsize(csv_path) / 1e6
    print(f"  reading {csv_path}  ({src_mb:,.0f} MB on disk)")

    # Chunked read keeps the parser's own buffers small; the typed chunks are
    # concatenated once, which is still far below an untyped single-shot read.
    chunks = [c for c in pd.read_csv(csv_path, dtype=dt, chunksize=chunksize,
                                     low_memory=False)]
    df = pd.concat(chunks, ignore_index=True)
    del chunks

    n_csv = len(df)
    df.to_parquet(out, engine="pyarrow", compression="zstd", index=False)

    # verification: row count and column set must survive the round trip
    back = pd.read_parquet(out)
    assert len(back) == n_csv, f"row count changed: {n_csv:,} -> {len(back):,}"
    assert list(back.columns) == list(df.columns), "column order changed"
    out_mb = os.path.getsize(out) / 1e6

    print(f"  {n_csv:,} rows x {len(df.columns)} cols")
    print(f"  in memory, typed : {mem_mb(df):,.0f} MB")
    print(f"  parquet on disk  : {out_mb:,.0f} MB  ({src_mb/max(out_mb,1e-9):.1f}x smaller)")
    print(f"  wrote {out} in {hms(time.time() - t0)}\n")
    del df, back
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="convert a single csv")
    ap.add_argument("--chunksize", type=int, default=250_000)
    a = ap.parse_args()

    try:
        import pyarrow  # noqa: F401
    except ImportError:
        sys.exit("pyarrow is required:  pip3 install pyarrow")

    rule("who-gets-funded  03d  feature matrices -> Parquet")
    targets = [a.only] if a.only else DEFAULT
    written = [p for p in (convert(t, a.chunksize) for t in targets) if p]

    rule("03d complete")
    print(f"  {len(written)} file(s) written. Every later step now prefers the")
    print("  .parquet automatically; the .csv files are left untouched.")


if __name__ == "__main__":
    main()
