#!/usr/bin/env python3
# Code written with the assistance of Claude (Anthropic).
"""
who-gets-funded — out-of-regime feature extract: 2020-01-01 to 2026-12-31.

WHAT
  The identical feat_v1 SELECT (single source of truth:
  src/features/03_features_select.sql) with only the two date bounds swapped, so
  the out-of-regime rows get exactly the same columns and transformations as the
  cohort. Same guarantees as 03_build_features.py: the database is opened
  mode=ro&immutable=1 — SQLite writes nothing, not even -wal/-shm sidecars.

  Same filters otherwise: LoanPartner, funded+expired. Expect roughly 1.25m rows
  and ~260 MB.

OUTPUT  outputs/tables/feat_oot_2020plus.csv

USAGE   python3 src/features/03c_build_oot_features.py
        (needs the database: ~/Downloads/kiva_raw.sqlite; run in your console)
"""
import argparse, csv, math, os, re, sqlite3, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SELECT_SQL = os.path.join(HERE, "03_features_select.sql")
BAR_W = 34


def hms(s):
    s = int(s)
    return f"{s}s" if s < 60 else (f"{s//60}m {s%60:02d}s" if s < 3600
                                   else f"{s//3600}h {(s%3600)//60:02d}m")


def bar(done, total, t0):
    frac = 0.0 if not total else min(1.0, done / total)
    fill = int(BAR_W * frac)
    el = time.time() - t0
    rate = done / el if el > 0 else 0
    eta = (total - done) / rate if rate > 0 and total else 0
    sys.stdout.write(f"\r  [{'#'*fill}{'.'*(BAR_W-fill)}] {frac*100:5.1f}%  "
                     f"{done:>9,} / {total:,} rows  {hms(el)} elapsed  "
                     f"{rate:>7,.0f}/s  ETA {hms(eta)}   ")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.expanduser("~/Downloads/kiva_raw.sqlite"))
    ap.add_argument("--out", default="outputs/tables/feat_oot_2020plus.csv")
    a = ap.parse_args()
    if not os.path.exists(a.db):
        sys.exit(f"database not found: {a.db}")
    if not os.path.exists(SELECT_SQL):
        sys.exit(f"missing {SELECT_SQL}")

    raw = open(SELECT_SQL, encoding="utf8").read()
    # swap the window in CODE lines only (the header comments also mention the
    # cohort dates); ORDER MATTERS — move the upper bound out of the way first
    code = [l for l in raw.splitlines() if not l.lstrip().startswith("--")]
    assert sum(l.count("'2020-01-01'") for l in code) == 1, "expected one upper bound"
    assert sum(l.count("'2013-01-01'") for l in code) == 1, "expected one lower bound"
    out = []
    for l in raw.splitlines():
        if not l.lstrip().startswith("--"):
            l = l.replace("'2020-01-01'", "'2027-01-01'")
            l = l.replace("'2013-01-01'", "'2020-01-01'")
        out.append(l)
    sql = "\n".join(out)
    assert not any("'2013-01-01'" in l for l in sql.splitlines()
                   if not l.lstrip().startswith("--"))

    print("=" * 78)
    print("who-gets-funded  out-of-regime feature extract (2020-01-01 .. 2026-12-31)")
    print("=" * 78)
    print(f"  database : {a.db}  (read-only, immutable=1)")
    uri = f"file:{a.db}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    con.create_function("ln", 1, lambda v: math.log(v) if v and v > 0 else None)

    print("  counting rows (one scan, ~80s) ...")
    t0 = time.time()
    total = con.execute(
        "SELECT COUNT(*) FROM loans WHERE typename='LoanPartner' "
        "AND status IN ('funded','expired') "
        "AND fundraising_date >= '2020-01-01' "
        "AND fundraising_date < '2027-01-01'").fetchone()[0]
    print(f"  {total:,} rows in {hms(time.time()-t0)}")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    t0 = time.time()
    cur = con.execute(sql)
    names = [d[0] for d in cur.description]
    done = 0
    with open(a.out, "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(names)
        bar(0, total, t0)
        while True:
            rows = cur.fetchmany(50_000)
            if not rows:
                break
            w.writerows(rows)
            done += len(rows)
            bar(done, total, t0)
    print()
    con.close()
    size = os.path.getsize(a.out)
    print(f"  wrote {done:,} rows x {len(names)} cols in {hms(time.time()-t0)} "
          f"({size/1e6:,.1f} MB) -> {a.out}")
    if done != total:
        print(f"  !! WARNING exported {done:,} but counted {total:,}")
    print("\n  next: python3 src/models/08_oot_regime_test.py")


if __name__ == "__main__":
    main()
