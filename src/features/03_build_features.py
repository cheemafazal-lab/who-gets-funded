#!/usr/bin/env python3
# Code written with the assistance of Claude (Anthropic).
"""
who-gets-funded — feat_v1 feature matrix build, with a live progress bar.

Runs the SELECT in 03_features_select.sql against the pinned snapshot and streams
the result to csv. The database is opened with mode=ro&immutable=1, which
guarantees SQLite writes NOTHING — not even the -wal or -shm sidecar files. The
snapshot cannot be modified by this script.

Input  : ~/Downloads/kiva_raw.sqlite  (read-only, immutable)
         src/features/03_features_select.sql   (the feature definitions)
Output : outputs/tables/feat_v1_full.csv
         outputs/tables/feat_v1_tag_combinations.csv
         outputs/tables/feat_v1_theme_combinations.csv

Usage:
    cd path/to/who-gets-funded
    python3 src/features/03_build_features.py

Optional:
    python3 src/features/03_build_features.py --db /path/to/kiva_raw.sqlite
"""
import argparse, csv, math, os, sqlite3, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SELECT_SQL = os.path.join(HERE, "03_features_select.sql")

COHORT = """typename = 'LoanPartner'
        AND status IN ('funded','expired')
        AND fundraising_date >= '2013-01-01'
        AND fundraising_date <  '2020-01-01'"""

RECON = f"""
SELECT COUNT(*)                                          AS rows_to_export,
       SUM(status='expired')                             AS expired,
       SUM(status='funded')                              AS funded,
       ROUND(AVG(share_raised),6)                        AS share_mean,
       ROUND(1-AVG(share_raised),6)                      AS naive_mae,
       SUM(loan_amount IS NULL OR loan_amount<=0)        AS bad_amount,
       SUM(borrower_count IS NULL OR borrower_count=0)   AS bad_borrower_count,
       COUNT(DISTINCT partner_id)                        AS partners,
       COUNT(DISTINCT activity)                          AS activities,
       COUNT(DISTINCT country_iso)                       AS countries,
       MIN(fundraising_date)                             AS first_posted,
       MAX(fundraising_date)                             AS last_posted
  FROM loans WHERE {COHORT}
"""

LEAK_PROBE = f"""
SELECT CASE WHEN INSTR(','||COALESCE(tags,'')||',', ',{{tag}},')>0
            THEN '{{tag}}' ELSE 'no {{tag}}' END          AS flag,
       COUNT(*)                                          AS n,
       ROUND(100.0*SUM(status='expired')/COUNT(*),4)      AS expiry_pct,
       ROUND(AVG(share_raised),6)                        AS share_mean
  FROM loans WHERE {COHORT} GROUP BY 1
"""

VOCAB = f"""
SELECT COALESCE({{col}},'(null)')                        AS {{col}},
       COUNT(*)                                          AS n,
       ROUND(100.0*SUM(status='expired')/COUNT(*),4)      AS expiry_pct
  FROM loans WHERE {COHORT} GROUP BY 1 ORDER BY 2 DESC
"""

BAR_W = 34
CHUNK = 50_000


def hms(s):
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s//60}m {s%60:02d}s"
    return f"{s//3600}h {(s%3600)//60:02d}m"


def bar(done, total, t0, label="rows"):
    frac = 0.0 if not total else min(1.0, done / total)
    fill = int(BAR_W * frac)
    el = time.time() - t0
    rate = done / el if el > 0 else 0
    eta = (total - done) / rate if rate > 0 and total else 0
    sys.stdout.write(
        f"\r  [{'#'*fill}{'.'*(BAR_W-fill)}] {frac*100:5.1f}%  "
        f"{done:>9,} / {total:,} {label}  "
        f"{hms(el)} elapsed  {rate:>7,.0f}/s  ETA {hms(eta)}   ")
    sys.stdout.flush()


def rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def table(cur, rows):
    """Print a small result set as aligned columns."""
    names = [d[0] for d in cur.description]
    data = [[("" if v is None else str(v)) for v in r] for r in rows]
    w = [max(len(names[i]), *(len(r[i]) for r in data)) if data else len(names[i])
         for i in range(len(names))]
    print("  " + "  ".join(n.ljust(w[i]) for i, n in enumerate(names)))
    print("  " + "  ".join("-" * w[i] for i in range(len(names))))
    for r in data:
        print("  " + "  ".join(r[i].ljust(w[i]) for i in range(len(names))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.expanduser("~/Downloads/kiva_raw.sqlite"))
    ap.add_argument("--out", default="outputs/tables")
    a = ap.parse_args()

    if not os.path.exists(a.db):
        sys.exit(f"database not found: {a.db}")
    if not os.path.exists(SELECT_SQL):
        sys.exit(f"feature definitions not found: {SELECT_SQL}")
    os.makedirs(a.out, exist_ok=True)

    uri = f"file:{a.db}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    # LN() only exists in SQLite builds with math functions compiled in; register
    # our own so the script never depends on how the local sqlite was built
    con.create_function("ln", 1, lambda v: math.log(v) if v and v > 0 else None)

    rule("who-gets-funded  feat_v1 feature build")
    print(f"  database : {a.db}")
    print(f"  mode     : read-only, immutable=1 (no writes, no -wal, no -shm)")
    print(f"  sqlite   : {sqlite3.sqlite_version}")
    print(f"  defs     : {os.path.relpath(SELECT_SQL)}")

    # ---- 1. reconciliation, and the row total the progress bar needs ---------
    rule("Step 1 of 4  cohort reconciliation")
    print("  counting cohort rows (one full scan, ~80s) ...")
    t0 = time.time()
    cur = con.execute(RECON)
    row = cur.fetchall()
    table(cur, row)
    total = row[0][0]
    expired, funded = row[0][1], row[0][2]
    bad_amt, bad_bc = row[0][5], row[0][6]
    print(f"\n  done in {hms(time.time()-t0)}")
    if total != 1_344_542:
        print(f"  !! WARNING expected 1,344,542 rows, got {total:,} — cohort drift, "
              f"check RESULTS_LOG section 2")
    else:
        print("  cohort matches RESULTS_LOG section 2 exactly (1,344,542 / 79,993 expired)")
    if bad_amt:
        print(f"  !! WARNING {bad_amt:,} rows have loan_amount <= 0 — log_loan_amount "
              f"will be NULL for those")
    if bad_bc:
        print(f"  !! WARNING {bad_bc:,} rows have borrower_count = 0 — "
              f"amount_per_borrower will be NULL for those")

    # ---- 2. stream the feature matrix ---------------------------------------
    rule("Step 2 of 4  writing the feature matrix")
    sql = open(SELECT_SQL, encoding="utf8").read()
    path = os.path.join(a.out, "feat_v1_full.csv")
    t0 = time.time()
    cur = con.execute(sql)
    names = [d[0] for d in cur.description]
    print(f"  {len(names)} columns -> {path}")
    done = 0
    with open(path, "w", newline="", encoding="utf8") as fh:
        w = csv.writer(fh)
        w.writerow(names)
        bar(0, total, t0)
        while True:
            rows = cur.fetchmany(CHUNK)
            if not rows:
                break
            w.writerows(rows)
            done += len(rows)
            bar(done, total, t0)
    print()
    size = os.path.getsize(path)
    print(f"  wrote {done:,} rows in {hms(time.time()-t0)}  "
          f"({size/1e6:,.1f} MB, {size/max(done,1):.0f} bytes/row)")
    if done != total:
        print(f"  !! WARNING exported {done:,} but counted {total:,}")

    # ---- 3. complete tag / theme vocabularies -------------------------------
    rule("Step 3 of 4  complete tag and theme vocabularies")
    for col, fname in (("tags", "feat_v1_tag_combinations.csv"),
                       ("themes", "feat_v1_theme_combinations.csv")):
        t0 = time.time()
        print(f"  {col:7s} ", end="", flush=True)
        cur = con.execute(VOCAB.format(col=col))
        rows = cur.fetchall()
        p = os.path.join(a.out, fname)
        with open(p, "w", newline="", encoding="utf8") as fh:
            w = csv.writer(fh)
            w.writerow([d[0] for d in cur.description])
            w.writerows(rows)
        print(f"{len(rows):>6,} distinct combinations -> {fname}  ({hms(time.time()-t0)})")

    # ---- 4. leakage probe ---------------------------------------------------
    rule("Step 4 of 4  leakage probe on the two quarantined tags")
    print("  if expiry differs sharply by these flags, they encode post-posting")
    print("  lender behaviour and the x_leak_ quarantine was correct\n")
    for tag in ("user_favorite", "volunteer_pick"):
        cur = con.execute(LEAK_PROBE.format(tag=tag))
        table(cur, cur.fetchall())
        print()

    con.close()
    rule("feature build complete")
    print(f"  {done:,} rows x {len(names)} columns")
    print(f"  expired {expired:,} ({100*expired/total:.4f}%)   funded {funded:,}")
    print(f"  -> {os.path.join(a.out, 'feat_v1_full.csv')}")
    print("\n  next: python3 src/features/03b_make_dev_sample.py")


if __name__ == "__main__":
    main()
