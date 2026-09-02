#!/usr/bin/env python3
# Code written with the assistance of Claude (Anthropic).
"""
kiva_harvest2.py — rebuild the full Kiva loan book from the GraphQL API.

Replaces the retired build.kiva.org / s3.kiva.org snapshot, whose distribution
host (a CNAME into Limelight/Edgio's footprint.net CDN) stopped resolving after
Edgio's network was shut down on 15 January 2025 following its Chapter 11
liquidation. Kiva's docs page still advertises the dead URL.

    Project: "Who Gets Funded?" -- F. Cheema, QUB MSc Business Analytics, 2026

CONFIRMED FACTS (verified against the live gateway, 29 July 2026)
----------------------------------------------------------------
* Endpoint: https://gateway.production.kiva.org/graphql  (POST)
* A User-Agent header is MANDATORY. Bare curl gets 403 from the WAF.
* LoanSearchFiltersInput.status defaults to `fundraising`. Leaving it unset
  returns only the ~7.2k live marketplace loans. Passing `status: all` returns
  totalCount = 3,152,795 -- the entire loan book, expired loans included.
* LoanSearchStatusEnum accepts only {fundraising, funded, all}. There is no
  `expired` search value; you request `all` and read each loan's own
  `status` field (LoanStatusEnum), which does include `expired`.
* `limit: 0` is rejected ("Limit must be positive") despite the schema
  docstring claiming otherwise. Minimum is 1.
* LoanBasic is an INTERFACE implemented by LoanPartner and LoanDirect.
  `themes`, `partnerId`, `partnerName` exist ONLY on LoanPartner.
* There is no `postedDate`. Use `fundraisingDate` as the posting timestamp.
* Filter `loanIds: [Int]` exists, so we walk IDs and never touch the offset
  ceiling that would otherwise cap us long before 3.15M records.

SELF-HEALING SCHEMA PROBE
-------------------------
Nested field names (Geocode, Money, Borrower, LoanTerm) are best-guess. Rather
than another round of trial and error, --probe reads the gateway's own
"Cannot query field X on type Y" errors, strips the offending fields, and
retries until the query validates. It then prints the surviving field set.

USAGE
-----
    python3 kiva_harvest2.py --probe                # settle the schema, do this first
    python3 kiva_harvest2.py --maxid                # find the highest live loan id
    python3 kiva_harvest2.py --harvest --limit-batches 5    # smoke test
    python3 kiva_harvest2.py --harvest --end 3700000       # the real run
    python3 kiva_harvest2.py --status
    python3 kiva_harvest2.py --export kiva_loans.csv

Python 3.9+, standard library only.
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = "https://gateway.production.kiva.org/graphql"
DB_DEFAULT = "kiva_raw.sqlite"
UA = "QUB-MSc-BusinessAnalytics-research/1.0 (academic dissertation; +https://www.qub.ac.uk)"

DELAY_SECONDS = 0.35        # be kind; Kiva is a nonprofit
MAX_RETRIES = 6
TIMEOUT = 90

FIELDS_CACHE = "kiva_fields.json"   # written by --probe, read by --harvest

# --- candidate field set ----------------------------------------------------
# One entry per line so the probe can strip individual offenders.
# Outcome fields (raisedDate, expiredDate, paidAmount, reservedAmount) are
# collected deliberately: they define the target and the audit, and must NOT be
# used as predictors. See PREDICTOR_SAFE below.
CANDIDATE_FIELDS = [
    "__typename",
    "id",
    "status",
    "statusLabel",
    "loanAmount",
    "loanFundraisingInfo { fundedAmount reservedAmount isExpiringSoon }",
    "borrowerCount",
    "gender",
    "borrowers { gender pictured isPrimary }",
    "sector { id name }",
    "activity { id name }",
    "geocode { city state latitude longitude country { isoCode name region } }",
    "fundraisingDate",
    "plannedExpirationDate",
    "raisedDate",
    "expiredDate",
    "endedDate",
    "disbursalDate",
    "defaultedDate",
    "refundedDate",
    "repaymentInterval",
    "lenderRepaymentTerm",
    "distributionModel",
    "anonymizationLevel",
    "minNoteSize",
    "paidAmount",
    "delinquent",
    "researchScore",
    "tags",
    "use",
    "description",
    "originalLanguage { id name isoCode }",
    "previousLoanId",
    "isMatchable",
    "matchRatio",
    "matcherName",
    "inPfp",
    "pfpMinLenders",
    "terms { disbursalAmount lossLiabilityCurrencyExchange lossLiabilityNonpayment }",
    "... on LoanPartner { partnerId partnerName themes journalCount }",
]

# Fields safe to use as predictors (known at posting time). Everything else in
# the table is outcome or metadata. Kept here so the modelling script can import
# it and you cannot leak by accident.
PREDICTOR_SAFE = [
    "loan_amount", "borrower_count", "gender", "borrower_genders", "is_group",
    "sector", "activity", "country_iso", "region", "fundraising_date",
    "fundraising_month", "repayment_interval", "lender_repayment_term",
    "distribution_model", "min_note_size", "tags", "themes", "partner_id",
    "use_text_length", "original_language", "previous_loan_id", "is_repeat_borrower",
    "is_matchable", "match_ratio", "in_pfp", "planned_expiration_date",
    "fundraising_window_days",
]


def build_query(fields):
    body = "\n        ".join(fields)
    return """
query Batch($ids: [Int], $lim: Int!) {
  lend {
    loans(filters: {loanIds: $ids, status: all}, limit: $lim) {
      totalCount
      values {
        %s
      }
    }
  }
}
""" % body


MAXID_QUERY = """
{ lend { loans(filters: {status: all}, limit: 1, sortBy: newest) {
    totalCount values { id fundraisingDate } } } }
"""

COUNT_QUERY = """
{ lend { loans(filters: {status: all}, limit: 1) { totalCount } } }
"""


# --- transport --------------------------------------------------------------

def gql(query, variables=None, raise_on_error=False):
    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    last = None
    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(
            ENDPOINT, data=payload,
            headers={"Content-Type": "application/json",
                     "Accept": "application/json",
                     "User-Agent": UA},          # omit this and the WAF 403s you
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                doc = json.loads(body)           # GraphQL validation errors: 400 + JSON
                if not raise_on_error:
                    return doc
                return doc
            except ValueError:
                last = "HTTP %s: %s" % (e.code, body[:200])
                if e.code in (403,):
                    raise RuntimeError(
                        "403 from the WAF. The User-Agent header is required.\n" + last)
                if e.code in (429, 500, 502, 503, 504):
                    wait = min(90, 2 ** attempt)
                    sys.stderr.write("  %s -- backing off %ss\n" % (last[:70], wait))
                    time.sleep(wait)
                    continue
                raise RuntimeError(last)
        except Exception as e:
            last = str(e)
            wait = min(90, 2 ** attempt)
            sys.stderr.write("  %s -- retry in %ss\n" % (last[:70], wait))
            time.sleep(wait)
    raise RuntimeError("gave up after %d attempts: %s" % (MAX_RETRIES, last))


# --- self-healing schema probe ---------------------------------------------

BAD_FIELD = re.compile(r'Cannot query field "([^"]+)" on type "([^"]+)"')
UNKNOWN_ARG = re.compile(r'Unknown argument "([^"]+)"')


def strip_field(fields, name):
    """Remove `name` wherever it appears -- as a whole line or a nested token."""
    out = []
    for line in fields:
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", line)
        if not tokens:
            out.append(line)
            continue
        # Leading token is the field itself -> drop the whole line/block.
        if tokens[0] == name or (line.startswith("...") and tokens[2:3] == [name]):
            continue
        if name in tokens:
            # Nested subfield: remove just that token from the block.
            new = re.sub(r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])" % re.escape(name),
                         "", line)
            new = re.sub(r"\s+", " ", new).strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*\{\s*\}$", new):
                continue          # block emptied out entirely
            out.append(new)
        else:
            out.append(line)
    return out


def cmd_probe():
    print("Endpoint: %s" % ENDPOINT)
    r = gql(COUNT_QUERY)
    total = (((r.get("data") or {}).get("lend") or {}).get("loans") or {}).get("totalCount")
    print("Platform total (status: all): %s" % (format(total, ",") if total else r))

    fields = list(CANDIDATE_FIELDS)
    removed = []
    for round_no in range(1, 16):
        r = gql(build_query(fields), {"ids": [88, 1000], "lim": 2})
        errs = r.get("errors") or []
        bad = []
        for e in errs:
            m = BAD_FIELD.search(e.get("message", ""))
            if m:
                bad.append(m.group(1))
        if not bad:
            vals = (((r.get("data") or {}).get("lend") or {})
                    .get("loans") or {}).get("values") or []
            if errs:
                print("\nNon-field errors remain:")
                for e in errs:
                    print("   !", e.get("message"))
            print("\nQuery validated after %d round(s)." % (round_no - 1))
            if removed:
                print("Fields the gateway rejected: %s" % ", ".join(sorted(set(removed))))
            with open(FIELDS_CACHE, "w", encoding="utf-8") as fh:
                json.dump(fields, fh, indent=1)
            print("Working field set cached to %s\n" % FIELDS_CACHE)
            if vals:
                print("Sample loan:")
                print(json.dumps(vals[0], indent=2)[:3000])
                print("\nFlattened preview:")
                row = flatten(vals[0])
                for k, v in list(zip(COLUMNS, row))[:26]:
                    print("  %-24s %r" % (k, v))
            else:
                print("No values returned. Raw:", json.dumps(r)[:1000])
            return
        for name in bad:
            fields = strip_field(fields, name)
        removed += bad
        print("  round %d: dropping %s" % (round_no, ", ".join(bad)))
    print("Still failing after 15 rounds. Last response:")
    print(json.dumps(r, indent=2)[:2000])


def cmd_maxid():
    r = gql(MAXID_QUERY)
    print(json.dumps(r, indent=2))
    print("\nUse the id above (rounded up) as --end for the harvest.")


# --- storage ----------------------------------------------------------------

COLUMNS = [
    "loan_id", "typename", "status", "status_label",
    "loan_amount", "funded_amount", "share_raised", "reserved_amount",
    "borrower_count", "is_group", "gender", "borrower_genders",
    "sector", "activity",
    "country_iso", "country_name", "region", "city", "state",
    "fundraising_date", "fundraising_month", "planned_expiration_date",
    "fundraising_window_days",
    "raised_date", "expired_date", "ended_date", "disbursal_date",
    "defaulted_date", "refunded_date",
    "repayment_interval", "lender_repayment_term", "distribution_model",
    "anonymization_level", "min_note_size", "paid_amount", "delinquent",
    "research_score", "tags", "themes", "use_text_length", "description_length",
    "original_language", "previous_loan_id", "is_repeat_borrower",
    "is_matchable", "match_ratio", "matcher_name", "in_pfp",
    "partner_id", "partner_name",
    "raw", "harvested_at",
]

SCHEMA = """
CREATE TABLE IF NOT EXISTS loans (
  loan_id INTEGER PRIMARY KEY, typename TEXT, status TEXT, status_label TEXT,
  loan_amount REAL, funded_amount REAL, share_raised REAL, reserved_amount REAL,
  borrower_count INTEGER, is_group INTEGER, gender TEXT, borrower_genders TEXT,
  sector TEXT, activity TEXT,
  country_iso TEXT, country_name TEXT, region TEXT, city TEXT, state TEXT,
  fundraising_date TEXT, fundraising_month TEXT, planned_expiration_date TEXT,
  fundraising_window_days REAL,
  raised_date TEXT, expired_date TEXT, ended_date TEXT, disbursal_date TEXT,
  defaulted_date TEXT, refunded_date TEXT,
  repayment_interval TEXT, lender_repayment_term INTEGER, distribution_model TEXT,
  anonymization_level TEXT, min_note_size REAL, paid_amount REAL, delinquent INTEGER,
  research_score REAL, tags TEXT, themes TEXT, use_text_length INTEGER,
  description_length INTEGER, original_language TEXT, previous_loan_id INTEGER,
  is_repeat_borrower INTEGER, is_matchable INTEGER, match_ratio INTEGER,
  matcher_name TEXT, in_pfp INTEGER, partner_id INTEGER, partner_name TEXT,
  raw TEXT, harvested_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_fund   ON loans(fundraising_date);
CREATE INDEX IF NOT EXISTS ix_status ON loans(status);
CREATE INDEX IF NOT EXISTS ix_ctry   ON loans(country_iso);
CREATE INDEX IF NOT EXISTS ix_sector ON loans(sector);

CREATE TABLE IF NOT EXISTS harvest_progress (
  batch_start INTEGER PRIMARY KEY, batch_end INTEGER,
  n_returned INTEGER, done_at TEXT
);
"""

INSERT = "INSERT OR REPLACE INTO loans VALUES (%s)" % ",".join("?" * len(COLUMNS))


def connect(path):
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.executescript(SCHEMA)
    return con


def money(v):
    """Money may serialise as a number, a numeric string, or {amount,...}."""
    if v is None:
        return None
    if isinstance(v, dict):
        v = v.get("amount", v.get("value"))
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _days(a, b):
    if not a or not b:
        return None
    try:
        import datetime as dt
        fmt = lambda s: dt.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return (fmt(b) - fmt(a)).total_seconds() / 86400.0
    except Exception:
        return None


def flatten(loan):
    def g(d, *path):
        cur = d
        for k in path:
            if not isinstance(cur, dict) or cur.get(k) is None:
                return None
            cur = cur[k]
        return cur

    requested = money(loan.get("loanAmount"))
    funded = money(g(loan, "loanFundraisingInfo", "fundedAmount"))
    share = None
    if requested and requested > 0 and funded is not None:
        share = min(1.0, funded / requested)

    borrowers = loan.get("borrowers") or []
    genders = ",".join((b.get("gender") or "") for b in borrowers if isinstance(b, dict))
    bcount = loan.get("borrowerCount") or (len(borrowers) or None)

    fdate = loan.get("fundraisingDate")
    tags = loan.get("tags")
    themes = loan.get("themes")
    use = loan.get("use") or ""
    desc = loan.get("description") or ""
    prev = loan.get("previousLoanId")

    return (
        int(loan["id"]),
        loan.get("__typename"),
        loan.get("status"),
        loan.get("statusLabel"),
        requested,
        funded,
        share,
        money(g(loan, "loanFundraisingInfo", "reservedAmount")),
        bcount,
        1 if (bcount or 0) > 1 else 0,
        loan.get("gender"),
        genders or None,
        g(loan, "sector", "name"),
        g(loan, "activity", "name"),
        g(loan, "geocode", "country", "isoCode"),
        g(loan, "geocode", "country", "name"),
        g(loan, "geocode", "country", "region"),
        g(loan, "geocode", "city"),
        g(loan, "geocode", "state"),
        fdate,
        str(fdate)[:7] if fdate else None,
        loan.get("plannedExpirationDate"),
        _days(fdate, loan.get("plannedExpirationDate")),
        loan.get("raisedDate"),
        loan.get("expiredDate"),
        loan.get("endedDate"),
        loan.get("disbursalDate"),
        loan.get("defaultedDate"),
        loan.get("refundedDate"),
        loan.get("repaymentInterval"),
        loan.get("lenderRepaymentTerm"),
        loan.get("distributionModel"),
        loan.get("anonymizationLevel"),
        money(loan.get("minNoteSize")),
        money(loan.get("paidAmount")),
        1 if loan.get("delinquent") else 0,
        loan.get("researchScore"),
        ",".join(map(str, tags)) if isinstance(tags, list) else tags,
        ",".join(map(str, themes)) if isinstance(themes, list) else themes,
        len(use) or None,
        len(desc) or None,
        g(loan, "originalLanguage", "isoCode") or g(loan, "originalLanguage", "name"),
        prev,
        1 if prev else 0,
        1 if loan.get("isMatchable") else 0,
        loan.get("matchRatio"),
        loan.get("matcherName"),
        1 if loan.get("inPfp") else 0,
        loan.get("partnerId"),
        loan.get("partnerName"),
        json.dumps(loan, separators=(",", ":")),
        time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


# --- harvest ----------------------------------------------------------------

def load_fields():
    if os.path.exists(FIELDS_CACHE):
        with open(FIELDS_CACHE, encoding="utf-8") as fh:
            return json.load(fh)
    sys.exit("No %s found. Run --probe first so the field set is verified."
             % FIELDS_CACHE)


def cmd_harvest(db, start, end, batch, limit_batches):
    fields = load_fields()
    query = build_query(fields)
    con = connect(db)
    done = {r[0] for r in con.execute("SELECT batch_start FROM harvest_progress")}
    new = batches = 0
    t0 = time.time()

    for lo in range(start, end + 1, batch):
        if lo in done:
            continue
        if limit_batches and batches >= limit_batches:
            print("\nStopped at --limit-batches %d." % limit_batches)
            break
        hi = min(lo + batch - 1, end)
        ids = list(range(lo, hi + 1))

        r = gql(query, {"ids": ids, "lim": len(ids)})
        if r.get("errors"):
            msgs = "; ".join(e.get("message", "?") for e in r["errors"])
            if BAD_FIELD.search(msgs):
                sys.exit("Schema drifted mid-run. Re-run --probe.\n" + msgs[:400])
            sys.stderr.write("  partial errors at %d: %s\n" % (lo, msgs[:180]))

        vals = (((r.get("data") or {}).get("lend") or {})
                .get("loans") or {}).get("values") or []
        rows = []
        for ln in vals:
            if isinstance(ln, dict) and ln.get("id") is not None:
                try:
                    rows.append(flatten(ln))
                except Exception as e:
                    sys.stderr.write("  flatten %s: %s\n" % (ln.get("id"), e))

        con.executemany(INSERT, rows)
        con.execute("INSERT OR REPLACE INTO harvest_progress VALUES (?,?,?,?)",
                    (lo, hi, len(rows), time.strftime("%Y-%m-%dT%H:%M:%S")))
        con.commit()
        new += len(rows)
        batches += 1

        if batches % 20 == 0:
            el = max(1e-9, time.time() - t0)
            pct = 100.0 * (lo - start) / max(1, end - start)
            eta = (end - lo) / max(1e-9, (lo - start) / el) / 3600 if lo > start else 0
            print("  id %9s | %5.1f%% | %9s loans | %5.1f/s | eta %.1fh"
                  % (format(lo, ","), pct, format(new, ","), new / el, eta))
        time.sleep(DELAY_SECONDS)

    n = con.execute("SELECT COUNT(*) FROM loans").fetchone()[0]
    print("\n+%s loans this run. Database holds %s." % (format(new, ","), format(n, ",")))
    con.close()


def cmd_status(db):
    if not os.path.exists(db):
        return print("No database at %s yet." % db)
    con = connect(db)
    n = con.execute("SELECT COUNT(*) FROM loans").fetchone()[0]
    print("Loans stored: %s  (platform total 3,152,795 => %.1f%%)"
          % (format(n, ","), 100.0 * n / 3152795))
    print("\nBy status:")
    for s, c in con.execute("SELECT COALESCE(status,'(null)'),COUNT(*) "
                            "FROM loans GROUP BY 1 ORDER BY 2 DESC"):
        print("  %-18s %10s" % (s, format(c, ",")))
    row = con.execute("SELECT MIN(fundraising_date),MAX(fundraising_date) FROM loans "
                      "WHERE fundraising_date IS NOT NULL").fetchone()
    print("\nFundraising-date range: %s -> %s" % row)
    print("\nMean share raised, by status:")
    for s, m, c in con.execute(
            "SELECT status, AVG(share_raised), COUNT(*) FROM loans "
            "WHERE share_raised IS NOT NULL GROUP BY 1 ORDER BY 3 DESC"):
        print("  %-18s %.4f  (n=%s)" % (s, m, format(c, ",")))
    print("\nExpiry rate by gender (the RQ3 headline):")
    for g, c, e in con.execute(
            "SELECT gender, COUNT(*), SUM(CASE WHEN status='expired' THEN 1 ELSE 0 END) "
            "FROM loans WHERE gender IS NOT NULL GROUP BY 1"):
        print("  %-10s n=%10s  expired=%8s  (%.2f%%)"
              % (g, format(c, ","), format(e, ","), 100.0 * e / max(1, c)))
    con.close()


def cmd_export(db, out):
    import csv
    con = connect(db)
    cols = [c for c in COLUMNS if c != "raw"]
    cur = con.execute("SELECT %s FROM loans ORDER BY loan_id" % ",".join(cols))
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        n = 0
        for row in cur:
            w.writerow(row)
            n += 1
    print("Wrote %s rows to %s" % (format(n, ","), out))
    print("Predictor-safe columns (no target leakage): %s" % ", ".join(PREDICTOR_SAFE))
    con.close()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default=DB_DEFAULT)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--maxid", action="store_true")
    p.add_argument("--harvest", action="store_true")
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=3_700_000)
    p.add_argument("--batch", type=int, default=50)
    p.add_argument("--limit-batches", type=int, default=0)
    p.add_argument("--status", action="store_true")
    p.add_argument("--export", metavar="CSV")
    a = p.parse_args()

    if a.probe:
        cmd_probe()
    elif a.maxid:
        cmd_maxid()
    elif a.harvest:
        cmd_harvest(a.db, a.start, a.end, max(1, min(a.batch, 100)), a.limit_batches)
    elif a.status:
        cmd_status(a.db)
    elif a.export:
        cmd_export(a.db, a.export)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
