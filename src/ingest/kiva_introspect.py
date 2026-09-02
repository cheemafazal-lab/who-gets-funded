#!/usr/bin/env python3
# Code written with the assistance of Claude (Anthropic).
"""
kiva_introspect.py — ask Kiva's GraphQL gateway what its schema actually is.

The gateway rejected several documented field names ("Cannot query field
'fundedAmount' on type 'LoanBasic'"), and the default lend.loans query returns
only ~7k currently-fundraising loans rather than the 3.1M loan book. Both
questions are answerable by introspection, so stop guessing and ask the server.

    python3 kiva_introspect.py            > kiva_schema.txt
    python3 kiva_introspect.py LoanBasic  # or any single type

Send me kiva_schema.txt and I'll pin the harvester's field list and filters.
Python 3.9 compatible, standard library only.
"""

import json
import sys
import time
import urllib.error
import urllib.request

ENDPOINT = "https://gateway.production.kiva.org/graphql"
UA = "QUB-MSc-BusinessAnalytics-research/1.0 (academic)"


def gql(query, variables=None):
    payload = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        ENDPOINT, data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "User-Agent": UA},
        method="POST")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                return json.loads(body)          # GraphQL errors arrive as 400 + JSON
            except ValueError:
                print("HTTP %s: %s" % (e.code, body[:300]), file=sys.stderr)
                return {}
        except Exception as e:
            print("  retry (%s)" % e, file=sys.stderr)
            time.sleep(2 ** attempt)
    return {}


TYPE_Q = """
query T($name: String!) {
  __type(name: $name) {
    name kind description
    fields {
      name
      args { name type { name kind ofType { name kind ofType { name kind } } } }
      type { name kind ofType { name kind ofType { name kind ofType { name kind } } } }
    }
    inputFields { name type { name kind ofType { name kind ofType { name kind } } } }
    enumValues { name }
  }
}
"""

ROOT_Q = """
{ __schema { queryType { name fields { name type { name kind ofType { name kind } } } } } }
"""

ALL_TYPES_Q = """
{ __schema { types { name kind } } }
"""


def tname(t):
    if not t:
        return "?"
    if t.get("name"):
        return t["name"]
    inner = tname(t.get("ofType"))
    if t.get("kind") == "LIST":
        return "[%s]" % inner
    if t.get("kind") == "NON_NULL":
        return "%s!" % inner
    return inner


def dump(name):
    r = gql(TYPE_Q, {"name": name})
    t = (r.get("data") or {}).get("__type")
    if not t:
        return False
    print("\n" + "=" * 70)
    print("%s   (%s)" % (t["name"], t["kind"]))
    print("=" * 70)
    if t.get("description"):
        print("  %s" % t["description"][:300])
    for key, label in (("fields", "FIELDS"), ("inputFields", "INPUT FIELDS")):
        for f in (t.get(key) or []):
            args = ""
            if f.get("args"):
                args = "(" + ", ".join(
                    "%s: %s" % (a["name"], tname(a.get("type"))) for a in f["args"]) + ")"
            print("  %-34s %s%s" % (f["name"], tname(f.get("type")), args))
        if t.get(key):
            print("  " + "-" * 40 + " (%s)" % label)
    if t.get("enumValues"):
        print("  ENUM: " + ", ".join(v["name"] for v in t["enumValues"]))
    return True


def main():
    if len(sys.argv) > 1:
        for n in sys.argv[1:]:
            if not dump(n):
                print("Type %r not found." % n)
        return

    print("ENDPOINT: %s" % ENDPOINT)
    print("\n" + "=" * 70)
    print("ROOT QUERY FIELDS")
    print("=" * 70)
    r = gql(ROOT_Q)
    qt = ((r.get("data") or {}).get("__schema") or {}).get("queryType") or {}
    for f in qt.get("fields") or []:
        print("  %-24s %s" % (f["name"], tname(f.get("type"))))

    # Every type whose name mentions a loan, so we catch whatever the real
    # filter/enum/connection types are called.
    r = gql(ALL_TYPES_Q)
    all_types = [t["name"] for t in
                 (((r.get("data") or {}).get("__schema") or {}).get("types") or [])
                 if t.get("name") and not t["name"].startswith("__")]
    print("\nTypes matching 'loan' / 'lend' / 'filter':")
    interesting = [n for n in all_types
                   if any(k in n.lower() for k in ("loan", "lend", "filter", "status"))]
    print("  " + ", ".join(sorted(interesting)))

    targets = [qt.get("name") or "Query"]
    targets += [n for n in interesting]
    seen = set()
    for n in targets:
        if n and n not in seen:
            seen.add(n)
            dump(n)


if __name__ == "__main__":
    main()
