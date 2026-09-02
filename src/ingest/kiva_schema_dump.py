#!/usr/bin/env python3
# Code written with the assistance of Claude (Anthropic).
"""
kiva_schema_dump.py — pull Kiva's ENTIRE GraphQL schema in one request, cache it,
and print any type you ask for.

Why this replaces kiva_introspect.py: per-type __type(name:) lookups were being
rejected by the gateway, and the previous script hid the error. One full
__schema query avoids the problem, costs a single request, and lets us inspect
any type offline afterwards.

    python3 kiva_schema_dump.py                 # fetch + show the key types
    python3 kiva_schema_dump.py LoanBasic Lend  # show specific types (from cache)
    python3 kiva_schema_dump.py --refetch       # force a fresh pull

Writes kiva_schema.json next to itself. Python 3.9+, standard library only.
"""

import json
import os
import sys
import urllib.error
import urllib.request

ENDPOINT = "https://gateway.production.kiva.org/graphql"
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kiva_schema.json")

# Types most relevant to rebuilding the loan book. Shown by default.
DEFAULT_SHOW = [
    "Lend",
    "LoanBasic",
    "LoanBasicCollection",
    "LoanSearchFiltersInput",
    "LoanSearchFilters",
    "LoanSearchStatusEnum",
    "LoanStatusEnum",
    "LoanSearchSortByEnum",
    "LoanFundraisingInfo",
    "LoanDirect",
    "LoanPartner",
]

FULL_INTROSPECTION = """
{
  __schema {
    queryType { name }
    types {
      name
      kind
      description
      fields(includeDeprecated: true) {
        name
        description
        isDeprecated
        args { name type { ...TypeRef } defaultValue }
        type { ...TypeRef }
      }
      inputFields { name description type { ...TypeRef } defaultValue }
      enumValues(includeDeprecated: true) { name description }
    }
  }
}

fragment TypeRef on __Type {
  kind name
  ofType { kind name
    ofType { kind name
      ofType { kind name
        ofType { kind name
          ofType { kind name
            ofType { kind name }
          }
        }
      }
    }
  }
}
"""


def fetch_schema():
    payload = json.dumps({"query": FULL_INTROSPECTION}).encode()
    req = urllib.request.Request(
        ENDPOINT, data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json",
                 "User-Agent": "QUB-MSc-BusinessAnalytics-research/1.0 (academic)"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            body = r.read().decode()
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print("HTTP %s -- reading body anyway" % e.code, file=sys.stderr)

    doc = json.loads(body)
    if doc.get("errors"):
        print("!! GraphQL errors returned:", file=sys.stderr)
        for err in doc["errors"]:
            print("   - %s" % err.get("message"), file=sys.stderr)
    if not (doc.get("data") or {}).get("__schema"):
        print("\nNo schema in response. Raw body (first 1500 chars):\n", file=sys.stderr)
        print(body[:1500], file=sys.stderr)
        sys.exit(1)
    return doc["data"]["__schema"]


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


def show(types_by_name, name):
    t = types_by_name.get(name)
    if not t:
        print("\n--- %s : NOT IN SCHEMA ---" % name)
        return
    print("\n" + "=" * 72)
    print("%s   (%s)" % (t["name"], t["kind"]))
    print("=" * 72)
    if t.get("description"):
        print("  %s" % " ".join(t["description"].split())[:300])

    for f in (t.get("fields") or []):
        args = ""
        if f.get("args"):
            args = "(" + ", ".join(
                "%s: %s" % (a["name"], tname(a.get("type"))) for a in f["args"]) + ")"
        dep = "  [DEPRECATED]" if f.get("isDeprecated") else ""
        print("  %-32s %s%s%s" % (f["name"], tname(f.get("type")), args, dep))

    for f in (t.get("inputFields") or []):
        dflt = "" if f.get("defaultValue") in (None, "null") else "  = %s" % f["defaultValue"]
        print("  %-32s %s%s" % (f["name"], tname(f.get("type")), dflt))

    if t.get("enumValues"):
        print("  ENUM VALUES:")
        for v in t["enumValues"]:
            d = (" -- " + " ".join(v["description"].split())[:80]) if v.get("description") else ""
            print("      %-28s%s" % (v["name"], d))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    refetch = "--refetch" in sys.argv

    if refetch or not os.path.exists(CACHE):
        print("Fetching full schema (one request) ...", file=sys.stderr)
        schema = fetch_schema()
        with open(CACHE, "w", encoding="utf-8") as fh:
            json.dump(schema, fh)
        print("Cached %d types to %s" % (len(schema.get("types") or []), CACHE),
              file=sys.stderr)
    else:
        with open(CACHE, encoding="utf-8") as fh:
            schema = json.load(fh)

    by_name = {t["name"]: t for t in (schema.get("types") or []) if t.get("name")}
    for n in (args or DEFAULT_SHOW):
        show(by_name, n)

    if not args:
        print("\n" + "=" * 72)
        print("Cache written to kiva_schema.json -- inspect any other type with:")
        print("  python3 kiva_schema_dump.py TypeName [TypeName ...]")
        print("=" * 72)


if __name__ == "__main__":
    main()
