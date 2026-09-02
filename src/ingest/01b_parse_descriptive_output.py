#!/usr/bin/env python3
# Code written with the assistance of Claude (Anthropic).
"""
who-gets-funded — split the D01-D23 descriptive-statistics console output into per-section CSVs.

Input  : outputs/tables/01_descriptive_statistics_output.txt
         (the tee'd console output of src/ingest/01_descriptive_statistics.sql)
Output : one CSV per result block in outputs/tables/, plus _timings.csv and _manifest.csv

Parses sqlite3 `.mode column` fixed-width output. Column boundaries are taken from
the dashed ruler line sqlite3 prints under every header row, so the parse is exact
rather than whitespace-guessed. Values containing commas are quoted by csv.writer.

Usage: python3 src/ingest/01b_parse_descriptive_output.py \
           outputs/tables/01_descriptive_statistics_output.txt outputs/tables
"""
import csv, re, sys, os

SECTION = re.compile(r'^== ([A-Z]\d+)\s+(.*?)\s*=+\s*$')
SUBHEAD = re.compile(r'^--\s*(.*?)\s*--\s*$')
RULER   = re.compile(r'^-[- ]*-$')
TIMER   = re.compile(r'^Run Time: real ([\d.]+) user ([\d.]+) sys ([\d.]+)')


def spans(ruler):
    return [(m.start(), m.end()) for m in re.finditer(r'-+', ruler)]


def cut(line, sp):
    out = []
    for i, (a, b) in enumerate(sp):
        seg = line[a:] if i == len(sp) - 1 else line[a:b]
        out.append(seg.strip())
    return out


def main(src, dst):
    os.makedirs(dst, exist_ok=True)
    lines = open(src, encoding='utf8', errors='replace').read().splitlines()

    section = title = sub = None
    blocks, timings = [], []
    i = 0
    while i < len(lines):
        line = lines[i]

        m = SECTION.match(line.strip())
        if m:
            section, title, sub = m.group(1), m.group(2).strip(), None
            i += 1; continue

        m = SUBHEAD.match(line.strip())
        if m and section:
            sub = m.group(1); i += 1; continue

        m = TIMER.match(line.strip())
        if m and section:
            timings.append((section, sub or '', float(m.group(1)),
                            float(m.group(2)), float(m.group(3))))
            i += 1; continue

        if (section and line.strip() and i + 1 < len(lines)
                and RULER.match(lines[i + 1].strip()) and '-' in lines[i + 1]):
            sp = spans(lines[i + 1])
            header = cut(line, sp)
            rows = []
            j = i + 2
            while j < len(lines):
                nxt = lines[j]
                if (not nxt.strip() or SECTION.match(nxt.strip())
                        or SUBHEAD.match(nxt.strip()) or TIMER.match(nxt.strip())):
                    break
                rows.append(cut(nxt, sp)); j += 1
            blocks.append((section, title, sub, header, rows))
            i = j; continue

        i += 1

    seen, manifest = {}, []
    for section, title, sub, header, rows in blocks:
        seen[section] = seen.get(section, 0) + 1
        n = seen[section]
        slug = re.sub(r'[^a-z0-9]+', '_', (sub or title).lower()).strip('_')[:48]
        name = f"{section}_{slug}.csv" if n == 1 else f"{section}{chr(96+n)}_{slug}.csv"
        with open(os.path.join(dst, name), 'w', newline='', encoding='utf8') as fh:
            w = csv.writer(fh); w.writerow(header); w.writerows(rows)
        manifest.append((section, name, title, sub or '', len(header), len(rows)))

    with open(os.path.join(dst, '_timings.csv'), 'w', newline='', encoding='utf8') as fh:
        w = csv.writer(fh)
        w.writerow(['section', 'subquery', 'real_s', 'user_s', 'sys_s']); w.writerows(timings)

    with open(os.path.join(dst, '_manifest.csv'), 'w', newline='', encoding='utf8') as fh:
        w = csv.writer(fh)
        w.writerow(['section', 'file', 'title', 'subquery', 'n_cols', 'n_rows'])
        w.writerows(manifest)

    total = sum(t[2] for t in timings)
    print(f"{len(blocks)} CSVs written to {dst}")
    print(f"total measured query time: {total:.1f}s ({total/60:.1f} min)")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
