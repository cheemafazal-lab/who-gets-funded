#!/usr/bin/env python3
# Code written with the assistance of Claude (Anthropic).
"""
who-gets-funded — descriptive figures F01-F09 from the D01-D23 CSVs.

Input  : outputs/tables/*.csv   (produced by 01b_parse_descriptive_output.py)
Output : outputs/figures/F01_*.png ... F09_*.png

Usage: python3 src/ingest/01c_make_descriptive_figures.py outputs/tables outputs/figures

Palette is the validated categorical default (blue slot 1, orange slot 2), checked
with the data-viz validator: adjacent CVD dE 24.7 protan / 32.7 tritan, normal-vision
33.6, both slots >= 3:1 on the light surface. Grey is de-emphasis, not a series.
No dual-axis plots anywhere; two measures are always two panels.
"""
import csv, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---- tokens -----------------------------------------------------------------
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_2     = "#52514e"
MUTED     = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"
S1        = "#2a78d6"   # categorical slot 1 - blue
S2        = "#eb6834"   # categorical slot 2 - orange
DEEMPH    = "#c3c2b7"   # de-emphasis grey

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.5,
    "figure.dpi": 200,
    "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})

TABLES, FIGS = sys.argv[1], sys.argv[2]
os.makedirs(FIGS, exist_ok=True)
made = []


def load(name):
    path = os.path.join(TABLES, name)
    return list(csv.DictReader(open(path, encoding="utf8")))


def f(v):
    v = (v or "").strip()
    return None if v in ("", "NULL") else float(v)


def thousands(x, _=None):
    return f"{int(x):,}"


def tidy(ax, xgrid=False):
    ax.grid(axis="x" if xgrid else "y", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name, title):
    p = os.path.join(FIGS, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    made.append((name, title))
    print(f"  wrote {name}")


# =============================================================================
# F01  the outcome variable is overwhelmingly a point mass at 1.0
# =============================================================================
d04 = load("D04_by_status.csv")
by = {r["status"]: r for r in d04}
comp = [
    ("Fully funded\n(share = 1.0)",           int(f(by["funded"]["n"]))),
    ("Expired\n(partial funding, 2,016 at 0)", int(f(by["expired"]["n"]))),
    ("Refunded\n(share = 0.0)",               int(f(by["refunded"]["n"]))),
    ("Still fundraising\n(no outcome yet)",   int(f(by["fundraising"]["n"]))),
]
T = sum(n for _, n in comp)
fig, ax = plt.subplots(figsize=(6.8, 2.9))
ys = list(range(len(comp)))[::-1]
# dot plot, not bars: on a log axis a bar's length from an arbitrary baseline
# misstates magnitude ratios, whereas a dot encodes position only.
ax.plot([n for _, n in comp], ys, "o", ms=9, color=S1,
        markeredgecolor=SURFACE, markeredgewidth=2, zorder=4, linestyle="none")
ax.set_xscale("log")
ax.set_yticks(ys, [l for l, _ in comp], fontsize=8)
ax.set_xlim(2e3, 6e7)
ax.set_ylim(-0.7, len(comp) - 0.3)
ax.set_xlabel("Loans (log scale)")
ax.xaxis.set_major_formatter(FuncFormatter(thousands))
for y, (_, n) in zip(ys, comp):
    ax.text(n * 1.35, y, f"{n:,}  ({100*n/T:.2f}%)", va="center", fontsize=8, color=INK)
tidy(ax, xgrid=True)
ax.set_title("F01  The modelling target is a point mass at 1.0\n"
             f"share_raised, all {T:,} harvested loans",
             fontsize=9.5, loc="left", color=INK, pad=8)
save(fig, "F01_outcome_composition.png", "Composition of share_raised by status")


# =============================================================================
# F02  expiry rate by posting year - the structural break
# =============================================================================
d08 = load("D08_by_posting_year.csv")
yrs = [int(r["yr"]) for r in d08]
rate = [f(r["expiry_pct"]) for r in d08]
n_yr = [int(f(r["n"])) for r in d08]


def regime(y):
    if y <= 2011:
        return DEEMPH          # expiry dates not recorded
    if y <= 2019:
        return S1              # high-expiry regime
    return S2                  # post-2019 regime


fig, ax = plt.subplots(figsize=(7.4, 3.4))
ax.bar(yrs, rate, width=0.56, color=[regime(y) for y in yrs], zorder=3)
tidy(ax)
ax.set_ylabel("Loans expiring unfunded (%)")
ax.set_xlabel("Year posted")
ax.set_xticks(yrs, [str(y) for y in yrs], rotation=90, fontsize=7.5)
ax.set_ylim(0, 8.4)
# selective direct labels only: the two regime extremes
for y, r in zip(yrs, rate):
    if y in (2016, 2019, 2020, 2024):
        ax.text(y, r + 0.18, f"{r:.2f}%", ha="center", fontsize=7.5, color=INK)
ax.axvline(2019.5, color=BASELINE, lw=0.8, zorder=2)
ax.text(2019.7, 7.9, "regime change", fontsize=7.5, color=INK_2, va="top")
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color=DEEMPH, label="2005–2011  no expiry dates recorded"),
    Patch(color=S1,     label="2012–2019  4.1–7.2% expiry"),
    Patch(color=S2,     label="2020–2026  0.23–0.97% expiry"),
], loc="upper left", fontsize=7.5, handlelength=1.1)
ax.set_title("F02  Expiry collapses after 2019 — earlier and later cohorts are not the same population\n"
             "Expiry rate by posting year, full harvest",
             fontsize=9.5, loc="left", color=INK, pad=8)
save(fig, "F02_expiry_by_year.png", "Expiry rate by posting year, showing the post-2019 break")


# =============================================================================
# F03  gender gap within every sector
# =============================================================================
d16 = load("D16_gender_x_sector_mean_share_and_expiry.csv")
sec = {}
for r in d16:
    sec.setdefault(r["sector"], {})[r["gender"]] = (f(r["expiry_pct"]), int(f(r["n"])))
order = sorted(sec, key=lambda s: sec[s].get("male", (0, 0))[0])
fig, ax = plt.subplots(figsize=(7.0, 6.2))
h = 0.36
for i, s in enumerate(order):
    fem = sec[s].get("female", (None, 0))
    mal = sec[s].get("male", (None, 0))
    if fem[0] is not None:
        ax.barh(i + h / 2 + 0.01, fem[0], height=h, color=S1, zorder=3)
    if mal[0] is not None:
        ax.barh(i - h / 2 - 0.01, mal[0], height=h, color=S2, zorder=3)
ax.set_yticks(range(len(order)), order, fontsize=8)
ax.set_xlabel("Loans expiring unfunded (%)")
ax.set_ylim(-0.8, len(order) - 0.2)
ax.set_xlim(0, 19.5)          # headroom so the direct label sits inside the axes
tidy(ax, xgrid=True)
ax.legend(handles=[Patch(color=S1, label="Female-tagged"), Patch(color=S2, label="Male-tagged")],
          loc="lower right", fontsize=8, handlelength=1.1)
# label the widest gap only
gaps = [(s, sec[s].get("male", (0,))[0] - sec[s].get("female", (0,))[0])
        for s in order if sec[s].get("male") and sec[s].get("female")]
worst = max(gaps, key=lambda t: t[1])
wi = order.index(worst[0])
ax.text(sec[worst[0]]["male"][0] + 0.25, wi - h / 2 - 0.01,
        f"{sec[worst[0]]['male'][0]:.1f}% vs {sec[worst[0]]['female'][0]:.1f}%  "
        f"(+{worst[1]:.1f}pp)", va="center", fontsize=7.5, color=INK)
ax.set_title("F03  Male-tagged loans expire more often in all 19 sectors\n"
             "Expiry rate by sector and borrower gender",
             fontsize=9.5, loc="left", color=INK, pad=8)
save(fig, "F03_expiry_gender_by_sector.png", "Expiry rate by sector and gender")


# =============================================================================
# F04  among expired loans, how far short did they fall?
# =============================================================================
d17 = load("D17_expired_loans_only_how_far_short_did_they_fall.csv")
lab = [r["band"].split(" ", 1)[1] for r in d17]
val = [f(r["pct_of_expired"]) for r in d17]
cnt = [int(f(r["n"])) for r in d17]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.bar(range(len(lab)), val, width=0.5, color=S1, zorder=3)
ax.set_xticks(range(len(lab)), lab, fontsize=8)
ax.set_ylabel("Share of expired loans (%)")
tidy(ax)
for i, (v, c) in enumerate(zip(val, cnt)):
    ax.text(i, v + 0.5, f"{v:.1f}%\n{c:,}", ha="center", fontsize=7.5, color=INK)
ax.set_ylim(0, max(val) * 1.28)
ax.set_title("F04  Failure is partial, not binary — only 2.2% of expired loans raised nothing\n"
             "Proportion of the requested amount reached, expired loans (n = 93,321)",
             fontsize=9.5, loc="left", color=INK, pad=8)
save(fig, "F04_expired_shortfall.png", "Distribution of share raised among expired loans")


# =============================================================================
# F05  loan size vs expiry, and the female share by size - two panels, one axis each
# =============================================================================
d15 = load("D15_loan_amount_bands_vs_outcome.csv")
bl = [r["band"].split(" ", 1)[1] for r in d15]
ex = [f(r["expiry_pct"]) for r in d15]
fp = [f(r["female_pct"]) for r in d15]
nn = [int(f(r["n"])) for r in d15]
fig, axes = plt.subplots(2, 1, figsize=(6.6, 4.9), sharex=True)
a = axes[0]
a.bar(range(len(bl)), ex, width=0.5, color=S1, zorder=3)
a.set_ylabel("Expiry rate (%)")
tidy(a)
for i, v in enumerate(ex):
    a.text(i, v + 0.13, f"{v:.2f}", ha="center", fontsize=7.5, color=INK)
a.set_ylim(0, max(ex) * 1.22)
a.set_title(r"F05  Expiry rises 130-fold to the \$1,000\u20132,499 band, then flattens between 6% and 8%\n"
            "The female share moves the opposite way. Two measures, two panels — never one dual axis.",
            fontsize=9.5, loc="left", color=INK, pad=8)
b = axes[1]
b.bar(range(len(bl)), fp, width=0.5, color=S2, zorder=3)
b.set_ylabel("Female-tagged (%)")
b.set_ylim(0, 100)
tidy(b)
for i, v in enumerate(fp):
    b.text(i, v + 1.6, f"{v:.1f}", ha="center", fontsize=7.5, color=INK)
b.set_xticks(range(len(bl)), [f"{s}\nn={n:,}" for s, n in zip(bl, nn)], fontsize=7.5)
b.set_xlabel(r"Requested amount band (US\$)")
save(fig, "F05_amount_band_expiry_and_gender.png", "Expiry rate and female share by loan-amount band")


# =============================================================================
# F06  country variation
# =============================================================================
d07 = load("D07_by_country_top_30_of_all_countries.csv")
top = sorted(d07, key=lambda r: f(r["expiry_pct"]))
names = [r["country_name"] for r in top]
rates = [f(r["expiry_pct"]) for r in top]
ns = [int(f(r["n"])) for r in top]
fold = rates[-1] / rates[0]
fig, ax = plt.subplots(figsize=(6.8, 6.4))
ax.barh(range(len(top)), rates, height=0.5, color=S1, zorder=3)
ax.set_yticks(range(len(top)), [f"{nm}   n={c:,}" for nm, c in zip(names, ns)], fontsize=7.5)
ax.set_xlabel("Loans expiring unfunded (%)")
tidy(ax, xgrid=True)
for i, (r, n) in enumerate(zip(rates, ns)):
    ax.text(r + 0.14, i, f"{r:.2f}%", va="center", fontsize=7.5, color=INK)
ax.set_xlim(0, max(rates) * 1.14)
ax.set_ylim(-0.8, len(top) - 0.2)
ax.set_title(f"F06  Expiry risk is overwhelmingly geographic — a {fold:.0f}-fold spread across the 30 largest markets\n"
             f"{names[0]} {rates[0]:.2f}% to {names[-1]} {rates[-1]:.2f}%, ranked by expiry rate",
             fontsize=9.5, loc="left", color=INK, pad=8)
save(fig, "F06_country_expiry.png", "Expiry rate by country, top 20 of the 30 largest")


# =============================================================================
# F07  partner concentration
# =============================================================================
d13 = load("D13_partner_concentration_top_20.csv")
pn = [r["partner_name"][:42] for r in d13][::-1]
pp = [f(r["pct"]) for r in d13][::-1]
fig, ax = plt.subplots(figsize=(6.8, 5.0))
cols = [S2 if p == max(pp) else S1 for p in pp]
ax.barh(range(len(pn)), pp, height=0.5, color=cols, zorder=3)
ax.set_yticks(range(len(pn)), pn, fontsize=7.5)
ax.set_xlabel("Share of all harvested loans (%)")
tidy(ax, xgrid=True)
ax.text(max(pp) - 0.35, len(pn) - 1, f"{max(pp):.1f}%", va="center", ha="right",
        fontsize=8, color="#ffffff", fontweight="bold")
ax.legend(handles=[Patch(color=S2, label="Largest single partner"),
                   Patch(color=S1, label="Partners 2–20")],
          loc="lower right", fontsize=8, handlelength=1.1)
ax.set_title(f"F07  One field partner accounts for {max(pp):.1f}% of the entire loan book\n"
             "Top 20 of 639 partners by loan volume",
             fontsize=9.5, loc="left", color=INK, pad=8)
save(fig, "F07_partner_concentration.png", "Partner concentration, top 20 of 639")


# =============================================================================
# F08  when does planned_expiration_date actually exist?
# =============================================================================
avail = [100 * (1 - int(f(r["no_expiry_date"])) / int(f(r["n"]))) for r in d08]
fig, ax = plt.subplots(figsize=(7.0, 2.9))
ax.bar(yrs, avail, width=0.56,
       color=[DEEMPH if a < 99 else S1 for a in avail], zorder=3)
tidy(ax)
ax.set_ylabel("Records with an expiry date (%)")
ax.set_xlabel("Year posted")
ax.set_xticks(yrs, [str(y) for y in yrs], rotation=90, fontsize=7.5)
ax.set_ylim(0, 108)
ax.legend(handles=[Patch(color=S1, label="Complete (2013 onward)"),
                   Patch(color=DEEMPH, label="Incomplete or absent")],
          loc="lower right", fontsize=7.5, handlelength=1.1)
ax.text(2013, 104, "2013: first complete cohort", fontsize=7.5, color=INK_2)
ax.set_title("F08  planned_expiration_date is absent before 2012 and complete from 2013\n"
             "371,810 records (11.9%) have no expiry date",
             fontsize=9.5, loc="left", color=INK, pad=8)
save(fig, "F08_expiry_date_availability.png", "Availability of planned_expiration_date by year")


# =============================================================================
# F09  loan_amount is severely right-skewed
# =============================================================================
q = load("D21_sort_quantiles_loan_amount.csv")[0]
d01 = load("D01_numeric_summary_all_3_132_059_rows.csv")[0]
P = {k: f(q[k]) for k in ("p01", "p05", "p25", "median", "p75", "p95", "p99")}
mean = f(d01["loan_amt_mean"])
mx = f(d01["loan_amt_max"])
# box plot built from the precomputed quantiles: box = p25-p75, whiskers = p05-p95
stats = [{"med": P["median"], "q1": P["p25"], "q3": P["p75"],
          "whislo": P["p05"], "whishi": P["p95"], "mean": mean, "fliers": []}]
fig, ax = plt.subplots(figsize=(7.2, 2.7))
bp = ax.bxp(stats, vert=False, widths=0.34, patch_artist=True, showmeans=True,
            meanline=False, showfliers=False, zorder=3)
for b in bp["boxes"]:
    b.set(facecolor=S1, alpha=0.30, edgecolor=S1, linewidth=1.4)
for w in bp["whiskers"] + bp["caps"]:
    w.set(color=S1, linewidth=1.4)
for m in bp["medians"]:
    m.set(color=INK, linewidth=2)
for m in bp["means"]:
    m.set(marker="D", markerfacecolor=S2, markeredgecolor=SURFACE,
          markeredgewidth=1.5, markersize=8)
ax.set_xscale("log")
ax.set_yticks([])
ax.set_ylim(0.70, 1.42)
ax.set_xlim(P["p01"] * 0.7, P["p99"] * 1.9)
ax.set_xlabel(r"Requested amount, US\$ (log scale)")
ax.spines["left"].set_visible(False)
ax.grid(axis="x", zorder=0)
ax.set_axisbelow(True)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: rf"\${int(x):,}"))
# selective labels only: the median and the mean are the story; the axis carries the rest
ax.annotate(rf"median \${P['median']:,.0f}", (P["median"], 1),
            textcoords="offset points", xytext=(-8, 26), ha="right",
            fontsize=8, color=INK)
ax.annotate(rf"mean \${mean:,.2f}", (mean, 1), textcoords="offset points",
            xytext=(10, -32), ha="left", fontsize=8, color=INK)
ax.legend(handles=[
    Patch(facecolor=S1, alpha=0.30, edgecolor=S1, label="p25–p75 (whiskers p05–p95)"),
    plt.Line2D([], [], color=INK, lw=2, label="median"),
    plt.Line2D([], [], marker="D", ls="none", markerfacecolor=S2,
               markeredgecolor=SURFACE, markersize=8, label="mean"),
], loc="upper left", fontsize=7.5, handlelength=1.3, ncol=3,
   bbox_to_anchor=(0, 1.04))
ax.set_title("F09  Requested amount is severely right-skewed — the mean sits between the median and p75\n"
             rf"p01 \${P['p01']:,.0f} · p25 \${P['p25']:,.0f} · median \${P['median']:,.0f} · "
             rf"p75 \${P['p75']:,.0f} · p95 \${P['p95']:,.0f} · p99 \${P['p99']:,.0f} · "
             rf"max \${mx:,.0f} · SD \$2,011",
             fontsize=9.5, loc="left", color=INK, pad=14)
save(fig, "F09_loan_amount_quantiles.png", "Quantiles of requested loan amount")


# ---- register ---------------------------------------------------------------
with open(os.path.join(FIGS, "_figure_register.csv"), "w", newline="", encoding="utf8") as fh:
    w = csv.writer(fh)
    w.writerow(["figure_id", "file", "title"])
    for name, title in made:
        w.writerow([name.split("_")[0], name, title])

print(f"\n{len(made)} figures written to {FIGS}")
