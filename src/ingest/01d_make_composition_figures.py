#!/usr/bin/env python3
# Code written with the assistance of Claude (Anthropic).
"""
who-gets-funded — composition-check figures F10-F12 from the C01-C10 CSVs.

Input  : outputs/tables/composition/*.csv
Output : outputs/figures/F10_*.png .. F12_*.png

Usage: python3 src/ingest/01d_make_composition_figures.py \
           outputs/tables/composition outputs/figures

Palette: validated categorical slots 1-2 (blue #2a78d6 / orange #eb6834) for the
two genders; the dumbbell uses two steps of the single blue sequential ramp
(step 250 #86b6ef, step 550 #1c5cab) per the before/after rule. No dual axes.
"""
import csv, glob, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
S1, S2, DEEMPH = "#2a78d6", "#eb6834", "#c3c2b7"
SEQ_LO, SEQ_HI = "#86b6ef", "#1c5cab"      # blue ramp steps 250 / 550

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8.5,
    "figure.dpi": 200, "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "legend.frameon": False, "axes.spines.top": False, "axes.spines.right": False,
})

TABLES, FIGS = sys.argv[1], sys.argv[2]
os.makedirs(FIGS, exist_ok=True)
made = []


def load(pat):
    return list(csv.DictReader(open(glob.glob(os.path.join(TABLES, pat))[0], encoding="utf8")))


def f(v):
    v = (v or "").strip()
    return None if v in ("", "NULL") else float(v)


def tidy(ax, xgrid=False):
    ax.grid(axis="x" if xgrid else "y", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name, title):
    fig.savefig(os.path.join(FIGS, name), bbox_inches="tight")
    plt.close(fig)
    made.append((name, title))
    print(f"  wrote {name}")


# =============================================================================
# F10  crude vs directly standardised gap — the headline equity result
# =============================================================================
specs = [("Country\n53 strata", "C05_*"),
         ("Field partner\n94 strata", "C06_*"),
         ("Country x sector\n202 strata", "C07_*")]
lab, crude, std, expl = [], [], [], []
for name, pat in specs:
    d = load(pat)
    cr = [r for r in d if r["method"].startswith("crude")][0]
    st = [r for r in d if not r["method"].startswith("crude")][0]
    lab.append(name); crude.append(f(cr["gap_pp"])); std.append(f(st["gap_pp"]))
    expl.append(100 * (f(cr["gap_pp"]) - f(st["gap_pp"])) / f(cr["gap_pp"]))

fig, ax = plt.subplots(figsize=(7.0, 3.1))
ys = list(range(len(lab)))[::-1]
for y, c, s in zip(ys, crude, std):
    ax.plot([s, c], [y, y], color=DEEMPH, lw=2.5, solid_capstyle="round", zorder=2)
ax.plot(crude, ys, "o", ms=10, color=SEQ_LO, markeredgecolor=SURFACE,
        markeredgewidth=2, zorder=4, linestyle="none")
ax.plot(std, ys, "o", ms=10, color=SEQ_HI, markeredgecolor=SURFACE,
        markeredgewidth=2, zorder=5, linestyle="none")
for y, c, s, e in zip(ys, crude, std, expl):
    ax.text(c + 0.22, y, f"{c:.2f}pp", va="center", fontsize=8, color=INK)
    ax.text(s - 0.22, y, f"{s:.2f}pp", va="center", ha="right", fontsize=8, color=INK)
    ax.text((c + s) / 2, y + 0.30, f"−{e:.0f}% composition",
            ha="center", fontsize=7.5, color=INK_2)
ax.set_yticks(ys, lab, fontsize=8)
ax.set_xlabel("Male minus female expiry rate (percentage points)")
ax.set_xlim(5.6, 11.4)
ax.set_ylim(-0.62, len(lab) - 0.28)
tidy(ax, xgrid=True)
ax.legend(handles=[Patch(color=SEQ_LO, label="Crude gap"),
                   Patch(color=SEQ_HI, label="Directly standardised gap")],
          loc="upper right", fontsize=8, handlelength=1.1)
ax.set_title("F10  The gender gap survives standardisation — composition explains 14–28% of it, not all\n"
             "Cohort 2013–2019, LoanPartner, funded+expired (n = 1,344,542). "
             "Crude and standardised computed on identical strata.",
             fontsize=9.5, loc="left", color=INK, pad=8)
save(fig, "F10_gap_crude_vs_standardised.png",
     "Crude vs directly standardised gender expiry gap, three stratifications")


# =============================================================================
# F11  the gap country by country — 51 of 53 point the same way
# =============================================================================
d4 = sorted(load("C04_*"), key=lambda r: f(r["gap_pp"]))
names = [r["country_name"] for r in d4]
gaps = [f(r["gap_pp"]) for r in d4]
vols = [int(f(r["n_total"])) for r in d4]
cols = [S2 if g <= 0 else S1 for g in gaps]
fig, ax = plt.subplots(figsize=(7.0, 8.4))
ax.barh(range(len(d4)), gaps, height=0.52, color=cols, zorder=3)
ax.set_yticks(range(len(d4)),
              [f"{nm}   n={v:,}" for nm, v in zip(names, vols)], fontsize=7)
ax.set_xlabel("Male minus female expiry rate (percentage points)")
ax.axvline(0, color=BASELINE, lw=0.8, zorder=2)
ax.set_ylim(-0.8, len(d4) - 0.2)
tidy(ax, xgrid=True)
nrev = sum(1 for g in gaps if g <= 0)
ax.legend(handles=[Patch(color=S1, label=f"Male-tagged worse ({len(gaps)-nrev} countries)"),
                   Patch(color=S2, label=f"No gap or reversed ({nrev})")],
          loc="lower right", fontsize=8, handlelength=1.1)
ax.set_xlim(min(min(gaps) * 1.6, -0.6), max(gaps) * 1.20)
top = max(range(len(gaps)), key=lambda i: gaps[i])
ax.text(gaps[top] + 0.3, top, f"+{gaps[top]:.1f}pp widest",
        va="center", ha="left", fontsize=7.5, color=INK)
bot = min(range(len(gaps)), key=lambda i: gaps[i])
ax.text(gaps[bot] - 0.3, bot, f"{gaps[bot]:.2f}pp", va="center", ha="right",
        fontsize=7.5, color=INK)
ax.set_title("F11  The direction is near-universal — the gap is not one country's artefact\n"
             "Countries with at least 500 loans of each gender in the cohort",
             fontsize=9.5, loc="left", color=INK, pad=8)
save(fig, "F11_gap_by_country.png", "Gender expiry gap by country, cohort")


# =============================================================================
# F12  within-cohort drift — two panels, one measure each
# =============================================================================
d10 = load("C10_*")
yrs = [int(r["yr"]) for r in d10]
ef = [f(r["expiry_female"]) for r in d10]
em = [f(r["expiry_male"]) for r in d10]
mae = [f(r["naive_mae"]) for r in d10]
fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.0), sharex=True)
a = axes[0]
a.plot(yrs, em, "-o", lw=2, ms=8, color=S2, markeredgecolor=SURFACE,
       markeredgewidth=2, label="Male-tagged", zorder=4)
a.plot(yrs, ef, "-o", lw=2, ms=8, color=S1, markeredgecolor=SURFACE,
       markeredgewidth=2, label="Female-tagged", zorder=4)
a.set_ylabel("Expiry rate (%)")
a.set_ylim(0, 19)
tidy(a)
a.legend(loc="upper left", fontsize=8)
a.text(yrs[-1] + 0.08, em[-1], f"{em[-1]:.1f}%", va="center", fontsize=8, color=INK)
a.text(yrs[-1] + 0.08, ef[-1], f"{ef[-1]:.1f}%", va="center", fontsize=8, color=INK)
a.set_title("F12  The cohort is not internally stable — the gap widens and the naive floor moves 4.4x\n"
            "Every temporal-CV fold needs its own naive benchmark; a single global figure would mislead.",
            fontsize=9.5, loc="left", color=INK, pad=8)
b = axes[1]
b.bar(yrs, mae, width=0.5, color=DEEMPH, zorder=3)
b.set_ylabel("Naive 'predict 1.0' MAE")
b.set_xlabel("Year posted")
b.set_xticks(yrs, [str(y) for y in yrs], fontsize=8)
tidy(b)
for y, m in zip(yrs, mae):
    if m in (min(mae), max(mae)):
        b.text(y, m + 0.0012, f"{m:.4f}", ha="center", fontsize=7.5, color=INK)
b.set_ylim(0, max(mae) * 1.22)
b.axhline(0.032912, color=S1, lw=1.4, zorder=4)
b.text(yrs[0] - 0.35, 0.0345, "cohort-wide 0.0329", fontsize=7.5, color=S1)
save(fig, "F12_within_cohort_drift.png", "Gender expiry and naive MAE by year within the cohort")


with open(os.path.join(FIGS, "_figure_register_composition.csv"), "w",
          newline="", encoding="utf8") as fh:
    w = csv.writer(fh)
    w.writerow(["figure_id", "file", "title"])
    for name, title in made:
        w.writerow([name.split("_")[0], name, title])
print(f"\n{len(made)} figures written to {FIGS}")
