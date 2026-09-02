#!/usr/bin/env python3
# Code written with the assistance of Claude (Anthropic).
"""
who-gets-funded — F24 pipeline architecture, drawn from the repo AS BUILT (20 Aug 2026).

Replaces prior_drafts/fig_architecture.png, which was drawn pre-feedback and
contains three claims that are no longer true: a Kickstarter RQ4 source, a
pinned build.kiva.org CSV snapshot, and DiCE counterfactuals.

Palette and typography match src/pipeline_common.py so F24 sits beside F01-F23.
Solid boxes are built and run. The dashed box is proposed, not implemented.

Run:  python3 make_architecture.py
Out:  F24_architecture.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ---- house palette, lifted from src/pipeline_common.py ----------------------
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
S1, S2, DEEMPH = "#2a78d6", "#eb6834", "#c3c2b7"
FILL = "#f2f5fa"      # faint blue wash for built boxes
FILL_HL = "#fdf1ea"   # faint orange wash for the novel-contribution box

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"], "figure.dpi": 200})

fig, ax = plt.subplots(figsize=(11.2, 8.9))
ax.set_position([0, 0, 1, 1])
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

COL = [0.025, 0.350, 0.675]      # left edges
ROW = [0.720, 0.497, 0.274, 0.051]  # bottom edges
W, H = 0.300, 0.180


def box(c, r, title, lines, edge=S1, fill=FILL, dashed=False, badge=None):
    x, y = COL[c], ROW[r]
    p = FancyBboxPatch((x, y), W, H,
                       boxstyle="round,pad=0.006,rounding_size=0.010",
                       linewidth=1.5 if not dashed else 1.3,
                       edgecolor=edge, facecolor=fill,
                       linestyle="--" if dashed else "-", zorder=3)
    ax.add_patch(p)
    tc = MUTED if dashed else INK
    ax.text(x + W / 2, y + H - 0.030, title, ha="center", va="center",
            fontsize=9.2, fontweight="bold", color=tc, zorder=4)
    for i, ln in enumerate(lines):
        ax.text(x + W / 2, y + H - 0.062 - i * 0.026, ln, ha="center", va="center",
                fontsize=7.3, color=MUTED if dashed else INK_2, zorder=4)
    if badge:
        ax.text(x + W / 2, y + 0.014, badge, ha="center", va="center",
                fontsize=7.2, fontweight="bold", color=edge, zorder=4)
    return x, y


def harrow(c_from, c_to, r, dashed=False):
    x0 = COL[c_from] + W
    x1 = COL[c_to]
    y = ROW[r] + H / 2
    ax.add_patch(FancyArrowPatch((x0 + 0.004, y), (x1 - 0.004, y),
                                 arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.2, color=BASELINE, zorder=2,
                                 linestyle=(0, (4, 3)) if dashed else "-"))


def elbow(c_from, r_from, c_to, r_to, dashed=False):
    """Down out of the last box in a row, across, down into the next row."""
    x0 = COL[c_from] + W / 2
    y0 = ROW[r_from]
    x1 = COL[c_to] + W / 2
    y1 = ROW[r_to] + H
    ym = (y0 + y1) / 2
    ls = (0, (4, 3)) if dashed else "-"
    ax.plot([x0, x0], [y0 - 0.004, ym], color=BASELINE, lw=1.2, ls=ls, zorder=2)
    ax.plot([x0, x1], [ym, ym], color=BASELINE, lw=1.2, ls=ls, zorder=2)
    ax.add_patch(FancyArrowPatch((x1, ym), (x1, y1 + 0.004),
                                 arrowstyle="-|>", mutation_scale=11,
                                 linewidth=1.2, color=BASELINE,
                                 linestyle=ls, zorder=2))


# ---------------------------------------------------------------- row 1 -----
box(0, 0, "SOURCE", [
    "Kiva GraphQL API",
    "gateway.production.kiva.org",
    "3,132,059 loans · 99.34% of book",
    "S3 snapshot dead — Edgio, 15 Jan 2025"])
box(1, 0, "1 · INGEST + VALIDATE", [
    "src/ingest/kiva_harvest2.py",
    "ID traversal, filters status: all",
    "schema introspection · User-Agent fix",
    "raw JSON retained verbatim"])
box(2, 0, "2 · RAW STORE", [
    "~/Downloads/kiva_raw.sqlite",
    "11.99 GiB · immutable archive",
    "4 indexes · never mutated",
    "cohort applied as a filter in code"])
harrow(0, 1, 0); harrow(1, 2, 0)

# ---------------------------------------------------------------- row 2 -----
box(0, 1, "3 · DESCRIPTIVE + COMPOSITION", [
    "src/ingest/01_*.sql · 02_*.sql",
    "D01–D23 tables · C01–C10 tables",
    "direct standardisation by country,",
    "partner, country × sector → F01–F12"])
box(1, 1, "4 · COHORT + FEATURES", [
    "src/features/03 · 03b · 03c",
    "2013–2019 · LoanPartner · funded+expired",
    "feat_v1_full 1,344,542 × 54",
    "dev 200,003 · out-of-regime 1,254,757"])
box(2, 1, "5 · MODELS", [
    "src/models/04 · 05 · 06 · 07",
    "R001 naive floors B0–B3",
    "R002 fractional logit, QMLE + HC0",
    "R003 LightGBM · R004 binary aux"])
elbow(2, 0, 0, 1)
harrow(0, 1, 1); harrow(1, 2, 1)

# ---------------------------------------------------------------- row 3 -----
box(0, 2, "6 · TEMPORAL EVALUATION", [
    "src/models/08 · 12",
    "expanding folds, test 2016–2019",
    "MAE · RMSE · shortfall-MAE · PR-AUC",
    "R005 out-of-regime 2020–2026"])
box(1, 2, "7 · EXPLAIN", [
    "src/explain/09 · 10",
    "R006 exact TreeSHAP, native pred_contrib",
    "R007 in-repo greedy counterfactual search",
    "actionable set only · protected frozen"])
box(2, 2, "8 · EQUITY AUDIT", [
    "src/equity/11_recourse_audit.py",
    "R008 fairness of recourse",
    "market outcomes ✓ · conditional disparity ✓",
    "model errors by group — pending"],
    edge=S2, fill=FILL_HL, badge="NOVEL CONTRIBUTION")
elbow(2, 1, 0, 2)
harrow(0, 1, 2); harrow(1, 2, 2)

# ---------------------------------------------------------------- row 4 -----
box(0, 3, "9 · OUTPUTS + COMPARISON", [
    "src/run_pipeline.py — single entry point",
    "outputs/ figures · tables · models",
    "F01–F23 · T1–T6 · runs R001–R009",
    "nothing enters the report unlogged"])
box(1, 3, "10 · STAKEHOLDER DASHBOARD", [
    "src/dashboard.py (Plotly)",
    "user: Kiva marketplace / impact analyst",
    "deferred extension per supervisor step 7"],
    edge=DEEMPH, fill=SURFACE, dashed=True,
    badge="COULD BE IMPLEMENTED — NOT BUILT")
elbow(2, 2, 0, 3)
harrow(0, 1, 3, dashed=True)

# ------------------------------------------------------------- legend -------
lx, ly = COL[2], ROW[3] + H
ax.text(lx, ly - 0.020, "Reading the figure", fontsize=8.6,
        fontweight="bold", color=INK, va="top")
for i, s in enumerate([
        "Solid outline — built, run, and logged",
        "Dashed outline — proposed, not implemented",
        "Orange — the novel contribution",
        "Seed 40465466 throughout; every run carries",
        "an R-number in RESULTS_LOG.md",
        "",
        "Fractional target: share of request raised,",
        "bounded [0,1]. Binary model is auxiliary,",
        "used only to drive the counterfactual search."]):
    ax.text(lx, ly - 0.048 - i * 0.021, s, fontsize=7.2, color=INK_2, va="top")

ax.text(0.025, 0.995,
        "who-gets-funded — pipeline architecture as built, 20 August 2026",
        fontsize=11.5, fontweight="bold", color=INK, ha="left", va="top")
ax.text(0.025, 0.966,
        "Drawn from the repository file tree. Supersedes the pre-feedback diagram in "
        "prior_drafts/, which showed a Kickstarter source, a pinned build.kiva.org "
        "CSV snapshot, and DiCE — none of which apply.",
        fontsize=7.6, color=MUTED, ha="left", va="top")

fig.savefig("/home/claude/out/F24_architecture.png", bbox_inches="tight")
print("wrote F24_architecture.png")
