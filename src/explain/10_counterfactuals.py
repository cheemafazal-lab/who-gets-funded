#!/usr/bin/env python3
"""
who-gets-funded — R007: counterfactual recourse search on the auxiliary binary model.

WHAT
  For test-2019 loans the binary model predicts will NOT fully fund
  (p_funded < threshold), search for the smallest set of ACTIONABLE changes that
  flips the prediction. Transparent greedy search, written in-repo rather than
  imported, so the constraint set is exactly the actionability table in
  RESULTS_LOG §3.1 and every step is inspectable.

THE PERMITTED ACTION SET (RESULTS_LOG §3.1; protected attributes frozen — gender,
country, partner, sector, activity and all realised outcomes can never move):
  amount   reduce the ask: ×0.9 ×0.8 ×0.7 ×0.6 ×0.5   (updates log_loan_amount
           and amount_per_borrower together; one action, one family)
  use_text lengthen the short narrative to the train-funded p50/p75/p90
  desc     lengthen the description to the train-funded p50/p75/p90
  month    move the posting month to any other month (posting_doy moved to the
           month's midpoint; day-of-week left unchanged — unknowable)
  tags     add one presentation tag from {eco_friendly, health_sanitation,
           technology, animals} (identity-linked tags are excluded: woman_owned,
           parent, elderly, repeat_borrower cannot honestly be "added")

DISTANCE  numeric action: |Δ| / MAD of the feature in train (robust units);
          month or tag action: fixed cost 1.0. Up to --max-changes actions.

INPUT   outputs/models/<run>_gbm_binary.pkl, outputs/tables/<run>_test2019_scores.csv,
        + the features file
OUTPUT  outputs/tables/<out-id>_counterfactuals.csv   one row per attempted loan
        outputs/tables/<out-id>_cf_summary.csv        both readings
        outputs/figures/F28_cf_actions.png

USAGE   python3 src/explain/10_counterfactuals.py
        python3 src/explain/10_counterfactuals.py --base flagged --n-instances 1500

Code written with the assistance of Claude (Anthropic).
"""
import argparse, os, sys, time, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from pipeline_common import (SEED, CANON_TEST, load_features, load_bundle,
                             predict_p_funded, bar, hms, rule, mpl, tidy,
                             save_fig, S1, DEEMPH)

AMOUNT_FACTORS = [0.9, 0.8, 0.7, 0.6, 0.5]
TEXT_QS = [0.50, 0.75, 0.90]
TAG_ADDS = ["tag_eco_friendly", "tag_health_sanitation", "tag_technology",
            "tag_animals"]


def mad(x):
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return 1.0
    m = np.median(x)
    return max(1e-9, 1.4826 * np.median(np.abs(x - m)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/tables/feat_v1_full.csv")
    ap.add_argument("--out", default="outputs/tables")
    ap.add_argument("--figdir", default="outputs/figures")
    ap.add_argument("--modeldir", default="outputs/models")
    ap.add_argument("--n-instances", type=int, default=0,
                    help="0 = every loan in the base")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="target p_funded that counts as recourse achieved")
    ap.add_argument("--max-changes", type=int, default=3)
    ap.add_argument("--base", choices=["actual-short", "flagged"],
                    default="actual-short")
    ap.add_argument("--model-run", default="R013",
                    help="R013 = tuned full-cohort binary model; R004 = untuned")
    ap.add_argument("--out-id", default="R015",
                    help="run id for the outputs; R007 was the untuned first pass")
    ap.add_argument("--figure", default="F28_cf_actions.png")
    a = ap.parse_args()

    rule("who-gets-funded  R007 counterfactual recourse search (binary aux model)")
    bundle = load_bundle(os.path.join(a.modeldir, f"{a.model_run}_gbm_binary.pkl"))
    scores_p = os.path.join(a.out, f"{a.model_run}_test2019_scores.csv")
    if not os.path.exists(scores_p):
        sys.exit(f"missing {scores_p} — run the binary model step first")
    scores = pd.read_csv(scores_p)
    df = load_features(a.features)
    feats = bundle["feat_cols"]

    # ---- the base of the audit ----------------------------------------
    if a.base == "actual-short":
        neg = scores[scores.y_short == 1]
        what = "loans that actually fell short"
    else:
        neg = scores[scores.p_funded < a.threshold]
        what = f"loans the model scores below p_funded {a.threshold}"
    n_already = int((neg.p_funded >= a.threshold).sum())
    print(f"  test-{CANON_TEST}: {len(scores):,} loans; base = {what}: {len(neg):,}")
    print(f"  of those, {n_already:,} already score at or above the target "
          f"({100*n_already/max(1,len(neg)):.1f}%) — kept, flagged, reported both ways")
    if len(neg) == 0:
        sys.exit("  nothing to explain — the base is empty")
    cap = a.n_instances if a.n_instances and a.n_instances > 0 else len(neg)
    take = (neg.groupby("gender", group_keys=False)
               .apply(lambda g_: g_.sample(
                   n=min(len(g_), max(1, int(round(cap * len(g_) / len(neg))))),
                   random_state=SEED))
            if len(neg) > cap else neg)
    inst = df[df.loan_id.isin(take.loan_id)].reset_index(drop=True)
    meta = inst[["loan_id", "gender", "country_iso", "sector"]].copy()
    print(f"  attempting {len(inst):,} instances "
          f"(gender mix preserved: {dict(meta.gender.value_counts())})")

    # train statistics for targets and distances
    tr = df[df.posting_year < CANON_TEST]
    tr_funded = tr[tr.is_fully_funded == 1]
    text_targets = {c: [int(tr_funded[c].quantile(q)) for q in TEXT_QS]
                    for c in ("use_text_length", "description_length")}
    mads = {"log_loan_amount": mad(tr.log_loan_amount.values),
            "use_text_length": mad(tr.use_text_length.values),
            "description_length": mad(tr.description_length.values)}
    print(f"  text targets (train-funded p50/p75/p90): {text_targets}")

    # action registry: (family, label, applicable(state)->mask, apply(sub)->None, cost(before,after)->vec)
    acts = []
    for f in AMOUNT_FACTORS:
        def _ap(sub, f=f):
            sub["log_loan_amount"] += np.log(f)
            sub["amount_per_borrower"] *= f
        acts.append(("amount", f"amount×{f}",
                     lambda st, f=f: np.ones(len(st), bool),
                     _ap,
                     lambda b, aft: np.abs(aft.log_loan_amount.values
                                           - b.log_loan_amount.values)
                     / mads["log_loan_amount"]))
    for col, short in (("use_text_length", "use_text"),
                       ("description_length", "desc")):
        for T in text_targets[col]:
            def _ap(sub, col=col, T=T):
                sub[col] = float(T)
                if col == "use_text_length":
                    sub["has_use_text"] = 1
                else:
                    sub["has_description"] = 1
            acts.append((short, f"{short}→{T}",
                         lambda st, col=col, T=T: ~(st[col] >= T),  # NaN counts as applicable
                         _ap,
                         lambda b, aft, col=col: np.abs(
                             aft[col].values - np.nan_to_num(
                                 b[col].values, nan=float(np.nanmedian(tr[col]))))
                         / mads[col]))
    for m_ in range(1, 13):
        def _ap(sub, m_=m_):
            sub["posting_month"] = m_
            sub["posting_doy"] = int((m_ - 0.5) * 30.44)
        acts.append(("month", f"month→{m_}",
                     lambda st, m_=m_: st.posting_month.values != m_,
                     _ap, lambda b, aft: np.full(len(b), 1.0)))
    for tg in TAG_ADDS:
        def _ap(sub, tg=tg):
            sub[tg] = 1
            sub["n_tags_desc"] += 1
        acts.append((tg.replace("tag_", "+"), f"+{tg[4:]}",
                     lambda st, tg=tg: st[tg].values == 0,
                     _ap, lambda b, aft: np.full(len(b), 1.0)))
    print(f"  action set: {len(acts)} candidate moves in "
          f"{len(set(x[0] for x in acts))} families; protected attributes frozen")

    # greedy batched search
    state = inst[feats].copy().reset_index(drop=True)
    p_now = predict_p_funded(bundle, state)
    p0 = p_now.copy()
    success = p_now >= a.threshold
    stalled = np.zeros(len(state), bool)
    cost = np.zeros(len(state))
    changes = [[] for _ in range(len(state))]
    fams_used = [set() for _ in range(len(state))]
    t0 = time.time()
    for depth in range(1, a.max_changes + 1):
        active = ~success & ~stalled
        if not active.mask.any() if hasattr(active, "mask") else not active.any():
            break
        idx_active = np.where(active)[0]
        cand_frames, cand_meta = [], []
        for ai, (fam, label, applicable, apply_fn, cost_fn) in enumerate(acts):
            fam_ok = np.array([fam not in fams_used[i] for i in idx_active])
            app = np.asarray(applicable(state.iloc[idx_active]), dtype=bool) & fam_ok
            if not app.any():
                continue
            rows = idx_active[app]
            sub = state.iloc[rows].copy()
            before = state.iloc[rows]
            apply_fn(sub)
            c = np.asarray(cost_fn(before, sub), dtype=float)
            cand_frames.append(sub)
            cand_meta.append(pd.DataFrame({"row": rows, "action": ai, "cost": c}))
        if not cand_frames:
            break
        big = pd.concat(cand_frames, ignore_index=True)
        metaC = pd.concat(cand_meta, ignore_index=True)
        bar(depth - 1, a.max_changes, t0,
            f"depth {depth}: {len(big):,} candidates")
        metaC["p"] = predict_p_funded(bundle, big)
        # choose per instance: cheapest crossing if any, else best improvement
        chosen = {}
        for row, grp in metaC.groupby("row"):
            cross = grp[grp.p >= a.threshold]
            pick = (cross.sort_values("cost").iloc[0] if len(cross)
                    else grp.sort_values("p", ascending=False).iloc[0])
            if not len(cross) and pick.p <= p_now[row] + 1e-9:
                stalled[row] = True
                continue
            chosen[int(row)] = (int(pick.action), float(pick.cost), float(pick.p),
                                int(pick.name))
        for row, (ai, c, p, big_ix) in chosen.items():
            state.iloc[row] = big.iloc[big_ix]
            cost[row] += c
            p_now[row] = p
            fam, label = acts[ai][0], acts[ai][1]
            fams_used[row].add(fam)
            changes[row].append(label)
            if p >= a.threshold:
                success[row] = True
        bar(depth, a.max_changes, t0,
            f"depth {depth}: {int(success.sum()):,} flipped")
    print()

    amt_pct = []
    for ch in changes:
        f = 1.0
        for lab in ch:
            if lab.startswith("amount×"):
                f *= float(lab.split("×")[1])
        amt_pct.append(round(100 * (1 - f), 1))
    res = meta.copy()
    res["p_start"] = np.round(p0, 6)
    res["p_final"] = np.round(p_now, 6)
    res["success"] = success.astype(int)
    res["n_changes"] = [len(c) for c in changes]
    res["distance"] = np.round(cost, 4)
    res["amount_cut_pct"] = amt_pct
    res["changes"] = [" | ".join(c) if c else "" for c in changes]
    res["already_at_target"] = (res.p_start >= a.threshold).astype(int)
    os.makedirs(a.out, exist_ok=True)
    res.to_csv(os.path.join(a.out, f"{a.out_id}_counterfactuals.csv"), index=False)

    def block(d, name):
        okd = d[d.success == 1]
        return {"reading": name, "attempted": len(d), "feasible": len(okd),
                "feasibility_pct": round(100 * len(okd) / max(1, len(d)), 2),
                "median_distance": round(float(okd.distance.median()), 4) if len(okd) else np.nan,
                "mean_distance": round(float(okd.distance.mean()), 4) if len(okd) else np.nan,
                "median_changes": float(okd.n_changes.median()) if len(okd) else np.nan,
                "median_amount_cut_pct": (round(float(
                    okd.loc[okd.amount_cut_pct > 0, "amount_cut_pct"].median()), 1)
                    if (okd.amount_cut_pct > 0).any() else 0.0),
                "base": a.base, "threshold": a.threshold,
                "max_changes": a.max_changes, "engine": bundle["engine"]}

    summ = pd.DataFrame([
        block(res, "all in base (already-at-target counted at distance 0)"),
        block(res[res.already_at_target == 0], "excluding already-at-target")])
    summ.to_csv(os.path.join(a.out, f"{a.out_id}_cf_summary.csv"), index=False)
    print("\n" + summ.to_string(index=False))
    print(f"\n  already at target: {int(res.already_at_target.sum()):,} of {len(res):,}"
          f" — the model did not flag these despite their actual shortfall")

    plt = mpl()
    ok = res[(res.success == 1) & (res.already_at_target == 0)]
    fam_counts = {}
    for ch in ok.changes:
        for lab in str(ch).split(" | "):
            if lab:
                fam = lab.split("×")[0].split("→")[0]
                fam_counts[fam] = fam_counts.get(fam, 0) + 1
    fc = pd.Series(fam_counts).sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    ax.barh(range(len(fc)), fc.values, height=0.5, color=S1, zorder=3)
    ax.set_yticks(range(len(fc)), fc.index, fontsize=8.5)
    ax.set_xlabel("Times used in a successful counterfactual")
    tidy(ax, xgrid=True)
    need = res[res.already_at_target == 0]
    ax.set_title(f"{a.figure.split('_')[0]}  Which actionable changes lift the "
                 f"prediction to the target\n"
                 f"{len(ok):,} of {len(need):,} loans needing a change were fixed "
                 f"({100*len(ok)/max(1,len(need)):.1f}%); protected attributes frozen",
                 fontsize=9.5, loc="left", pad=8)
    save_fig(fig, a.figdir, a.figure,
             "Action families used in successful counterfactuals")

    rule(f"{a.out_id} complete — the recourse audit step reads these by group")


if __name__ == "__main__":
    main()
