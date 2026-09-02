# Who Gets Funded

An automated, explainable machine-learning pipeline for predicting funding outcomes and equity of access in prosocial crowdfunding.

MSc Business Analytics dissertation, Queen's Business School, Queen's University Belfast.

**Stack:** Python · SQLite · LightGBM · statsmodels · scikit-learn · matplotlib
**Data:** Kiva official research snapshot (~3.1M loans). Not distributed here — see below.

---

## What this is

Kiva is a prosocial crowdfunding platform: borrowers post loan requests, lenders fund them in increments, and a request that isn't fully funded before it expires returns nothing. This pipeline asks who gets funded, whether it can be predicted from what is visible at launch, why the model decides what it decides, and — the part that matters most — whether the ability to *act* on that prediction is distributed equally.

The code runs end to end: harvest, ingest, cohort construction, feature build, baseline and gradient-boosted models, explanation, counterfactuals, and a set of equity audits.

---

## The research chain

The stages exist in this order for a reason. Each question only makes sense once the previous one is answered.

1. **Who gets funded?** Descriptive and compositional analysis of outcomes across gender, country, sector and loan size.
2. **Can it be predicted?** A naive benchmark first, then a fractional logit baseline, then gradient boosting — each measured against the floor below it.
3. **Why?** Feature attribution, so the model's reasoning is inspectable rather than asserted.
4. **What could have changed the outcome?** Counterfactual search restricted to features an applicant could realistically alter.
5. **Is recourse equally available?** Whether the changes the model demands are equally achievable across groups.

The fifth question is the contribution. A model can be well calibrated and still hand one group a cheaper route to a better outcome than another.

---

## Design decisions

**The outcome is fractional, not binary.** The target is the share of the requested amount raised — a value in [0,1], not funded/not-funded. Collapsing it to a binary throws away the difference between a request that raised 5% and one that raised 95%. The baseline is therefore a fractional logit (Papke–Wooldridge quasi-MLE), which is built for a bounded proportional outcome, rather than an ordinary logit on a manufactured threshold.

**Only information available at launch is used.** A feature that encodes how funding actually progressed would predict the outcome trivially and mean nothing. The feature build is constrained to what a lender could see the moment the request went live.

**The cohort stops before a known regime change.** Kiva ended all-or-nothing funding partway through the period covered by the data. Training and evaluation sit entirely inside the comparable earlier regime; the later period is held back and used as an out-of-regime robustness check, not as a conventional temporal test. Treating a rule change as ordinary drift would be a mistake dressed up as validation.

**Counterfactuals are restricted to what an applicant can actually change.** Loan amount, description, duration — yes. Gender, country, sector — never. The actionability set is defined explicitly in the code rather than left to the search to discover, because a counterfactual that tells someone to change their nationality is not advice, it's an artefact.

**Identity variables are used for measurement, not prescription.** They appear in the disparity audits and nowhere else.

**The equity questions are kept separate.** Outcome disparities, prediction disparities, error-rate disparities and recourse cost are four different things with four different remedies. They are reported separately and never pooled into a single "fairness" number.

**TreeSHAP comes from LightGBM directly.** Exact attributions via the model's native contribution output rather than an additional dependency. Counterfactual search is implemented in-repo so its constraints match the actionability table exactly, rather than being approximated by a general-purpose library.

---

## Pipeline

Stages are numbered in execution order. `run_full_pipeline.py` drives the whole thing; `pipeline_common.py` holds shared configuration, paths and helpers.

| Stage | Directory | What it does |
|---|---|---|
| 00–02 | `src/ingest/` | Schema introspection, GraphQL harvest, load to SQLite, descriptive statistics, cohort composition checks |
| 03 | `src/features/` | Feature construction in SQL, development sample, out-of-regime features, Parquet conversion |
| 04–05 | `src/models/` | Naive benchmark floor, then the fractional logit baseline |
| 06–08 | `src/models/` | Gradient boosting with temporal cross-validation, binary auxiliary model, out-of-regime regime test |
| 09–10 | `src/explain/` | TreeSHAP attribution; constrained counterfactual search |
| 11 | `src/equity/` | Recourse audit |
| 12–14 | `src/models/` | Model comparison, hyperparameter tuning, final fits on the full cohort |
| 15 | `src/equity/` | Group calibration and error-rate audit |
| 16 | `src/models/` | Final model comparison |

Validation folds are strictly temporal. The held-out test period and the per-fold naive floors are fixed and never re-derived from a tuned model.

---

## Repository layout

```
who-gets-funded/
├── src/
│   ├── ingest/        harvest, load, describe, compositional checks
│   ├── features/      feature construction and sampling
│   ├── models/        benchmark, baseline, boosting, tuning, comparison
│   ├── explain/       attribution and counterfactuals
│   ├── equity/        recourse and calibration audits
│   ├── pipeline_common.py
│   └── run_full_pipeline.py
├── outputs/
│   ├── figures/       F01–F36
│   ├── tables/        result tables
│   ├── model_comparison/
│   └── final_comparison/
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

---

## What isn't here

**The data.** The Kiva snapshot is obtained from Kiva's own research portal under their terms; it is not redistributed here. The harvest and ingest code in `src/ingest/` fetches and loads it. The working database is ~13GB and would not belong in a repository regardless.

**The dissertation.** The research report and technical report are assessed work and are not published here. This repository is the technical artefact — the code, the pipeline, and the outputs it produces.

**Results and interpretation.** Findings belong to the written work and to a paper in preparation. What is published here is the machinery that produced them.

---

## Reproducing

```bash
pip install -r requirements.txt
python src/run_full_pipeline.py
```

Obtain the Kiva research snapshot yourself and configure the path in `pipeline_common.py`. Stages can be run individually in numeric order; each writes its outputs and logs before the next begins.

---

## Licence

MIT — see [`LICENSE`](LICENSE).

---

## Citation

If you use this code, please cite it via [`CITATION.cff`](CITATION.cff), or use the "Cite this repository" button in the sidebar.

---

## AI assistance

The code in this repository was written with the assistance of Claude (Anthropic) and, in earlier stages, ChatGPT, working to my specifications. The research question, cohort definition, outcome variable, model specification, actionability constraints and audit design are mine; every result was run and checked by me. Each source file carries an assistance header, and the use is disclosed in the accompanying technical report.

---

**Fazal Ur Rehman Cheema** · MSc Business Analytics, Queen's University Belfast
