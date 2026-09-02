#!/usr/bin/env python3
# Code written with the assistance of Claude (Anthropic).
"""
who-gets-funded — shared plumbing for pipeline steps 06-12.

Nothing analytical lives here: the progress bar, the figure palette, metric
helpers, the feat_v1 column contract, the temporal folds, the R001 naive floors,
and a gradient-boosting engine wrapper (lightgbm preferred, xgboost second,
scikit-learn HistGradientBoosting as the always-available fallback). Every
numbered step stays readable on its own; this module only keeps them short.

Run everything from the repo root. Nothing here writes to the database or edits
any input file.
"""
import os, pickle, sys, time, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

SEED = 40465466            # student number — every random draw is reproducible

# ---------------------------------------------------------------- terminal --
BAR_W = 34


def hms(s):
    s = int(s)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s//60}m {s%60:02d}s"
    return f"{s//3600}h {(s%3600)//60:02d}m"


def bar(done, total, t0, label=""):
    frac = 0.0 if not total else min(1.0, done / total)
    fill = int(BAR_W * frac)
    el = time.time() - t0
    eta = (total - done) * el / done if done else 0
    sys.stdout.write(f"\r  [{'#'*fill}{'.'*(BAR_W-fill)}] {frac*100:5.1f}%  "
                     f"{done:>7,}/{total:,}  {hms(el)} elapsed  ETA {hms(eta)}  {label:<30}")
    sys.stdout.flush()


def rule(t):
    print(f"\n{'='*90}\n{t}\n{'='*90}")


# ---------------------------------------------------------------- metrics ---
def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


# R001 floors (B0 = constant 1.0). Keys: test year -> (mae, rmse, mae_on_short)
B0 = {2016: (0.039403, 0.159988, 0.5551),
      2017: (0.020863, 0.113719, 0.5212),
      2018: (0.047102, 0.185991, 0.6705),
      2019: (0.048465, 0.190227, 0.6888)}

# B3 binary floors: always predict "funded". AUC 0.5 by construction; the PR-AUC
# floor for positive class = SHORTFALL equals the shortfall prevalence.
B3_ACC = {2016: 92.9014, 2017: 95.9969, 2018: 92.9753, 2019: 92.9636}

FOLD_TEST_YEARS = [2016, 2017, 2018, 2019]   # expanding window: train = all earlier
CANON_TEST = 2019                            # canonical split: train<=2018, test 2019

# ------------------------------------------------------- feat_v1 contract ---
TARGET_FRAC = "share_raised"
TARGET_BIN = "is_fully_funded"          # 1 = fully funded; SHORTFALL = 1 - this
ID_COLS = ["loan_id", "fundraising_date", "posting_year"]
DROP_ALWAYS = ["loan_amount",           # monotone duplicate of log_loan_amount
               "x_excl_research_score", "x_leak_user_favorite", "x_leak_volunteer_pick"]
CAT_COLS = ["gender", "country_iso", "sector", "activity", "region", "partner_id",
            "original_language", "repayment_interval", "anonymization_level"]
PROTECTED = ["gender", "country_iso"]   # in the model for auditing; frozen in recourse


def feature_cols(df):
    """Model features = every column minus ids, targets and the always-dropped."""
    skip = set(ID_COLS + [TARGET_FRAC, TARGET_BIN] + DROP_ALWAYS)
    return [c for c in df.columns if c not in skip]


# feat_v1 storage schema. Loading 1.34m rows without this materialises the nine
# categorical columns as Python objects and peaks several GB; with it the frame
# sits in a few hundred MB. Any column not listed falls back to pandas inference.
DTYPES = {
    "loan_id": "int32", "posting_year": "int16",
    "share_raised": "float32", "is_fully_funded": "int8",
    "gender": "category", "country_iso": "category", "sector": "category",
    "activity": "category", "region": "category", "original_language": "category",
    "repayment_interval": "category", "anonymization_level": "category",
    "partner_id": "int32",
    "loan_amount": "float32", "log_loan_amount": "float32",
    "amount_per_borrower": "float32", "borrower_count": "int16",
    "is_group": "int8", "has_use_text": "int8", "has_description": "int8",
    "use_text_length": "float32", "description_length": "float32",
    "posting_month": "int8", "posting_dow": "int8", "posting_doy": "int16",
    "fundraising_window_days": "float32", "lender_repayment_term": "int16",
    "min_note_size": "float32", "n_tags_desc": "int8", "n_themes": "int8",
    "is_repeat_borrower": "int8", "has_previous_loan": "int8",
    "x_excl_research_score": "float32",
    "x_leak_user_favorite": "int8", "x_leak_volunteer_pick": "int8",
}
for _t in ("woman_owned", "parent", "repeat_borrower", "elderly", "animals",
           "eco_friendly", "health_sanitation", "technology"):
    DTYPES[f"tag_{_t}"] = "int8"
for _t in ("underfunded", "rural_exclusion", "startup", "crop_insurance",
           "vulnerable", "conflict_zones", "water_sanitation",
           "higher_education", "refugees", "clean_energy"):
    DTYPES[f"theme_{_t}"] = "int8"


def load_features(path, years=None, columns=None, typed=True):
    """Load feat_v1 from .parquet if a sibling exists (or the path names one),
    otherwise CSV with the DTYPES schema applied.

    years    : optional iterable of posting_year values to keep. With parquet
               this is pushed into the reader, so unwanted years are never
               materialised at all.
    columns  : optional column subset.
    """
    pq = path[:-4] + ".parquet" if path.endswith(".csv") else path
    if os.path.exists(pq) and pq.endswith(".parquet"):
        flt = [("posting_year", "in", list(years))] if years else None
        df = pd.read_parquet(pq, columns=columns, filters=flt)
        if years and "posting_year" in df.columns:      # older pyarrow ignores filters
            df = df[df.posting_year.isin(list(years))]
        return df.reset_index(drop=True)

    if not os.path.exists(path):
        sys.exit(f"missing {path} — run the feature build first")
    if not typed:
        return pd.read_csv(path, low_memory=False)

    head = pd.read_csv(path, nrows=0)
    dt = {c: t for c, t in DTYPES.items() if c in head.columns}
    use = [c for c in head.columns if columns is None or c in columns]
    # NaN-bearing integer columns must be read as float, never as int
    df = pd.read_csv(path, dtype=dt, usecols=use, low_memory=False)
    if years is not None and "posting_year" in df.columns:
        df = df[df.posting_year.isin(list(years))].reset_index(drop=True)
    return df


def mem_mb(df):
    return df.memory_usage(deep=True).sum() / 1e6


def _as_str(s):
    """Categorical column to plain strings, with missing mapped to the literal
    '(missing)' so pandas 3 (which keeps NaN through astype(str)) cannot mix
    floats and strings."""
    return s.astype(object).where(s.notna(), "(missing)").astype(str)


def build_categories(df, cols, engine=None):
    """Category levels from train. For the sklearn engine, columns beyond its
    255-level limit keep only their most frequent SKLEARN_MAX_CAT levels; the
    rest map to NaN = missing (partner_id has ~296 levels in a train fold, which
    crashed HistGradientBoosting outright)."""
    cats = {}
    for c in cols:
        vc = _as_str(df[c]).value_counts()
        levels = vc.index.tolist()
        if engine == "sklearn" and len(levels) > SKLEARN_MAX_CAT:
            print(f"  note: '{c}' capped to its {SKLEARN_MAX_CAT} most frequent of "
                  f"{len(levels)} levels (sklearn 255-category limit); rest -> missing")
            levels = levels[:SKLEARN_MAX_CAT]
        cats[c] = sorted(levels)
    return cats


def apply_categories(X, cats):
    """Set stored category mappings; unseen levels become NaN = missing to the
    trees. Builds the frame column-by-column rather than copying it wholesale —
    on the full cohort the old unconditional .copy() doubled peak memory at
    exactly the moment LightGBM was also allocating."""
    data = {}
    for c in X.columns:
        if c in cats:
            data[c] = pd.Categorical(_as_str(X[c]), categories=cats[c])
        else:
            data[c] = X[c].to_numpy(copy=False)
    return pd.DataFrame(data, index=X.index, copy=False)


# ----------------------------------------------------------- GBM engines ----
SKLEARN_MAX_CAT = 250   # HistGradientBoosting hard-caps categoricals at 255 levels


def detect_engine():
    """lightgbm > xgboost > sklearn. A failed lightgbm import is REPORTED, not
    swallowed — on macOS the usual cause is the missing OpenMP runtime, fixed by
    `brew install libomp`. Set who-gets-funded_ENGINE=sklearn|xgboost|lightgbm to force."""
    forced = os.environ.get("who-gets-funded_ENGINE")
    if forced:
        print(f"  note: engine forced to '{forced}' via who-gets-funded_ENGINE")
        return forced
    try:
        import lightgbm  # noqa
        return "lightgbm"
    except Exception as e:
        first = str(e).splitlines()[0][:100] if str(e) else type(e).__name__
        print(f"  note: lightgbm installed? import FAILED — {type(e).__name__}: {first}")
        print( "        on macOS this is almost always the OpenMP runtime:  brew install libomp")
    try:
        import xgboost  # noqa
        return "xgboost"
    except Exception:
        pass
    return "sklearn"


# Fixed, deliberately UNTUNED defaults — tuning is a later, separate stage.
GBM_PARAMS = dict(n_estimators=2000, learning_rate=0.05, num_leaves=63,
                  min_child_samples=100, subsample=0.9, subsample_freq=1,
                  colsample_bytree=0.9, early_stopping_rounds=100)


def fit_gbm(Xtr, ytr, kind, engine, n_estimators=None, seed=SEED,
            params=None, valid=None):
    """Fit a GBM. kind='frac' (cross-entropy on a [0,1] target — the natural GBM
    analogue of the fractional logit) or 'binary'. Early stopping uses the LAST
    10% of the training rows as validation — chronological, because feat_v1 is
    sorted by fundraising_date, so the validation slice is the most recent data.
    Returns (model, best_iteration)."""
    P = dict(GBM_PARAMS)
    if params:
        P.update(params)
    if n_estimators:
        P["n_estimators"] = n_estimators
    if valid is not None:
        Xf, yf = Xtr, ytr
        Xv, yv = valid
    else:
        n_val = max(1000, int(0.1 * len(Xtr)))
        Xf, Xv = Xtr.iloc[:-n_val], Xtr.iloc[-n_val:]
        yf, yv = ytr[:-n_val], ytr[-n_val:]

    if engine == "lightgbm":
        import lightgbm as lgb
        cls = lgb.LGBMRegressor if kind == "frac" else lgb.LGBMClassifier
        obj = "cross_entropy" if kind == "frac" else "binary"
        extra = {k: v for k, v in P.items()
                 if k not in ("n_estimators", "learning_rate", "num_leaves",
                              "min_child_samples", "subsample", "subsample_freq",
                              "colsample_bytree", "early_stopping_rounds")}
        m = cls(objective=obj, n_estimators=P["n_estimators"],
                learning_rate=P["learning_rate"], num_leaves=P["num_leaves"],
                min_child_samples=P["min_child_samples"], subsample=P["subsample"],
                subsample_freq=P["subsample_freq"], colsample_bytree=P["colsample_bytree"],
                random_state=seed, verbosity=-1, **extra)
        m.fit(Xf, yf, eval_set=[(Xv, yv)],
              callbacks=[lgb.early_stopping(P["early_stopping_rounds"], verbose=False),
                         lgb.log_evaluation(0)])
        return m, int(m.best_iteration_ or P["n_estimators"])

    if engine == "xgboost":
        import xgboost as xgb
        cls = xgb.XGBRegressor if kind == "frac" else xgb.XGBClassifier
        obj = "reg:logistic" if kind == "frac" else "binary:logistic"
        m = cls(objective=obj, n_estimators=P["n_estimators"],
                learning_rate=P["learning_rate"], max_depth=8,
                subsample=P["subsample"], colsample_bytree=P["colsample_bytree"],
                tree_method="hist", enable_categorical=True, random_state=seed,
                early_stopping_rounds=P["early_stopping_rounds"], verbosity=0)
        m.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
        return m, int(getattr(m, "best_iteration", P["n_estimators"]) or P["n_estimators"])

    # sklearn fallback: histogram GBM. No cross-entropy loss for a fractional
    # target, so squared error + clipping — noted in every output it produces.
    from sklearn.ensemble import (HistGradientBoostingRegressor,
                                  HistGradientBoostingClassifier)
    cls = HistGradientBoostingRegressor if kind == "frac" else HistGradientBoostingClassifier
    m = cls(max_iter=min(P["n_estimators"], 500), learning_rate=0.06,
            max_leaf_nodes=63, min_samples_leaf=100,
            categorical_features="from_dtype", random_state=seed,
            early_stopping=True, validation_fraction=0.1)
    m.fit(Xtr, ytr)
    return m, int(getattr(m, "n_iter_", 0))


def predict_frac(bundle, Xdf):
    X = apply_categories(Xdf[bundle["feat_cols"]], bundle["categories"])
    p = bundle["model"].predict(X)
    return np.clip(np.asarray(p, dtype=float), 0.0, 1.0)


def predict_p_funded(bundle, Xdf):
    X = apply_categories(Xdf[bundle["feat_cols"]], bundle["categories"])
    m = bundle["model"]
    if hasattr(m, "predict_proba"):
        return np.asarray(m.predict_proba(X)[:, 1], dtype=float)
    return np.clip(np.asarray(m.predict(X), dtype=float), 0.0, 1.0)


def shap_contribs(bundle, Xdf):
    """Exact TreeSHAP via the boosters' native pred_contrib — no shap package
    needed. Returns (matrix [n, p+1], names + ['bias']) or (None, reason)."""
    X = apply_categories(Xdf[bundle["feat_cols"]], bundle["categories"])
    eng = bundle["engine"]
    if eng == "lightgbm":
        booster = bundle["model"].booster_
        M = booster.predict(X, pred_contrib=True)
        if isinstance(M, list):
            M = M[1] if len(M) > 1 else M[0]
        return np.asarray(M), bundle["feat_cols"] + ["bias"]
    if eng == "xgboost":
        import xgboost as xgb
        dm = xgb.DMatrix(X, enable_categorical=True)
        M = bundle["model"].get_booster().predict(dm, pred_contribs=True)
        return np.asarray(M), bundle["feat_cols"] + ["bias"]
    return None, ("engine 'sklearn' has no native TreeSHAP — install lightgbm "
                  "(pip3 install lightgbm) and re-run step 06/07")


def gain_importance(bundle):
    m, eng = bundle["model"], bundle["engine"]
    if eng == "lightgbm":
        g = m.booster_.feature_importance(importance_type="gain")
        return pd.DataFrame({"feature": m.booster_.feature_name(), "gain": g})
    if eng == "xgboost":
        d = m.get_booster().get_score(importance_type="gain")
        return pd.DataFrame({"feature": list(d), "gain": list(d.values())})
    return pd.DataFrame({"feature": bundle["feat_cols"],
                         "gain": getattr(m, "feature_importances_",
                                         np.zeros(len(bundle["feat_cols"])))})


def save_bundle(bundle, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"  model saved -> {path}  ({os.path.getsize(path)/1e6:.1f} MB)")


def load_bundle(path):
    if not os.path.exists(path):
        sys.exit(f"missing model {path} — run the earlier step first")
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ------------------------------------------------------------- figures ------
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
S1, S2, DEEMPH = "#2a78d6", "#eb6834", "#c3c2b7"   # blue=female/model-1, orange=male/model-2
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#1c5cab", "#104281"]


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"], "font.size": 8.5, "figure.dpi": 200,
        "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "legend.frameon": False, "axes.spines.top": False,
        "axes.spines.right": False})
    return plt


def tidy(ax, xgrid=False):
    ax.grid(axis="x" if xgrid else "y", zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save_fig(fig, figdir, name, title):
    os.makedirs(figdir, exist_ok=True)
    p = os.path.join(figdir, name)
    fig.savefig(p, bbox_inches="tight")
    reg = os.path.join(figdir, "_figure_register_pipeline.csv")
    new = not os.path.exists(reg)
    with open(reg, "a", encoding="utf8") as fh:
        if new:
            fh.write("figure_id,file,title\n")
        fh.write(f"{name.split('_')[0]},{name},\"{title}\"\n")
    print(f"  figure -> {p}")


# =============================================================== TUNING =======
# Added at the tuning stage (runs R010-R011). Validation happens strictly inside
# the training years: inner folds validate on 2017 and 2018, so the 2019 test
# fold and the R001 per-fold floors in RESULTS_LOG 4.1 are never touched.

INNER_VAL_YEARS = [2017, 2018]      # train 2013..N-1, validate N
TUNED_FRAC_PATH = "outputs/models/R010_best_params_frac.json"
TUNED_BIN_PATH = "outputs/models/R011_best_params_binary.json"


def sample_params(rng, kind):
    """One random draw from the search space. n_estimators is deliberately NOT
    tuned: it is fixed high and settled by early stopping on the inner
    validation year, so tree count is chosen by the data, not by the sweep.
    cat_smooth / cat_l2 / max_cat_threshold are included because partner_id
    carries 459 levels and roughly a ninth of the SHAP signal."""
    p = {
        "learning_rate":     float(np.exp(rng.uniform(np.log(0.02), np.log(0.10)))),
        "num_leaves":        int(rng.choice([15, 31, 63, 95, 127, 191, 255])),
        "min_child_samples": int(np.exp(rng.uniform(np.log(20), np.log(500)))),
        "colsample_bytree":  float(rng.uniform(0.50, 1.00)),
        "subsample":         float(rng.uniform(0.50, 1.00)),
        "subsample_freq":    1,
        "reg_alpha":         float(rng.choice([0.0, 0.1, 0.5, 1.0, 2.0, 5.0])),
        "reg_lambda":        float(rng.choice([0.0, 0.5, 1.0, 5.0, 10.0, 20.0])),
        "min_split_gain":    float(rng.choice([0.0, 0.0, 0.05, 0.2, 0.5, 1.0])),
        "cat_smooth":        float(rng.choice([10, 25, 50, 100, 200])),
        "cat_l2":            float(rng.choice([1, 5, 10, 25, 50])),
        "max_cat_threshold": int(rng.choice([16, 32, 48, 64])),
        "n_estimators":      3000,
    }
    return p


def frac_metrics(y, p, short_mask):
    """The three fractional metrics, always reported together (RESULTS_LOG 13)."""
    return {"mae": mae(y, p), "rmse": rmse(y, p),
            "mae_short": mae(y[short_mask], p[short_mask])}


def composite_score(m, floor):
    """Selection metric for the fractional model: the mean of the percentage
    improvement over the B0 naive floor on RMSE and on shortfall-MAE. NEGATIVE
    IS BETTER, and it is a straight average of the two metrics the model is
    actually able to beat B0 on. Raw MAE is deliberately excluded — the median
    of share_raised is exactly 1.0, so B0 wins raw MAE by construction and
    selecting on it would reward a model that ignores shortfall entirely."""
    b0_mae, b0_rmse, b0_short = floor
    return 0.5 * (100 * (m["rmse"] / b0_rmse - 1)
                  + 100 * (m["mae_short"] / b0_short - 1))


def b0_floor_from(y):
    """B0 = predict 1.0 for every loan, evaluated on whatever slice is passed.
    Inner validation years have no entry in the R001 table (which covers the
    2016-2019 outer test folds only), so the floor is recomputed on the slice."""
    y = np.asarray(y, dtype=float)
    short = y < 1.0
    resid = 1.0 - y
    return (float(np.mean(np.abs(resid))),
            float(np.sqrt(np.mean(resid ** 2))),
            float(np.mean(np.abs(resid[short]))) if short.any() else float("nan"))


def load_tuned(path):
    """Tuned parameters if the sweep has been run, else None (untuned defaults)."""
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)["params"]
    return None
