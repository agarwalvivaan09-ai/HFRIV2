"""Shared settings and functions for every HFRI v2 script.

Every script imports this file, so the formulas, weights, seed and data locations
are defined in exactly one place.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ folders
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DERIVED = ROOT / "data" / "derived"
MASTER = ROOT / "data" / "hfri_master_dataset.csv"
OUT = ROOT / "output"
RES = OUT / "results"
FIG = OUT / "figures"
TAB = OUT / "tables"
for _d in (RAW, DERIVED, RES, FIG, TAB):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ settings
SEED = 42
BOOTSTRAP_REPS = 400        # bootstrap resamples for confidence intervals
SPLITS = 10                 # random half-splits for the held-out weight test (about 30-40 s each)

# Weights in the order: Savings Strength, Debt Sustainability, Emergency Coverage, Expense Stability
W_ORIGINAL = (0.25, 0.25, 0.25, 0.25)
W_FINAL = (0.05, 0.30, 0.55, 0.10)
W_EMERGENCY_ONLY = (0.0, 0.0, 1.0, 0.0)
VERSIONS = {"original": W_ORIGINAL, "final": W_FINAL, "emergency_only": W_EMERGENCY_ONLY}

# ------------------------------------------------------------------ data sources
# File fingerprints (SHA-256) of the exact files used in the paper.
SOURCES = {
    "scf2025": {
        "name": "Survey of Consumer Finances 2025, public summary extract",
        "url": "https://www.federalreserve.gov/econres/files/scfp2025excel.zip",
        "zip": "scfp2025excel.zip", "file": "SCFP2025.csv",
        "sha256": "8f6de8f528de484eeca6eb01a19566a0f084ba21452c1198e8eebf03fc43a8f8",
    },
    "scf2022": {
        "name": "Survey of Consumer Finances 2022, public summary extract",
        "url": "https://www.federalreserve.gov/econres/files/scfp2022excel.zip",
        "zip": "scfp2022excel.zip", "file": "SCFP2022.csv",
        "sha256": "ddc45f83327dfb2fe431fcd128b662cdfb965a140405bab28e59356aa4eeebe8",
    },
    "panel": {
        "name": "2007-09 Panel Survey of Consumer Finances, public summary extract",
        "url": "https://www.federalreserve.gov/econres/files/rscfp2009panelexcel.zip",
        "zip": "rscfp2009panelexcel.zip", "file": "SCFP2009panel.xlsx",
        "sha256": "e6a692ac3a0cb35ec9c9ecd44b353e34b3ea841c4a158729bd56883706ea690f",
    },
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ HFRI scoring functions (paper, Section 4)
def savings_score(sr):
    sr = np.asarray(sr, dtype=float)
    return np.clip(np.where(sr <= -0.10, 0.0, 100 * (1 - np.exp(-5.89681 * (sr + 0.10)))), 0, 100)


def dti_score(dti):
    dti = np.clip(np.asarray(dti, dtype=float), 0, None)
    return np.clip(100 * np.exp(-0.0573534 * dti ** 2.13509), 0, 100)


def dsti_score(dsti):
    dsti = np.asarray(dsti, dtype=float)
    return np.clip(-521.45598 * dsti ** 2 + 9.08245 * dsti + 100.00523, 0, 100)


def debt_score(dti, dsti):
    return 0.40 * dti_score(dti) + 0.60 * dsti_score(dsti)


def emergency_score(ec):
    ec = np.clip(np.asarray(ec, dtype=float), 0, None)
    return np.clip(100 * (1 - np.exp(-0.37028 * ec ** 0.901391)), 0, 100)


def stability_score(cv):
    cv = np.clip(np.asarray(cv, dtype=float), 0, None)
    return np.clip(100 * np.exp(-3.13957 * cv ** 1.30837), 0, 100)


def hfri(df, w):
    """Weighted HFRI. If Expense Stability (X) is missing for a household (2007-09 panel),
    the remaining three weights are rescaled to sum to one."""
    S, D, E = df["S"].to_numpy(float), df["D"].to_numpy(float), df["E"].to_numpy(float)
    X = df["X"].to_numpy(float)
    has_x = ~np.isnan(X)
    num = w[0] * S + w[1] * D + w[2] * E + np.where(has_x, w[3] * np.nan_to_num(X), 0.0)
    den = w[0] + w[1] + w[2] + np.where(has_x, w[3], 0.0)
    return np.clip(num / den, 0, 100)


# ------------------------------------------------------------------ survey helpers
def after_tax(income):
    """Approximate disposable income: average U.S. tax rate (federal + state + payroll) by income band."""
    y = np.asarray(income, dtype=float)
    rate = np.select([y < 30000, y < 60000, y < 100000, y < 200000, y < 500000],
                     [0.08, 0.14, 0.18, 0.22, 0.27], 0.32)
    return y * (1 - rate)


def auc(y, risk, w=None):
    """Weighted area under the ROC curve (Mann-Whitney form, ties count one half).
    Identical to sklearn.metrics.roc_auc_score(y, risk, sample_weight=w), but much faster."""
    y = np.asarray(y).astype(bool)
    r = np.asarray(risk, dtype=float)
    w = np.ones(len(r)) if w is None else np.asarray(w, dtype=float)
    o = np.argsort(r, kind="mergesort")
    r, y, w = r[o], y[o], w[o]
    _, start = np.unique(r, return_index=True)
    wn = np.where(~y, w, 0.0)
    wp = np.where(y, w, 0.0)
    ng = np.add.reduceat(wn, start)
    pg = np.add.reduceat(wp, start)
    below = np.cumsum(ng) - ng
    return float((pg * (below + 0.5 * ng)).sum() / (wp.sum() * wn.sum()))


def distress_auc(y, score, w=None):
    """AUC for an outcome where 1 = distress: a LOW HFRI should mean HIGH risk."""
    return auc(y, -np.asarray(score, float), w)


def wmean(x, w):
    x, w = np.asarray(x, float), np.asarray(w, float)
    return float(np.sum(x * w) / np.sum(w))


def wquantile(x, w, q):
    x, w = np.asarray(x, float), np.asarray(w, float)
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return float(np.interp(q, c, x[o]))


def implicates(df, cols, id_col="hh_id"):
    """List of aligned arrays, one per survey implicate, households in the same order
    (needed for paired bootstrap resampling across the five implicates)."""
    out = []
    for _, g in df.groupby("implicate"):
        g = g.sort_values(id_col)
        out.append({c: g[c].to_numpy() for c in cols})
    return out


def rubin_logit(df, y, xs, weight_col, scale=None):
    """Weighted logistic regression fitted on each implicate and combined with Rubin's rules.
    Returns odds ratios, 95% CIs and p-values. `scale` divides each x (e.g. 10 = per 10 points)."""
    import statsmodels.api as sm
    from scipy.stats import norm
    scale = scale or {}
    est, var = [], []
    for _, g in df.groupby("implicate"):
        X = pd.DataFrame({x: g[x] / scale.get(x, 1.0) for x in xs})
        X = sm.add_constant(X, has_constant="add")
        fit = sm.GLM(g[y], X, family=sm.families.Binomial(),
                     var_weights=g[weight_col] / g[weight_col].mean()).fit()
        est.append(fit.params[xs].to_numpy())
        var.append(fit.bse[xs].to_numpy() ** 2)
    est, var = np.array(est), np.array(var)
    m = len(est)
    q = est.mean(0)
    t = var.mean(0) + (1 + 1 / m) * est.var(0, ddof=1)
    se = np.sqrt(t)
    return {x: {"odds_ratio": float(np.exp(q[i])),
                "ci_low": float(np.exp(q[i] - 1.96 * se[i])),
                "ci_high": float(np.exp(q[i] + 1.96 * se[i])),
                "p": float(2 * norm.sf(abs(q[i] / se[i])))} for i, x in enumerate(xs)}


# ------------------------------------------------------------------ results
def save_json(name, obj):
    path = RES / f"{name}.json"
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)
    print(f"Saved {path.relative_to(ROOT)}")


def load_json(name):
    with open(RES / f"{name}.json") as f:
        return json.load(f)


def load_master(dataset=None):
    df = pd.read_csv(MASTER, low_memory=False)
    return df if dataset is None else df[df["dataset"] == dataset].copy()
