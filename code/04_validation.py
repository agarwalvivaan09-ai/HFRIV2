"""04 - Phase III: validation against current financial distress (SCF 2025), with 2022 replication.

Distribution, distress by HFRI decile, AUC with bootstrap 95% CIs, paired significance tests,
single-dimension AUCs and logistic regressions combined across implicates (Rubin's rules).
Output: output/results/validation.json
"""
import numpy as np

from hfri_common import (BOOTSTRAP_REPS, SEED, VERSIONS, distress_auc, hfri, implicates, load_master,
                         rubin_logit, save_json, wmean, wquantile)

OUTCOMES = {"out_late60": "60+ days late", "out_late": "Any late payment", "out_payday": "Payday loan",
            "out_bankruptcy": "Bankruptcy (5 years)", "out_credit_constrained": "Denied or feared credit"}
rng = np.random.default_rng(SEED)

s = load_master("scf2025")
for name, w in VERSIONS.items():
    s[name] = hfri(s, w)
for dim in ["S", "D", "E", "X"]:
    s[f"only_{dim}"] = s[dim]
s["only_income"] = np.log1p(s["income"])
scores = list(VERSIONS) + ["only_S", "only_D", "only_E", "only_X", "only_income"]
imps = implicates(s, scores + ["weight"] + list(OUTCOMES))
n = len(imps[0]["weight"])
res = {"n_households": int(s["hh_id"].nunique()), "n_implicates": int(s["implicate"].nunique())}

# ---------------- outcome prevalence and HFRI distribution (weighted, averaged over implicates)
res["prevalence"] = {o: float(np.mean([wmean(a[o], a["weight"]) for a in imps])) for o in OUTCOMES}
res["distribution"] = {}
for v in ["original", "final"]:
    res["distribution"][v] = {k: float(np.mean([f(a) for a in imps])) for k, f in {
        "mean": lambda a: wmean(a[v], a["weight"]),
        "median": lambda a: wquantile(a[v], a["weight"], 0.5),
        "p10": lambda a: wquantile(a[v], a["weight"], 0.1),
        "p90": lambda a: wquantile(a[v], a["weight"], 0.9),
        "min": lambda a: a[v].min(), "max": lambda a: a[v].max()}.items()}
res["component_means"] = {d: float(np.mean([wmean(a[f"only_{d}"], a["weight"]) for a in imps])) for d in "SDEX"}

# ---------------- serious delinquency by HFRI decile
res["deciles"] = {}
for v in ["original", "final"]:
    rows = []
    for k in range(10):
        vals_h, vals_l = [], []
        for a in imps:
            rank = np.argsort(np.argsort(a[v], kind="mergesort"), kind="mergesort")
            m = (rank * 10 // n) == k
            vals_h.append(wmean(a[v][m], a["weight"][m]))
            vals_l.append(wmean(a["out_late60"][m], a["weight"][m]))
        rows.append({"decile": k + 1, "mean_hfri": float(np.mean(vals_h)), "late60": float(np.mean(vals_l))})
    res["deciles"][v] = rows


# ---------------- AUC with bootstrap CIs and paired differences
def avg_auc(o, col, idx=None):
    vals = []
    for a in imps:
        i = slice(None) if idx is None else idx
        vals.append(distress_auc(a[o][i], a[col][i], a["weight"][i]))
    return float(np.mean(vals))


boot_idx = [rng.integers(0, n, n) for _ in range(BOOTSTRAP_REPS)]
PAIRS = [("final", "original"), ("final", "emergency_only"), ("original", "emergency_only")]
res["auc"] = {}
for o, label in OUTCOMES.items():
    point = {c: avg_auc(o, c) for c in scores}
    reps = {c: np.array([avg_auc(o, c, i) for i in boot_idx]) for c in VERSIONS}
    entry = {"label": label, "auc": {}, "diff": {}}
    for c in scores:
        entry["auc"][c] = {"est": point[c]}
        if c in reps:
            entry["auc"][c].update(ci_low=float(np.percentile(reps[c], 2.5)), ci_high=float(np.percentile(reps[c], 97.5)))
    for a_, b_ in PAIRS:
        d = reps[a_] - reps[b_]
        p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / BOOTSTRAP_REPS)
        entry["diff"][f"{a_}-{b_}"] = {"est": point[a_] - point[b_], "ci_low": float(np.percentile(d, 2.5)),
                                       "ci_high": float(np.percentile(d, 97.5)), "p": float(p)}
    res["auc"][o] = entry
    print(f"{label:26s} original {point['original']:.3f} | final {point['final']:.3f} | emergency only {point['emergency_only']:.3f}")

# ---------------- logistic regressions (Rubin's rules)
s["hfri_original"] = s["original"]
s["hfri_final"] = s["final"]
s["log_income"] = np.log1p(s["income"])
res["logit_controls"] = {v: rubin_logit(s, "out_late60", [f"hfri_{v}", "log_income", "age", "kids", "married"],
                                        "weight", scale={f"hfri_{v}": 10}) for v in ["original", "final"]}
res["logit_dimensions"] = {o: rubin_logit(s, o, ["S", "D", "E", "X"], "weight", scale={d: 10 for d in "SDEX"})
                           for o in OUTCOMES}

# ---------------- replication in SCF 2022 (Debt and Emergency only; no spending data)
t = load_master("scf2022")
r22 = {}
for label, sc in {"debt_only": t["D"], "emergency_only": t["E"], "debt_and_emergency": (t["D"] + t["E"]) / 2}.items():
    r22[label] = float(np.mean([distress_auc(g["out_late60"], sc[g.index], g["weight"]) for _, g in t.groupby("implicate")]))
r22["n_households"] = int(t["hh_id"].nunique())
res["replication_2022"] = r22
save_json("validation", res)
