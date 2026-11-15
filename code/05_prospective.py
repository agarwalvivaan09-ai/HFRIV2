"""05 - Phase III: prospective test. Does the HFRI measured in 2007 predict NEW financial distress by 2009?

Uses the 2007-09 Panel SCF. Outcomes are restricted to households not already in that state in 2007.
Expense Stability is not observed in 2007, so the other weights are rescaled (see hfri_common.hfri).
Output: output/results/prospective.json
"""
import numpy as np

from hfri_common import (BOOTSTRAP_REPS, SEED, VERSIONS, distress_auc, hfri, implicates, load_master,
                         rubin_logit, save_json, wmean)

rng = np.random.default_rng(SEED)
p = load_master("panel")
for name, w in VERSIONS.items():
    p[name] = hfri(p, w)
for dim in ["S", "D", "E"]:
    p[f"only_{dim}"] = p[dim]
p["new_late60_mask"] = (p["pan_late60_07"] == 0).astype(int)
p["new_credit_mask"] = (p["pan_credit_constrained_07"] == 0).astype(int)
scores = list(VERSIONS) + ["only_S", "only_D", "only_E"]
imps = implicates(p, scores + ["weight", "pan_late60_09", "pan_credit_constrained_09", "new_late60_mask", "new_credit_mask"])
n = len(imps[0]["weight"])

TESTS = {"new_late60": ("pan_late60_09", "new_late60_mask", "New 60+ days late by 2009"),
         "new_credit_constrained": ("pan_credit_constrained_09", "new_credit_mask", "New credit denial or fear by 2009")}


def avg_auc(outcome, mask, col, idx=None):
    vals = []
    for a in imps:
        i = np.arange(n) if idx is None else idx
        m = a[mask][i] == 1
        vals.append(distress_auc(a[outcome][i][m], a[col][i][m], a["weight"][i][m]))
    return float(np.mean(vals))


res = {"n_households": int(p["hh_id"].nunique())}
res["rates"] = {
    "late60_2009_new": float(np.mean([wmean(a["pan_late60_09"][a["new_late60_mask"] == 1], a["weight"][a["new_late60_mask"] == 1]) for a in imps])),
    "credit_2009_new": float(np.mean([wmean(a["pan_credit_constrained_09"][a["new_credit_mask"] == 1], a["weight"][a["new_credit_mask"] == 1]) for a in imps])),
}
boot_idx = [rng.integers(0, n, n) for _ in range(BOOTSTRAP_REPS)]
PAIRS = [("original", "final"), ("original", "emergency_only"), ("final", "emergency_only")]
res["auc"] = {}
for key, (outcome, mask, label) in TESTS.items():
    point = {c: avg_auc(outcome, mask, c) for c in scores}
    reps = {c: np.array([avg_auc(outcome, mask, c, i) for i in boot_idx]) for c in VERSIONS}
    entry = {"label": label, "auc": {}, "diff": {}}
    for c in scores:
        entry["auc"][c] = {"est": point[c]}
        if c in reps:
            entry["auc"][c].update(ci_low=float(np.percentile(reps[c], 2.5)), ci_high=float(np.percentile(reps[c], 97.5)))
    for a_, b_ in PAIRS:
        d = reps[a_] - reps[b_]
        p_val = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / BOOTSTRAP_REPS)
        entry["diff"][f"{a_}-{b_}"] = {"est": point[a_] - point[b_], "ci_low": float(np.percentile(d, 2.5)),
                                       "ci_high": float(np.percentile(d, 97.5)), "p": float(p_val)}
    res["auc"][key] = entry
    print(f"{label:36s} original {point['original']:.3f} | final {point['final']:.3f} | emergency only {point['emergency_only']:.3f}")

# Which dimensions predict new delinquency? (logistic, all three together, Rubin's rules)
res["logit_dimensions"] = rubin_logit(p[p["new_late60_mask"] == 1], "pan_late60_09", ["S", "D", "E"], "weight",
                                      scale={d: 10 for d in "SDE"})
save_json("prospective", res)
