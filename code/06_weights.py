"""06 - Choosing the weights (the slow step: about 10 minutes on a laptop).

Three yardsticks, each an average AUC:
  now      - 5 current distress outcomes, SCF 2025
  future   - new 60+ days late and new credit denial/fear by 2009, 2007-09 panel
  sim      - the 3 simulation flags (ONS-style resilient, survives income shock, survives expense shock)
  combined = (now + future) / 2   <- the quantity that is maximised

A. Emergency weight dial, 0-100% in 1% steps (other three share the rest equally)
B. Bootstrap of the best combined Emergency weight
C. All four weights free, 5% steps; top 10
D. 1% refinement around the 5% optimum
E. Held-out test: choose weights on a random half of households, grade them on the other half (30 splits)
Output: output/results/weights.json
"""
import itertools
import time

import numpy as np

from hfri_common import (SEED, SPLITS, W_FINAL, W_ORIGINAL, auc, implicates, load_master, save_json)

BOOT_WEIGHTS = 200            # bootstrap re-runs in part B (set to 20 for a quick test)
rng = np.random.default_rng(SEED)
t0 = time.time()

NOW_OUT = ["out_late60", "out_late", "out_payday", "out_bankruptcy", "out_credit_constrained"]
now_a = implicates(load_master("scf2025"), ["S", "D", "E", "X", "weight"] + NOW_OUT)
p = load_master("panel")
fut_a = implicates(p, ["S", "D", "E", "weight", "pan_late60_07", "pan_late60_09",
                       "pan_credit_constrained_07", "pan_credit_constrained_09"])
sim = load_master("simulation")
n_now, n_fut = len(now_a[0]["weight"]), len(fut_a[0]["weight"])


def now_auc(w, idx=None):
    vals = []
    for a in now_a:
        k = slice(None) if idx is None else idx
        sc = w[0] * a["S"][k] + w[1] * a["D"][k] + w[2] * a["E"][k] + w[3] * a["X"][k]
        vals += [auc(a[o][k], -sc, a["weight"][k]) for o in NOW_OUT]
    return float(np.mean(vals))


def fut_auc(w, idx=None):
    t = w[0] + w[1] + w[2]
    if t == 0:
        return np.nan
    ws = (w[0] / t, w[1] / t, w[2] / t)          # Expense Stability not observed in 2007
    vals = []
    for a in fut_a:
        k = np.arange(n_fut) if idx is None else idx
        sc = ws[0] * a["S"][k] + ws[1] * a["D"][k] + ws[2] * a["E"][k]
        for before, after in [("pan_late60_07", "pan_late60_09"),
                              ("pan_credit_constrained_07", "pan_credit_constrained_09")]:
            m = a[before][k] == 0
            vals.append(auc(a[after][k][m], -sc[m], a["weight"][k][m]))
    return float(np.mean(vals))


def sim_auc(w):
    sc = w[0] * sim["S"] + w[1] * sim["D"] + w[2] * sim["E"] + w[3] * sim["X"]
    return float(np.mean([auc(1 - sim[o], -sc) for o in
                          ["sim_ons_resilient", "sim_income_survival", "sim_expense_survival"]]))


def evaluate(w):
    a, b = now_auc(w), fut_auc(w)
    return {"S": w[0], "D": w[1], "E": w[2], "X": w[3], "now": a, "future": b, "combined": (a + b) / 2}


def dial(e):
    return ((1 - e) / 3, (1 - e) / 3, e, (1 - e) / 3)


def grid(step_pct):
    """All weight combinations in steps of step_pct %, summing to 100% (Stability takes the remainder)."""
    k = 100 // step_pct
    out = []
    for c in itertools.product(range(k + 1), repeat=3):
        if sum(c) <= k and c[0] + c[1] + c[2] > 0:
            out.append(tuple(round(x * step_pct / 100, 2) for x in c) + (round(1 - sum(c) * step_pct / 100, 2),))
    return out


res = {"reference": {"original": {**evaluate(W_ORIGINAL), "sim": sim_auc(W_ORIGINAL)},
                     "final": {**evaluate(W_FINAL), "sim": sim_auc(W_FINAL)}}}

# ---------------- A. Emergency dial
print("A. Emergency weight dial (1% steps)")
dial_rows = []
for e in np.round(np.arange(0, 1.0001, 0.01), 2):
    r = evaluate(dial(e))
    r.update(E_weight=float(e), sim=sim_auc(dial(e)))
    dial_rows.append(r)
best = {k: max(dial_rows, key=lambda r: r[k])["E_weight"] for k in ["now", "future", "combined", "sim"]}
res["dial"] = {"rows": dial_rows, "best_E_weight": best}
print(f"   best Emergency weight - now {best['now']:.0%}, future {best['future']:.0%}, "
      f"combined {best['combined']:.0%}, simulation {best['sim']:.0%}   [{time.time() - t0:.0f}s]")

# ---------------- B. Bootstrap of the combined optimum
print(f"B. Bootstrap of the best combined Emergency weight ({BOOT_WEIGHTS} re-runs)")
grid_b = np.round(np.arange(0.10, 0.81, 0.01), 2)
boot_best = []
for b in range(BOOT_WEIGHTS):
    i1, i2 = rng.integers(0, n_now, n_now), rng.integers(0, n_fut, n_fut)
    vals = [(now_auc(dial(e), i1) + fut_auc(dial(e), i2)) / 2 for e in grid_b]
    boot_best.append(float(grid_b[int(np.argmax(vals))]))
    if (b + 1) % 50 == 0:
        print(f"   {b + 1} of {BOOT_WEIGHTS}   [{time.time() - t0:.0f}s]")
boot_best = np.array(boot_best)
res["dial_bootstrap"] = {"median": float(np.median(boot_best)), "ci_low": float(np.percentile(boot_best, 2.5)),
                         "ci_high": float(np.percentile(boot_best, 97.5)), "draws": boot_best.tolist()}
print(f"   median {np.median(boot_best):.0%}, 95% range {np.percentile(boot_best, 2.5):.0%}-{np.percentile(boot_best, 97.5):.0%}")

# ---------------- C. Full 5% grid
print("C. All four weights free, 5% steps")
G5 = grid(5)
full = sorted((evaluate(w) for w in G5), key=lambda r: -r["combined"])
res["grid5"] = {"n_combinations": len(G5), "top10": full[:10],
                "best_now": max(full, key=lambda r: r["now"]), "best_future": max(full, key=lambda r: r["future"])}
b5 = full[0]
print(f"   {len(G5)} combinations; best S{b5['S']:.0%} D{b5['D']:.0%} E{b5['E']:.0%} X{b5['X']:.0%} "
      f"combined {b5['combined']:.4f}   [{time.time() - t0:.0f}s]")

# ---------------- D. 1% refinement around the 5% optimum
print("D. 1% refinement (S 0-10, D 20-40, E 45-70, X 0-20)")
fine = []
for s_, d_, e_ in itertools.product(range(0, 11), range(20, 41), range(45, 71)):
    x_ = 100 - s_ - d_ - e_
    if 0 <= x_ <= 20:
        fine.append(evaluate((s_ / 100, d_ / 100, e_ / 100, x_ / 100)))
fine.sort(key=lambda r: -r["combined"])
res["grid1"] = {"n_combinations": len(fine), "top10": fine[:10],
                "final_rank": 1 + sum(r["combined"] > res["reference"]["final"]["combined"] for r in fine)}
b1 = fine[0]
print(f"   {len(fine)} combinations; best S{b1['S']:.0%} D{b1['D']:.0%} E{b1['E']:.0%} X{b1['X']:.0%} "
      f"combined {b1['combined']:.5f}; final 5/30/55/10 combined {res['reference']['final']['combined']:.5f}   [{time.time() - t0:.0f}s]")

# ---------------- E. Held-out test of the whole method
print(f"E. Held-out test ({SPLITS} random half-splits)")
rows = []
for k in range(SPLITS):
    pn, pf = rng.permutation(n_now), rng.permutation(n_fut)
    trn, ten = np.sort(pn[: n_now // 2]), np.sort(pn[n_now // 2:])
    trf, tef = np.sort(pf[: n_fut // 2]), np.sort(pf[n_fut // 2:])
    train = [(now_auc(w, trn) + fut_auc(w, trf)) / 2 for w in G5]
    w_best = G5[int(np.argmax(train))]
    row = {"split": k + 1, "S": w_best[0], "D": w_best[1], "E": w_best[2], "X": w_best[3]}
    for name, w in [("chosen", w_best), ("original", W_ORIGINAL), ("final", W_FINAL)]:
        a, b = now_auc(w, ten), fut_auc(w, tef)
        row.update({f"{name}_now": a, f"{name}_future": b, f"{name}_combined": (a + b) / 2})
    rows.append(row)
    print(f"   split {k + 1:2d}: chose S{w_best[0]:.0%} D{w_best[1]:.0%} E{w_best[2]:.0%} X{w_best[3]:.0%} | "
          f"unseen gain {row['chosen_combined'] - row['original_combined']:+.4f}   [{time.time() - t0:.0f}s]")
summary = {"mean_weights": {c: float(np.mean([r[c] for r in rows])) for c in "SDEX"},
           "min_weights": {c: float(np.min([r[c] for r in rows])) for c in "SDEX"},
           "max_weights": {c: float(np.max([r[c] for r in rows])) for c in "SDEX"}}
for part in ["now", "future", "combined"]:
    g = np.array([r[f"chosen_{part}"] - r[f"original_{part}"] for r in rows])
    summary[part] = {"chosen": float(np.mean([r[f"chosen_{part}"] for r in rows])),
                     "original": float(np.mean([r[f"original_{part}"] for r in rows])),
                     "final": float(np.mean([r[f"final_{part}"] for r in rows])),
                     "gain_mean": float(g.mean()), "gain_min": float(g.min()), "gain_max": float(g.max()),
                     "splits_better": int((g > 0).sum())}
    print(f"   {part:8s} unseen: chosen {summary[part]['chosen']:.4f} vs original {summary[part]['original']:.4f} "
          f"(gain {g.mean():+.4f}, better in {(g > 0).sum()}/{SPLITS})")
res["held_out"] = {"splits": rows, "summary": summary}
res["runtime_seconds"] = time.time() - t0
save_json("weights", res)
