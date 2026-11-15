"""09 - Interpretation bands calibrated on real outcomes.

A smooth risk curve (logistic regression of 60+ days late on HFRI and HFRI squared, SCF 2025,
averaged across implicates) gives the score where risk falls to 2x, 1x and 0.5x the population
average. Cut-offs are rounded to the nearest 5 and checked on the 2007-09 panel (new distress by 2009).
Output: output/results/bands.json
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from hfri_common import VERSIONS, hfri, load_master, save_json, wmean

LABELS = ["High risk", "Elevated risk", "Moderate risk", "Low risk"]
s = load_master("scf2025")
p = load_master("panel")
for v in ["original", "final"]:
    s[v] = hfri(s, VERSIONS[v])
    p[v] = hfri(p, VERSIONS[v])
grid = np.arange(0, 101)


def risk_curve(df, v):
    curves = []
    for _, g in df.groupby("implicate"):
        X = sm.add_constant(pd.DataFrame({"h": g[v] / 10, "h2": (g[v] / 10) ** 2}))
        fit = sm.GLM(g["out_late60"], X, family=sm.families.Binomial(),
                     var_weights=g["weight"] / g["weight"].mean()).fit()
        Xg = sm.add_constant(pd.DataFrame({"h": grid / 10, "h2": (grid / 10) ** 2}), has_constant="add")
        curves.append(np.asarray(fit.predict(Xg)))
    return np.mean(curves, axis=0)


def band_table(df, v, cuts, outcome, mask=None):
    df = df if mask is None else df[mask]
    edges = [0] + cuts + [101]
    rows = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], LABELS):
        share, rate = [], []
        for _, g in df.groupby("implicate"):
            m = (g[v] >= lo) & (g[v] < hi)
            share.append(g["weight"][m].sum() / g["weight"].sum())
            rate.append(wmean(g[outcome][m], g["weight"][m]) if m.any() else np.nan)
        rows.append({"band": f"{lo}-{min(hi, 100)}", "label": lab, "share": float(np.mean(share)),
                     "rate": float(np.nanmean(rate))})
    return rows


avg = float(np.mean([wmean(g["out_late60"], g["weight"]) for _, g in s.groupby("implicate")]))
res = {"population_late60_rate": avg}
for v in ["original", "final"]:
    curve = risk_curve(s, v)
    raw = {k: int(grid[np.argmax(curve <= f * avg)]) for k, f in [("2x", 2), ("1x", 1), ("0.5x", 0.5)]}
    cuts = [int(5 * round(raw[k] / 5)) for k in ["2x", "1x", "0.5x"]]
    res[v] = {"raw_cut_points": raw, "cuts": cuts, "curve": curve.tolist(),
              "scf2025_late60": band_table(s, v, cuts, "out_late60"),
              "panel_new_late60": band_table(p, v, cuts, "pan_late60_09", p["pan_late60_07"] == 0),
              "panel_new_credit": band_table(p, v, cuts, "pan_credit_constrained_09", p["pan_credit_constrained_07"] == 0)}
    print(f"{v}: cut-offs {cuts} (raw {raw})")
    for a, b in zip(res[v]["scf2025_late60"], res[v]["panel_new_late60"]):
        print(f"   {a['band']:>7s} {a['label']:14s} share {a['share']:5.1%} | 2025 60+ late {a['rate']:5.1%} | "
              f"new by 2009 {b['rate']:5.1%}")
save_json("bands", res)
