"""08 - Agreement with established resilience benchmarks and distress outcomes.

For each HFRI version: Pearson and Spearman correlation with each benchmark (sign set so that
positive = agrees) and AUC (how well a low score picks out the vulnerable households).
Simulation (paper), SCF 2025 and the 2007-09 panel.
Output: output/results/benchmarks.json
"""
import numpy as np
from scipy.stats import pearsonr, spearmanr

from hfri_common import VERSIONS, distress_auc, hfri, load_master, save_json

m = load_master()
for name, w in VERSIONS.items():
    m[name] = hfri(m, w)
p = m[m["dataset"] == "panel"]

# (label, group, dataset rows, benchmark column, True if 1 = resilient / False if 1 = distress)
BENCHMARKS = [
    ("ONS-style resilient (paper)", "Simulation", m[m["dataset"] == "simulation"], "sim_ons_resilient", True),
    ("Survives 25% income shock", "Simulation", m[m["dataset"] == "simulation"], "sim_income_survival", True),
    ("Survives 25% expense shock", "Simulation", m[m["dataset"] == "simulation"], "sim_expense_survival", True),
    ("ONS-style resilient", "SCF 2025", m[m["dataset"] == "scf2025"], "bench_ons", True),
    ("3 months of spending in liquid savings (Fed)", "SCF 2025", m[m["dataset"] == "scf2025"], "bench_liquid_3m", True),
    ("6 months of spending in liquid savings", "SCF 2025", m[m["dataset"] == "scf2025"], "bench_liquid_6m", True),
    ("Debt payments below 40% of income (Fed)", "SCF 2025", m[m["dataset"] == "scf2025"], "bench_debt_ok", True),
    ("Would cope with job loss from savings", "SCF 2025", m[m["dataset"] == "scf2025"], "bench_cope_savings", True),
    ("Would need to borrow if job lost", "SCF 2025", m[m["dataset"] == "scf2025"], "bench_cope_borrow", False),
    ("60+ days late", "SCF 2025 distress", m[m["dataset"] == "scf2025"], "out_late60", False),
    ("Any late payment", "SCF 2025 distress", m[m["dataset"] == "scf2025"], "out_late", False),
    ("Payday loan", "SCF 2025 distress", m[m["dataset"] == "scf2025"], "out_payday", False),
    ("Bankruptcy (5 years)", "SCF 2025 distress", m[m["dataset"] == "scf2025"], "out_bankruptcy", False),
    ("Denied or feared credit", "SCF 2025 distress", m[m["dataset"] == "scf2025"], "out_credit_constrained", False),
    ("ONS-style resilient (2007)", "Panel 2007", p, "bench_ons", True),
    ("Debt payments below 40% of income (2007)", "Panel 2007", p, "bench_debt_ok", True),
    ("New 60+ days late by 2009", "Panel future", p[p["pan_late60_07"] == 0], "pan_late60_09", False),
    ("New credit denial or fear by 2009", "Panel future", p[p["pan_credit_constrained_07"] == 0],
     "pan_credit_constrained_09", False),
]

rows = []
for label, group, df, col, good in BENCHMARKS:
    for v in VERSIONS:
        r, rho, a = [], [], []
        for _, g in df.groupby("implicate"):
            y = g[col].astype(int).to_numpy()
            sc = g[v].to_numpy()
            sign = 1 if good else -1
            r.append(sign * pearsonr(sc, y)[0])
            rho.append(sign * spearmanr(sc, y)[0])
            a.append(distress_auc(1 - y if good else y, sc, g["weight"]))
        rows.append({"benchmark": label, "group": group, "version": v,
                     "pearson": float(np.mean(r)), "spearman": float(np.mean(rho)), "auc": float(np.mean(a))})

summary = {}
for group in dict.fromkeys(r["group"] for r in rows):
    summary[group] = {v: float(np.mean([r["auc"] for r in rows if r["group"] == group and r["version"] == v]))
                      for v in VERSIONS}
print(f"{'Average AUC':20s}" + "".join(f"{v:>16s}" for v in VERSIONS))
for group, vals in summary.items():
    print(f"{group:20s}" + "".join(f"{vals[v]:16.3f}" for v in VERSIONS))
save_json("benchmarks", {"rows": rows, "summary_auc": summary})
