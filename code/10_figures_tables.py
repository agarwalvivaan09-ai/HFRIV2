"""10 - Figures (PDF), LaTeX tables (booktabs) and number macros for the paper.

Reads only output/results/*.json, so it runs in seconds and never changes a number.
Outputs:
  output/figures/fig01_scoring_curves.pdf ... fig10_bands.pdf
  output/tables/tab01_data.tex ... tab09_bands.tex
  output/numbers.tex  (\\newcommand macros, e.g. \\AUCfinalLate, so the text always matches the results)
In LaTeX:  \\usepackage{booktabs}  \\input{numbers.tex}  \\input{tables/tab03_auc_now.tex}
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hfri_common import (FIG, OUT, SOURCES, TAB, W_FINAL, dsti_score, dti_score, emergency_score, load_json,
                         savings_score, stability_score)

sim, val, pro = load_json("simulation"), load_json("validation"), load_json("prospective")
wts, sav, ben, bnd = load_json("weights"), load_json("savings"), load_json("benchmarks"), load_json("bands")

# ------------------------------------------------------------------ style
COL = {"original": "#2a78d6", "final": "#eb6834", "emergency_only": "#1baf7a"}
NAME = {"original": "Original (25/25/25/25)", "final": "Final (5/30/55/10)", "emergency_only": "Emergency only"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "lines.linewidth": 2,
    "savefig.bbox": "tight", "pdf.fonttype": 42,
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  figures/{name}.pdf")


def pct(x, d=1):
    return f"{100 * x:.{d}f}\\%"


print("Figures")
# 1. scoring curves (paper, Section 4)
fig, ax = plt.subplots(1, 5, figsize=(11, 2.4))
curves = [("Savings Strength", "Savings rate", np.linspace(-0.2, 0.6, 300), savings_score),
          ("Debt: DTI score", "Debt-to-income", np.linspace(0, 4, 300), dti_score),
          ("Debt: DSTI score", "Debt service-to-income", np.linspace(0, 0.6, 300), dsti_score),
          ("Emergency Coverage", "Months of essential spending", np.linspace(0, 12, 300), emergency_score),
          ("Expense Stability", "Coefficient of variation", np.linspace(0, 0.6, 300), stability_score)]
for a, (t, xl, x, f) in zip(ax, curves):
    a.plot(x, f(x), color=COL["original"])
    a.set(title=t, xlabel=xl, ylim=(0, 105))
ax[0].set_ylabel("Score (0-100)")
fig.tight_layout()
save(fig, "fig01_scoring_curves")

# 2. simulation distribution, original vs final (histogram needs household data)
import pandas as pd  # noqa: E402

from hfri_common import DERIVED  # noqa: E402

h = pd.read_csv(DERIVED / "simulation_households.csv")
fig, ax = plt.subplots(figsize=(6, 3))
bins = np.arange(0, 102.5, 2.5)
for v in ["original", "final"]:
    ax.hist(h[f"HFRI_{v}"], bins=bins, histtype="step", linewidth=2, color=COL[v], label=NAME[v])
ax.set(xlabel="HFRI", ylabel="Households", title="Simulated households (n = 1,000)")
ax.legend(loc="upper left")
save(fig, "fig02_simulation_distribution")

# 3. serious delinquency by decile (SCF 2025)
fig, ax = plt.subplots(figsize=(6, 3))
x = np.arange(1, 11)
for i, v in enumerate(["original", "final"]):
    r = [d["late60"] * 100 for d in val["deciles"][v]]
    ax.bar(x + (i - 0.5) * 0.38, r, width=0.36, color=COL[v], label=NAME[v])
ax.set(xticks=x, xlabel="HFRI decile (1 = lowest score)", ylabel="60+ days late (%)",
       title="Serious delinquency by HFRI decile, SCF 2025")
ax.legend()
save(fig, "fig03_deciles")


def auc_dots(ax, entries, labels, title):
    y = np.arange(len(entries))[::-1]
    for j, v in enumerate(["original", "final", "emergency_only"]):
        off = (j - 1) * 0.22
        for yi, e in zip(y, entries):
            a = e["auc"][v]
            ax.plot([a["ci_low"], a["ci_high"]], [yi + off] * 2, color=COL[v], linewidth=1.5)
            ax.plot(a["est"], yi + off, "o", color=COL[v], markersize=6, label=NAME[v] if yi == y[0] else None)
    ax.set(yticks=y, yticklabels=labels, xlabel="AUC (95% bootstrap CI)", title=title)
    ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
    ax.legend(loc="lower right", fontsize=8)


# 4. AUC now
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ents = list(val["auc"].values())
auc_dots(ax, ents, [e["label"] for e in ents], "Current distress, SCF 2025")
ax.set_xlim(0.6, 0.85)
save(fig, "fig04_auc_now")

# 5. AUC future
fig, ax = plt.subplots(figsize=(6.5, 2.4))
ents = list(pro["auc"].values())
auc_dots(ax, ents, [e["label"] for e in ents], "New distress by 2009 (2007 HFRI), Panel SCF")
ax.set_xlim(0.55, 0.8)
save(fig, "fig05_auc_future")

# 6. emergency dial
d = wts["dial"]["rows"]
e = [r["E_weight"] * 100 for r in d]
fig, ax = plt.subplots(figsize=(6.5, 3.2))
for k, c, lab in [("now", COL["original"], "Now (SCF 2025)"), ("future", COL["emergency_only"], "Future (panel 2007-09)"),
                  ("combined", INK, "Combined")]:
    ax.plot(e, [r[k] for r in d], color=c, label=lab, linewidth=2.5 if k == "combined" else 2)
ax.axvline(W_FINAL[2] * 100, color=COL["final"], linewidth=1.2, linestyle="--")
ax.text(W_FINAL[2] * 100 + 1, ax.get_ylim()[0] + 0.005, "final: 55%", color=INK, fontsize=8)
ax.axvline(25, color=MUTED, linewidth=0.8, linestyle=":")
ax.text(26, ax.get_ylim()[0] + 0.005, "original: 25%", color=MUTED, fontsize=8)
ax.set(xlabel="Emergency Coverage weight (%), other three equal", ylabel="Average AUC",
       title="Current and future distress pull the Emergency weight in opposite directions")
ax.legend(loc="upper left")
save(fig, "fig06_emergency_dial")

# 7. held-out gains
sp = wts["held_out"]["splits"]
fig, ax = plt.subplots(figsize=(6.5, 2.6))
for i, (part, lab) in enumerate([("now", "Now"), ("future", "Future"), ("combined", "Combined")]):
    g = [r[f"chosen_{part}"] - r[f"original_{part}"] for r in sp]
    jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(g))
    ax.plot(g, i + jitter, "o", color=COL["final"], markersize=4, alpha=0.8)
    ax.plot([np.mean(g)] * 2, [i - 0.3, i + 0.3], color=INK, linewidth=2)
ax.axvline(0, color=MUTED, linewidth=1)
ax.set(yticks=[0, 1, 2], yticklabels=["Now", "Future", "Combined"],
       xlabel="AUC gain over original weights on unseen households",
       title=f"Held-out test: {len(sp)} random half-splits (bar = mean)")
save(fig, "fig07_held_out")

# 8. benchmarks
groups = list(ben["summary_auc"])
fig, ax = plt.subplots(figsize=(6.5, 3.2))
x = np.arange(len(groups))
for j, v in enumerate(["original", "final", "emergency_only"]):
    ax.bar(x + (j - 1) * 0.26, [ben["summary_auc"][g][v] for g in groups], width=0.24, color=COL[v], label=NAME[v])
ax.set(xticks=x, xticklabels=groups, ylim=(0.5, 1.05), ylabel="Average AUC", title="Agreement with benchmarks")
ax.legend(loc="upper center", fontsize=8, ncol=3, bbox_to_anchor=(0.5, -0.12))
plt.setp(ax.get_xticklabels(), fontsize=8)
save(fig, "fig08_benchmarks")

# 9. savings strength
fig, ax = plt.subplots(1, 2, figsize=(8, 3))
rb = sav["rebuilt_rate_low_buffer"]
labs = ["Spent more\nthan income", "Spent about\nincome", "Spent less\nthan income"]
ax[0].bar(labs, [v * 100 for v in rb.values()], color=COL["final"], width=0.55)
ax[0].set(ylabel="Rebuilt to 1+ month by 2009 (%)", title="Low-buffer households in 2007")
tb = sav["saving_by_buffer"]
for j, buf in enumerate(["little buffer", "some buffer"]):
    vals = [r["new_late60_2009"] * 100 for r in tb if r["buffer"] == buf]
    ax[1].bar(np.arange(3) + (j - 0.5) * 0.38, vals, width=0.36,
              color=[COL["original"], COL["emergency_only"]][j], label=f"{buf.capitalize()} (2007)")
ax[1].set(xticks=range(3), xticklabels=labs, ylabel="New 60+ days late by 2009 (%)", title="Saving behaviour x buffer")
ax[1].legend(fontsize=8)
fig.tight_layout()
save(fig, "fig09_savings")

# 10. bands (final)
b = bnd["final"]
fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.plot(np.arange(101), np.array(b["curve"]) * 100, color=COL["final"], label="Fitted risk, final HFRI")
avg = bnd["population_late60_rate"] * 100
for f, lab in [(2, "2x average"), (1, "average"), (0.5, "half average")]:
    ax.axhline(avg * f, color=MUTED, linewidth=0.8, linestyle=":")
    ax.text(101, avg * f, lab, fontsize=7, color=MUTED, va="center")
for c in b["cuts"]:
    ax.axvline(c, color=INK, linewidth=1, linestyle="--")
ax.set(xlabel="HFRI (final weights)", ylabel="Probability 60+ days late (%)", xlim=(0, 100),
       title=f"Calibrated bands: cut-offs at {', '.join(map(str, b['cuts']))}")
save(fig, "fig10_bands")

# ------------------------------------------------------------------ tables
print("Tables")


def table(name, caption, label, header, rows, align):
    lines = ["\\begin{table}[htbp]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(str(c) for c in r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n")
    print(f"  tables/{name}.tex")


def ci(a):
    return f"{a['est']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]"


def pv(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


table("tab01_data", "Data sources (Board of Governors of the Federal Reserve System).", "tab:data",
      ["Dataset", "File", "Households", "SHA-256 (first 16)"],
      [["SCF 2025", SOURCES["scf2025"]["file"], f"{val['n_households']:,}", SOURCES["scf2025"]["sha256"][:16]],
       ["SCF 2022", SOURCES["scf2022"]["file"], f"{val['replication_2022']['n_households']:,}", SOURCES["scf2022"]["sha256"][:16]],
       ["Panel SCF 2007--09", SOURCES["panel"]["file"], f"{pro['n_households']:,}", SOURCES["panel"]["sha256"][:16]],
       ["Simulation", "simulation\\_households.csv", f"{sim['n_households']:,}", "seed 42"]], "llrl")

table("tab02_distribution", "HFRI distribution, SCF 2025 (survey-weighted).", "tab:dist",
      ["Version", "Mean", "Median", "P10", "P90"],
      [[NAME[v], *(f"{val['distribution'][v][k]:.1f}" for k in ["mean", "median", "p10", "p90"])]
       for v in ["original", "final"]], "lrrrr")

table("tab03_auc_now", "AUC for current distress, SCF 2025 (95\\% bootstrap CI).", "tab:aucnow",
      ["Outcome", "Rate", "Original", "Final", "Emergency only", "Final $-$ original", "$p$"],
      [[e["label"], pct(val["prevalence"][o]), ci(e["auc"]["original"]), ci(e["auc"]["final"]),
        ci(e["auc"]["emergency_only"]), f"{e['diff']['final-original']['est']:+.3f}", pv(e["diff"]["final-original"]["p"])]
       for o, e in val["auc"].items()], "lrcccrr")

table("tab04_auc_future", "AUC for new distress by 2009, 2007 HFRI (Panel SCF, 95\\% bootstrap CI).", "tab:aucfut",
      ["Outcome", "Rate", "Original", "Final", "Emergency only", "Final $-$ E only", "$p$"],
      [[e["label"], pct(pro["rates"]["late60_2009_new" if k == "new_late60" else "credit_2009_new"]),
        ci(e["auc"]["original"]), ci(e["auc"]["final"]), ci(e["auc"]["emergency_only"]),
        f"{e['diff']['final-emergency_only']['est']:+.3f}", pv(e["diff"]["final-emergency_only"]["p"])]
       for k, e in pro["auc"].items()], "lrcccrr")

rows = []
for o, e in val["logit_dimensions"].items():
    rows.append([val["auc"][o]["label"], *(f"{e[d]['odds_ratio']:.3f} ({pv(e[d]['p'])})" for d in "SDEX")])
fd = pro["logit_dimensions"]
rows.append(["New 60+ late by 2009 (panel)", *(f"{fd[d]['odds_ratio']:.3f} ({pv(fd[d]['p'])})" for d in "SDE"), "--"])
table("tab05_dimensions", "Odds ratio per 10 points of each dimension, all four in one logistic model "
      "(Rubin's rules; $p$ in brackets).", "tab:dims", ["Outcome", "Savings", "Debt", "Emergency", "Stability"],
      rows, "lcccc")

table("tab06_grid", "Top 10 weight combinations (5\\% grid), ranked by combined AUC.", "tab:grid",
      ["Savings", "Debt", "Emergency", "Stability", "Now", "Future", "Combined"],
      [[*(f"{r[c] * 100:.0f}\\%" for c in "SDEX"), f"{r['now']:.4f}", f"{r['future']:.4f}", f"{r['combined']:.4f}"]
       for r in wts["grid5"]["top10"]], "rrrrrrr")

s_ = wts["held_out"]["summary"]
table("tab07_held_out", f"Held-out test ({len(sp)} half-splits): weights chosen on one half, AUC on the other.",
      "tab:heldout", ["Yardstick", "Chosen", "Original", "Final (fixed)", "Gain", "Splits better"],
      [[k.capitalize(), f"{s_[k]['chosen']:.4f}", f"{s_[k]['original']:.4f}", f"{s_[k]['final']:.4f}",
        f"{s_[k]['gain_mean']:+.4f}", f"{s_[k]['splits_better']}/{len(sp)}"] for k in ["now", "future", "combined"]],
      "lrrrrr")

table("tab08_benchmarks", "Agreement with benchmarks: AUC (Spearman $\\rho$).", "tab:bench",
      ["Group", "Benchmark", "Original", "Final", "Emergency only"],
      [[r0["group"], r0["benchmark"], *(f"{r['auc']:.3f} ({r['spearman']:.2f})" for r in
                                        [x for x in ben["rows"] if x["benchmark"] == r0["benchmark"] and x["group"] == r0["group"]])]
       for r0 in ben["rows"] if r0["version"] == "original"], "llccc")

table("tab09_bands", "Interpretation bands, final weights: share of households and serious delinquency.", "tab:bands",
      ["Band", "Label", "Share (2025)", "60+ late (2025)", "New 60+ late by 2009"],
      [[a["band"].replace("-", "--"), a["label"], pct(a["share"]), pct(a["rate"]), pct(c["rate"])]
       for a, c in zip(bnd["final"]["scf2025_late60"], bnd["final"]["panel_new_late60"])], "llrrr")

# ------------------------------------------------------------------ number macros
m = {
    "MeanSimOriginal": f"{sim['original']['distribution']['mean']:.2f}",
    "MeanSimFinal": f"{sim['final']['distribution']['mean']:.2f}",
    "ONSrOriginal": f"{sim['original']['ons_pearson']:.3f}",
    "ONSrFinal": f"{sim['final']['ons_pearson']:.3f}",
    "NHouseholdsTwentyFive": f"{val['n_households']:,}",
    "NHouseholdsPanel": f"{pro['n_households']:,}",
    "RateLateSixty": pct(val["prevalence"]["out_late60"]),
    "AUCoriginalLate": f"{val['auc']['out_late60']['auc']['original']['est']:.3f}",
    "AUCfinalLate": f"{val['auc']['out_late60']['auc']['final']['est']:.3f}",
    "AUCemergencyLate": f"{val['auc']['out_late60']['auc']['emergency_only']['est']:.3f}",
    "AUCgainLate": f"{val['auc']['out_late60']['diff']['final-original']['est']:+.3f}",
    "AUCfutOriginal": f"{pro['auc']['new_late60']['auc']['original']['est']:.3f}",
    "AUCfutFinal": f"{pro['auc']['new_late60']['auc']['final']['est']:.3f}",
    "AUCfutEmergency": f"{pro['auc']['new_late60']['auc']['emergency_only']['est']:.3f}",
    "ORhfriOriginal": f"{val['logit_controls']['original']['hfri_original']['odds_ratio']:.3f}",
    "ORhfriFinal": f"{val['logit_controls']['final']['hfri_final']['odds_ratio']:.3f}",
    "BestEnow": f"{wts['dial']['best_E_weight']['now'] * 100:.0f}\\%",
    "BestEfuture": f"{wts['dial']['best_E_weight']['future'] * 100:.0f}\\%",
    "BestEcombined": f"{wts['dial']['best_E_weight']['combined'] * 100:.0f}\\%",
    "BootEmedian": f"{wts['dial_bootstrap']['median'] * 100:.0f}\\%",
    "BootElow": f"{wts['dial_bootstrap']['ci_low'] * 100:.0f}\\%",
    "BootEhigh": f"{wts['dial_bootstrap']['ci_high'] * 100:.0f}\\%",
    "CombinedFinal": f"{wts['reference']['final']['combined']:.4f}",
    "CombinedOriginal": f"{wts['reference']['original']['combined']:.4f}",
    "HeldOutGain": f"{s_['combined']['gain_mean']:+.4f}",
    "HeldOutBetter": f"{s_['combined']['splits_better']}",
    "CorrSE": f"{sav['correlation_S_E']['scf2025']:.2f}",
    "CutHigh": str(bnd["final"]["cuts"][0]), "CutMid": str(bnd["final"]["cuts"][1]), "CutLow": str(bnd["final"]["cuts"][2]),
}
(OUT / "numbers.tex").write_text("% generated by 10_figures_tables.py - do not edit by hand\n" +
                                 "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()))
print(f"  numbers.tex ({len(m)} macros)")
