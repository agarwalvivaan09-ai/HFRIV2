"""07 - Why keep Savings Strength? Evidence that it carries information Emergency Coverage does not.

(a) Overlap: correlation of Savings Strength with Emergency Coverage
(b) Independent effect on current distress (2025, all four dimensions in one logistic model)
(c) Rebuilding: do 2007 savers rebuild their buffer by 2009? (panel)
(d) Saving behaviour x buffer -> new 60+ days late by 2009 (panel)
Output: output/results/savings.json
"""
import numpy as np

from hfri_common import load_json, load_master, rubin_logit, save_json, wmean

LOW_BUFFER = 30   # Emergency score below 30 = less than about one month of essential spending
SAVED_LABELS = {1: "spent more than income", 2: "spent about income", 3: "spent less than income"}

s = load_master("scf2025")
p = load_master("panel")
res = {}

# (a) overlap
res["correlation_S_E"] = {
    "scf2025": float(np.mean([np.corrcoef(g["S"], g["E"])[0, 1] for _, g in s.groupby("implicate")])),
    "panel2007": float(np.mean([np.corrcoef(g["S"], g["E"])[0, 1] for _, g in p.groupby("implicate")])),
}

# (b) independent effect, from the validation step (all four dimensions together)
dims = load_json("validation")["logit_dimensions"]
res["independent_effect_2025"] = {o: dims[o]["S"] for o in dims}

# (c) rebuilding the buffer, 2007 -> 2009
p["log_income"] = np.log(p["income"].clip(lower=1)) * 10      # x10 because rubin_logit divides by 10 below
p["buffer_up"] = (p["pan_E09"] > p["E"]).astype(int)
p["rebuilt"] = ((p["pan_E09"] >= LOW_BUFFER) & (p["E"] < LOW_BUFFER)).astype(int)
sc = {c: 10 for c in ["S", "D", "E", "log_income"]}
res["buffer_up_logit"] = rubin_logit(p, "buffer_up", ["S", "E", "D", "log_income"], "weight", scale=sc)
low = p[p["E"] < LOW_BUFFER]
res["rebuilt_logit_low_buffer"] = rubin_logit(low, "rebuilt", ["S", "D", "log_income"], "weight", scale=sc)
res["rebuilt_rate_low_buffer"] = {
    SAVED_LABELS[c]: float(np.mean([wmean(g["rebuilt"][g["pan_wsaved07"] == c], g["weight"][g["pan_wsaved07"] == c])
                                    for _, g in low.groupby("implicate")]))
    for c in SAVED_LABELS}

# (d) saving behaviour x buffer -> new serious delinquency
pn = p[p["pan_late60_07"] == 0]
table = []
for c, lab in SAVED_LABELS.items():
    for buf, cond in [("little buffer", pn["E"] < LOW_BUFFER), ("some buffer", pn["E"] >= LOW_BUFFER)]:
        g_all = pn[cond & (pn["pan_wsaved07"] == c)]
        rate = float(np.mean([wmean(g["pan_late60_09"], g["weight"]) for _, g in g_all.groupby("implicate")]))
        table.append({"saving": lab, "buffer": buf, "new_late60_2009": rate,
                      "households": int(g_all["hh_id"].nunique())})
res["saving_by_buffer"] = table
res["low_buffer_future_logit"] = rubin_logit(pn[pn["E"] < LOW_BUFFER], "pan_late60_09", ["S", "D"], "weight",
                                             scale={"S": 10, "D": 10})

print(f"Correlation S-E: 2025 {res['correlation_S_E']['scf2025']:.2f}, 2007 {res['correlation_S_E']['panel2007']:.2f}")
for o, v in res["independent_effect_2025"].items():
    print(f"  2025 {o:24s} Savings OR per 10 pts {v['odds_ratio']:.3f} (p={v['p']:.4f})")
for k, v in res["rebuilt_rate_low_buffer"].items():
    print(f"  low buffer, {k:24s} rebuilt by 2009: {v:.1%}")
for r in table:
    print(f"  {r['buffer']:14s} {r['saving']:24s} new 60+ late by 2009: {r['new_late60_2009']:.1%} (n={r['households']})")
save_json("savings", res)
