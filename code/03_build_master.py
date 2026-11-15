"""03 - Build the master dataset: one file holding every household used in the paper.

Rows come from four sources (column `dataset`):
  simulation  - the paper's 1,000 simulated households (Phase I-II)
  scf2025     - Survey of Consumer Finances 2025 (4,360 households x 5 implicates)
  scf2022     - Survey of Consumer Finances 2022 (4,595 households x 5 implicates; replication)
  panel       - 2007-09 Panel SCF (3,857 households x 5 implicates; 2007 scores, 2009 outcomes)
Every later analysis reads only this file: data/hfri_master_dataset.csv
"""
import numpy as np
import pandas as pd

from hfri_common import (DERIVED, MASTER, RAW, W_FINAL, W_ORIGINAL, after_tax, debt_score, emergency_score,
                         hfri, savings_score, stability_score)

ESSENTIAL_SHARE = 0.60     # share of non-housing spending treated as essential (SCF 2025)
INCOME_SHARE_ESSENTIAL = 0.70  # essential spending proxy where spending is not recorded (SCF 2022, panel 2007)
CV_PROXY = {3: 0.10, 2: 0.20, 1: 0.40}   # EXPENSHILO: 3 normal, 2 unusually low, 1 unusually high spending
SR_PROXY = {1: -0.05, 2: 0.00, 3: 0.10}  # WSAVED: 1 spent more than income, 2 about equal, 3 spent less

frames = []

# ------------------------------------------------------------------ simulation
sim = pd.read_csv(DERIVED / "simulation_households.csv")
frames.append(pd.DataFrame({
    "dataset": "simulation", "hh_id": sim["household_id"], "implicate": 1, "weight": 1.0,
    "S": sim["S"], "D": sim["D"], "E": sim["E"], "X": sim["X"],
    "sim_ons_resilient": sim["ONS_resilient"], "sim_income_survival": sim["income_survival"],
    "sim_expense_survival": sim["expense_survival"],
}))

# ------------------------------------------------------------------ SCF 2025
s = pd.read_csv(RAW / "SCFP2025.csv")
y_net = after_tax(s["INCOME"].clip(lower=1))
spend_annual = s["TOTSPENDNH"] + 12 * s["RENT"] + 12 * s["TPAY"]
essential = (s["RENT"] + s["TPAY"] + ESSENTIAL_SHARE * s["TOTSPENDNH"] / 12).clip(lower=50)
frames.append(pd.DataFrame({
    "dataset": "scf2025", "hh_id": s["YY1"], "implicate": s["Y1"] % 10, "weight": s["WGT"],
    "S": savings_score(((y_net - spend_annual) / y_net).clip(-5, 1)),
    "D": debt_score(s["DEBT"] / y_net, 12 * s["TPAY"] / y_net),
    "E": emergency_score(s["LIQ"] / essential),
    "X": stability_score(s["EXPENSHILO"].map(CV_PROXY)),
    "income": s["INCOME"], "age": s["AGE"], "kids": s["KIDS"], "married": (s["MARRIED"] == 1).astype(int),
    "out_late60": s["LATE60"], "out_late": s["LATE"], "out_payday": s["HPAYDAY"],
    "out_bankruptcy": s["BNKRUPLAST5"], "out_credit_constrained": s["TURNFEAR"],
    "bench_ons": (s["LIQ"] >= 0.25 * (y_net / 12) * 3).astype(int),
    "bench_liquid_3m": (s["LIQ"] >= 3 * spend_annual / 12).astype(int),
    "bench_liquid_6m": (s["LIQ"] >= 6 * spend_annual / 12).astype(int),
    "bench_debt_ok": (s["PIR40"] == 0).astype(int),
    "bench_cope_savings": s["EMERGSAV"].astype(int),
    "bench_cope_borrow": s["EMERGBORR"].astype(int),
}))

# ------------------------------------------------------------------ SCF 2022 (no total spending recorded)
t = pd.read_csv(RAW / "SCFP2022.csv")
y22 = after_tax(t["INCOME"].clip(lower=1))
frames.append(pd.DataFrame({
    "dataset": "scf2022", "hh_id": t["YY1"], "implicate": t["Y1"] % 10, "weight": t["WGT"],
    "S": np.nan, "D": debt_score(t["DEBT"] / y22, 12 * t["TPAY"] / y22),
    "E": emergency_score(t["LIQ"] / (INCOME_SHARE_ESSENTIAL * y22 / 12)), "X": np.nan,
    "income": t["INCOME"], "out_late60": t["LATE60"],
}))

# ------------------------------------------------------------------ 2007-09 panel (2007 inputs, 2009 outcomes)
p = pd.read_csv(RAW / "SCFP2009panel.csv")
y07 = after_tax(p["INCOME07"].clip(lower=1))
y09 = after_tax(p["INCOME09"].clip(lower=1))
frames.append(pd.DataFrame({
    "dataset": "panel", "hh_id": p["YY1"], "implicate": p["Y1"] % 10, "weight": p["WGT09"],
    "S": savings_score(p["WSAVED07"].map(SR_PROXY)),
    "D": debt_score(p["DEBT07"] / y07, 12 * p["TPAY07"] / y07),
    "E": emergency_score(p["LIQ07"] / (INCOME_SHARE_ESSENTIAL * y07 / 12)), "X": np.nan,
    "income": p["INCOME07"], "pan_wsaved07": p["WSAVED07"],
    "pan_late60_07": p["LATE6007"], "pan_late60_09": p["LATE6009"],
    "pan_credit_constrained_07": ((p["DENIED07"] + p["FEARED07"]) > 0).astype(int),
    "pan_credit_constrained_09": ((p["DENIED09"] + p["FEARED09"]) > 0).astype(int),
    "pan_E09": emergency_score(p["LIQ09"] / (INCOME_SHARE_ESSENTIAL * y09 / 12)),
    "bench_ons": (p["LIQ07"] >= 0.25 * (y07 / 12) * 3).astype(int),
    "bench_debt_ok": (p["HIPIR07"] == 0).astype(int),
}))

master = pd.concat(frames, ignore_index=True)
master["HFRI_original"] = hfri(master, W_ORIGINAL)
master["HFRI_final"] = hfri(master, W_FINAL)
master.to_csv(MASTER, index=False)

print(f"Saved {MASTER.name}: {len(master):,} rows")
print(master.groupby("dataset").agg(rows=("hh_id", "size"), households=("hh_id", "nunique"),
                                    mean_hfri_original=("HFRI_original", "mean")).round(2).to_string())
