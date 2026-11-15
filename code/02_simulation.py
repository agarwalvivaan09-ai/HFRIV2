"""02 - Phase I and II: the paper's 1,000-household, 12-month simulation, ONS-style benchmark and shock tests.

This reproduces the original Colab notebook exactly (same parameters, same order of random draws, seed 42),
and additionally scores every household with the final v2 weights.
Output: data/derived/simulation_households.csv and output/results/simulation.json
"""
import numpy as np
import pandas as pd

from hfri_common import (DERIVED, SEED, W_FINAL, W_ORIGINAL, debt_score, dsti_score, dti_score,
                         emergency_score, hfri, savings_score, save_json, stability_score)

# =====================================================================
# STEP 1 - household characteristics
# =====================================================================
N = 1000
rng = np.random.default_rng(SEED)
hh = pd.DataFrame({"household_id": np.arange(1, N + 1)})

income = np.clip(rng.lognormal(np.log(24000), 0.70, N), 6000, 150000)
hh["monthly_income"] = income
log_income = np.log(hh["monthly_income"])
z_income = (log_income - log_income.mean()) / log_income.std()

hh["expenditure_income_ratio"] = np.clip(0.80 - 0.05 * z_income + rng.normal(0, 0.08, N), 0.55, 1.25)
hh["monthly_expenditure"] = hh["monthly_income"] * hh["expenditure_income_ratio"]
hh["expenditure_cv"] = np.clip(rng.lognormal(np.log(0.15), 0.50, N), 0.05, 0.45)
hh["essential_expenditure_share"] = np.clip(0.65 - 0.06 * z_income + rng.normal(0, 0.05, N), 0.45, 0.90)
hh["monthly_essential_expenditure"] = hh["monthly_expenditure"] * hh["essential_expenditure_share"]

zero_savings = rng.random(N) < 0.25
savings_months = np.clip(rng.lognormal(np.log(1.5), 1.0, N), 0.0, 24.0)
savings_months[zero_savings] = rng.uniform(0, 0.25, zero_savings.sum())
hh["starting_liquid_savings"] = savings_months * hh["monthly_essential_expenditure"]

has_debt = rng.random(N) < 0.35
initial_dti = np.clip(rng.lognormal(np.log(0.90), 0.90, N), 0.05, 6.0)
initial_dti[~has_debt] = 0
hh["starting_debt"] = initial_dti * hh["monthly_income"] * 12

rates = np.clip(rng.lognormal(np.log(0.11), 0.30, N), 0.08, 0.30)
rates[~has_debt] = 0
maturity = np.round(np.clip(rng.lognormal(np.log(60), 0.80, N), 12, 240)).astype(int)
maturity[~has_debt] = 0

# =====================================================================
# STEP 2 - 12-month simulation
# =====================================================================
savings = hh["starting_liquid_savings"].to_numpy().copy()
tranches = [[] for _ in range(N)]
for i in range(N):
    if hh["starting_debt"].iloc[i] > 0:
        tranches[i].append({"principal": float(hh["starting_debt"].iloc[i]),
                            "annual_rate": float(rates[i]), "remaining_months": int(maturity[i])})


def payment_parts(t):
    P, r, n = t["principal"], t["annual_rate"] / 12, t["remaining_months"]
    payment = P * r * (1 + r) ** n / ((1 + r) ** n - 1) if r > 0 else P / n
    interest = P * r
    return interest, min(P, max(0, payment - interest))


records = []
base_income = hh["monthly_income"].to_numpy()
base_spend = hh["monthly_expenditure"].to_numpy()
cv = hh["expenditure_cv"].to_numpy()
ess_share = hh["essential_expenditure_share"].to_numpy()
for month in range(1, 13):
    m_income = base_income * np.exp(rng.normal(0, 0.04, N))
    m_spend = base_spend * np.exp(rng.normal(0, np.sqrt(np.log(1 + cv ** 2)), N))
    m_essential = m_spend * ess_share

    debt_service = np.zeros(N)
    for i in range(N):
        for t in tranches[i]:
            if t["principal"] > 0 and t["remaining_months"] > 0:
                interest, principal = payment_parts(t)
                debt_service[i] += interest + principal

    balance = m_income - m_spend - debt_service
    savings[balance > 0] += balance[balance > 0]
    deficit = np.maximum(0, -balance)
    used = np.minimum(savings, deficit)
    savings -= used
    new_borrowing = np.minimum(deficit - used, base_income * 6)

    for i in range(N):
        for t in tranches[i]:
            if t["principal"] > 0 and t["remaining_months"] > 0:
                _, principal = payment_parts(t)
                t["principal"] = max(0, t["principal"] - principal)
                t["remaining_months"] -= 1
    for i in range(N):
        if new_borrowing[i] > 0:
            tranches[i].append({"principal": float(new_borrowing[i]),
                                "annual_rate": float(rng.uniform(0.18, 0.24)),
                                "remaining_months": int(rng.integers(6, 13))})
    ending_debt = np.array([sum(t["principal"] for t in tranches[i] if t["principal"] > 0) for i in range(N)])
    records.append(pd.DataFrame({"household_id": np.arange(1, N + 1), "month": month, "income": m_income,
                                 "expenditure": m_spend, "essential": m_essential, "debt_service": debt_service,
                                 "savings": savings.copy(), "debt": ending_debt}))
monthly = pd.concat(records, ignore_index=True)

# =====================================================================
# STEP 3 - indicators, dimension scores and HFRI
# =====================================================================
g = monthly.sort_values(["household_id", "month"]).groupby("household_id")
ind = pd.DataFrame({
    "annual_income": g["income"].sum(), "annual_expenditure": g["expenditure"].sum(),
    "monthly_income": g["income"].mean(), "monthly_expenditure": g["expenditure"].mean(),
    "expenditure_sd": g["expenditure"].std(), "monthly_essential": g["essential"].mean(),
    "annual_debt_service": g["debt_service"].sum(),
    "final_savings": g["savings"].last(), "final_debt": g["debt"].last(),
})
ind["SR"] = (ind["annual_income"] - ind["annual_expenditure"]) / ind["annual_income"]
ind["DTI"] = ind["final_debt"] / ind["annual_income"]
ind["DSTI"] = ind["annual_debt_service"] / ind["annual_income"]
ind["EC"] = ind["final_savings"] / ind["monthly_essential"]
ind["CV"] = ind["expenditure_sd"] / ind["monthly_expenditure"]

ind["S"] = savings_score(ind["SR"])
ind["D"] = np.clip(debt_score(ind["DTI"], ind["DSTI"]), 0, 100)
ind["E"] = emergency_score(ind["EC"])
ind["X"] = stability_score(ind["CV"])
ind["HFRI_original"] = hfri(ind, W_ORIGINAL)
ind["HFRI_final"] = hfri(ind, W_FINAL)

# =====================================================================
# PHASE II - ONS-style benchmark and shock tests
# =====================================================================
ind["ONS_resilient"] = (ind["final_savings"] >= 0.25 * ind["monthly_income"] * 3).astype(int)
ind["post_income_shock_savings"] = ind["final_savings"] - 0.25 * ind["monthly_income"] * 3
ind["post_expense_shock_savings"] = ind["final_savings"] - 0.25 * ind["monthly_expenditure"] * 3
ind["income_survival"] = (ind["post_income_shock_savings"] >= 0).astype(int)
ind["expense_survival"] = (ind["post_expense_shock_savings"] >= 0).astype(int)
ind["income_borrowing_need"] = (-ind["post_income_shock_savings"]).clip(lower=0)
ind["expense_borrowing_need"] = (-ind["post_expense_shock_savings"]).clip(lower=0)


def stressed(post_savings, w):
    """HFRI after a shock: only liquid savings (Emergency Coverage) change (paper, Section 7.5)."""
    tmp = ind[["S", "D", "X"]].copy()
    tmp["E"] = emergency_score(post_savings.clip(lower=0) / ind["monthly_essential"])
    return hfri(tmp, w)


results = {"n_households": N, "n_household_months": len(monthly)}
for name, w in [("original", W_ORIGINAL), ("final", W_FINAL)]:
    h = ind[f"HFRI_{name}"]
    q = pd.qcut(h, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    bands = pd.cut(h, [0, 60, 65, 70, 75, 101], right=False, labels=["<60", "60-65", "65-70", "70-75", "75-100"])
    results[name] = {
        "distribution": {"mean": h.mean(), "median": h.median(), "sd": h.std(), "min": h.min(), "max": h.max(),
                         "p10": h.quantile(.10), "p25": h.quantile(.25), "p75": h.quantile(.75), "p90": h.quantile(.90)},
        "ons_pearson": float(np.corrcoef(h, ind["ONS_resilient"])[0, 1]),
        "ons_spearman": float(pd.Series(h).corr(ind["ONS_resilient"], method="spearman")),
        "quartiles": ind.groupby(q, observed=True).agg(
            mean_hfri=(f"HFRI_{name}", "mean"), ons=("ONS_resilient", "mean"),
            income_survival=("income_survival", "mean"), expense_survival=("expense_survival", "mean"),
            income_borrowing=("income_borrowing_need", "mean"), expense_borrowing=("expense_borrowing_need", "mean")
        ).reset_index(names="quartile").to_dict("records"),
        "bands": ind.groupby(bands, observed=False).agg(
            households=("income_survival", "size"), income_survival=("income_survival", "mean"),
            expense_survival=("expense_survival", "mean")).reset_index(names="band").to_dict("records"),
        "stress_change_income": float((stressed(ind["post_income_shock_savings"], w) - h).mean()),
        "stress_change_expense": float((stressed(ind["post_expense_shock_savings"], w) - h).mean()),
    }
results["component_means"] = {c: ind[c].mean() for c in ["S", "D", "E", "X"]}
results["component_correlations"] = ind[["S", "D", "E", "X"]].corr().round(4).to_dict()
results["ons_rate"] = ind["ONS_resilient"].mean()
results["rank_correlation_original_final"] = float(ind["HFRI_original"].corr(ind["HFRI_final"], method="spearman"))
results["quartile_changes"] = int((pd.qcut(ind["HFRI_original"], 4, labels=False) != pd.qcut(ind["HFRI_final"], 4, labels=False)).sum())

ind.reset_index().to_csv(DERIVED / "simulation_households.csv", index=False)
save_json("simulation", results)
print(f"Simulation: mean HFRI original {results['original']['distribution']['mean']:.3f} "
      f"(paper: 75.839), ONS Pearson {results['original']['ons_pearson']:.4f} (paper: 0.8729)")
