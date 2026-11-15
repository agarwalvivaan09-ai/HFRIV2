# HFRI v2: Household Financial Resilience Index, validated on real data

Reproducible code and data for version 2 of the Household Financial Resilience Index (HFRI).
Version 1 built the index from a 1,000-household simulation with equal weights (25/25/25/25).
Version 2 tests it on Federal Reserve survey data and revises the weights to:

| Savings Strength | Debt Sustainability | Emergency Coverage | Expense Stability |
|---:|---:|---:|---:|
| 5% | 30% | 55% | 10% |

## Run it

```bash
python3 -m pip install -r requirements.txt
python3 run_all.py
```

The run takes about 10 minutes; step 06 is the slow one. Everything uses seed 42, so repeated runs give identical numbers.
To run a single step, go into `code/` first, for example `cd code && python3 04_validation.py`.

## Data

| Dataset | Source | Used for |
|---|---|---|
| Survey of Consumer Finances 2025 | https://www.federalreserve.gov/econres/files/scfp2025excel.zip | Current distress (main validation) |
| Survey of Consumer Finances 2022 | https://www.federalreserve.gov/econres/files/scfp2022excel.zip | Replication |
| 2007-09 Panel SCF | https://www.federalreserve.gov/econres/files/rscfp2009panelexcel.zip | 2007 score, new distress by 2009 |
| Simulation (v1) | `code/02_simulation.py`, seed 42 | Phases I-II, ONS-style benchmark |

The raw files are not stored in this repository. `code/01_download_data.py` downloads them into `data/raw/`
and checks each file's SHA-256 fingerprint against the version used in the paper (fingerprints are listed in `code/hfri_common.py`).

`data/hfri_master_dataset.csv` is the single master file: every household from all four sources
(65,060 rows; column `dataset` = `simulation`, `scf2025`, `scf2022` or `panel`), with dimension scores,
both HFRI versions, outcomes and benchmarks. Every analysis reads only this file.

## Pipeline

| Script | What it does | Output |
|---|---|---|
| `hfri_common.py` | Formulas, weights, seed, data sources, shared helpers | - |
| `01_download_data.py` | Download, unzip, SHA-256 check | `data/raw/` |
| `02_simulation.py` | v1 simulation and shock tests, exact replica (mean 75.839, ONS r 0.8729) | `simulation.json` |
| `03_build_master.py` | Build the master dataset | `data/hfri_master_dataset.csv` |
| `04_validation.py` | AUC, deciles and logistic regressions for current distress (SCF 2025); 2022 replication | `validation.json` |
| `05_prospective.py` | 2007 HFRI and new distress by 2009 | `prospective.json` |
| `06_weights.py` | Emergency dial, bootstrap, full grids, held-out test (10 half-splits) | `weights.json` |
| `07_savings.py` | Evidence for keeping Savings Strength | `savings.json` |
| `08_benchmarks.py` | Agreement with ONS-style, Fed and self-reported benchmarks | `benchmarks.json` |
| `09_bands.py` | Risk bands calibrated on real outcomes | `bands.json` |
| `10_figures_tables.py` | Figures (PDF and PNG), LaTeX tables, number macros | `output/` |

## Outputs for the paper

- `output/figures/`: 10 figures (PDF for LaTeX, PNG for preview)
- `output/tables/`: 9 booktabs tables; add `\usepackage{booktabs}` and `\input{tables/tab03_auc_now.tex}`
- `output/numbers.tex`: macros such as `\AUCfinalLate`, so the text always matches the results
- `output/results/*.json`: every number in machine-readable form

## Methods

- Survey weights (`WGT`, `WGT09`) throughout. Results are averaged across the 5 SCF implicates; regressions are combined with Rubin's rules.
- AUC uses the weighted Mann-Whitney form. Confidence intervals come from 400 household bootstrap resamples, paired across versions.
- The weights are chosen to maximise the average of current-distress AUC (SCF 2025) and future-distress AUC (panel 2007-09).
  The held-out test checks that this choice is not overfitting: weights are chosen on one half of the households and graded on the other.
