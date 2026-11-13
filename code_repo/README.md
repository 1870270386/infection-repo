# Predicting the need for broad-spectrum empiric antibiotics at hospital admission

Analysis code for a retrospective cohort study using MIMIC-IV v3.0 (2008–2022).

**Research question.** Among adults started on systemic antibiotics within 24 h of admission, can data routinely available in the first 6 h identify those who will go on to receive broad-spectrum coverage, defined operationally as ≥3 distinct antibiotic classes within 24 h?

**Cohort.** 184,645 hospital admissions with antibiotics started within 24 h of admission; 32,431 (17.6%) received ≥3 antibiotic classes. Development and hold-out sets were split at the 70th percentile of the de-identified admission year (129,268 / 55,377 admissions). Variable selection was performed in the development set only.

**Models.** Logistic regression and gradient boosting, compared head-to-head with NEWS2 and qSOFA on the subset where those scores are computable. Performance: AUROC, AUPRC, Brier score, calibration slope/intercept, decision-curve analysis, bootstrap confidence intervals, subgroup and data-availability strata, four alternative outcome definitions, and an ICU-anchored cohort.

## What is in this repository

```
code/python/            Main cohort pipeline (Python 3.9)
code/python/icu_anchor/ ICU-anchored sensitivity pipeline
code/R/                 Table generation (R 4.5)
code/sql/               Antibiotic class definitions (mimic-code antibiotic.sql and local variants)
docs/                   Data access notes
requirements.txt        Pinned Python packages
sessionInfo.txt         R session used for the tables
CITATION.cff            Citation metadata (add the DOI after archiving)
```

## What is NOT in this repository

- **No patient-level data.** MIMIC-IV is distributed under a PhysioNet credentialed data use agreement and cannot be redistributed. See `docs/DATA_ACCESS.md`.
- **No intermediate or output files.** `results/`, `figures/`, and the DuckDB database are excluded by `.gitignore`.
- **Figure-generation scripts are not included**, because they depend on an internal plotting helper that is not redistributed. All numbers plotted in the manuscript are produced by the Python scripts and the R table scripts here.

## Data access

1. Complete CITI "Data or Specimens Only Research" training and sign the PhysioNet credentialed data use agreement for MIMIC-IV v3.0.
2. Download the MIMIC-IV `hosp/` and `icu/` modules and place them under a local directory, e.g. `D:/mimic-iv-3`.
3. Set the two path constants at the top of the pipeline scripts:
   - `MIMIC_ROOT` (directory containing `hosp/` and `icu/`)
   - `DB_PATH` (path for the DuckDB database, e.g. `D:/mimic_iv.duckdb`)

The scripts currently contain example Windows paths (`D:/mimic-iv-3`, `D:/mimic_iv.duckdb`); change them to your own locations before running.

## Software

- Python 3.9.13 — see `requirements.txt` (duckdb 1.4.5, pandas 1.4.4, numpy 1.21.5, scikit-learn 1.6.1, statsmodels 0.13.2, matplotlib 3.5.2, joblib 1.5.3)
- R 4.5.3 — see `sessionInfo.txt` (data.table, dplyr, ggplot2, flextable, officer, patchwork, readxl, scales, colorspace)

## Running the pipeline

Run scripts in numeric order from the repository root. Each script writes to `results/` (created automatically where needed).

| Step | Script | Purpose |
|---|---|---|
| 1 | `code/python/01_build_db.py` | Build the DuckDB database from MIMIC-IV CSVs |
| 2 | `code/python/02_label_landscape.py` | Define antibiotic exposure and the broad-spectrum label |
| 3 | `code/python/03_find_itemids.py` | Resolve laboratory and vital-sign item IDs |
| 4 | `code/python/04_build_features.py` | Extract first-6-h features |
| 5 | `code/python/04b_aggregate_features.py` | Aggregate to one row per admission |
| 6 | `code/python/04c_extend_vitals.py` | Extend vital-sign coverage |
| 7 | `code/python/05a_diag_selection.py`, `05b_diag_l1.py` | Admission-diagnosis feature selection |
| 8 | `code/python/05_model.py` | Preprocessing, variable selection, main models, calibration, bootstrap, DCA, subgroups, sensitivity, value chain |
| 9 | `code/python/06_news2_compare.py` | Head-to-head comparison with NEWS2 and qSOFA |
| 10 | `code/python/07_descriptive.py`, `07b`, `07d`, `07e`, `07f`, `07g`, `07i`, `07j` | Descriptive analyses, ROC/PR, subset models, value chain |
| 11 | `code/python/11_fix_dca_ci.py` | Recompute decision-curve confidence intervals |
| 12 | `code/python/12_riskgroup_profile.py` | Observed outcomes by model-defined risk group |
| 13 | `code/R/09_tables.R` | Main tables |
| 14 | `code/R/10_supplementary_tables.R` | Supplementary tables |

ICU-anchored sensitivity analysis: run `code/python/icu_anchor/00_setup_db.py` through `06_robustness.py`, then `code/R/icu_anchor/05_tables.R`.

Random seed: `20261105` (set in `05_model.py`).

## Reproducibility notes

- MIMIC-IV v3.0 is a fixed release; dates are shifted into the future and are consistent within a patient. The development/hold-out split uses the de-identified admission year, as described in the manuscript.
- Analyses are at the level of hospital admissions, not patients; repeated admissions of the same patient are not clustered in the main models (stated as a limitation in the manuscript).
- Results depend on the DuckDB build and on the `mimic-code` antibiotic definitions shipped in `code/sql/`.

## Citation

After archiving a release (see below), add the DOI to `CITATION.cff` and to the manuscript's Data Availability statement.

```
Wang J, et al. Predicting the need for broad-spectrum empiric antibiotics at hospital admission:
analysis code. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
```

## License

Code is released under the MIT License (`LICENSE`). MIMIC-IV data remain governed by the PhysioNet data use agreement and are not covered by this license.
