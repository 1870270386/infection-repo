# Data access

This repository contains no patient-level data. All analyses use **MIMIC-IV v3.0**, which is available only under a credentialed data use agreement.

## Steps

1. Create a PhysioNet account and complete the required CITI training ("Data or Specimens Only Research").
2. Request access to MIMIC-IV (https://physionet.org/content/mimiciv/3.0/) and sign the data use agreement.
3. Download the `hosp/` and `icu/` modules. The `core/` module is not required by these scripts.
4. Place the files under a local directory, for example:

```
D:/mimic-iv-3/
  hosp/
    admissions.csv
    patients.csv
    diagnoses_icd.csv
    labevents.csv
    microbiologyevents.csv
    prescriptions.csv
    ...
  icu/
    icustays.csv
    chartevents.csv
    ...
```

5. Edit the path constants at the top of the pipeline scripts:

- `MIMIC_ROOT` — directory containing `hosp/` and `icu/`
- `DB_PATH` — path of the DuckDB database to be created (e.g. `D:/mimic_iv.duckdb`)

6. Run `code/python/01_build_db.py` to build the database.

## Why the data are not shared

The MIMIC-IV data use agreement prohibits redistribution of the data, in whole or in part, including derived row-level files. The scripts therefore expect a local copy obtained under the same agreement. Aggregate results reported in the manuscript can be reproduced from the code and the public data release.

## Antibiotic definitions

`code/sql/antibiotic.sql` is the antibiotic class definition distributed with the public `mimic-code` repository (MIT licensed). `code/sql/_abx_select.sql` and `code/sql/antibiotic_patterns.json` are local variants used to enumerate distinct antibiotic classes. If the upstream definition changes, results may differ slightly; the version used here is the one included in this repository.
