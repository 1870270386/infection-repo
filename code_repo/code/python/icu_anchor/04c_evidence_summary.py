# -*- coding: utf-8 -*-
"""分支【ICU 锚点为主分析】：把"感染证据充分性"等关键比例落成一张小结果表（供正文数字溯源）。"""
import duckdb, os, sys
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

BRANCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(BRANCH_ROOT, "results")
con = duckdb.connect("D:/mimic_iv_branch_icu.duckdb", read_only=True)
df = con.execute("SELECT * FROM icu_cohort2").df()
con.close()

n = len(df)
rows = [
    {"quantity": "Patients analysed", "n": n, "percent": 100.0},
    {"quantity": "Broad-spectrum outcome (>=3 antimicrobial classes within 24 h of ICU admission)",
     "n": int(df.broad_ge3.sum()), "percent": 100 * df.broad_ge3.mean()},
    {"quantity": "Positive culture before the first antibiotic",
     "n": int((df.micro_evidence_before_abx == 1).sum()), "percent": 100 * df.micro_evidence_before_abx.mean()},
    {"quantity": "No positive culture before the first antibiotic",
     "n": int((df.micro_evidence_before_abx == 0).sum()), "percent": 100 * (df.micro_evidence_before_abx == 0).mean()},
    {"quantity": "No positive culture at any time during the admission",
     "n": int(df.t_first_pos.isna().sum()), "percent": 100 * df.t_first_pos.isna().mean()},
    {"quantity": "In-hospital death", "n": int(df.hospital_expire_flag.sum()),
     "percent": 100 * df.hospital_expire_flag.mean()},
    {"quantity": "Median hours from hospital admission to ICU admission (IQR)",
     "n": int(df.adm_to_icu_h.median()), "percent": float(df.adm_to_icu_h.quantile(.25))},
]
out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "icu_evidence_summary.csv"), index=False)
print(out.to_string(index=False))
print("\n已导出 icu_evidence_summary.csv")
