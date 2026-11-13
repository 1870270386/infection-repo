# -*- coding: utf-8 -*-
"""分支【ICU 锚点为主分析】：算清"决策点数据可得性"的真实数字，出图直接读 CSV（不硬编码）。

两套锚点：
  住院入院锚点（feats 表，empiric_24h=1）→ 窗口 = 住院入院后 0–6 h
  ICU 入院锚点（icu_cohort2 + 0–6 h 聚合）→ 窗口 = ICU 入院后 0–6 h
输出：results/icu_data_availability.csv、results/icu_variable_availability.csv
"""
import duckdb, os, sys
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

BRANCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(BRANCH_ROOT, "results")
import duckdb as _d
con = _d.connect(":memory:")
con.execute("ATTACH 'D:/mimic_iv_branch_icu.duckdb' AS b (READ_ONLY)")
con.execute("USE b")

# 目标变量集合：用于判定"是否有任一实验室 / 生命体征记录"
LAB_SET = ["creatinine", "lactate", "wbc", "bicarbonate", "hgb", "platelet", "sodium",
           "potassium", "glucose", "bun", "albumin", "bilirubin", "neut", "ph", "pco2",
           "po2", "hct", "ast", "alt", "anion_gap", "trop", "crp"]
VIT_SET = ["hr", "rr", "sbp", "spo2_pulse", "spo2_arterial", "tempC", "tempF"]

con.execute("""CREATE OR REPLACE TEMP TABLE _tmp_avail AS
SELECT c.hadm_id,
  COALESCE(v.v_hr_n,0) v_hr_n, COALESCE(v.v_rr_n,0) v_rr_n, COALESCE(v.v_sbp_n,0) v_sbp_n,
  COALESCE(v.v_spo2_pulse_n,0) v_spo2_pulse_n, COALESCE(v.v_spo2_arterial_n,0) v_spo2_arterial_n,
  COALESCE(v.v_tempC_n,0) v_tempC_n, COALESCE(v.v_tempF_n,0) v_tempF_n,
  CASE WHEN v.v_gcs_total IS NOT NULL THEN 1 ELSE 0 END v_gcs_total_n,
  COALESCE(v.v_o2_n,0) v_o2_n,
  COALESCE(l.l_glucose_n,0) l_glucose_n, COALESCE(l.l_bun_n,0) l_bun_n,
  COALESCE(l.l_creatinine_n,0) l_creatinine_n, COALESCE(l.l_bicarbonate_n,0) l_bicarbonate_n,
  COALESCE(l.l_hgb_n,0) l_hgb_n, COALESCE(l.l_platelet_n,0) l_platelet_n,
  COALESCE(l.l_wbc_n,0) l_wbc_n, COALESCE(l.l_sodium_n,0) l_sodium_n,
  COALESCE(l.l_potassium_n,0) l_potassium_n, COALESCE(l.l_lactate_n,0) l_lactate_n,
  COALESCE(l.l_ph_n,0) l_ph_n, COALESCE(l.l_bilirubin_n,0) l_bilirubin_n,
  COALESCE(l.l_neut_n,0) l_neut_n
FROM icu_cohort2 c
LEFT JOIN vit_agg_icu v ON c.hadm_id = v.hadm_id
LEFT JOIN lab_agg_icu l ON c.hadm_id = l.hadm_id""")
con.execute("""CREATE OR REPLACE TEMP TABLE _tmp_avail2 AS
SELECT *, (CASE WHEN v_tempC_n>0 OR v_tempF_n>0 THEN 1 ELSE 0 END) v_temp_any_n FROM _tmp_avail""")

lab_icu = " + ".join([f"COALESCE(l_{k}_n,0)" for k in LAB_SET])
VIT_COMMON = ["hr", "rr", "sbp", "tempC", "tempF"]   # 两套锚点共有的生命体征
vit_icu = " + ".join([f"COALESCE(v_{k}_n,0)" for k in VIT_COMMON])
# feats 表（住院锚点，0–6 h）的列与 ICU 表不完全一致，按实际存在的列来判定
_feats_cols = [r[0] for r in con.execute("SELECT column_name FROM information_schema.columns "
                                         "WHERE table_name='feats'").fetchall()]
def _present(keys, pref):
    return [k for k in keys if f"{pref}_{k}_n" in _feats_cols]
lab_hosp = " + ".join([f"COALESCE(l_{k}_n,0)" for k in _present(LAB_SET, "l")])
vit_hosp = " + ".join([f"COALESCE(v_{k}_n,0)" for k in _present(VIT_SET, "v")])
print("feats 表中用于判定的实验室变量:", _present(LAB_SET, "l"))
print("feats 表中用于判定的生命体征变量:", _present(VIT_SET, "v"))

lab_icu_f = " + ".join([f"COALESCE(l_{k}_n,0)" for k in LAB_SET])
vit_icu_f = " + ".join([f"COALESCE(v_{k}_n,0)" for k in VIT_COMMON])
r = con.execute(f"""
SELECT count(*) FILTER (WHERE ({lab_icu_f})>0),
       count(*) FILTER (WHERE ({vit_icu_f})>0),
       count(*) FILTER (WHERE ({lab_icu_f})>0 AND ({vit_icu_f})>0),
       count(*) FROM icu_feats""").fetchone()
n_icu = r[3]
icu = {"anchor": "ICU-admission anchor", "n": n_icu,
       "Laboratory data": 100 * r[0] / n_icu,
       "Vital signs": 100 * r[1] / n_icu,
       "Both available": 100 * r[2] / n_icu}

r = con.execute(f"""
SELECT count(*) FILTER (WHERE ({lab_hosp})>0),
       count(*) FILTER (WHERE ({vit_hosp})>0),
       count(*) FILTER (WHERE ({lab_hosp})>0 AND ({vit_hosp})>0),
       count(*) FROM feats WHERE empiric_24h = 1""").fetchone()
n_hosp = r[3]
hosp = {"anchor": "Hospital-admission anchor", "n": n_hosp,
        "Laboratory data": 100 * r[0] / n_hosp,
        "Vital signs": 100 * r[1] / n_hosp,
        "Both available": 100 * r[2] / n_hosp}

df = pd.DataFrame([hosp, icu])
print("=== 决策点 0–6 h 数据可得性（实算）===")
print(df.to_string(index=False))
df.to_csv(os.path.join(OUT, "icu_data_availability.csv"), index=False)

items = [
    ("Heart rate", "v_hr_n"), ("Respiratory rate", "v_rr_n"), ("SpO2 (pulse oximetry)", "v_spo2_pulse_n"),
    ("Systolic blood pressure", "v_sbp_n"), ("GCS total", "v_gcs_total_n"),
    ("Body temperature", "v_temp_any_n"), ("Supplemental oxygen", "v_o2_n"),
    ("Glucose", "l_glucose_n"), ("BUN", "l_bun_n"), ("Creatinine", "l_creatinine_n"),
    ("Bicarbonate", "l_bicarbonate_n"), ("Haemoglobin", "l_hgb_n"), ("Platelet count", "l_platelet_n"),
    ("White cell count", "l_wbc_n"), ("Sodium", "l_sodium_n"), ("Potassium", "l_potassium_n"),
    ("Lactate", "l_lactate_n"), ("pH", "l_ph_n"), ("Bilirubin", "l_bilirubin_n"),
    ("Neutrophils", "l_neut_n"),
]
n = con.execute("SELECT count(*) FROM _tmp_avail2").fetchone()[0]
rows = []
for name, col in items:
    k = con.execute(f"SELECT count(*) FILTER (WHERE {col} > 0) FROM _tmp_avail2").fetchone()[0]
    rows.append({"variable": name, "pct": 100 * k / n, "n_with_value": k})
va = pd.DataFrame(rows).sort_values("pct", ascending=False)
print("\n=== ICUadm 0–6 h 各变量可得率（实算）===")
print(va.to_string(index=False))
va.to_csv(os.path.join(OUT, "icu_variable_availability.csv"), index=False)
con.close()
print("\n已导出 icu_data_availability.csv、icu_variable_availability.csv")
