# -*- coding: utf-8 -*-
"""MIMIC-IV：描述性统计（Table 1、Figure 1 面板所需的全部数字）。

产出
  results/desc_table1.csv     Table 1：目标人群特征，按"入院 24h 内 ≥3 类抗生素"分组
  results/desc_labels.csv     5 个"大包围"定义的阳性率与两两一致性
  results/desc_evidence.csv   "感染证据不足"的量化（培养与抗生素的先后）
  results/desc_availability.csv 决策时点数据可得性分层（全队列 + 留出验证集）

脚本：07_descriptive.py
"""
import duckdb, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

OUT = "studies/infection-sepsis-2026-11/results"
con = duckdb.connect("D:/mimic_iv.duckdb", read_only=True)
df = con.execute("SELECT * FROM feats2 WHERE empiric_24h=1").df()
con.close()
N = len(df)
df["_g"] = np.where(df.broad_ge3 == 1, "Broad-spectrum", "Narrower")
g1, g0 = df[df.broad_ge3 == 1], df[df.broad_ge3 == 0]
print(f"全队列 n={N:,}  大包围 {len(g1):,}（{100*df.broad_ge3.mean():.2f}%）")

def fmt_cont(s):
    s = pd.to_numeric(s, errors="coerce")
    return f"{s.median():.1f} ({s.quantile(.25):.1f}–{s.quantile(.75):.1f})", s.isna().mean()

def fmt_cat(s):
    s = s.astype(str)
    n = s.value_counts()
    tot = len(s)
    return "n (%)", n, tot

rows = []
def _grp():
    return df[df.broad_ge3 == 1], df[df.broad_ge3 == 0]

def add_c(label, col, unit=""):
    g1, g0 = _grp()
    v1, miss1 = fmt_cont(g1[col]); v0, miss0 = fmt_cont(g0[col]); va, _ = fmt_cont(df[col])
    rows.append({"variable": label, "type": "continuous", "overall": va,
                 "broad_spectrum": v1, "no_broad_spectrum": v0,
                 "missing_pct": round(100 * miss1, 1), "unit": unit})

def add_cat(label, col, levels=None, ref=None):
    g1, g0 = _grp()
    if levels is None:
        levels = sorted(df[col].astype(str).unique())
    else:
        levels = [l for l in levels if (df[col].astype(str) == l).any()]
    for i, lv in enumerate(levels):
        c1 = int((g1[col].astype(str) == lv).sum()); c0 = int((g0[col].astype(str) == lv).sum())
        ct = int((df[col].astype(str) == lv).sum())
        rows.append({"variable": (label if i == 0 else ""), "type": "categorical",
                     "level": lv, "overall": f"{ct} ({100*ct/N:.1f})",
                     "broad_spectrum": f"{c1} ({100*c1/len(g1):.1f})",
                     "no_broad_spectrum": f"{c0} ({100*c0/len(g0):.1f})",
                     "missing_pct": round(100 * df[col].isna().mean(), 1), "unit": ""})

# ---- 人口学
add_c("Age, years", "anchor_age")
df["_sex"] = df.gender.astype(str).str.upper().map({"M": "Male", "F": "Female"}).fillna("Unknown")
add_cat("Sex", "_sex")
df["_race"] = df.race.astype(str).str.upper()
RACE_MAP = {"WHITE": "White", "BLACK": "Black", "ASIAN": "Asian",
            "HISPANIC": "Hispanic", "OTHER": "Other"}
df["_race"] = df._race.map(lambda v: RACE_MAP.get(v, "Other/unknown"))
add_cat("Race", "_race")
ATYPE_MAP = {"EW EMER.": "Emergency", "DIRECT EMER.": "Emergency",
             "EU OBSERVATION": "Observation", "OBSERVATION ADMIT": "Observation",
             "DIRECT OBSERVATION": "Observation", "AMBULATORY OBSERVATION": "Observation",
             "URGENT": "Urgent", "SURGICAL SAME DAY ADMISSION": "Surgical same-day",
             "ELECTIVE": "Elective"}
df["_atype"] = df.admission_type.astype(str).str.upper().map(ATYPE_MAP).fillna("Other")
add_cat("Admission type", "_atype")

# ---- 合并症
for c in [c for c in df.columns if c.startswith("cm_")]:
    df["_" + c] = (df[c] == 1).map({True: "Yes", False: "No"})
    add_cat(c.replace("cm_", "Comorbidity: ").replace("_", " "), "_" + c, levels=["Yes", "No"])

# ---- 决策时点数据可得性
_lab_n = [c for c in df.columns if c.startswith("l_") and c.endswith("_n")]
_vit_keys = [c for c in ["v_hr_n", "v_rr_n", "v_sbp_n", "v_nibp_sbp_n",
                          "v_spo2_pls_n", "v_spo2_art_n"] if c in df.columns]
df["has_lab_6h"] = (df[_lab_n].sum(axis=1) > 0).astype(int)
df["has_vit_6h"] = (df[_vit_keys].sum(axis=1) > 0).astype(int)
df["_avail"] = np.where((df.has_lab_6h == 1) & (df.has_vit_6h == 1), "Both laboratory and vital signs",
                 np.where(df.has_lab_6h == 1, "Laboratory only",
                 np.where(df.has_vit_6h == 1, "Vital signs only", "Neither")))
add_cat("Objective data available at 0-6 h", "_avail")

# ---- 关键生命体征（0–6 h）
for col, lab, unit in [("v_hr_max", "Heart rate, maximum, beats/min", "beats/min"),
                        ("v_rr_max", "Respiratory rate, maximum, breaths/min", "breaths/min"),
                        ("v_sbp_min", "Systolic blood pressure, minimum, mmHg", "mmHg"),
                        ("v_spo2_pls_min", "SpO2, minimum, %", "%"),
                        ("v_gcs_total_min", "Glasgow Coma Scale, minimum", "score")]:
    if col in df.columns:
        add_c(lab, col, unit)
add_c("Temperature, maximum, °C", "v_tempC_max", "°C")
add_c("Temperature, maximum, °F", "v_tempF_max", "°F")

# ---- 关键实验室（0–6 h）
for col, lab, unit in [("l_creatinine_max", "Creatinine, maximum, mg/dL", "mg/dL"),
                        ("l_bun_max", "Urea nitrogen, maximum, mg/dL", "mg/dL"),
                        ("l_lactate_max", "Lactate, maximum, mmol/L", "mmol/L"),
                        ("l_bicarbonate_min", "Bicarbonate, minimum, mmol/L", "mmol/L"),
                        ("l_wbc_max", "White blood cell count, maximum, 10⁹/L", "10⁹/L"),
                        ("l_hgb_min", "Haemoglobin, minimum, g/dL", "g/dL"),
                        ("l_platelet_min", "Platelet count, minimum, 10⁹/L", "10⁹/L"),
                        ("l_sodium_min", "Sodium, minimum, mmol/L", "mmol/L"),
                        ("l_glucose_max", "Glucose, maximum, mg/dL", "mg/dL"),
                        ("l_albumin_min", "Albumin, minimum, g/dL", "g/dL"),
                        ("l_ph_min", "pH, minimum", ""),
                        ("l_bilirubin_max", "Total bilirubin, maximum, mg/dL", "mg/dL")]:
    if col in df.columns:
        add_c(lab, col, unit)

# ---- 结局
df["_died"] = np.where(df.hospital_expire_flag == 1, "Died", "Survived")
add_cat("In-hospital outcome", "_died")
add_c("Length of stay, h", "los_h", "h")
add_c("Distinct antibiotic classes in first 24 h", "n24")

t1 = pd.DataFrame(rows)
t1.to_csv(f"{OUT}/desc_table1.csv", index=False)
print(f"Table 1: {len(t1)} 行 → results/desc_table1.csv")

# ---- 标签定义
lab_rows = []
for lab, col in [("≥3 antibiotic classes in 24 h (primary)", "broad_ge3"),
                 ("Any carbapenem", "broad_carbapenem"),
                 ("Anti-MRSA plus anti-gram-negative broad spectrum", "broad_mrsa_plus_gnr"),
                 ("Carbapenem plus vancomycin/linezolid", "broad_carb_vanc"),
                 ("≥3 classes including anti-MRSA", "broad_ge3_mrsa")]:
    k = int(df[col].sum())
    lab_rows.append({"definition": lab, "n": k, "rate_pct": round(100 * k / N, 2)})
pd.DataFrame(lab_rows).to_csv(f"{OUT}/desc_labels.csv", index=False)
print("\n=== 5 个『大包围』定义 ===")
for r in lab_rows:
    print(f"  {r['definition']:46s} {r['n']:>7,} ({r['rate_pct']:5.2f}%)")

# ---- 感染证据不足
ev = []
ev.append({"item": "Positive culture before the first antibiotic dose", "n": int((df.micro_evidence_before_abx == 1).sum()),
           "pct": round(100 * (df.micro_evidence_before_abx == 1).mean(), 2)})
ev.append({"item": "No microbiological evidence at the decision point", "n": int((df.micro_evidence_before_abx == 0).sum()),
           "pct": round(100 * (df.micro_evidence_before_abx == 0).mean(), 2)})
ev.append({"item": "No culture sent during the admission", "n": int(df.t_first_pos.isna().sum()),
           "pct": round(100 * df.t_first_pos.isna().mean(), 2)})
ev.append({"item": "Admitted to ICU during the admission",
           "n": int(df.hadm_id.isin([]).sum()), "pct": 0.0})
pd.DataFrame(ev).to_csv(f"{OUT}/desc_evidence.csv", index=False)
print("\n=== 感染证据不足 ===")
for r in ev:
    print(f"  {r['item']:56s} {r['n']:>7,} ({r['pct']:5.2f}%)")

# ---- 可得性分层（全队列 + 留出验证集）
cut = int(np.quantile(df.adm_year.values, 0.70))
val = df.adm_year.values > cut
av = []
for k in ["Both laboratory and vital signs", "Laboratory only", "Vital signs only", "Neither"]:
    m = (df._avail.values == k)
    mv = m & val
    av.append({"stratum": k,
               "n_all": int(m.sum()), "pct_all": round(100 * m.mean(), 1),
               "broad_rate_all": round(100 * df.loc[m, "broad_ge3"].mean(), 2),
               "mortality_all": round(100 * df.loc[m, "hospital_expire_flag"].mean(), 2),
               "n_val": int(mv.sum()),
               "broad_rate_val": round(100 * df.loc[mv, "broad_ge3"].mean(), 2)})
pd.DataFrame(av).to_csv(f"{OUT}/desc_availability.csv", index=False)
print("\n=== 数据可得性分层 ===")
for r in av:
    print(f"  {r['stratum']:34s} n={r['n_all']:>7,} ({r['pct_all']:4.1f}%)  "
          f"大包围率 {r['broad_rate_all']:5.2f}%  死亡 {r['mortality_all']:5.2f}%")

# ICU 收治率补上
con = duckdb.connect("D:/mimic_iv.duckdb", read_only=True)
icu_n = con.execute("""SELECT count(*) FROM feats2 f WHERE f.empiric_24h=1 AND
                        EXISTS (SELECT 1 FROM icustays i WHERE i.hadm_id=f.hadm_id)""").fetchone()[0]
con.close()
df["_icu"] = 0
ev2 = pd.read_csv(f"{OUT}/desc_evidence.csv")
ev2.loc[ev2.item == "Admitted to ICU during the admission", ["n", "pct"]] = [icu_n, round(100*icu_n/N, 2)]
ev2.to_csv(f"{OUT}/desc_evidence.csv", index=False)
print(f"  ICU 收治 {icu_n:,} ({100*icu_n/N:.2f}%)")
print("\n已导出 desc_table1 / desc_labels / desc_evidence / desc_availability")
