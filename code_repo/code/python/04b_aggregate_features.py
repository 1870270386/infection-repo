# -*- coding: utf-8 -*-
"""MIMIC-IV：0–6h 特征聚合（条件聚合单次扫描，不用相关子查询）。

依赖：mimic_iv.duckdb 中已存在 cv_vitals / cv_labs（由 04_build_features.py 抽取）。
输出：feats 表 + studies/infection-sepsis-2026-11/results/mimic_features.csv
"""
import duckdb, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
con = duckdb.connect("D:/mimic_iv.duckdb")
con.execute("PRAGMA threads=4"); con.execute("PRAGMA memory_limit='10GB'")

VITALS = {"hr": [220045], "sbp": [220050], "dbp": [220051], "map": [220052],
          "tempC": [223762], "tempF": [223761], "rr": [220210],
          "spo2_art": [220227], "spo2_pls": [220277],
          "gcs": [226755, 227013], "weight": [226512], "height": [226707]}
LABELS = {
    "creatinine": ["Creatinine"], "lactate": ["Lactate"], "bicarbonate": ["Bicarbonate"],
    "ph": ["pH"], "pco2": ["pCO2"], "po2": ["pO2"],
    "sodium": ["Sodium"], "chloride": ["Chloride"], "potassium": ["Potassium"],
    "glucose": ["Glucose"], "bun": ["Urea Nitrogen"], "albumin": ["Albumin"],
    "bilirubin": ["Bilirubin, Total"], "platelet": ["Platelet Count"],
    "wbc": ["WBC Count", "White Blood Cells"], "hct": ["Hematocrit"], "hgb": ["Hemoglobin"],
    "neut": ["Absolute Neutrophil Count"], "ast": ["AST", "Aspartate Aminotransferase"],
    "alt": ["Alanine Aminotransferase"], "anion_gap": ["Anion Gap"],
    "trop": ["Troponin I", "Troponin T"], "crp": ["C-Reactive Protein"],
}
LAB = {}
for key, labs in LABELS.items():
    ids = []
    for lb in labs:
        ids += [x[0] for x in con.execute(
            """SELECT itemid FROM d_labitems WHERE LOWER(label)=? AND fluid='Blood'
               ORDER BY itemid LIMIT 2""", [lb.lower()]).fetchall()]
    if ids:
        LAB[key] = ids
print("实验室 itemid 解析:", {k: v for k, v in list(LAB.items())[:6]}, "...")

def pivot(table, spec, prefix):
    sel = []
    for k, ids in spec.items():
        idl = ",".join(str(i) for i in ids)
        cond = f"z.itemid IN ({idl})"
        sel += [
          f"COUNT(*) FILTER (WHERE {cond}) AS {prefix}_{k}_n",
          f"AVG(z.valuenum) FILTER (WHERE {cond}) AS {prefix}_{k}_mean",
          f"MIN(z.valuenum) FILTER (WHERE {cond}) AS {prefix}_{k}_min",
          f"MAX(z.valuenum) FILTER (WHERE {cond}) AS {prefix}_{k}_max",
          f"STDDEV(z.valuenum) FILTER (WHERE {cond}) AS {prefix}_{k}_sd",
          f"ARG_MAX(z.valuenum, z.charttime) FILTER (WHERE {cond}) AS {prefix}_{k}_last",
        ]
    q = f"""
    SELECT z.hadm_id, {', '.join(sel)}
    FROM {table} z JOIN admissions a ON z.hadm_id = a.hadm_id
    WHERE z.charttime >= a.admittime AND z.charttime < a.admittime + INTERVAL 6 HOUR
    GROUP BY z.hadm_id"""
    return q

for tbl, spec, pref, out in [("cv_vitals", VITALS, "v", "vit_agg"),
                             ("cv_labs", LAB, "l", "lab_agg")]:
    t0 = time.time()
    con.execute(f"CREATE OR REPLACE TABLE {out} AS " + pivot(tbl, spec, pref))
    print(f"  {out}: {con.execute(f'SELECT count(*) FROM {out}').fetchone()[0]:,} 行 "
          f"({time.time()-t0:.0f}s)")

t0 = time.time()
con.execute("""CREATE OR REPLACE TABLE feats AS
SELECT w.*, v.* EXCLUDE (hadm_id), l.* EXCLUDE (hadm_id)
FROM cohort2 w
LEFT JOIN vit_agg v ON w.hadm_id = v.hadm_id
LEFT JOIN lab_agg l ON w.hadm_id = l.hadm_id""")
print(f"  feats: {con.execute('SELECT count(*) FROM feats').fetchone()[0]:,} 行 ({time.time()-t0:.0f}s)")

# 合并症
CM = {
    "cm_hf": ["428"], "cm_ckd": ["584", "585", "403"], "cm_dm": ["250"],
    "cm_copd": ["496"], "cm_cirrhosis": ["571"],
    "cm_cancer": [str(i) for i in range(140, 240)] + ["201", "202"],
    "cm_sepsis_adm": ["995", "A41"], "cm_pneumonia": ["486", "J18"],
    "cm_uti": ["599", "N39"], "cm_bacteremia": ["041", "A48"],
    "cm_immunocomp": ["279", "446", "042", "V08"],
    "cm_stroke": ["430", "431", "432", "433", "434", "435", "436", "437", "438", "439", "I63"],
    "cm_alcohol": ["291"], "cm_obesity": ["278"],
}
print("\n=== 合并症（入院诊断 ICD）===")
for k, codes in CM.items():
    cond = " OR ".join([f"d.icd_code LIKE '{c}%'" for c in codes])
    con.execute(f"CREATE OR REPLACE TABLE _cm AS SELECT DISTINCT hadm_id FROM diagnoses_icd d WHERE {cond}")
    con.execute(f"ALTER TABLE feats ADD COLUMN {k} INTEGER DEFAULT NULL")
    con.execute(f"UPDATE feats SET {k} = CASE WHEN hadm_id IN (SELECT hadm_id FROM _cm) THEN 1 ELSE 0 END")
    print(f"  {k:18s} {con.execute('SELECT count(*) FROM _cm').fetchone()[0]:,}")

con.execute("ALTER TABLE feats ADD COLUMN adm_year INTEGER")
con.execute("UPDATE feats SET adm_year = EXTRACT(YEAR FROM admittime)")
print("\n目标人群按入院年份（用于时间外切分）:")
for r in con.execute("""SELECT adm_year, count(*), sum(broad_ge3), round(100.0*sum(broad_ge3)/count(*),2)
                        FROM feats WHERE empiric_24h=1 GROUP BY 1 ORDER BY 1""").fetchall():
    print("   ", r)

con.execute("""COPY (SELECT * FROM feats WHERE empiric_24h=1)
                TO 'studies/infection-sepsis-2026-11/results/mimic_features.csv' (HEADER)""")
print("\n已导出 studies/infection-sepsis-2026-11/results/mimic_features.csv")
con.close()
