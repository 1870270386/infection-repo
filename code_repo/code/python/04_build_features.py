# -*- coding: utf-8 -*-
"""MIMIC-IV：为"入院时是否需要大包围"构建预测变量（窗口 = 入院后 0–6h）。

设计：
  预测时点   = 入院后 6 小时（对应"初步评估"这个临床决策点）
  预测变量   = 该窗口内的生命体征与实验室统计量 + 人口学 + 入院诊断合并症
  标签       = 入院 24h 内 ≥3 类抗生素（broad_ge3），在 02_label_landscape.py 已定义
  泄漏控制   = 变量窗口(0–6h)严格早于标签窗口(0–24h)；不含任何用药信息

输出：mimic_iv.duckdb 中的表 feats / feats_csv
"""
import duckdb, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
MIMIC = "D:/mimic-iv-3"
con = duckdb.connect("D:/mimic_iv.duckdb")
con.execute("PRAGMA threads=4"); con.execute("PRAGMA memory_limit='10GB'")

# ---------------------------------------------------------------- itemid 定位
VITALS = {
    "hr": 220045, "sbp": 220050, "dbp": 220051, "map": 220052,
    "tempC": 223762, "tempF": 223761, "rr": 220210,
    "spo2_arterial": 220227, "spo2_pulse": 220277,
    "gcs": 226755, "gcs_iv": 227013, "weight": 226512, "height": 226707,
}
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
        r = con.execute("""SELECT itemid FROM d_labitems WHERE LOWER(label)=? AND fluid='Blood'
                           ORDER BY itemid LIMIT 2""", [lb.lower()]).fetchall()
        ids += [x[0] for x in r]
    if ids:
        LAB[key] = ids
print("生命体征 itemid:", VITALS)
print("实验室 itemid:", LAB)
print("未找到的实验室项:", [k for k in LABELS if k not in LAB])

# ---------------------------------------------------------------- 抽取窄表
t0 = time.time()
print("\n=== 抽取 chartevents 窄表（入院后 0–24h，仅目标 itemid）===")
v_ids = ",".join(str(v) for v in VITALS.values())
con.execute(f"""
CREATE OR REPLACE TABLE cv_vitals AS
SELECT c.subject_id, c.hadm_id, c.charttime, c.itemid, c.valuenum
FROM read_csv_auto('{MIMIC}/icu/chartevents.csv.gz', header=true, escape='"',
                   columns={{'subject_id':'BIGINT','hadm_id':'BIGINT','stay_id':'BIGINT',
                             'caregiver_id':'BIGINT','charttime':'TIMESTAMP','storetime':'TIMESTAMP',
                             'itemid':'INTEGER','value':'VARCHAR','valuenum':'DOUBLE',
                             'valueuom':'VARCHAR','warning':'VARCHAR'}}) c
JOIN admissions a ON c.hadm_id = a.hadm_id
WHERE c.itemid IN ({v_ids})
  AND c.charttime >= a.admittime AND c.charttime < a.admittime + INTERVAL 24 HOUR""")
print("  cv_vitals:", con.execute("SELECT count(*) FROM cv_vitals").fetchone()[0], "行  (%.0fs)" % (time.time()-t0))

t0 = time.time()
print("=== 抽取 labevents 窄表（入院后 0–24h，仅目标 itemid）===")
l_ids = ",".join(str(i) for ids in LAB.values() for i in ids)
con.execute(f"""
CREATE OR REPLACE TABLE cv_labs AS
SELECT l.subject_id, l.hadm_id, l.charttime, l.itemid, l.valuenum
FROM read_csv_auto('{MIMIC}/hosp/labevents.csv.gz', header=true, escape='"',
                   columns={{'labevent_id':'BIGINT','subject_id':'BIGINT','hadm_id':'BIGINT',
                             'specimen_id':'BIGINT','itemid':'INTEGER','order_provider_id':'BIGINT',
                             'charttime':'TIMESTAMP','storetime':'TIMESTAMP','value':'VARCHAR',
                             'valuenum':'DOUBLE','valueuom':'VARCHAR','ref_range_lower':'DOUBLE',
                             'ref_range_upper':'DOUBLE','flag':'VARCHAR','priority':'VARCHAR',
                             'comments':'VARCHAR'}}) l
JOIN admissions a ON l.hadm_id = a.hadm_id
WHERE l.itemid IN ({l_ids})
  AND l.charttime >= a.admittime AND l.charttime < a.admittime + INTERVAL 24 HOUR""")
print("  cv_labs:", con.execute("SELECT count(*) FROM cv_labs").fetchone()[0], "行  (%.0fs)" % (time.time()-t0))

# ---------------------------------------------------------------- 0–6h 特征
def agg(src, tag):
    parts = []
    for k, ids in (VITALS.items() if src == "cv_vitals" else LAB.items()):
        idlist = ",".join(str(i) for i in (ids if isinstance(ids, list) else [ids]))
        nm = f"{tag}_{k}"
        parts.append(f"""
        (SELECT COUNT(*) FROM {src} z WHERE z.hadm_id = w.hadm_id AND z.itemid IN ({idlist})
           AND z.charttime >= w.admittime AND z.charttime < w.admittime + INTERVAL 6 HOUR) AS {nm}_n,
        (SELECT AVG(z.valuenum) FROM {src} z WHERE z.hadm_id = w.hadm_id AND z.itemid IN ({idlist})
           AND z.charttime >= w.admittime AND z.charttime < w.admittime + INTERVAL 6 HOUR) AS {nm}_mean,
        (SELECT MIN(z.valuenum) FROM {src} z WHERE z.hadm_id = w.hadm_id AND z.itemid IN ({idlist})
           AND z.charttime >= w.admittime AND z.charttime < w.admittime + INTERVAL 6 HOUR) AS {nm}_min,
        (SELECT MAX(z.valuenum) FROM {src} z WHERE z.hadm_id = w.hadm_id AND z.itemid IN ({idlist})
           AND z.charttime >= w.admittime AND z.charttime < w.admittime + INTERVAL 6 HOUR) AS {nm}_max,
        (SELECT STDDEV(z.valuenum) FROM {src} z WHERE z.hadm_id = w.hadm_id AND z.itemid IN ({idlist})
           AND z.charttime >= w.admittime AND z.charttime < w.admittime + INTERVAL 6 HOUR) AS {nm}_sd,
        (SELECT ARG_MAX(z.valuenum, z.charttime) FROM {src} z WHERE z.hadm_id = w.hadm_id AND z.itemid IN ({idlist})
           AND z.charttime >= w.admittime AND z.charttime < w.admittime + INTERVAL 6 HOUR) AS {nm}_last""")
    return ",\n".join(parts)

print("\n=== 聚合 0–6h 特征 ===")
t0 = time.time()
con.execute(f"""
CREATE OR REPLACE TABLE feats AS
SELECT w.*,
{agg("cv_vitals", "v")},
{agg("cv_labs", "l")}
FROM cohort2 w""")
print("  feats:", con.execute("SELECT count(*) FROM feats").fetchone()[0], "行  (%.0fs)" % (time.time()-t0))

# ---------------------------------------------------------------- 合并症（入院诊断）
print("\n=== 合并症（入院诊断 ICD）===")
CM = {
    "cm_hf": ["428"], "cm_ckd": ["584", "585", "403"], "cm_dm": ["250"],
    "cm_copd": ["496"], "cm_cirrhosis": ["571"], "cm_cancer": ["140", "141", "142", "143", "144", "145",
                                                       "146", "147", "148", "149", "150", "151", "152",
                                                       "153", "154", "155", "156", "157", "158", "159",
                                                       "160", "161", "162", "163", "164", "165", "166",
                                                       "167", "168", "169", "170", "171", "172", "173",
                                                       "174", "175", "176", "177", "178", "179", "180",
                                                       "181", "182", "183", "184", "185", "186", "187",
                                                       "188", "189", "190", "191", "192", "193", "194",
                                                       "195", "199", "200", "201", "202", "203", "204",
                                                       "205", "206", "207", "208", "209", "210", "211",
                                                       "212", "213", "214", "215", "216", "217", "218",
                                                       "219", "220", "221", "222", "223", "230", "231",
                                                       "232", "233", "234", "235", "236", "237", "238",
                                                       "239", "241"],
    "cm_sepsis_adm": ["995", "A41"], "cm_pneumonia": ["486", "J18"],
    "cm_uti": ["599", "N39"], "cm_bacteremia": ["041", "A48"],
    "cm_immunocomp": ["279", "446", "042", "V08"],
    "cm_stroke": ["430", "431", "432", "433", "434", "435", "436", "437", "438", "439", "I63"],
    "cm_alcohol": ["291"], "cm_obesity": ["278"],
}
for k, codes in CM.items():
    cond = " OR ".join([f"d.icd_code LIKE '{c}%'" for c in codes])
    con.execute(f"""CREATE OR REPLACE TABLE _cm_{k} AS
                    SELECT DISTINCT hadm_id FROM diagnoses_icd d WHERE {cond}""")
    con.execute(f"""ALTER TABLE feats ADD COLUMN {k} INTEGER DEFAULT NULL""")
    con.execute(f"""UPDATE feats SET {k} = CASE WHEN hadm_id IN (SELECT hadm_id FROM _cm_{k}) THEN 1 ELSE 0 END""")
    print(f"  {k:18s} 命中住院 {con.execute(f'SELECT count(*) FROM _cm_{k}').fetchone()[0]:,}")

# 年份（用于时间外切分）
con.execute("ALTER TABLE feats ADD COLUMN adm_year INTEGER")
con.execute("UPDATE feats SET adm_year = EXTRACT(YEAR FROM admittime)")
print("\n目标人群按入院年份:")
for r in con.execute("SELECT adm_year, count(*), sum(broad_ge3) FROM feats WHERE empiric_24h=1 GROUP BY 1 ORDER BY 1").fetchall():
    print("   ", r)

con.execute(f"""COPY (SELECT * FROM feats WHERE empiric_24h=1)
                TO 'studies/infection-sepsis-2026-11/results/mimic_features.csv' (HEADER)""")
print("\n已导出 studies/infection-sepsis-2026-11/results/mimic_features.csv")
con.close()
