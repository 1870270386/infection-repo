# -*- coding: utf-8 -*-
"""分支【ICU 锚点为主分析】：抽取 ICU 锚点窄表并构建 0–6h 预测变量。

锚点 = ICU 入院时刻（intime）
窗口 = [intime, intime + 6 h]   ← 预测变量
窗口 = [intime, intime + 24 h]  ← 结局（大包围）
生命体征用 stay_id 限定（只取该次 ICU 收治的记录），实验室用 hadm_id + 时间窗。

输出：分支库表 icu_feats（= icu_cohort2 + 0–6h 特征 + 合并症）
"""
import duckdb, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")

BRANCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(BRANCH_ROOT, "results")
MIMIC = "D:/mimic-iv-3"
BRANCH_DB = "D:/mimic_iv_branch_icu.duckdb"

con = duckdb.connect(BRANCH_DB)
con.execute("PRAGMA threads=4"); con.execute("PRAGMA memory_limit='12GB'")

# ---------------------------------------------------------------- itemid 定位
VITALS = {"hr": [220045],
          "sbp": [220050, 220179, 224167, 227243],
          "dbp": [220051, 220180, 224643, 227242],
          "map": [220052, 220181],
          "tempC": [223762], "tempF": [223761], "rr": [220210],
          "spo2_arterial": [220227], "spo2_pulse": [220277],
          "gcs_eye": [220739], "gcs_verbal": [223900], "gcs_motor": [223901],
          "gcs_apache": [226755, 227013],
          "weight": [226512], "height": [226707]}
O2_IDS = [223834, 223835, 226732]          # O2 Flow / Inspired O2 Fraction / O2 Delivery Device
LABELS = {
    "creatinine": ["Creatinine"], "lactate": ["Lactate"], "bicarbonate": ["Bicarbonate"],
    "ph": ["pH"], "pco2": ["pCO2"], "po2": ["pO2"],
    "sodium": ["Sodium"], "chloride": ["Chloride"], "potassium": ["Potassium"],
    "glucose": ["Glucose"], "bun": ["Urea Nitrogen"], "albumin": ["Albumin"],
    "bilirubin": ["Bilirubin, Total"], "platelet": ["Platelet Count"],
    "wbc": ["WBC Count", "White Blood Cells"], "hct": ["Hematocrit"], "hgb": ["Hemoglobin"],
    "neut": ["Absolute Neutrophil Count"],
    "alt": ["Alanine Aminotransferase"],
    "ast": ["AST", "Asparate Aminotransferase (AST)", "Aspartate Aminotransferase"], "anion_gap": ["Anion Gap"],
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
print("实验室 itemid 覆盖:", len(LAB), "项；未找到:", [k for k in LABELS if k not in LAB])

con.execute("""CREATE OR REPLACE TABLE icu_stay AS
SELECT hadm_id, stay_id, intime FROM icustays
WHERE (hadm_id, intime) IN (SELECT hadm_id, MIN(intime) FROM icustays GROUP BY hadm_id)""")
n = con.execute("SELECT count(*) FROM icu_stay").fetchone()[0]
print("ICU 收治（每次住院首段）:", n)

# ---------------------------------------------------------------- 抽取窄表
t0 = time.time()
print("\n=== 抽取 chartevents 窄表（ICU 锚点 0–24h）===")
SKIP = con.execute("SELECT count(*) FROM information_schema.tables "
                   "WHERE table_name='cv_vitals_icu'").fetchone()[0] > 0
v_ids = ",".join(str(x) for v in VITALS.values() for x in (v if isinstance(v, list) else [v])) + \
         "," + ",".join(map(str, O2_IDS))
if not SKIP:
  con.execute(f"""
CREATE OR REPLACE TABLE cv_vitals_icu AS
SELECT c.hadm_id, c.charttime, c.itemid, c.valuenum,
       (c.charttime < s.intime + INTERVAL 6 HOUR) AS in6h
FROM read_csv_auto('{MIMIC}/icu/chartevents.csv.gz', header=true, escape='"',
                   columns={{'subject_id':'BIGINT','hadm_id':'BIGINT','stay_id':'BIGINT',
                             'caregiver_id':'BIGINT','charttime':'TIMESTAMP','storetime':'TIMESTAMP',
                             'itemid':'INTEGER','value':'VARCHAR','valuenum':'DOUBLE',
                             'valueuom':'VARCHAR','warning':'VARCHAR'}}) c
JOIN icu_stay s ON c.stay_id = s.stay_id
WHERE c.itemid IN ({v_ids})
  AND c.charttime >= s.intime AND c.charttime < s.intime + INTERVAL 24 HOUR""")
print("  cv_vitals_icu:", con.execute("SELECT count(*) FROM cv_vitals_icu").fetchone()[0],
      "行  (%.0fs)" % (time.time()-t0), flush=True)

t0 = time.time()
print("\n=== 抽取 labevents 窄表（ICU 锚点 0–24h）===")
SKIP2 = con.execute("SELECT count(*) FROM information_schema.tables "
                    "WHERE table_name='cv_labs_icu'").fetchone()[0] > 0
l_ids = ",".join(str(i) for ids in LAB.values() for i in ids)
if not SKIP2:
  con.execute(f"""
CREATE OR REPLACE TABLE cv_labs_icu AS
SELECT l.hadm_id, l.charttime, l.itemid, l.valuenum,
       (l.charttime < s.intime + INTERVAL 6 HOUR) AS in6h
FROM read_csv_auto('{MIMIC}/hosp/labevents.csv.gz', header=true, escape='"',
                   columns={{'labevent_id':'BIGINT','subject_id':'BIGINT','hadm_id':'BIGINT',
                             'specimen_id':'BIGINT','itemid':'INTEGER','order_provider_id':'BIGINT',
                             'charttime':'TIMESTAMP','storetime':'TIMESTAMP','value':'VARCHAR',
                             'valuenum':'DOUBLE','valueuom':'VARCHAR','ref_range_lower':'DOUBLE',
                             'ref_range_upper':'DOUBLE','flag':'VARCHAR','priority':'VARCHAR',
                             'comments':'VARCHAR'}}) l
JOIN icu_stay s ON l.hadm_id = s.hadm_id
WHERE l.itemid IN ({l_ids})
  AND l.charttime >= s.intime AND l.charttime < s.intime + INTERVAL 24 HOUR""")
print("  cv_labs_icu:", con.execute("SELECT count(*) FROM cv_labs_icu").fetchone()[0],
      "行  (%.0fs)" % (time.time()-t0), flush=True)

# ---------------------------------------------------------------- 聚合 0–6h 特征
def agg_sql(src, keys, tag):
    parts = []
    for k, ids in keys.items():
        idlist = ",".join(str(i) for i in (ids if isinstance(ids, list) else [ids]))
        nm = f"{tag}_{k}"
        parts.append(
            f"COUNT(*) FILTER (WHERE itemid IN ({idlist}) AND in6h) AS {nm}_n, "
            f"AVG(valuenum) FILTER (WHERE itemid IN ({idlist}) AND in6h) AS {nm}_mean, "
            f"MIN(valuenum) FILTER (WHERE itemid IN ({idlist}) AND in6h) AS {nm}_min, "
            f"MAX(valuenum) FILTER (WHERE itemid IN ({idlist}) AND in6h) AS {nm}_max, "
            f"STDDEV(valuenum) FILTER (WHERE itemid IN ({idlist}) AND in6h) AS {nm}_sd, "
            f"ARG_MAX(valuenum, charttime) FILTER (WHERE itemid IN ({idlist}) AND in6h) AS {nm}_last")
    return ",\n       ".join(parts)


print("\n=== 聚合 0–6h 特征（GROUP BY + FILTER，避免相关子查询）===")
t0 = time.time()
con.execute(f"""
CREATE OR REPLACE TABLE vit_agg_icu AS
SELECT hadm_id,
       {agg_sql("cv_vitals_icu", VITALS, "v")},
       COUNT(*) FILTER (WHERE itemid IN ({','.join(map(str, O2_IDS))}) AND in6h) AS v_o2_n,
       MAX(CASE WHEN itemid = 223834 AND valuenum > 0 AND in6h THEN 1 ELSE 0 END) AS on_o2_flow,
       MAX(CASE WHEN itemid = 223835 AND valuenum > 0.21 AND in6h THEN 1 ELSE 0 END) AS on_o2_fio2,
       MAX(CASE WHEN itemid = 226732 AND in6h THEN 1 ELSE 0 END) AS on_o2_device
FROM cv_vitals_icu GROUP BY hadm_id""")
print("  vit_agg_icu:", con.execute("SELECT count(*) FROM vit_agg_icu").fetchone()[0],
      "行  (%.0fs)" % (time.time()-t0), flush=True)

# 派生床旁 GCS 总分（睁眼+言语+运动，三分量齐全才求和，取各分量最差值）
con.execute("""ALTER TABLE vit_agg_icu ADD COLUMN v_gcs_total DOUBLE""")
con.execute("""UPDATE vit_agg_icu SET v_gcs_total =
  CASE WHEN v_gcs_eye_n > 0 AND v_gcs_verbal_n > 0 AND v_gcs_motor_n > 0
       THEN LEAST(15.0, GREATEST(3.0,
            COALESCE(v_gcs_eye_min,0) + COALESCE(v_gcs_verbal_min,0) + COALESCE(v_gcs_motor_min,0)))
       ELSE NULL END""")
print("  床旁 GCS 总分可得(ICU 0-6h):",
      con.execute("SELECT count(v_gcs_total) FROM vit_agg_icu").fetchone()[0], flush=True)

t0 = time.time()
con.execute(f"""
CREATE OR REPLACE TABLE lab_agg_icu AS
SELECT hadm_id,
       {agg_sql("cv_labs_icu", LAB, "l")}
FROM cv_labs_icu GROUP BY hadm_id""")
print("  lab_agg_icu:", con.execute("SELECT count(*) FROM lab_agg_icu").fetchone()[0],
      "行  (%.0fs)" % (time.time()-t0), flush=True)

# ---------------------------------------------------------------- 合并
print("\n=== 合并到 icu_feats ===")
vcols = []
for k in VITALS:
    for suf in ["n", "mean", "min", "max", "sd", "last"]:
        vcols.append(f"v.v_{k}_{suf}")
vcols += ["v.v_o2_n", "v.on_o2_flow", "v.on_o2_fio2", "v.on_o2_device", "v.v_gcs_total"]
lcols = []
for k in LAB:
    for suf in ["n", "mean", "min", "max", "sd", "last"]:
        lcols.append(f"l.l_{k}_{suf}")
con.execute("""CREATE OR REPLACE TABLE icu_feats AS
SELECT c.*,
       """ + ", ".join(vcols) + ",\n       " + ", ".join(lcols) + """
FROM icu_cohort2 c
LEFT JOIN vit_agg_icu v ON c.hadm_id = v.hadm_id
LEFT JOIN lab_agg_icu l ON c.hadm_id = l.hadm_id
-- 必须固定行序：DuckDB 并行哈希连接的输出顺序在两次运行间会变，
-- 而模型按行号做分层切分 → 行序不定会让同一份数据得到不同的切分与性能估计
ORDER BY c.hadm_id""")
print("  icu_feats:", con.execute("SELECT count(*) FROM icu_feats").fetchone()[0], "行")

# ---------------------------------------------------------------- 合并症
print("\n=== 合并症（入院诊断 ICD）===")
CM = {
    "cm_hf": ["428"], "cm_ckd": ["584", "585", "403"], "cm_dm": ["250"],
    "cm_copd": ["496"], "cm_cirrhosis": ["571"],
    "cm_cancer": ["140", "141", "142", "143", "144", "145", "146", "147", "148", "149", "150",
                  "151", "152", "153", "154", "155", "156", "157", "158", "159", "160", "161",
                  "162", "163", "164", "165", "166", "167", "168", "169", "170", "171", "172",
                  "173", "174", "175", "176", "177", "178", "179", "180", "181", "182", "183",
                  "184", "185", "186", "187", "188", "189", "190", "191", "192", "193", "194",
                  "195", "199", "200", "201", "202", "203", "204", "205", "206", "207", "208",
                  "209", "210", "211", "212", "213", "214", "215", "216", "217", "218", "219",
                  "220", "221", "222", "223", "230", "231", "232", "233", "234", "235", "236",
                  "237", "238", "239", "241"],
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
    con.execute(f"""ALTER TABLE icu_feats ADD COLUMN {k} INTEGER DEFAULT NULL""")
    con.execute(f"""UPDATE icu_feats SET {k} = CASE WHEN hadm_id IN
                    (SELECT hadm_id FROM _cm_{k}) THEN 1 ELSE 0 END""")
print("  合并症标记完成")

# ---------------------------------------------------------------- 数据可得性体检
print("\n=== ICU 入院 0–6h 的数据可得性（分支队列 n=37,145）===")
n = con.execute("SELECT count(*) FROM icu_feats").fetchone()[0]
for lab, col in [("生命体征（任一）", "v_hr_n"), ("床旁 GCS 总分", "v_gcs_total"),
                 ("收缩压(有创+无创)", "v_sbp_n"), ("呼吸频率", "v_rr_n"), ("收缩压", "v_sbp_n"),
                 ("SpO2 脉氧", "v_spo2_pulse_n"), ("体温", "v_tempC_n"),
                 ("肌酐", "l_creatinine_n"), ("乳酸", "l_lactate_n"), ("白细胞", "l_wbc_n"),
                 ("碳酸氢盐", "l_bicarbonate_n"), ("血红蛋白", "l_hgb_n"),
                 ("血小板", "l_platelet_n"), ("钠", "l_sodium_n"), ("钾", "l_potassium_n"),
                 ("血糖", "l_glucose_n"), ("尿素氮", "l_bun_n"), ("白蛋白", "l_albumin_n"),
                 ("胆红素", "l_bilirubin_n"), ("中性粒细胞", "l_neut_n"), ("pH", "l_ph_n")]:
    k = con.execute(f"SELECT count(*) FILTER (WHERE {col} > 0) FROM icu_feats").fetchone()[0]
    print(f"  {lab:14s} {k:>7,}  ({100*k/n:5.1f}%)")
both = con.execute("""SELECT count(*) FROM icu_feats
    WHERE (v_hr_n>0 OR v_rr_n>0 OR v_sbp_n>0 OR v_spo2_pulse_n>0)
      AND (l_creatinine_n>0 OR l_wbc_n>0 OR l_lactate_n>0 OR l_hgb_n>0)""").fetchone()[0]
print(f"  生命体征与实验室都有: {both:,} ({100*both/n:.1f}%)")

con.execute(f"COPY (SELECT * FROM icu_feats) TO '{OUT}/icu_feats.csv' (HEADER)")
print(f"\n已导出 {OUT}/icu_feats.csv")
con.close()
print("[2/3] 完成：ICU 锚点特征已建立")
