# -*- coding: utf-8 -*-
"""MIMIC-IV：为"入院时感染证据不足 → 是否需要一入院就大包围"这个决策建立标签与人群。

临床问题（用户提出）：院前病史不详细 → 感染证据不足 → 入院治疗不是延用院前方案就是"大包围"；
等待培养太慢，病原宏基因组学不适用于推广。需要一个模型，在入院时就能判断"是否需要大包围"。

本脚本产出：
  1) 目标人群的规模（疑似感染并启动经验性抗生素）
  2) "大包围"的若干可操作定义及其事件率
  3) "感染证据不足"的可操作性：抗生素前有无阳性培养（病原学证据）、入院诊断有无明确感染（临床证据）
  4) 各定义的一致性、结局（死亡/LOS）
"""
import duckdb, os, sys, io
sys.stdout.reconfigure(encoding="utf-8")
OUT = "studies/infection-sepsis-2026-11/results"
os.makedirs(OUT, exist_ok=True)
con = duckdb.connect("D:/mimic_iv.duckdb")
con.execute("PRAGMA threads=4"); con.execute("PRAGMA memory_limit='8GB'")

# ---------------------------------------------------------------- 基础派生
print("=== 派生分析用表 ===")
con.execute("""
CREATE OR REPLACE TABLE adm_abx AS
SELECT a.subject_id, a.hadm_id, a.admittime, a.dischtime, a.admission_type,
       a.admission_location, a.race, a.hospital_expire_flag, a.deathtime,
       p.gender, p.anchor_age, p.dod,
       EXTRACT(EPOCH FROM (a.dischtime - a.admittime))/3600.0 AS los_h,
       (SELECT MIN(b.starttime) FROM abx b WHERE b.hadm_id = a.hadm_id) AS t_abx_first,
       (SELECT COUNT(DISTINCT b.drug) FROM abx b WHERE b.hadm_id = a.hadm_id
          AND b.starttime >= a.admittime AND b.starttime < a.admittime + INTERVAL 24 HOUR) AS n24,
       (SELECT COUNT(DISTINCT b.drug) FROM abx b WHERE b.hadm_id = a.hadm_id
          AND b.starttime >= a.admittime AND b.starttime < a.admittime + INTERVAL 72 HOUR) AS n72,
       (SELECT COUNT(DISTINCT b.drug) FROM abx b WHERE b.hadm_id = a.hadm_id) AS nall
FROM admissions a JOIN patients p USING (subject_id)""")
print("  adm_abx:", con.execute("SELECT count(*) FROM adm_abx").fetchone()[0], "次住院")

# 入院 24h 内的关键药物标记
KEY = {"meropenem": "carbapenem", "vancomycin": "mrsa_vanc", "linezolid": "mrsa_lzd",
       "ceftazidime": "gnr_ceftaz", "piperacillin": "gnr_piptazo", "cefepime": "gnr_cefepime",
       "gentamicin": "amino_gen", "amikacin": "amino_ami", "tobramycin": "amino_tob",
       "ceftriaxone": "ceph3", "cefazolin": "ceph1"}
cols = []
for k in KEY:
    cols.append(f"MAX(CASE WHEN LOWER(d.drug) LIKE '%{k}%' AND d.starttime >= x.admittime "
                f"AND d.starttime < x.admittime + INTERVAL 24 HOUR THEN 1 ELSE 0 END) AS ab_{KEY[k]}")
con.execute(f"""
CREATE OR REPLACE TABLE adm_abx2 AS
SELECT x.*, {', '.join(cols)}
FROM adm_abx x JOIN abx d ON d.hadm_id = x.hadm_id
GROUP BY ALL""")
n = con.execute("SELECT count(*) FROM adm_abx2").fetchone()[0]
print("  adm_abx2:", n)

# 大包围定义
con.execute("""
CREATE OR REPLACE TABLE cohort AS
SELECT *,
  CASE WHEN ab_carbapenem = 1 THEN 1 ELSE 0 END AS broad_carbapenem,
  CASE WHEN (ab_mrsa_vanc = 1 OR ab_mrsa_lzd = 1)
        AND (ab_gnr_ceftaz = 1 OR ab_gnr_piptazo = 1 OR ab_gnr_cefepime = 1 OR ab_carbapenem = 1)
       THEN 1 ELSE 0 END AS broad_mrsa_plus_gnr,
  CASE WHEN ab_carbapenem = 1 AND (ab_mrsa_vanc = 1 OR ab_mrsa_lzd = 1) THEN 1 ELSE 0 END AS broad_carb_vanc,
  CASE WHEN n24 >= 3 THEN 1 ELSE 0 END AS broad_ge3,
  CASE WHEN n24 >= 3 AND (ab_mrsa_vanc = 1 OR ab_mrsa_lzd = 1) THEN 1 ELSE 0 END AS broad_ge3_mrsa,
  CASE WHEN t_abx_first IS NOT NULL
        AND t_abx_first >= admittime AND t_abx_first < admittime + INTERVAL 24 HOUR THEN 1 ELSE 0 END AS empiric_24h
FROM adm_abx2""")
pop = con.execute("SELECT count(*) FROM cohort WHERE empiric_24h = 1").fetchone()[0]
print(f"\n目标人群（入院 24h 内启动经验性抗生素）: {pop:,}")

print("\n=== 各『大包围』定义的阳性率（分母 = 目标人群）===")
for lab, col in [("碳青霉烯", "broad_carbapenem"), ("万古/利奈唑胺 + 抗革兰阴性广谱", "broad_mrsa_plus_gnr"),
                 ("碳青霉烯 + 万古/利奈唑胺", "broad_carb_vanc"), ("入院 24h 内 ≥3 类抗生素", "broad_ge3"),
                 ("≥3 类抗生素 + 含抗 MRSA 药", "broad_ge3_mrsa")]:
    k = con.execute(f"SELECT sum({col}) FROM cohort WHERE empiric_24h=1").fetchone()[0]
    print(f"  {lab:30s} {k:>8,}  ({100*k/pop:5.2f}%)")

print("\n=== 定义间一致性（目标人群内）===")
con.execute("""CREATE OR REPLACE TABLE _d AS SELECT broad_carbapenem c, broad_mrsa_plus_gnr m,
                      broad_ge3 g, broad_ge3_mrsa g3 FROM cohort WHERE empiric_24h=1""")
for r in con.execute("""SELECT c, m, g, g3, count(*) n FROM _d GROUP BY 1,2,3,4
                        ORDER BY n DESC LIMIT 12""").fetchall():
    print("   碳青/万古+GNR/≥3类/≥3类+MRSA =", r)

# 感染证据：抗生素前是否有阳性培养
print("\n=== 『感染证据不足』的可操作性 ===")
con.execute("""CREATE OR REPLACE TABLE micro_agg AS
SELECT hadm_id, MIN(COALESCE(charttime, chartdate)) AS t_first_pos
FROM micro_pos GROUP BY hadm_id""")
con.execute("""CREATE OR REPLACE TABLE cohort2 AS
SELECT c.*, m.t_first_pos,
  CASE WHEN m.t_first_pos IS NOT NULL AND c.t_abx_first IS NOT NULL
        AND m.t_first_pos <= c.t_abx_first THEN 1 ELSE 0 END AS micro_evidence_before_abx
FROM cohort c LEFT JOIN micro_agg m ON c.hadm_id = m.hadm_id""")
for lab, q in [
    ("目标人群", "SELECT count(*) FROM cohort2 WHERE empiric_24h=1"),
    ("抗生素前已有阳性培养（病原学证据充分）", "SELECT count(*) FROM cohort2 WHERE empiric_24h=1 AND micro_evidence_before_abx=1"),
    ("抗生素前无阳性培养（病原学证据不足）", "SELECT count(*) FROM cohort2 WHERE empiric_24h=1 AND micro_evidence_before_abx=0"),
    ("全程无培养记录", "SELECT count(*) FROM cohort2 WHERE empiric_24h=1 AND t_first_pos IS NULL"),
]:
    k = con.execute(q).fetchone()[0]
    print(f"  {lab:34s} {k:>8,}")

print("\n=== 在『病原学证据不足』子群里，各大包围定义的阳性率 ===")
for lab, col in [("碳青霉烯", "broad_carbapenem"), ("万古/利奈唑胺+抗GNR广谱", "broad_mrsa_plus_gnr"),
                 ("碳青+万古/利奈唑胺", "broad_carb_vanc"), ("≥3 类抗生素", "broad_ge3"),
                 ("≥3 类 + 抗MRSA", "broad_ge3_mrsa")]:
    k, tot = con.execute(f"SELECT sum({col}), count(*) FROM cohort2 WHERE empiric_24h=1 AND micro_evidence_before_abx=0").fetchone()
    print(f"  {lab:26s} {k:>7,} / {tot:,}  ({100*k/tot:5.2f}%)")

print("\n=== 结局 ===")
for lab, q in [
    ("院内死亡（目标人群）", "SELECT avg(hospital_expire_flag), sum(hospital_expire_flag) FROM cohort2 WHERE empiric_24h=1"),
    ("住院时长中位 h（目标人群）", "SELECT median(los_h) FROM cohort2 WHERE empiric_24h=1"),
]:
    print("  ", lab, con.execute(q).fetchone())
print("\n  死亡 × 大包围（碳青霉烯）:")
for r in con.execute("""SELECT broad_carbapenem, count(*), avg(hospital_expire_flag), median(los_h)
                         FROM cohort2 WHERE empiric_24h=1 GROUP BY 1""").fetchall():
    print("    carbapenem=%s  n=%s  死亡率=%.3f  LOS中位=%.1f h" % r)

print("\n=== 入院类型 / 科室分布（目标人群）===")
for r in con.execute("SELECT admission_type, count(*) FROM cohort2 WHERE empiric_24h=1 GROUP BY 1 ORDER BY 2 DESC").fetchall():
    print("   ", r)
con.execute(f"COPY (SELECT * FROM cohort2) TO '{OUT}/mimic_cohort_v1.csv' (HEADER)")
print(f"\n已导出 {OUT}/mimic_cohort_v1.csv")
con.close()
