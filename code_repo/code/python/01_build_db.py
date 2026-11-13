# -*- coding: utf-8 -*-
"""MIMIC-IV 构建：派生抗生素医嘱表 + 入院期抗生素用药谱盘点。

对应研究问题：在"入院时感染证据不足"的患者中，预测是否需要一入院就"大包围"
（经验性广谱抗生素）。本脚本负责把"大包围"这个临床决策的候选标签量化出来。

输入：D:\\mimic-iv-3\\{hosp,icu}\\*.csv.gz（只读）
参考：MIT-LCP/mimic-code 的 antibiotic.sql（抗生素清单与排除条件，本地副本 antibiotic.sql）
输出：D:\\mimic_iv.duckdb
      studies/infection-sepsis-2026-11/results/mimic_abx_landscape.csv
"""
import duckdb, json, io, os, re, sys, datetime

sys.stdout.reconfigure(encoding="utf-8")
MIMIC = "D:/mimic-iv-3"
DB = "D:/mimic_iv.duckdb"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = "studies/infection-sepsis-2026-11/results"
os.makedirs(OUTDIR, exist_ok=True)

# ---------------------------------------------------------------- 官方抗生素规则
sql = io.open(os.path.join(HERE, "antibiotic.sql"), encoding="utf-8", errors="ignore").read()
PATS = [p for p in re.findall(r"LOWER\(\s*drug\s*\)\s+LIKE\s+'([^']+)'", sql, re.I) if p != "%"]
ROUTE_EXCL = []
m = re.search(r"NOT\s+route\s+IN\s*\(([^)]*)\)", sql, re.I)
if m:
    ROUTE_EXCL = [x.strip().strip("'") for x in m.group(1).split(",") if x.strip()]
ROUTE_LIKE_EXCL = re.findall(r"LOWER\(\s*route\s*\)\s+NOT\s+LIKE\s+'([^']+)'", sql, re.I)
DRUG_EXCL = re.findall(r"LOWER\(\s*drug\s*\)\s+NOT\s+LIKE\s+'([^']+)'", sql, re.I)
DRUG_EXCL = [x for x in DRUG_EXCL if x != "%"]
print("抗生素模式 %d 条；途径排除 %s；药名排除 %s" % (len(PATS), ROUTE_EXCL, DRUG_EXCL))

con = duckdb.connect(DB)
con.execute("PRAGMA threads=4")
con.execute("PRAGMA memory_limit='8GB'")

def load(name, path, cols=None, sample=-1):
    if con.execute("SELECT count(*) FROM information_schema.tables WHERE table_name=?", [name]).fetchone()[0]:
        n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
        print(f"  {name}: 已存在 {n:,} 行，跳过")
        return
    sel = "SELECT * FROM read_csv_auto(?, header=true, sample_size=%d)" % sample if sample > 0 \
          else "SELECT * FROM read_csv_auto(?, header=true)"
    con.execute(f"CREATE TABLE {name} AS {sel}", [path])
    n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
    c = [r[0] for r in con.execute(f"DESCRIBE {name}").fetchall()]
    print(f"  {name}: {n:,} 行，{len(c)} 列")

print("\n=== 载入 hosp/icu 表 ===")
load("admissions", f"{MIMIC}/hosp/admissions.csv.gz")
load("patients", f"{MIMIC}/hosp/patients.csv.gz")
load("prescriptions", f"{MIMIC}/hosp/prescriptions.csv.gz", sample=200000)
load("microbiologyevents", f"{MIMIC}/hosp/microbiologyevents.csv.gz", sample=200000)
load("diagnoses_icd", f"{MIMIC}/hosp/diagnoses_icd.csv.gz", sample=200000)
load("d_icd_diagnoses", f"{MIMIC}/hosp/d_icd_diagnoses.csv.gz")
load("icustays", f"{MIMIC}/icu/icustays.csv.gz")

# ---------------------------------------------------------------- 派生抗生素医嘱
print("\n=== 派生 antibiotics ===")
conds = " OR ".join(["LOWER(p.drug) LIKE '%s'" % p.lower() for p in PATS])
sql_abx = f"""
SELECT p.subject_id, p.hadm_id, p.drug, p.route, p.starttime, p.stoptime,
       a.admittime, a.dischtime, a.admission_type, a.admission_location, a.race,
       a.hospital_expire_flag, a.deathtime
FROM prescriptions p
JOIN admissions a USING (hadm_id)
WHERE {conds}
  AND COALESCE(p.drug_type, '') <> 'BASE'
"""
if ROUTE_EXCL:
    sql_abx += " AND COALESCE(p.route,'') NOT IN (%s)" % ",".join("'%s'" % x for x in ROUTE_EXCL)
for x in ROUTE_LIKE_EXCL:
    sql_abx += " AND LOWER(COALESCE(p.route,'')) NOT LIKE '%s'" % x.lower()
for x in DRUG_EXCL:
    sql_abx += " AND LOWER(p.drug) NOT LIKE '%s'" % x.lower()
print("  途径排除:", ROUTE_EXCL, "| 途径 LIKE 排除:", ROUTE_LIKE_EXCL)
con.execute("DROP TABLE IF EXISTS abx")
con.execute("CREATE TABLE abx AS " + sql_abx)
n = con.execute("SELECT count(*) FROM abx").fetchone()[0]
if n == 0:
    raise RuntimeError("abx 建表后为 0 行——检查模式连接方式（应为 OR 而非 AND）与排除条件")
nh = con.execute("SELECT count(DISTINCT hadm_id) FROM abx").fetchone()[0]
print(f"  抗生素医嘱 {n:,} 条，覆盖 {nh:,} 次住院")

# ---------------------------------------------------------------- 用药谱盘点
print("\n=== 入院期抗生素用药谱 ===")
con.execute("""CREATE OR REPLACE TABLE abx_adm AS
SELECT subject_id, hadm_id, admission_type, admission_location, race,
       hospital_expire_flag, admittime, dischtime, deathtime,
       COUNT(DISTINCT drug) AS n_drug_all,
       COUNT(DISTINCT drug) FILTER (WHERE starttime >= admittime AND starttime < admittime + INTERVAL 24 HOUR) AS n_drug_24h,
       COUNT(DISTINCT drug) FILTER (WHERE starttime >= admittime AND starttime < admittime + INTERVAL 72 HOUR) AS n_drug_72h,
       MIN(starttime) AS t_abx_first
FROM abx GROUP BY 1,2,3,4,5,6,7,8,9""")
print("  有抗生素医嘱的住院:", con.execute("SELECT count(*) FROM abx_adm").fetchone()[0])
print("  其中入院 24h 内即有抗生素医嘱:",
      con.execute("SELECT count(*) FROM abx_adm WHERE n_drug_24h>0").fetchone()[0])
print("  入院 24h 抗生素种类数分布:")
for r in con.execute("""SELECT CAST(CASE WHEN n_drug_24h=0 THEN '0' WHEN n_drug_24h=1 THEN '1'
                                  WHEN n_drug_24h=2 THEN '2' WHEN n_drug_24h=3 THEN '3' ELSE '4+' END AS VARCHAR) k,
                                 count(*) n FROM abx_adm GROUP BY 1 ORDER BY 1""").fetchall():
    print("     ", r)

print("\n  入院 24h 内最常用药 top 30:")
for r in con.execute("""SELECT drug, count(DISTINCT hadm_id) n FROM abx
                        WHERE starttime >= admittime AND starttime < admittime + INTERVAL 24 HOUR
                        GROUP BY 1 ORDER BY n DESC LIMIT 30""").fetchall():
    print(f"     {r[0][:44]:46s} {r[1]:,}")

print("\n  入院 24h 内的关键广谱药使用率（按住院计）:")
KEY = {"meropenem":"碳青霉烯(美罗培南)", "vancomycin":"万古霉素", "ceftazidime":"头孢他啶",
       "piperacillin":"哌拉西林", "linezolid":"利奈唑胺", "gentamicin":"庆大霉素",
       "amikacin":"阿米卡星", "ceftriaxone":"头孢曲松", "cefazolin":"头孢唑林",
       "imipenem":"亚胺培南", "ertapenem":"厄他培南", "tobramycin":"妥布霉素",
       "aztreonam":"阿奇霉氮", "cefoxitin":"头孢西丁"}
for k, cn in KEY.items():
    r = con.execute(f"""SELECT count(DISTINCT hadm_id) FROM abx
                        WHERE LOWER(drug) LIKE '%{k}%'
                        AND starttime >= admittime AND starttime < admittime + INTERVAL 24 HOUR""").fetchone()[0]
    r2 = con.execute(f"""SELECT count(DISTINCT hadm_id) FROM abx
                         WHERE LOWER(drug) LIKE '%{k}%'
                         AND starttime >= admittime AND starttime < admittime + INTERVAL 72 HOUR""").fetchone()[0]
    print(f"     {cn:18s} 24h: {r:,}   72h: {r2:,}")

print("\n  培养送检与病原:")
for q, lab in [("SELECT count(DISTINCT hadm_id) FROM microbiologyevents", "送检过的住院"),
               ("SELECT count(DISTINCT hadm_id) FROM microbiologyevents WHERE chartdate IS NOT NULL", "有 chartdate 的住院")]:
    print(f"     {lab}: {con.execute(q).fetchone()[0]:,}")
con.execute("""CREATE OR REPLACE TABLE micro_pos AS
SELECT subject_id, hadm_id, micro_specimen_id, chartdate, charttime, spec_type_desc,
       org_itemid, org_name, ab_itemid, ab_name, interpretation, isolate_num
FROM microbiologyevents
WHERE NOT (org_itemid IN (90856, 90760) OR org_name IS NULL OR org_name='' OR org_name='CANCELLED')""")
print("     培养阳性记录:", con.execute("SELECT count(*) FROM micro_pos").fetchone()[0])
print("     培养阳性住院:", con.execute("SELECT count(DISTINCT hadm_id) FROM micro_pos").fetchone()[0])
print("     阳性标本类型 top 10:")
for r in con.execute("SELECT spec_type_desc, count(*) n FROM micro_pos GROUP BY 1 ORDER BY n DESC LIMIT 10").fetchall():
    print(f"        {str(r[0])[:40]:42s} {r[1]:,}")

# 导出盘点表
con.execute(f"COPY (SELECT * FROM abx_adm) TO '{OUTDIR}/mimic_abx_landscape.csv' (HEADER, DELIMITER ',')")
print(f"\n已导出 {OUTDIR}/mimic_abx_landscape.csv")
con.close()
print("完成。")
