# -*- coding: utf-8 -*-
"""分支【ICU 锚点为主分析】：建立分支专属数据库与 ICU 锚点队列。

本脚本做四件事：
  1) 修正标签缺陷：主线用"不同药品名"计数（`Vancomycin` / `Vancomycin Oral Liquid` /
     `vancomycin` 被算成 3 类；`Metronidazole (Flagyl)` / `MetroNIDAZOLE` /
     `MetronidAZOLE Topical 1 % Gel` 同样被算成 3 类），事件率被高估。
     本分支改为**按抗菌药物类别（class）**计数，并排除局部用/眼用/雾化/肠道准备等非全身用药。
  2) 建立 ICU 锚点队列：住院后 24 小时内收治入 ICU 的疑似感染患者，
     锚点 = ICU 入院时刻，结局 = ICUadm 后 24 h 内使用 ≥3 类抗生素。
  3) 抽取 ICU 锚点窄表（生命体征 / 实验室 / 吸氧），窗口 [intime, intime+24h]。
  4) 导出分支队列与类别映射 CSV。

输出数据库：D:/mimic_iv_branch_icu.duckdb（分支专属；主线库只读挂载，不做任何修改）
"""
import duckdb, os, re, sys, time
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

BRANCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(BRANCH_ROOT, "results")
os.makedirs(OUT, exist_ok=True)
BRANCH_DB = "D:/mimic_iv_branch_icu.duckdb"
SRC_DB = "D:/mimic_iv.duckdb"

for p in (BRANCH_DB, BRANCH_DB + ".wal"):
    if os.path.exists(p):
        os.remove(p)

con = duckdb.connect(BRANCH_DB)
con.execute("PRAGMA threads=4"); con.execute("PRAGMA memory_limit='12GB'")

# 主线库可能正被主线任务占用（Windows 文件锁），加重试
attached = False
for attempt in range(40):
    try:
        con.execute(f"ATTACH '{SRC_DB}' AS src (READ_ONLY)")
        attached = True
        break
    except Exception as e:
        print(f"  挂载主线库失败（第 {attempt+1} 次）：{str(e)[:80]}")
        import time as _t
        _t.sleep(10)
if not attached:
    raise SystemExit("无法挂载主线库，请稍后重试")
print("已只读挂载主线库：", SRC_DB)

# ---------------------------------------------------------------- 0. 复制分支所需表（之后不再依赖主线库）
print("\n=== 复制分支所需表到分支库（一次性）===")
COPY = [
    ("admissions", "SELECT * FROM src.admissions"),
    ("patients", "SELECT * FROM src.patients"),
    ("icustays", "SELECT * FROM src.icustays"),
    ("diagnoses_icd", "SELECT * FROM src.diagnoses_icd"),
    ("d_labitems", "SELECT * FROM src.d_labitems"),
    ("d_items", "SELECT * FROM src.d_items"),
    ("micro_pos", "SELECT * FROM src.micro_pos"),
    ("feats", "SELECT * FROM src.feats"),
    ("abx", "SELECT * FROM src.abx"),
    ("prescriptions", "SELECT hadm_id, subject_id, starttime, drug, drug_type, route FROM src.prescriptions"),
]
for name, q in COPY:
    t0 = time.time()
    con.execute(f"CREATE OR REPLACE TABLE {name} AS {q}")
    n = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
    print(f"  {name:16s} {n:>10,} 行  ({time.time()-t0:.0f}s)")
con.execute("DETACH src")
print("已分离主线库；分支库自此独立")

# ---------------------------------------------------------------- 1. 抗菌药物类别映射
# 原则：按"临床覆盖广度"归类；同一类内的不同商品名只算 1 类。
# 非全身用药（外用/眼用/鼻腔/雾化/口服肠道准备/万古灌流与锁）排除，不计入广谱覆盖。
CLASS_PATTERNS = [
    ("glycopeptide_vanc", [r"vancomycin", r"^vancocin"]),
    ("oxazolidinone_linezolid", [r"linezolid"]),
    ("lipoglycopeptide_dapto", [r"daptomycin"]),
    ("carbapenem", [r"meropenem", r"imipenem", r"ertapenem", r"doripenem"]),
    ("antipseudomonal_bli_ceftaz_avibactam", [r"ceftazidime-avibactam", r"avibactam"]),
    ("antipseudomonal_3gc_ceftazidime", [r"ceftazidime"]),
    ("antipseudomonal_4gc_cefepime", [r"cefepime"]),
    ("antipseudomonal_piptazo", [r"piperacillin"]),
    ("antipseudomonal_aztreonam", [r"aztreonam", r"^cayston"]),
    ("aminoglycoside", [r"gentamicin", r"amikacin", r"tobramycin", r"streptomycin"]),
    ("anti_mrsa_ceftaroline", [r"ceftaroline"]),
    ("bl_bli_ampicillin_sulbactam", [r"ampicillin-sulbactam", r"ampicillin sulbactam"]),
    ("bl_bli_amox_clav", [r"amoxicillin-clavulanate", r"amoxicillin clavulanate", r"^augmentin"]),
    ("penicillin_narrow", [r"^penicillin", r"nafcillin", r"oxacillin", r"dicloxacillin",
                           r"amoxicillin", r"^amoxicill$", r"ampicillin"]),
    ("ceph_1_2gen", [r"cefazolin", r"cephalexin", r"^keflex", r"cefadroxil",
                     r"cefaclor", r"cefprozil"]),
    ("ceph_3gen", [r"ceftriaxone", r"cefotaxime", r"cefpodoxime", r"cefdinir",
                   r"cefuroxime", r"cefixime"]),
    ("ceph_other", [r"^cef[a-z]", r"cefoxitin"]),
    ("fluoroquinolone", [r"ciprofloxacin", r"levofloxacin", r"moxifloxacin",
                         r"ofloxacin", r"norfloxacin", r"^cipro$", r"^ciproflox$",
                         r"^ciprofl$", r"^avelox$"]),
    ("macrolide", [r"azithromycin", r"clarithromycin", r"erythromycin", r"^zithromax",
                   r"^ery-tab$"]),
    ("clindamycin", [r"clindamycin"]),
    ("tetracycline", [r"doxycycline", r"minocycline", r"tetracycline", r"^doxycy$"]),
    ("tmp_smx", [r"sulfameth", r"trimethoprim", r"^bactrim"]),
    ("nitroimidazole_metronidazole", [r"metronidazole", r"tinidazole", r"^flagyl"]),
    ("nitrofurantoin", [r"nitrofurantoin", r"^macrodantin"]),
    ("rifamycin", [r"rifampin", r"rifabutin", r"rifapentine", r"rifaximin"]),
    ("polymyxin_colistin", [r"polymyxin", r"colistin"]),
    ("fosfomycin", [r"fosfomycin"]),
    ("sulfonamide_other", [r"sulfadiazine"]),
    ("chloramphenicol", [r"chloramphenicol"]),
    ("antifungal_polyene_amphotericin", [r"amphotericin", r"^abelcet"]),
    ("antifungal_echinocandin", [r"anidulafungin", r"caspofungin", r"micafungin"]),
    ("other_antibiotic", [r"antibiotic"]),
]
EXCLUDE_PATTERNS = [
    r"ophth", r"eye", r"nasal", r"cream", r"ointment", r"topical", r" gel", r"gel$",
    r"polymyxin-bacitracin", r"mupirocin", r"bactroban", r"silver sulfadiazine",
    r"inhalation", r"inhaled", r"nebuliz", r"enema", r"antibiotic lock",
    r"desensitization", r"vancomycin oral", r"neomycin",
    r"clindamycin 1% solution", r"cleocin t", r"oral solution$",
    r"tivantinib", r"ciprodex",
]
# 类别 → 是否计入"抗菌药物类别数"主标签（抗真菌单列，做敏感性分析）
ANTIFUNGAL = {"antifungal_polyene_amphotericin", "antifungal_echinocandin"}


def classify(drug: str):
    d = re.sub(r"\s+", " ", drug.strip().lower())
    for p in EXCLUDE_PATTERNS:
        if re.search(p, d):
            return None, "non_systemic"
    for cls, pats in CLASS_PATTERNS:
        for p in pats:
            if re.search(p, d):
                return cls, "mapped"
    return None, "UNMAPPED"


print("\n=== 抗菌药物类别映射（基于主线 abx 表的全部药名）===")
src_drugs = con.execute("SELECT DISTINCT drug FROM abx ORDER BY drug").df()
src_drugs["drug_key"] = src_drugs["drug"].map(lambda s: re.sub(r"\s+", " ", s.strip().lower()))
src_drugs["cls"], src_drugs["status"] = zip(*src_drugs["drug"].map(classify))
unmapped = src_drugs[src_drugs.status == "UNMAPPED"]
print(f"  药名总数 {len(src_drugs)}；已映射 {int((src_drugs.status=='mapped').sum())}；"
      f"判为非全身用药 {int((src_drugs.status=='non_systemic').sum())}；未映射 {len(unmapped)}")
if len(unmapped):
    print("  未映射药名（需人工归类）:")
    for d in unmapped.drug.tolist():
        print("     ", d)
src_drugs.to_csv(os.path.join(OUT, "abx_class_map.csv"), index=False)
if len(unmapped):
    raise SystemExit("仍有未映射药名，先补映射再继续")

# ---------------------------------------------------------------- 2. class 级抗菌药物表
print("\n=== 重建 class 级抗菌药物表（prescriptions + 类别映射 + 非全身排除）===")
con.register("map_df", src_drugs[["drug_key", "cls"]].dropna().drop_duplicates(subset="drug_key"))
con.execute("""CREATE TABLE abx_class (hadm_id BIGINT, starttime TIMESTAMP, drug VARCHAR, cls VARCHAR)""")
con.execute("""INSERT INTO abx_class
SELECT p.hadm_id, p.starttime, p.drug, m.cls
FROM prescriptions p
JOIN map_df m ON LOWER(TRIM(p.drug)) = m.drug_key""")
n = con.execute("SELECT count(*) FROM abx_class").fetchone()[0]
print(f"  abx_class: {n:,} 行")
for r in con.execute("SELECT cls, count(*) n, count(DISTINCT hadm_id) adm FROM abx_class "
                      "GROUP BY 1 ORDER BY n DESC").fetchall():
    print(f"     {r[0]:38s} {r[1]:>8,}  涉及住院 {r[2]:>7,}")
print("  映射覆盖率检查：prescriptions 中匹配到映射表的医嘱条数")
chk = con.execute("""SELECT count(*) FROM prescriptions p
    JOIN map_df m ON LOWER(TRIM(p.drug)) = m.drug_key""").fetchone()[0]
print(f"     prescriptions 匹配条数 {chk:,}；abx_class {n:,}；"
      f"一致={'是' if chk == n else '否（存在扇出）'}")

# ---------------------------------------------------------------- 3. ICU 锚点队列
print("\n=== ICU 锚点队列构建 ===")
con.execute("""CREATE OR REPLACE TABLE icu_first AS
SELECT hadm_id, MIN(intime) AS intime FROM icustays GROUP BY hadm_id""")
con.execute("""
CREATE OR REPLACE TABLE icu_cohort AS
SELECT f.hadm_id, f.subject_id, a.admittime, a.dischtime, a.admission_type,
       a.admission_location, a.race, a.hospital_expire_flag, a.deathtime,
       p.gender, p.anchor_age,
       EXTRACT(EPOCH FROM (a.dischtime - a.admittime))/3600.0 AS los_h,
       i.intime AS icu_intime,
       EXTRACT(EPOCH FROM (i.intime - a.admittime))/3600.0 AS adm_to_icu_h,
       (SELECT COUNT(DISTINCT b.drug) FROM abx_class b
          WHERE b.hadm_id = f.hadm_id AND b.starttime >= a.admittime
            AND b.starttime < a.admittime + INTERVAL 24 HOUR) AS n_drug_24h_hosp,
       (SELECT COUNT(DISTINCT b.cls) FROM abx_class b
          WHERE b.hadm_id = f.hadm_id AND b.starttime >= a.admittime
            AND b.starttime < a.admittime + INTERVAL 24 HOUR
            AND b.cls NOT IN ('antifungal_polyene_amphotericin','antifungal_echinocandin'))
            AS n_cls_24h_hosp,
       (SELECT COUNT(DISTINCT b.cls) FROM abx_class b
          WHERE b.hadm_id = f.hadm_id AND b.starttime >= i.intime
            AND b.starttime < i.intime + INTERVAL 24 HOUR
            AND b.cls NOT IN ('antifungal_polyene_amphotericin','antifungal_echinocandin'))
            AS n_cls_24h_icu,
       (SELECT COUNT(DISTINCT b.cls) FROM abx_class b
          WHERE b.hadm_id = f.hadm_id AND b.starttime >= i.intime
            AND b.starttime < i.intime + INTERVAL 24 HOUR) AS n_cls_all_24h_icu,
       (SELECT COUNT(DISTINCT b.cls) FROM abx_class b
          WHERE b.hadm_id = f.hadm_id AND b.starttime >= i.intime
            AND b.starttime < i.intime + INTERVAL 72 HOUR
            AND b.cls NOT IN ('antifungal_polyene_amphotericin','antifungal_echinocandin'))
            AS n_cls_72h_icu,
       (SELECT MIN(b.starttime) FROM abx_class b
          WHERE b.hadm_id = f.hadm_id AND b.starttime >= i.intime
            AND b.starttime < i.intime + INTERVAL 24 HOUR) AS t_abx_icu
FROM feats f
JOIN admissions a ON f.hadm_id = a.hadm_id
JOIN patients p ON f.subject_id = p.subject_id
JOIN icu_first i ON f.hadm_id = i.hadm_id
WHERE f.empiric_24h = 1
  AND i.intime >= a.admittime AND i.intime < a.admittime + INTERVAL 24 HOUR""")
n = con.execute("SELECT count(*) FROM icu_cohort").fetchone()[0]
print(f"  ICU 锚点队列（住院后 24h 内入 ICU，且住院 24h 内启动经验性抗生素）: {n:,}")

KEY = [("meropenem", "carbapenem"), ("vancomycin", "glycopeptide_vanc"),
       ("linezolid", "oxazolidinone_linezolid"), ("ceftazidime", "ceftaz"),
       ("piperacillin", "piptazo"), ("cefepime", "cefepime"), ("gentamicin", "amino"),
       ("amikacin", "amino"), ("tobramycin", "amino"), ("ceftriaxone", "ceph3"),
       ("cefazolin", "ceph12")]
seen = set()
for k, cls in KEY:
    if cls in seen:
        continue
    seen.add(cls)
    con.execute(f"ALTER TABLE icu_cohort ADD COLUMN ab_{cls} INTEGER DEFAULT NULL")
    con.execute(f"""UPDATE icu_cohort SET ab_{cls} = CASE WHEN EXISTS
        (SELECT 1 FROM abx_class b WHERE b.hadm_id = icu_cohort.hadm_id
           AND b.starttime >= icu_cohort.icu_intime
           AND b.starttime < icu_cohort.icu_intime + INTERVAL 24 HOUR
           AND LOWER(b.drug) LIKE '%{k}%') THEN 1 ELSE 0 END""")

con.execute("""
CREATE OR REPLACE TABLE icu_cohort2 AS
SELECT c.*,
  CASE WHEN c.n_cls_24h_icu >= 3 THEN 1 ELSE 0 END AS broad_ge3,
  CASE WHEN c.n_cls_24h_icu >= 2 THEN 1 ELSE 0 END AS broad_ge2,
  CASE WHEN c.ab_carbapenem = 1 THEN 1 ELSE 0 END AS broad_carbapenem,
  CASE WHEN (c.ab_glycopeptide_vanc = 1 OR c.ab_oxazolidinone_linezolid = 1)
        AND (c.ab_ceftaz = 1 OR c.ab_piptazo = 1
             OR c.ab_cefepime = 1 OR c.ab_carbapenem = 1) THEN 1 ELSE 0 END AS broad_mrsa_plus_gnr,
  CASE WHEN c.ab_carbapenem = 1 AND (c.ab_glycopeptide_vanc = 1 OR c.ab_oxazolidinone_linezolid = 1)
       THEN 1 ELSE 0 END AS broad_carb_vanc,
  CASE WHEN c.n_cls_24h_icu >= 3 AND (c.ab_glycopeptide_vanc = 1 OR c.ab_oxazolidinone_linezolid = 1)
       THEN 1 ELSE 0 END AS broad_ge3_mrsa,
  (SELECT MIN(COALESCE(mp.charttime, mp.chartdate)) FROM micro_pos mp
     WHERE mp.hadm_id = c.hadm_id) AS t_first_pos
FROM icu_cohort c""")
con.execute("""ALTER TABLE icu_cohort2 ADD COLUMN micro_evidence_before_abx INTEGER""")
con.execute("""UPDATE icu_cohort2 SET micro_evidence_before_abx =
  CASE WHEN t_first_pos IS NOT NULL AND t_abx_icu IS NOT NULL AND t_first_pos <= t_abx_icu
       THEN 1 ELSE 0 END""")
con.execute("ALTER TABLE icu_cohort2 ADD COLUMN adm_year INTEGER")
con.execute("UPDATE icu_cohort2 SET adm_year = EXTRACT(YEAR FROM admittime)")

print("\n=== 标签口径对比 ===")
k, tot = con.execute("SELECT sum(CASE WHEN n_drug_24h_hosp>=3 THEN 1 ELSE 0 END), count(*) "
                      "FROM icu_cohort2").fetchone()
print(f"  住院锚点 ≥3 个不同药品名（主线口径）  {k:>7,} / {tot:,}  ({100*k/tot:5.2f}%)")
k, tot = con.execute("SELECT sum(CASE WHEN n_drug_24h_hosp>=3 THEN 1 ELSE 0 END), count(*) "
                      "FROM icu_cohort2").fetchone()
k2 = con.execute("SELECT sum(CASE WHEN n_cls_24h_hosp>=3 THEN 1 ELSE 0 END) FROM icu_cohort2").fetchone()[0]
print(f"  住院锚点 ≥3 类抗生素（修正口径）      {k2:>7,} / {tot:,}  ({100*k2/tot:5.2f}%)")
print("\n=== ICU 锚点各『大包围』定义阳性率 ===")
for lab, col in [("≥3 类抗生素（主标签）", "broad_ge3"), ("≥2 类抗生素", "broad_ge2"),
                 ("碳青霉烯", "broad_carbapenem"), ("抗MRSA + 抗GNR广谱", "broad_mrsa_plus_gnr"),
                 ("碳青 + 万古/利奈唑胺", "broad_carb_vanc"), ("≥3 类 + 含抗MRSA", "broad_ge3_mrsa")]:
    k = con.execute(f"SELECT sum({col}) FROM icu_cohort2").fetchone()[0]
    print(f"  {lab:26s} {k:>7,}  ({100*k/n:5.2f}%)")

print("\n=== 感染证据充分性（ICU 锚点）===")
for lab, q in [("抗生素前已有阳性培养", "SELECT count(*) FROM icu_cohort2 WHERE micro_evidence_before_abx=1"),
               ("抗生素前无阳性培养", "SELECT count(*) FROM icu_cohort2 WHERE micro_evidence_before_abx=0"),
               ("全程无阳性培养", "SELECT count(*) FROM icu_cohort2 WHERE t_first_pos IS NULL")]:
    k = con.execute(q).fetchone()[0]
    print(f"  {lab:22s} {k:>7,}  ({100*k/n:5.2f}%)")

print("\n=== 结局与时间结构 ===")
print("  住院→ICU 时间(h) P25/P50/P75:", con.execute(
    "SELECT quantile_cont(adm_to_icu_h,0.25), quantile_cont(adm_to_icu_h,0.5), "
    "quantile_cont(adm_to_icu_h,0.75) FROM icu_cohort2").fetchone())
print("  院内死亡:", con.execute(
    "SELECT sum(hospital_expire_flag), round(100.0*avg(hospital_expire_flag),2) FROM icu_cohort2").fetchone())
print("  住院时长中位(h):", con.execute("SELECT median(los_h) FROM icu_cohort2").fetchone()[0])
print("  按入院年份:")
for r in con.execute("SELECT adm_year, count(*), sum(broad_ge3), round(100.0*avg(broad_ge3),2) "
                      "FROM icu_cohort2 GROUP BY 1 ORDER BY 1").fetchall():
    print(f"     {r[0]}  n={r[1]:>7,}  事件 {r[2]:>6,}  ({r[3]}%)")

con.execute(f"COPY (SELECT * FROM icu_cohort2) TO '{OUT}/icu_cohort.csv' (HEADER)")
print(f"\n已导出 {OUT}/icu_cohort.csv")
con.close()
print("[1/3] 完成：分支数据库与 ICU 队列已建立")
