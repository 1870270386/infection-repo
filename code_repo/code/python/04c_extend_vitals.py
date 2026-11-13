# -*- coding: utf-8 -*-
"""MIMIC-IV：补抽三类缺失的关键变量，并生成 feats2（= feats + 新列）。

补抽原因（均为已核实的真实缺陷）
  1) **无创血压**：原 cv_vitals 只抽了有创血压 itemid 220050，NIBP(220179/220180/220181)
     完全没抽，导致收缩压覆盖被严重低估——而血压是 NEWS2 与脓毒症评分的主要成分。
  2) **GCS**：原代码用的 226755/227013 是 APACHE II/IV 的**计算分**，全库仅 8 行；
     真正的床旁 GCS 由三个分量构成——睁眼 220739、言语 223900、运动 223901。
  3) **吸氧指征**：NEWS2 的"吸氧 +2 分"分量需要 O2 Flow(223834)/
     Inspired O2 Fraction(223835)/ O2 Delivery Device(226732)。

输出：D:/mimic_iv.duckdb 的表 cv_vitals_ext、vit_ext、o2_flag、feats2
"""
import duckdb, os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
MIMIC = "D:/mimic-iv-3"
con = duckdb.connect("D:/mimic_iv.duckdb")
con.execute("PRAGMA threads=4"); con.execute("PRAGMA memory_limit='10GB'")

EXTRA = {
    "nibp_sbp": [220179], "nibp_dbp": [220180], "nibp_map": [220181],
    "gcseye": [220739], "gcsverbal": [223900], "gcsmotor": [223901],
    "o2flow": [223834], "o2fio2": [223835], "o2dev": [226732],
}

t0 = time.time()
ids = ",".join(str(i) for v in EXTRA.values() for i in v)
have = con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name='cv_vitals_ext'").fetchone()[0]
if not have:
    con.execute(f"""
CREATE OR REPLACE TABLE cv_vitals_ext AS
SELECT c.subject_id, c.hadm_id, c.charttime, c.itemid, c.valuenum
FROM read_csv_auto('{MIMIC}/icu/chartevents.csv.gz', header=true, escape='"',
                   columns={{'subject_id':'BIGINT','hadm_id':'BIGINT','stay_id':'BIGINT',
                             'caregiver_id':'BIGINT','charttime':'TIMESTAMP','storetime':'TIMESTAMP',
                             'itemid':'INTEGER','value':'VARCHAR','valuenum':'DOUBLE',
                             'valueuom':'VARCHAR','warning':'VARCHAR'}}) c
JOIN admissions a ON c.hadm_id = a.hadm_id
WHERE c.itemid IN ({ids})
  AND c.charttime >= a.admittime AND c.charttime < a.admittime + INTERVAL 24 HOUR""")
else:
    print("cv_vitals_ext 已存在，跳过重扫")
n = con.execute("SELECT count(*) FROM cv_vitals_ext").fetchone()[0]
print(f"cv_vitals_ext: {n:,} 行  ({time.time()-t0:.0f}s)")
for r in con.execute("""SELECT itemid, count(*) n, count(valuenum) nv, min(valuenum), max(valuenum)
                        FROM cv_vitals_ext GROUP BY 1 ORDER BY n DESC""").fetchall():
    print("   ", r)

# ---- 0–6h 聚合（与 feats 的窗口一致）
def agg(spec, prefix):
    sel = []
    for k, ids_ in spec.items():
        idl = ",".join(str(i) for i in ids_)
        cond = f"z.itemid IN ({idl})"
        sel += [
            f"COUNT(*) FILTER (WHERE {cond}) AS {prefix}_{k}_n",
            f"AVG(z.valuenum) FILTER (WHERE {cond}) AS {prefix}_{k}_mean",
            f"MIN(z.valuenum) FILTER (WHERE {cond}) AS {prefix}_{k}_min",
            f"MAX(z.valuenum) FILTER (WHERE {cond}) AS {prefix}_{k}_max",
            f"ARG_MAX(z.valuenum, z.charttime) FILTER (WHERE {cond}) AS {prefix}_{k}_last",
        ]
    return ",\n".join(sel)

VIT_EXT = {k: v for k, v in EXTRA.items() if not k.startswith("o2")}
con.execute(f"""CREATE OR REPLACE TABLE vit_ext AS
SELECT z.hadm_id, {agg(VIT_EXT, 'v')}
FROM cv_vitals_ext z JOIN admissions a ON z.hadm_id = a.hadm_id
WHERE z.charttime >= a.admittime AND z.charttime < a.admittime + INTERVAL 6 HOUR
GROUP BY z.hadm_id""")

# GCS 总分：三个分量齐全才求和（否则置空）
for s in ["mean", "min", "max", "last"]:
    con.execute(f"ALTER TABLE vit_ext ADD COLUMN IF NOT EXISTS v_gcs_total_{s} DOUBLE")
con.execute("""UPDATE vit_ext SET v_gcs_total_mean = v_gcseye_mean + v_gcsverbal_mean + v_gcsmotor_mean
                WHERE v_gcseye_mean IS NOT NULL AND v_gcsverbal_mean IS NOT NULL AND v_gcsmotor_mean IS NOT NULL""")
con.execute("""UPDATE vit_ext SET v_gcs_total_min = v_gcseye_min + v_gcsverbal_min + v_gcsmotor_min
                WHERE v_gcseye_min IS NOT NULL AND v_gcsverbal_min IS NOT NULL AND v_gcsmotor_min IS NOT NULL""")
con.execute("""UPDATE vit_ext SET v_gcs_total_max = v_gcseye_max + v_gcsverbal_max + v_gcsmotor_max
                WHERE v_gcseye_max IS NOT NULL AND v_gcsverbal_max IS NOT NULL AND v_gcsmotor_max IS NOT NULL""")
con.execute("""UPDATE vit_ext SET v_gcs_total_last = v_gcseye_last + v_gcsverbal_last + v_gcsmotor_last
                WHERE v_gcseye_last IS NOT NULL AND v_gcsverbal_last IS NOT NULL AND v_gcsmotor_last IS NOT NULL""")

# 吸氧标记
con.execute("""CREATE OR REPLACE TABLE o2_flag AS
SELECT hadm_id,
       MAX(CASE WHEN itemid = 223834 AND valuenum > 0 THEN 1 ELSE 0 END) AS on_o2_flow,
       MAX(CASE WHEN itemid = 223835 AND valuenum > 0.21 THEN 1 ELSE 0 END) AS on_o2_fio2,
       MAX(CASE WHEN itemid = 226732 THEN 1 ELSE 0 END) AS on_o2_device
FROM cv_vitals_ext GROUP BY hadm_id""")

# ---- 合并进 feats2
con.execute("""CREATE OR REPLACE TABLE feats2 AS
SELECT f.*, e.* EXCLUDE (hadm_id),
       CASE WHEN COALESCE(o.on_o2_flow,0)+COALESCE(o.on_o2_fio2,0)+COALESCE(o.on_o2_device,0) > 0
            THEN 1 ELSE 0 END AS on_o2
FROM feats f
LEFT JOIN vit_ext e ON f.hadm_id = e.hadm_id
LEFT JOIN o2_flag o ON f.hadm_id = o.hadm_id""")
print("\nfeats2:", con.execute("SELECT count(*) FROM feats2").fetchone()[0], "行")

print("\n=== 目标人群：新增变量的覆盖与关键校验 ===")
for c in ["v_nibp_sbp_n", "v_gcseye_n", "v_gcs_total_mean", "on_o2"]:
    r = con.execute(f"""SELECT count({c}), min({c}), quantile_cont({c},0.5), max({c})
                         FROM feats2 WHERE empiric_24h=1""").fetchone()
    print(f"  {c:20s} 非空 {r[0]:>7,} ({100*r[0]/184645:5.1f}%)  min {r[1]}  中位 {r[2]}  max {r[3]}")
print("  GCS 总分分布（0–6h 有值者）:",
      con.execute("""SELECT quantile_cont(v_gcs_total_min,0.25), quantile_cont(v_gcs_total_min,0.5),
                            quantile_cont(v_gcs_total_min,0.75), min(v_gcs_total_min), max(v_gcs_total_min)
                     FROM feats2 WHERE empiric_24h=1 AND v_gcs_total_min IS NOT NULL""").fetchone())
print("  有创 SBP 或 NIBP 覆盖:",
      con.execute("""SELECT count(*) FROM feats2 WHERE empiric_24h=1
                     AND (v_sbp_n>0 OR v_nibp_sbp_n>0)""").fetchone()[0])
print("  原仅有创 SBP 覆盖:",
      con.execute("""SELECT count(*) FROM feats2 WHERE empiric_24h=1 AND v_sbp_n>0""").fetchone()[0])
con.close()
