# -*- coding: utf-8 -*-
"""补齐手稿数字溯源所需的结果表：
  desc_flow.csv        纳排流程各层人数（546,028 / 241,281 / 184,645 / 排除数）
  desc_dateshift.csv   去标识化日期平移的三项核验（秩相关、顺序一致率、LOS 与再入院间隔）
  desc_missingness.csv 0–6 h 各候选变量的缺失率（给出最小/最大）
脚本：07g_descriptive_extra.py
"""
import duckdb, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
OUT = "studies/infection-sepsis-2026-11/results"

con = duckdb.connect("D:/mimic_iv.duckdb", read_only=True)
q = lambda s: con.execute(s).fetchall()

# ---------------- 纳排流程 ----------------
n_all = q("SELECT count(*) FROM admissions")[0][0]
n_abx = q("SELECT count(DISTINCT hadm_id) FROM abx")[0][0]
n_tgt = q("SELECT count(*) FROM feats2 WHERE empiric_24h=1")[0][0]
n_late = n_abx - n_tgt
n_none = n_all - n_abx
flow = pd.DataFrame([
    {"stage": "All MIMIC-IV v3.0 admissions", "n": n_all, "excluded_at_this_step": 0},
    {"stage": "Admissions with any systemic antibiotic order", "n": n_abx,
     "excluded_at_this_step": n_all - n_abx},
    {"stage": "Antibiotic started within 24 h of admission (study population)", "n": n_tgt,
     "excluded_at_this_step": n_late},
])
flow["excluded_reason"] = ["", "No systemic antibiotic order",
                           "First antibiotic started later than 24 h after admission"]
flow.to_csv(f"{OUT}/desc_flow.csv", index=False)
print(f"纳排流程: {n_all:,} → {n_abx:,} → {n_tgt:,}（排除无抗生素医嘱 {n_none:,}；排除首剂晚于 24h {n_late:,}）")

# ---------------- 去标识化日期平移核验 ----------------
rho = q("""WITH r AS (SELECT rank() OVER (ORDER BY hadm_id) rh,
                          rank() OVER (ORDER BY admittime) ra FROM admissions)
           SELECT corr(rh, ra) FROM r""")[0][0]
pairs, same = q("""SELECT count(*), sum(CASE WHEN t1.admittime < t2.admittime THEN 1 ELSE 0 END)
                   FROM admissions t1 JOIN admissions t2
                   ON t1.subject_id=t2.subject_id AND t1.hadm_id < t2.hadm_id""")[0]
gap_pos, gap_med, n_gap = q("""WITH p AS (SELECT subject_id, hadm_id, admittime, dischtime,
        ROW_NUMBER() OVER (PARTITION BY subject_id ORDER BY admittime) rn FROM admissions),
   x AS (SELECT a.dischtime, b.admittime FROM p a JOIN p b ON a.subject_id=b.subject_id AND b.rn=a.rn+1)
   SELECT sum(CASE WHEN admittime>dischtime THEN 1 ELSE 0 END),
          median(EXTRACT(EPOCH FROM (admittime-dischtime))/3600.0), count(*) FROM x""")[0]
los_neg, los_gt30 = q("""SELECT sum(CASE WHEN los_h<0 THEN 1 ELSE 0 END),
        sum(CASE WHEN los_h>24*30 THEN 1 ELSE 0 END) FROM
        (SELECT EXTRACT(EPOCH FROM (dischtime-admittime))/3600.0 los_h FROM admissions)""")[0]
n_adm = n_all
ds = pd.DataFrame([
    {"metric": "Spearman rank correlation between hadm_id and admittime, whole database",
     "value": float(rho), "unit": "correlation", "denominator": n_adm},
    {"metric": "Same-patient admission pairs", "value": int(pairs), "unit": "pairs", "denominator": pairs},
    {"metric": "Same-patient pairs whose shifted admission times keep the true order",
     "value": round(100 * same / pairs, 2), "unit": "percent", "denominator": pairs},
    {"metric": "Readmission intervals positive after ordering admissions by admission time",
     "value": round(100 * gap_pos / n_gap, 2), "unit": "percent", "denominator": n_gap},
    {"metric": "Median discharge-to-next-admission interval", "value": round(float(gap_med), 1),
     "unit": "hours", "denominator": n_gap},
    {"metric": "Admissions with negative length of stay", "value": round(100 * los_neg / n_adm, 3),
     "unit": "percent", "denominator": n_adm},
    {"metric": "Admissions with length of stay longer than 30 days",
     "value": round(100 * los_gt30 / n_adm, 3), "unit": "percent", "denominator": n_adm},
])
ds.to_csv(f"{OUT}/desc_dateshift.csv", index=False)
print("\n日期平移核验:")
print(ds.to_string(index=False))

# ---------------- 缺失率 ----------------
con.close()
df = pd.read_csv(f"{OUT}/mimic_features.csv") if os.path.exists(f"{OUT}/mimic_features.csv") else None
con2 = duckdb.connect("D:/mimic_iv.duckdb", read_only=True)
cols = [c[0] for c in con2.execute(
    "SELECT column_name FROM duckdb_columns() WHERE table_name='feats2'").fetchall()]
num = [c for c in cols if (c.startswith("l_") or c.startswith("v_")) and not c.endswith("_n")
       and c not in ("v_gcseye_mean", "v_gcseye_min", "v_gcseye_max", "v_gcseye_last",
                     "v_gcsverbal_mean", "v_gcsverbal_min", "v_gcsverbal_max", "v_gcsverbal_last",
                     "v_gcsmotor_mean", "v_gcsmotor_min", "v_gcsmotor_max", "v_gcsmotor_last")]
miss = []
tgt = con2.execute("SELECT * FROM feats2 WHERE empiric_24h=1").df()
for c in num:
    miss.append({"variable": c, "missing_pct": round(100 * tgt[c].isna().mean(), 1)})
con2.close()
mdf = pd.DataFrame(miss).sort_values("missing_pct")
mdf.to_csv(f"{OUT}/desc_missingness.csv", index=False)
print(mdf.head(6).to_string(index=False))
print(f"\n0–6 h 候选变量缺失率：最小 {mdf.missing_pct.min()}%，最大 {mdf.missing_pct.max()}%"
      f"（{len(mdf)} 个变量）")
