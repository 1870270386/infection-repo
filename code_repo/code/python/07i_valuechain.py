# -*- coding: utf-8 -*-
"""价值链第一环：按预测风险分组（**按秩分组**，不用概率阈值），并量化预测概率的并列现象。

为什么要改口径
  05_model.py 原按"概率 ≤ 留出集 P25"定义低风险组，但 60.0% 的患者在决策时点没有任何客观数据，
  模型对他们只剩 2 个合并症二值变量可用（cm_pneumonia、cm_sepsis_adm），预测概率大量并列，
  于是 P25 这条阈值线之下其实站了 65.4% 的人——按数值阈值分组会被并列扭曲。
  本脚本改按**秩**分组：最低风险四分位 = 预测概率排序后的前 25%。

输出：results/mimic_valuechain.csv（覆盖 05 的同名文件）、results/mimic_prediction_ties.csv
脚本：07i_valuechain.py
"""
import duckdb, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
OUT = "studies/infection-sepsis-2026-11/results"

con = duckdb.connect("D:/mimic_iv.duckdb", read_only=True)
df = con.execute("SELECT * FROM feats2 WHERE empiric_24h=1").df()
con.close()
Y = df.broad_ge3.values.astype(int)
cut = int(np.quantile(df.adm_year.values, 0.70))
val = df.adm_year.values > cut
p = np.load(f"{OUT}/_p_lr.npy")
assert len(p) == int(val.sum())
v = df[val].copy(); v["p"] = p
v["rank"] = v.p.rank(method="first")

n = len(v)
q = pd.qcut(v["rank"], 5, labels=False, duplicates="drop")
v["risk_q"] = q
rows = []
for k, g in v.groupby("risk_q"):
    rows.append({"group": f"风险 Q{int(k)+1}", "n": len(g),
                 "broad_rate": float(g.broad_ge3.mean()),
                 "mortality": float(g.hospital_expire_flag.mean()),
                 "los_median_h": float(g.los_h.median())})
lowmask = v["rank"] <= int(round(0.25 * n))
v["lowrisk"] = lowmask
for lab, g in [("低风险四分位", v[lowmask]), ("其余人群", v[~lowmask])]:
    rows.append({"group": lab, "n": len(g), "broad_rate": float(g.broad_ge3.mean()),
                 "mortality": float(g.hospital_expire_flag.mean()),
                 "los_median_h": float(g.los_h.median())})
low = v[lowmask]
for lab, g in [("低风险组内·实际用了大包围", low[low.broad_ge3 == 1]),
               ("低风险组内·实际未用大包围", low[low.broad_ge3 == 0])]:
    if len(g) == 0:
        continue
    rows.append({"group": lab, "n": len(g), "broad_rate": float(g.broad_ge3.mean()),
                 "mortality": float(g.hospital_expire_flag.mean()),
                 "los_median_h": float(g.los_h.median())})
out = pd.DataFrame(rows)
out.to_csv(f"{OUT}/mimic_valuechain.csv", index=False)
print("=== 价值链（按秩分组，留出集 n=%s）===" % f"{n:,}")
for _, r in out.iterrows():
    print(f"  {r.group:24s} n={int(r.n):>7,}  大包围率 {100*r.broad_rate:5.2f}%  "
          f"死亡 {100*r.mortality:5.2f}%  LOS中位 {r.los_median_h:6.1f} h")

# ---- 预测概率并列现象
vc = v.p.value_counts()
ties = pd.DataFrame({
    "distinct_predicted_probability": vc.index[:10],
    "n_admissions": vc.values[:10],
    "share_pct": (100 * vc.values[:10] / n).round(2),
})
ties.to_csv(f"{OUT}/mimic_prediction_ties.csv", index=False)
print(f"\n预测概率的并列：留出集 {n:,} 例只有 {v.p.nunique():,} 个不同的预测概率值；"
      f"众数覆盖 {vc.iloc[0]:,} 例（{100*vc.iloc[0]/n:.1f}%）")
print(ties.head(6).to_string(index=False))
# 无客观数据者的并列
_lab_n = [c for c in df.columns if c.startswith("l_") and c.endswith("_n")]
_vit_k = [c for c in ["v_hr_n","v_rr_n","v_sbp_n","v_nibp_sbp_n","v_spo2_pls_n","v_spo2_art_n"] if c in df.columns]
_hl = (df[_lab_n].sum(axis=1) > 0).values[val]
_hv = (df[_vit_k].sum(axis=1) > 0).values[val]
v["no_data"] = (~_hl) & (~_hv)
print(f"\n无任何客观数据者（n={int(v.no_data.sum()):,}）：不同预测概率值仅 {v.loc[v.no_data,'p'].nunique():,} 个，"
      f"众数覆盖 {v.loc[v.no_data,'p'].value_counts().iloc[0]:,} 例")
