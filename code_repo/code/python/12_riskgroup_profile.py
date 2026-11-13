# -*- coding: utf-8 -*-
"""各预测风险组的临床画像：回答"哪一部分人需要大包围、哪一部分不需要"。

做法
  留出集内按模型定义分组（共享单一预测值组 + 模型可区分者五分位），
  对每个组算关键临床指标的中位数（IQR），供正文"临床落地"一节直接引用。
  只读 feats2 与 _p_lr.npy，不重新拟合模型。

输出：results/mimic_riskgroup_profile.csv
"""
import duckdb, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

S = "studies/infection-sepsis-2026-11/results"

con = duckdb.connect("D:/mimic_iv.duckdb", read_only=True)
df = con.execute("SELECT * FROM feats2 WHERE empiric_24h=1").df()
con.close()

p = np.load(f"{S}/_p_lr.npy")
cut = int(np.quantile(df.adm_year.values, 0.70))
v = df[df.adm_year.values > cut].reset_index(drop=True)
assert len(v) == len(p), (len(v), len(p))
v["p"] = p

vals, cnts = np.unique(np.round(v.p.values, 10), return_counts=True)
modal = vals[np.argmax(cnts)]
v["tied"] = np.isclose(v.p, modal, rtol=0, atol=1e-9)

disc = v[~v.tied].copy()
disc["q"] = pd.qcut(disc.p, 5, labels=False, duplicates="drop") + 1
v.loc[~v.tied, "q"] = disc["q"].values
v.loc[v.tied, "q"] = 0

VARS = {
    "v_rr_max": "Respiratory rate, max (breaths/min)",
    "v_spo2_pls_min": "SpO2, min (%)",
    "v_sbp_min": "Systolic BP, min (mmHg)",
    "v_hr_max": "Heart rate, max (beats/min)",
    "v_gcs_total_min": "Glasgow Coma Scale, min",
    "v_tempC_max": "Temperature, max (°C)",
    "l_lactate_max": "Lactate, max (mmol/L)",
    "l_albumin_min": "Albumin, min (g/dL)",
    "l_bicarbonate_min": "Bicarbonate, min (mmol/L)",
    "l_bun_max": "Urea nitrogen, max (mg/dL)",
    "l_creatinine_max": "Creatinine, max (mg/dL)",
    "l_wbc_max": "White cell count, max (10⁹/L)",
    "l_platelet_min": "Platelet count, min (10⁹/L)",
    "l_bilirubin_max": "Total bilirubin, max (mg/dL)",
}
VARS = {k: lab for k, lab in VARS.items() if k in v.columns}

rows = []
for q, g in v.groupby("q"):
    label = "No objective data (single prediction)" if q == 0 else f"Predicted-risk Q{int(q)}"
    rec = {"group": label, "n": len(g),
           "broad_rate_pct": round(100 * g.broad_ge3.mean(), 2),
           "mortality_pct": round(100 * g.hospital_expire_flag.mean(), 2),
           "los_median_h": round(g.los_h.median(), 1),
           "mean_pred_pct": round(100 * g.p.mean(), 2)}
    for k, lab in VARS.items():
        s = g[k].dropna()
        rec[k] = "" if len(s) < 30 else f"{s.median():.2f} ({s.quantile(.25):.2f}-{s.quantile(.75):.2f})"
        rec[k + "__n"] = len(s)
    rows.append(rec)

out = pd.DataFrame(rows).sort_values("group")
out.to_csv(f"{S}/mimic_riskgroup_profile.csv", index=False)

print("=== 各风险组临床画像（中位数，IQR）===")
show = ["group", "n", "mean_pred_pct", "broad_rate_pct", "mortality_pct",
        "l_lactate_max", "v_gcs_total_min", "l_albumin_min", "l_bicarbonate_min",
        "l_bun_max", "v_rr_max", "v_spo2_pls_min", "v_sbp_min"]
show = [c for c in show if c in out.columns]
print(out[show].to_string(index=False))
print("\n已导出", f"{S}/mimic_riskgroup_profile.csv")
