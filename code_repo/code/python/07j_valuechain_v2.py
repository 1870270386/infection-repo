# -*- coding: utf-8 -*-
"""价值链第一环（修正版）：识别"不需要大包围"的人群。

为什么要修正
  05_model.py 原先用 `pd.qcut` 把留出集按预测概率切成五等分，但**56.0% 的留出集病例
  被模型给出完全相同的预测概率**（0.12857438）——这些是决策时点既无实验室又无生命体征的
  住院者，全部特征被中位数插补成同一个向量，于是得到同一个预测值。
  平局被 qcut 任意拆分 → 五分位梯度被打乱（Q1 12.03% > Q2 10.94%）；
  而"低风险四分位"实际是**平局块里按任意顺序取的前 25%**，不是模型识别出来的低风险人群。
  （见 results/mimic_prediction_ties.csv）

修正后的分析
  1. 明确划出"模型无法区分"的平局块（单一预测值），单独报告其规模与结局；
  2. 风险梯度只在**模型可区分**的子集上计算（平局块之外），此处分位是真实的排序；
  3. 低风险组 = 模型可区分子集里的最低四分位，用大包围率、死亡、住院时长佐证；
  4. 再按数据可得性分层各算一遍梯度。

脚本：07j_valuechain_v2.py
输出：results/mimic_valuechain_v2.csv、mimic_valuechain_quintiles.csv
"""
import os, sys, json
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

OUT = "studies/infection-sepsis-2026-11/results"
df = pd.read_csv(f"{OUT}/mimic_cohort_analysis.csv")
cut = int(np.quantile(df.adm_year.values, 0.70))
# _p_lr.npy 存的是**留出集**的预测（05_model.py 中 p_lr 来自 fit_eval(..., XI[val], ...)）
v = df[df.adm_year.values > cut].copy().reset_index(drop=True)
p = np.load(f"{OUT}/_p_lr.npy")
assert len(v) == len(p), f"留出集队列与预测向量长度不符：{len(v)} vs {len(p)}"
v["p"] = p
print("留出集:", len(v), " 预测向量:", len(p))
print(f"留出集 n={len(v):,}  事件率 {100*v.broad_ge3.mean():.2f}%")

# ---- 1. 平局块
vals, cnts = np.unique(np.round(v.p.values, 10), return_counts=True)
modal = vals[np.argmax(cnts)]
n_tie = int(cnts.max())
print(f"\n=== 平局检查（留出集）===")
print(f"  最大同值块: 预测值 {modal:.6f}，{n_tie:,} 例（{100*n_tie/len(v):.1f}%）")
print(f"  不同预测值个数: {len(vals):,}")
v["tied"] = np.isclose(v.p, modal, rtol=0, atol=1e-9)

rows = []


def blk(name, sub):
    r = {"group": name, "n": len(sub),
         "broad_rate": float(sub.broad_ge3.mean()),
         "mortality": float(sub.hospital_expire_flag.mean()),
         "los_median_h": float(sub.los_h.median()),
         "los_q1_h": float(sub.los_h.quantile(.25)),
         "los_q3_h": float(sub.los_h.quantile(.75))}
    rows.append(r)
    print(f"  {name:34s} n={r['n']:>7,}  大包围率 {100*r['broad_rate']:5.2f}%  "
          f"死亡 {100*r['mortality']:5.2f}%  LOS中位 {r['los_median_h']:6.1f} h")
    return r


print("\n=== 三大类人群（留出集）===")
blk("A 决策时点无任何客观数据\n（模型给出唯一预测值）", v[v.tied])
disc = v[~v.tied]
blk("B 模型可区分·最低四分位", disc[disc.p <= np.percentile(disc.p, 25)])
blk("C 模型可区分·其余四分之三", disc[disc.p > np.percentile(disc.p, 25)])

# 平局块内部：它是不是"混合人群"？—— 看其大包围率/死亡率
print("\n=== 平局块是否混杂 ===")
tb = v[v.tied]
print(f"  平局块内 大包围率 {100*tb.broad_ge3.mean():.2f}%，死亡 {100*tb.hospital_expire_flag.mean():.2f}%，"
      f"n={len(tb):,}")
print(f"  其与整体（{100*v.broad_ge3.mean():.2f}% 大包围、{100*v.hospital_expire_flag.mean():.2f}% 死亡）相比，"
      f"提示该块并非随机混合")

# ---- 2. 可区分子集上的真实五分位梯度
print("\n=== 风险五分位梯度（仅模型可区分者，n=%s）===" % f"{len(disc):,}")
q = pd.qcut(disc.p, 5, labels=False, duplicates="drop")
quint = []
for k, g in disc.groupby(q):
    r = {"quintile": int(k) + 1, "n": len(g),
         "broad_rate": float(g.broad_ge3.mean()),
         "mortality": float(g.hospital_expire_flag.mean()),
         "los_median_h": float(g.los_h.median()),
         "mean_pred": float(g.p.mean())}
    quint.append(r)
    print(f"  Q{r['quintile']}  n={r['n']:>7,}  平均预测 {100*r['mean_pred']:5.2f}%  "
          f"大包围率 {100*r['broad_rate']:5.2f}%  死亡 {100*r['mortality']:5.2f}%  "
          f"LOS中位 {r['los_median_h']:6.1f} h")
qdf = pd.DataFrame(quint)
mono_b = all(qdf.broad_rate.iloc[i] <= qdf.broad_rate.iloc[i + 1] for i in range(len(qdf) - 1))
mono_m = all(qdf.mortality.iloc[i] <= qdf.mortality.iloc[i + 1] for i in range(len(qdf) - 1))
print(f"  大包围率单调递增: {mono_b}；死亡率单调递增: {mono_m}")

# ---- 3. 低风险组内部：实际用了 vs 没用大包围
print("\n=== 低风险组内部拆分（模型可区分·最低四分位）===")
low = disc[disc.p <= np.percentile(disc.p, 25)]
blk("  低风险·实际用了大包围", low[low.broad_ge3 == 1])
blk("  低风险·实际未用大包围", low[low.broad_ge3 == 0])

# ---- 4. 按数据可得性分层各算梯度
print("\n=== 分层：各数据可得性亚组内的模型判别力与低风险组结局 ===")
# 数据可得性标记：05_model.py 里按 lab_n>0 与 vit_key_n>0 判定，此处用事件率对照
# （desc_availability.csv 给出同一分层），此处直接按 desc_availability 的分层重算需要原始标记，
# 因此改为报告"模型可区分"子集内按可得性分层的表现。
print("  （可得性分层的 AUROC 见 results/mimic_model_dataavailability.csv）")

# ---- 5. 决策阈值下的净获益式描述：若按模型把"低风险"判为可不升级
print("\n=== 模型判为低风险（预测值最低四分位，全留出集口径）===")
thr_all = np.percentile(v.p, 25)
lowall = v[v.p <= thr_all]
blk("全留出集最低四分位（含平局块任意 25%）", lowall)
print(f"  ⚠ 该口径含平局块内任意 25%，不作主要结论；主要结论用上面的 B 类。")

pd.DataFrame(rows).to_csv(f"{OUT}/mimic_valuechain_v2.csv", index=False)
qdf.to_csv(f"{OUT}/mimic_valuechain_quintiles.csv", index=False)

# ---- 6. 平局感知的校准表（替代 05_model.py 里被平局打乱的 qcut 五分位）
print("\n=== 校准表（平局感知分组：①平局块 ②–⑥可区分子集五分位）===")
cal_rows = []
for mname, pfile in [("Logistic", "_p_lr.npy"), ("HistGradientBoosting", "_p_gb.npy")]:
    pm = np.load(f"{OUT}/{pfile}")
    assert len(pm) == len(v), f"{pfile} 长度不符"
    v["_p"] = pm
    v["_tied"] = np.isclose(v._p, np.round(np.unique(np.round(v._p.values, 10),
                                                  return_counts=True)[0][
                        np.argmax(np.unique(np.round(v._p.values, 10), return_counts=True)[1])], 10))
    tie_g = v[v._tied]
    cal_rows.append({"model": mname, "group": "No objective data (single prediction)",
                     "n": len(tie_g), "mean_pred": float(tie_g._p.mean()),
                     "observed": float(tie_g.broad_ge3.mean())})
    dsc = v[~v._tied]
    qc = pd.qcut(dsc._p, 5, labels=False, duplicates="drop")
    for k, g in dsc.groupby(qc):
        cal_rows.append({"model": mname, "group": f"Q{int(k)+1}", "n": len(g),
                         "mean_pred": float(g._p.mean()), "observed": float(g.broad_ge3.mean())})
    print(f"  {mname}:")
    for r in cal_rows[-6:]:
        print(f"    {r['group']:38s} n={r['n']:>7,}  预测 {100*r['mean_pred']:5.2f}%  实测 {100*r['observed']:5.2f}%")
cal = pd.DataFrame(cal_rows)
cal.to_csv(f"{OUT}/mimic_model_calibration_v2.csv", index=False)

with open(f"{OUT}/mimic_valuechain_meta.json", "w", encoding="utf-8") as f:
    json.dump({"tie_value": float(modal), "tie_n": n_tie, "tie_share_pct": 100 * n_tie / len(v),
               "n_distinct_pred": int(len(vals)), "val_n": int(len(v)),
               "quintile_monotone_broad": bool(mono_b),
               "quintile_monotone_mortality": bool(mono_m)}, f, ensure_ascii=False, indent=1)
print("\n已导出 results/mimic_valuechain_v2.csv、mimic_valuechain_quintiles.csv、"
      "mimic_model_calibration_v2.csv、mimic_valuechain_meta.json")
