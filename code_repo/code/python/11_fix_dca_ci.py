# -*- coding: utf-8 -*-
"""审核意见修正：决策曲线公式、bootstrap 区间口径、校准偏差、首剂时点分布。

修正内容（对应审核团 critical/major）
  1) **决策曲线**：05_model.py 原实现 `w=y/p; tp=(w>=t)` 等价于 y>=t*p，
     对任何 t<1 都把**全部阳性**计入 tp（t=0.20 时 tp=9,618 = 全部事件），
     真阳性被高估、净获益失真。本脚本按 Vickers 定义重算：
        pred = p >= t ; tp = Σ(pred & y=1) ; fp = Σ(pred & y=0)
        NB = tp/N - (fp/N) * t/(1-t)
     并给出 treat-all 与 treat-none 参照。
  2) **AUROC 的 95%CI**：原 bootstrap 只重抽开发集、验证集固定，区间只反映
     "模型拟合"的变异（宽 0.003），不能当成 AUROC 的抽样区间。此处另算
     **留出集非参数 bootstrap**（2000 次，重抽验证集本身），并保留原口径作为乐观度参考。
  3) **校准偏差**：逐组算 |实测-预测| 的最大值，供正文如实报告。
  4) **首剂时点**：报告首剂抗生素距入院的小时数分布，以及延迟 >6 h 子集的比例与判别力
     （预测窗口内已用药者占多数，属设计局限，须如实披露）。

输出：results/mimic_model_dca.csv（覆盖）、mimic_model_bootstrap_holdout.csv、
      mimic_first_dose_timing.csv；并打印校准偏差与子集判别力。
"""
import json, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.metrics import roc_auc_score, average_precision_score

S = "studies/infection-sepsis-2026-11/results"
SEED = 20261105
rng = np.random.default_rng(SEED)

d = pd.read_csv(f"{S}/mimic_cohort_analysis.csv")
p_lr = np.load(f"{S}/_p_lr.npy"); p_gb = np.load(f"{S}/_p_gb.npy")
cut = int(np.quantile(d.adm_year.values, 0.70))
v = d[d.adm_year.values > cut].reset_index(drop=True)
assert len(v) == len(p_lr) == len(p_gb)
y = v.broad_ge3.values.astype(int); N = len(y)
v["p_lr"] = p_lr; v["p_gb"] = p_gb

# ---------------------------------------------------------------- 1. 决策曲线（修正）
def dca(y, p, grid):
    out = []
    for t in grid:
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        nb = tp / len(y) - fp / len(y) * t / (1 - t)
        nb_all = y.sum() / len(y) - (len(y) - y.sum()) / len(y) * t / (1 - t)
        out.append({"threshold": t, "net_benefit": nb, "treat_all": nb_all, "treat_none": 0.0,
                    "tp": tp, "fp": fp, "n_predicted": int(pred.sum())})
    return pd.DataFrame(out)

grid = np.round(np.arange(0.05, 0.605, 0.01), 2)
rows = []
for nm, p in [("Logistic", p_lr), ("HistGradientBoosting", p_gb)]:
    dd = dca(y, p, grid); dd.insert(0, "model", nm); rows.append(dd)
dca_tab = pd.concat(rows, ignore_index=True)
dca_tab.to_csv(f"{S}/mimic_model_dca.csv", index=False)
print("=== 决策曲线（修正后）===")
for nm, p in [("Logistic", p_lr), ("HistGradientBoosting", p_gb)]:
    dd = dca_tab[dca_tab.model == nm]
    gain = dd.net_benefit - dd.treat_all
    i = int(gain.idxmax())
    r10 = dd[np.isclose(dd.threshold, 0.10)].iloc[0]
    r20 = dd[np.isclose(dd.threshold, 0.20)].iloc[0]
    print(f"  {nm:20s} 阈0.10: 模型 NB {r10.net_benefit:.4f} vs treat-all {r10.treat_all:.4f}"
          f" | 阈0.20: 模型 {r20.net_benefit:.4f} vs treat-all {r20.treat_all:.4f}"
          f" | 最大增益 {gain.loc[i]:+.4f} @ t={dd.loc[i,'threshold']:.2f}")

# ---------------------------------------------------------------- 2. 留出集 bootstrap 区间
print("\n=== 留出集非参数 bootstrap（2000 次，重抽验证集）===")
boot = []
for b in range(2000):
    idx = rng.integers(0, N, N)
    if len(set(y[idx])) < 2:
        continue
    boot.append((roc_auc_score(y[idx], p_lr[idx]), average_precision_score(y[idx], p_lr[idx]),
                 roc_auc_score(y[idx], p_gb[idx]), average_precision_score(y[idx], p_gb[idx])))
boot = np.array(boot)
ci = pd.DataFrame([
    {"model": "Logistic", "reps": len(boot),
     "auroc": roc_auc_score(y, p_lr), "auroc_lo": np.percentile(boot[:, 0], 2.5), "auroc_hi": np.percentile(boot[:, 0], 97.5),
     "auprc": average_precision_score(y, p_lr), "auprc_lo": np.percentile(boot[:, 1], 2.5), "auprc_hi": np.percentile(boot[:, 1], 97.5)},
    {"model": "HistGradientBoosting", "reps": len(boot),
     "auroc": roc_auc_score(y, p_gb), "auroc_lo": np.percentile(boot[:, 2], 2.5), "auroc_hi": np.percentile(boot[:, 2], 97.5),
     "auprc": average_precision_score(y, p_gb), "auprc_lo": np.percentile(boot[:, 3], 2.5), "auprc_hi": np.percentile(boot[:, 3], 97.5)},
])
ci.to_csv(f"{S}/mimic_model_bootstrap_holdout.csv", index=False)
for _, r in ci.iterrows():
    print(f"  {r.model:20s} AUROC {r.auroc:.4f} (95%CI {r.auroc_lo:.4f}–{r.auroc_hi:.4f})  "
          f"AUPRC {r.auprc:.4f} ({r.auprc_lo:.4f}–{r.auprc_hi:.4f})")

# ---------------------------------------------------------------- 3. 校准偏差
cal = pd.read_csv(f"{S}/mimic_model_calibration_v2.csv")
cal["abs_dev_pp"] = 100 * (cal.observed - cal.mean_pred).abs()
print("\n=== 校准偏差（百分点）===")
for m, g in cal.groupby("model"):
    w = g.loc[g.abs_dev_pp.idxmax()]
    print(f"  {m:22s} 最大偏差 {w.abs_dev_pp:.2f} pp（组 {w.group}，预测 {100*w.mean_pred:.2f}% vs 实测 {100*w.observed:.2f}%）")

# ---------------------------------------------------------------- 4. 首剂时点
v["delay_h"] = (pd.to_datetime(v.t_abx_first) - pd.to_datetime(v.admittime)).dt.total_seconds() / 3600
print("\n=== 首剂抗生素距入院时间 ===")
print("  中位 %.2f h（IQR %.2f–%.2f），<=6 h 占 %.1f%%（n=%d）" % (
    v.delay_h.median(), v.delay_h.quantile(.25), v.delay_h.quantile(.75),
    100 * (v.delay_h <= 6).mean(), int((v.delay_h <= 6).sum())))
rows = []
for lab, m in [("All hold-out", np.ones(N, bool)), ("First dose after 6 h", (v.delay_h > 6).values),
               ("First dose within 6 h", (v.delay_h <= 6).values)]:
    if m.sum() < 100 or len(set(y[m])) < 2: continue
    rows.append({"stratum": lab, "n": int(m.sum()), "events": int(y[m].sum()),
                 "prevalence": float(y[m].mean()),
                 "auroc_lr": roc_auc_score(y[m], p_lr[m]), "auprc_lr": average_precision_score(y[m], p_lr[m]),
                 "auroc_gb": roc_auc_score(y[m], p_gb[m]), "auprc_gb": average_precision_score(y[m], p_gb[m])})
tim = pd.DataFrame(rows); tim.to_csv(f"{S}/mimic_first_dose_timing.csv", index=False)
for _, r in tim.iterrows():
    print(f"  {r.stratum:22s} n={r.n:>7,} 事件率 {100*r.prevalence:5.2f}%  AUROC(LR) {r.auroc_lr:.4f}  AUROC(HGB) {r.auroc_gb:.4f}")

# 同步到根目录
import shutil, os
for f in ["mimic_model_dca.csv", "mimic_model_bootstrap_holdout.csv", "mimic_first_dose_timing.csv"]:
    shutil.copy(os.path.join(S, f), os.path.join("results", f))
print("\n已覆盖/新增：mimic_model_dca.csv、mimic_model_bootstrap_holdout.csv、mimic_first_dose_timing.csv")
