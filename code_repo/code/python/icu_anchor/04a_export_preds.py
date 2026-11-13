# -*- coding: utf-8 -*-
"""分支【ICU 锚点为主分析】：把留出集的预测与结局导出为单一 CSV，供 R 出图与制表。

目的：图与表只读这一份文件，保证 figures/、results/、manuscript/tables/ 三方数字同源。
"""
import os, sys
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

BRANCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RES = os.path.join(BRANCH_ROOT, "results")
SEED = 20261105

coh = pd.read_csv(os.path.join(RES, "icu_cohort_analysis.csv"))
n2 = pd.read_csv(os.path.join(RES, "icu_news2_scores.csv"))
Y = coh.broad_ge3.values.astype(int)
rng = np.random.default_rng(SEED)
dev = np.zeros(len(coh), bool)
for cls in (0, 1):
    idx = np.where(Y == cls)[0]
    k = int(round(0.70 * len(idx)))
    dev[rng.permutation(idx)[:k]] = True
val = ~dev
vidx = np.where(val)[0]

preds = {}
for key, fn in [("p_lr", "_icu_p_lr.npy"), ("p_gb", "_icu_p_gb.npy"),
                 ("p_portable", "_icu_p_portable.npy"), ("p_news2", "_icu_p_news2.npy"),
                 ("p_qsofa", "_icu_p_qsofa.npy")]:
    p = np.load(os.path.join(RES, fn))
    assert len(p) == len(vidx), f"{fn} 长度 {len(p)} ≠ 留出集 {len(vidx)}"
    preds[key] = p

out = coh.iloc[vidx].reset_index(drop=True)
for k, v in preds.items():
    out[k] = v
# NEWS2 / qSOFA 分数（按 hadm_id 对齐）
out = out.merge(n2[["hadm_id", "news2", "news2_comps", "qsofa"]], on="hadm_id", how="left")
assert out.news2.notna().all(), "NEWS2 分数未能全部对齐"
out["y"] = Y[vidx]
out.to_csv(os.path.join(RES, "icu_holdout_predictions.csv"), index=False)
print("留出集预测已导出：", os.path.join(RES, "icu_holdout_predictions.csv"))
print("  n =", len(out), " 事件 =", int(out.y.sum()))
for k in preds:
    from sklearn.metrics import roc_auc_score, average_precision_score
    print(f"  {k:12s} AUROC {roc_auc_score(out.y, out[k]):.4f}  "
          f"AUPRC {average_precision_score(out.y, out[k]):.4f}")
