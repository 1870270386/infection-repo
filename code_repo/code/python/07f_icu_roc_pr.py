# -*- coding: utf-8 -*-
"""ICU 锚点敏感性分析：生成 ROC / PR 曲线点，供补充材料出图。

输入：.branches/ICU 锚点为主分析（推荐）/results/_icu_p_lr.npy、_icu_p_gb.npy
输出：该分支 results/icu_roc_pr.csv

脚本：07f_icu_roc_pr.py
"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.metrics import roc_curve, precision_recall_curve

BR = ".branches/ICU 锚点为主分析（推荐）"
RES = f"{BR}/results"
coh = pd.read_csv(f"{RES}/icu_cohort_analysis.csv")
y = coh.broad_ge3.values.astype(int)

rows = []
for name, f in [("Logistic", "_icu_p_lr.npy"), ("HistGradientBoosting", "_icu_p_gb.npy")]:
    p = np.load(f"{RES}/{f}")
    assert len(p) == len(y), f"{f} 长度与队列不符"
    fpr, tpr, _ = roc_curve(y, p)
    for a, b in zip(fpr, tpr):
        rows.append({"model": name, "curve": "roc", "x": a, "y": b})
    pr, rc, _ = precision_recall_curve(y, p)
    for a, b in zip(rc, pr):
        rows.append({"model": name, "curve": "pr", "x": a, "y": b})
pd.DataFrame(rows).to_csv(f"{RES}/icu_roc_pr.csv", index=False)
print(f"已导出 {RES}/icu_roc_pr.csv（{len(rows):,} 行，队列 n = {len(y):,}）")
