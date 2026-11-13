# -*- coding: utf-8 -*-
"""从留出验证集的预测向量生成 ROC / PR 曲线点，供 R 端出图使用。

输入：results/_p_lr.npy、results/_p_gb.npy（05_model.py 末尾保存）
输出：results/mimic_roc_pr.csv（model, curve, x, y）

脚本：07d_roc_pr.py
"""
import duckdb, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.metrics import roc_curve, precision_recall_curve

OUT = "studies/infection-sepsis-2026-11/results"
con = duckdb.connect("D:/mimic_iv.duckdb", read_only=True)
df = con.execute("SELECT hadm_id, adm_year, broad_ge3 FROM feats2 WHERE empiric_24h=1").df()
con.close()
cut = int(np.quantile(df.adm_year.values, 0.70))
val = df.adm_year.values > cut
y = df.broad_ge3.values[val]

rows = []
for name, f in [("Logistic", "_p_lr.npy"), ("HistGradientBoosting", "_p_gb.npy")]:
    p = np.load(f"{OUT}/{f}")
    assert len(p) == len(y), f"{f} 与验证集长度不符"
    fpr, tpr, _ = roc_curve(y, p)
    for a, b in zip(fpr, tpr):
        rows.append({"model": name, "curve": "roc", "x": a, "y": b})
    pr, rc, _ = precision_recall_curve(y, p)
    for a, b in zip(rc, pr):
        rows.append({"model": name, "curve": "pr", "x": a, "y": b})

pd.DataFrame(rows).to_csv(f"{OUT}/mimic_roc_pr.csv", index=False)
print(f"已导出 results/mimic_roc_pr.csv（{len(rows):,} 行）")
