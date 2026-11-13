# -*- coding: utf-8 -*-
"""MIMIC-IV：训练数据子集对照（照 PMID 41079949 的 Table 2 做法）。

对照三个模型，全部在开发集拟合、在留出验证集评估：
  1) 完整模型（人口学 + 合并症 + 生命体征 + 实验室）
  2) 可迁移子模型（人口学 + 合并症 + 实验室，**不含生命体征**）——面向无连续监测的医院（如本院队列）
  3) 基线模型（仅人口学 + 合并症）

预处理与 05_model.py 完全一致（同一 winsorize/插补/标准化口径，均在开发集上拟合），
变量集合取自 05 导出的 mimic_model_variables.json。

脚本：07b_subset_models.py  |  输出：results/mimic_model_subset.csv
"""
import duckdb, json, os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = "studies/infection-sepsis-2026-11/results"
SEED = 20261105

con = duckdb.connect("D:/mimic_iv.duckdb", read_only=True)
df = con.execute("SELECT * FROM feats2 WHERE empiric_24h=1").df()
con.close()

with open(f"{OUT}/mimic_model_variables.json", encoding="utf-8") as f:
    mv = json.load(f)
SEL = mv["features"]

Y = df.broad_ge3.values.astype(int)
cut = int(np.quantile(df.adm_year.values, 0.70))
dev = df.adm_year.values <= cut
val = ~dev

X = df[SEL].astype(float).values
keep = ~np.all(np.isnan(X), axis=0)
X = X[:, keep]
FN = [f for f, k in zip(SEL, keep) if k]
lo = np.nanpercentile(X[dev], 1, axis=0)
hi = np.nanpercentile(X[dev], 99, axis=0)
Xw = np.clip(X, lo, hi)
imp = SimpleImputer(strategy="median").fit(Xw[dev])
X_imp = imp.transform(Xw)
X_imp = StandardScaler().fit(X_imp[dev]).transform(X_imp)

def subset(cols):
    idx = [FN.index(c) for c in cols]
    return X_imp[:, idx]

def is_vital(c):
    return c.startswith("v_")
def is_lab(c):
    return c.startswith("l_")
def is_base(c):
    return c.startswith("cm_") or c.startswith("race_") or c.startswith("atype_") or c in ("age", "gender")

FULL = SEL
PORTABLE = [c for c in SEL if not is_vital(c)]
BASELINE = [c for c in SEL if is_base(c)]
print(f"完整模型 {len(FULL)} 个变量；可迁移子模型（无生命体征）{len(PORTABLE)} 个；基线 {len(BASELINE)} 个")

rows = []
for name, cols in [("All data (full model)", FULL),
                   ("Laboratory and demographics only (portable model)", PORTABLE),
                   ("Demographics and comorbidities only (baseline model)", BASELINE)]:
    if not cols:
        continue
    Xs = subset(cols)
    m = LogisticRegression(solver="lbfgs", max_iter=3000, C=1.0).fit(Xs[dev], Y[dev])
    p = m.predict_proba(Xs[val])[:, 1]
    r = {"model": name, "n_features": len(cols),
         "auroc": float(roc_auc_score(Y[val], p)),
         "auprc": float(average_precision_score(Y[val], p))}
    rows.append(r)
    print(f"  {name:58s} p={len(cols):>3}  AUROC {r['auroc']:.4f}  AUPRC {r['auprc']:.4f}")

pd.DataFrame(rows).to_csv(f"{OUT}/mimic_model_subset.csv", index=False)
print("已导出 results/mimic_model_subset.csv")
