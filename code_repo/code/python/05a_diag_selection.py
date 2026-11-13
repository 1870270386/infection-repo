# -*- coding: utf-8 -*-
"""诊断：为什么 L1-LASSO 在开发集上只选中 1 个变量？

三个待查点
  1) 极端值（labs 的 max/min 列可达数千）经标准化后形成超高杠杆点，是否把 L1 解压塌；
  2) liblinear L1 在本数据上是否真的收敛（n_iter_ / 选中数）；
  3) 计数类（*_n）观测过程特征是否主导了筛选。

脚本：05a_diag_selection.py（诊断用，不产出交付结果）
"""
import duckdb, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif

SEED = 20261105
rng = np.random.default_rng(SEED)

con = duckdb.connect("D:/mimic_iv.duckdb", read_only=True)
df = con.execute("SELECT * FROM feats WHERE empiric_24h=1").df()
con.close()

CM  = [c for c in df.columns if c.startswith("cm_")]
VIT = [c for c in df.columns if c.startswith("v_") and not c.endswith("_n")]
LAB = [c for c in df.columns if c.startswith("l_") and not c.endswith("_n")]
COV = [c for c in df.columns if c.startswith(("v_", "l_")) and c.endswith("_n")]
df["age"] = df["anchor_age"].astype(float)
FEATS = ["age", "gender"] + CM + VIT + LAB + COV
FEATS = [f for f in FEATS if f in df.columns and df[f].dtype != object]

Y = df.broad_ge3.values.astype(int)
YEAR = df.adm_year.values
cut = int(np.quantile(YEAR, 0.70))
dev = YEAR <= cut

X = df[FEATS].astype(float).values
keep = ~np.all(np.isnan(X), axis=0)
X = X[:, keep]
FN = [f for f, k in zip(FEATS, keep) if k]
imp = SimpleImputer(strategy="median").fit(X[dev])
Xi = imp.transform(X)
sc = StandardScaler().fit(Xi[dev])
Xi = sc.transform(Xi)

# ---- 1. 极端值体检
print("=== 极端值体检（标准化后 |z| 的最大值 / >20 的单元格数）===")
z = np.abs(Xi[dev])
order = np.argsort(-z.max(axis=0))
for i in order[:12]:
    col = FN[i]
    raw = df[col].values.astype(float)
    print(f"  {col:24s} 原值 min {np.nanmin(raw):>10.1f}  max {np.nanmax(raw):>10.1f}  "
          f"|z|max {z[:, i].max():>8.1f}  |z|>20 的单元格 {int((z[:, i] > 20).sum()):>6,}")
print(f"  全表 |z|>20 单元格数: {int((z > 20).sum()):,} / {z.size:,}")

# ---- 2. L1 在不同 C 下的收敛与选中数（30k 子样本）
dev_idx = np.where(dev)[0]
sub = rng.choice(len(dev_idx), 30000, replace=False)
Xd, Yd = Xi[dev_idx][sub], Y[dev_idx][sub]

print("\n=== liblinear L1：不同 C 的收敛与选中数（30k 子样本，原始标准化数据）===")
for C in [0.005, 0.02, 0.1, 0.5, 2.0, 10.0]:
    t0 = time.time()
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=5000,
                           tol=1e-4, random_state=SEED).fit(Xd, Yd)
    nz = np.abs(m.coef_[0]) > 1e-8
    top = np.argsort(-np.abs(m.coef_[0]))[:6]
    print(f"  C={C:<6} n_iter={m.n_iter_[0]:<5} 选中 {int(nz.sum()):>3}  "
          f"耗时 {time.time()-t0:>5.1f}s  top: " +
          ", ".join(f"{FN[i]}={m.coef_[0][i]:+.3f}" for i in top))

# ---- 3. 去掉计数类变量后重试
print("\n=== 去掉 *_n 计数类（观测过程特征）后 ===")
keep2 = np.array([not f.endswith("_n") for f in FN])
Xd2, Yd2 = Xi[dev_idx][sub][:, keep2], Y[dev_idx][sub]
FN2 = [f for f, k in zip(FN, keep2) if k]
for C in [0.005, 0.02, 0.1, 0.5, 2.0]:
    t0 = time.time()
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=5000,
                           tol=1e-4, random_state=SEED).fit(Xd2, Yd2)
    nz = np.abs(m.coef_[0]) > 1e-8
    top = np.argsort(-np.abs(m.coef_[0]))[:6]
    print(f"  C={C:<6} n_iter={m.n_iter_[0]:<5} 选中 {int(nz.sum()):>3}  "
          f"耗时 {time.time()-t0:>5.1f}s  top: " +
          ", ".join(f"{FN2[i]}={m.coef_[0][i]:+.3f}" for i in top))

# ---- 4. 去极端值（按开发集 1/99 分位 winsorize）后再试
print("\n=== 去极端值后（开发集 1%/99% 分位 winsorize，保留计数类）===")
lo = np.nanpercentile(Xi[dev], 1, axis=0)
hi = np.nanpercentile(Xi[dev], 99, axis=0)
Xw = np.clip(Xi, lo, hi)
Xwd = Xw[dev_idx][sub]
for C in [0.005, 0.02, 0.1, 0.5, 2.0]:
    t0 = time.time()
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=5000,
                           tol=1e-4, random_state=SEED).fit(Xwd, Yd)
    nz = np.abs(m.coef_[0]) > 1e-8
    top = np.argsort(-np.abs(m.coef_[0]))[:6]
    print(f"  C={C:<6} n_iter={m.n_iter_[0]:<5} 选中 {int(nz.sum()):>3}  "
          f"耗时 {time.time()-t0:>5.1f}s  top: " +
          ", ".join(f"{FN[i]}={m.coef_[0][i]:+.3f}" for i in top))

# ---- 5. F 检验 top-50 的构成（看是否被计数类与实验室统计量占满）
print("\n=== F 检验 top-50 构成 ===")
from sklearn.feature_selection import SelectKBest
skb = SelectKBest(f_classif, k=50).fit(Xi[dev], Y[dev])
sel = [FN[i] for i in np.where(skb.get_support())[0]]
print("  计数类:", sum(f.endswith('_n') for f in sel), " 生命体征:",
      sum(f.startswith('v_') for f in sel), " 实验室:", sum(f.startswith('l_') for f in sel),
      " 其他:", [f for f in sel if not f.startswith(('v_', 'l_'))])
print("  top-50 列表:", sel)
