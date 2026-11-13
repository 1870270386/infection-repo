# -*- coding: utf-8 -*-
"""诊断 2：L1-LASSO 为何只选中 1 个变量？

逐项排查：
  a) 系数向量是否含 NaN / 是否只是幅度极小（阈值问题）
  b) 是否存在（准）完全分离——无惩罚 L2 拟合的系数是否爆掉、训练集预测概率是否贴到 0/1
  c) liblinear 与 saga 的行为差异
  d) 逐个剔除变量族（实验室 / 生命体征 / 合并症）后 L1 的选中数
"""
import duckdb, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

SEED = 20261105
rng = np.random.default_rng(SEED)

con = duckdb.connect("D:/mimic_iv.duckdb", read_only=True)
src = "feats2" if con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name='feats2'").fetchone()[0] else "feats"
df = con.execute(f"SELECT * FROM {src} WHERE empiric_24h=1").df()
con.close()

CM  = [c for c in df.columns if c.startswith("cm_")]
VIT = [c for c in df.columns if c.startswith("v_") and not c.endswith("_n")
       and not any(k in c for k in ("gcseye", "gcsverbal", "gcsmotor"))]
LAB = [c for c in df.columns if c.startswith("l_") and not c.endswith("_n")]
df["age"] = df["anchor_age"].astype(float)
NUM = ["age", "gender"] + CM + VIT + LAB
NUM = [f for f in NUM if f in df.columns and df[f].dtype != object]
RACE_REF, ATYPE_REF = "WHITE", "EMERGENCY"
for lv in ["BLACK", "ASIAN", "HISPANIC", "OTHER"]:
    df[f"race_{lv}"] = (df.race.astype(str).str.upper() == lv).astype(int)
for lv in ["ELECTIVE", "URGENT", "NEWBORN"]:
    df[f"atype_{lv}"] = (df.admission_type.astype(str).str.upper() == lv).astype(int)
DUM = [f"race_{lv}" for lv in ["BLACK", "ASIAN", "HISPANIC", "OTHER"]] + \
      [f"atype_{lv}" for lv in ["ELECTIVE", "URGENT", "NEWBORN"]]
FEATS = NUM + DUM

Y = df.broad_ge3.values.astype(int)
dev = df.adm_year.values <= int(np.quantile(df.adm_year.values, 0.70))
X = df[FEATS].astype(float).values
keep = ~np.all(np.isnan(X), axis=0)
X = X[:, keep]; FN = [f for f, k in zip(FEATS, keep) if k]
lo = np.nanpercentile(X[dev], 1, axis=0); hi = np.nanpercentile(X[dev], 99, axis=0)
Xw = np.clip(X, lo, hi)
imp = SimpleImputer(strategy="median").fit(Xw[dev])
Xi = StandardScaler().fit(imp.transform(Xw)[dev]).transform(imp.transform(Xw))

dev_idx = np.where(dev)[0]
sub = rng.choice(len(dev_idx), 30000, replace=False)
Xd, Yd = Xi[dev_idx][sub], Y[dev_idx][sub]
print(f"子样本 {Xd.shape}, 事件率 {Yd.mean():.3f}")

# a) 系数细节
print("\n=== a) liblinear L1 系数向量细节（C=0.5, 30k 子样本）===")
m = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=500,
                       tol=1e-3, random_state=SEED).fit(Xd, Yd)
c = m.coef_[0]
print(f"  n_iter={m.n_iter_[0]}  NaN 系数 {int(np.isnan(c).sum())}  |coef| 最大 {np.nanmax(np.abs(c)):.6g}")
print(f"  |coef|>1e-8 的个数: {int((np.abs(c) > 1e-8).sum())}")
print(f"  |coef|>1e-4 的个数: {int((np.abs(c) > 1e-4).sum())}")
order = np.argsort(-np.abs(c))[:12]
print("  top12 |coef|:", ", ".join(f"{FN[i]}={c[i]:+.5f}" for i in order))
print(f"  截距 {m.intercept_[0]:.4f}")

# b) 分离检查：无惩罚 L2
print("\n=== b) 分离检查（无惩罚 L2，30k 子样本）===")
m2 = LogisticRegression(penalty="none", solver="lbfgs", max_iter=5000).fit(Xd, Yd)
p = m2.predict_proba(Xd)[:, 1]
print(f"  训练集最大 |coef| {np.max(np.abs(m2.coef_[0])):.3f}  系数 >10 的个数 {int((np.abs(m2.coef_[0])>10).sum())}")
print(f"  训练集预测概率: max {p.max():.6f}  min {p.min():.6f}  "
      f">0.999 的样本 {int((p>0.999).sum())}  <0.001 的样本 {int((p<0.001).sum())}")

# c) saga 对照
print("\n=== c) saga（L1, elasticnet=False）===")
for C in [0.01, 0.1, 1.0]:
    t0 = time.time()
    ms = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=3000,
                            tol=1e-3, random_state=SEED).fit(Xd, Yd)
    cs = ms.coef_[0]
    print(f"  C={C:<6} n_iter={ms.n_iter_[0]:<5} 选中 {int((np.abs(cs)>1e-8).sum()):>3}  "
          f"|coef|max {np.max(np.abs(cs)):.4f}  {time.time()-t0:.0f}s")

# d) 逐族剔除
print("\n=== d) 逐族剔除后 L1 选中数（C=0.5）===")
def nsel(cols, tag):
    idx = [FN.index(c) for c in cols]
    mm = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=500,
                            tol=1e-3, random_state=SEED).fit(Xd[:, idx], Yd)
    cc = mm.coef_[0]
    print(f"  {tag:28s} p={len(idx):>4}  选中 {int((np.abs(cc)>1e-8).sum()):>4}  "
          f"|coef|max {np.max(np.abs(cc)):.5f}  n_iter={mm.n_iter_[0]}")
    return cc

nsel([c for c in FN if c.startswith("l_")], "仅实验室统计量")
nsel([c for c in FN if c.startswith("v_")], "仅生命体征统计量")
nsel([c for c in FN if c.startswith("cm_") or c in ("age", "gender")], "仅人口学+合并症")
nsel([c for c in FN if not c.startswith("l_")], "去掉全部实验室")
nsel([c for c in FN if not c.startswith("v_")], "去掉全部生命体征")
nsel([c for c in FN if c.startswith("l_") and c.endswith(("_mean", "_max", "_min", "_last"))], "实验室统计量(去掉计数)")


# e) 规模扫描：L1 选中数随样本量的变化
print("\n=== e) 规模扫描（C=0.5, 全开发集内的随机子样本）===")
dev_all_idx = dev_idx
for frac in [30000, 60000, 90000, len(dev_all_idx)]:
    sub2 = rng.choice(len(dev_all_idx), frac, replace=False)
    idx2 = dev_all_idx[sub2]
    mm = LogisticRegression(penalty="l1", solver="liblinear", C=0.5, max_iter=500,
                            tol=1e-3, random_state=SEED).fit(Xi[idx2], Y[idx2])
    cc = mm.coef_[0]
    print(f"  n={frac:>7,}  n_iter={mm.n_iter_[0]:<5} 选中 {int((np.abs(cc)>1e-8).sum()):>4}  "
          f"|coef|max {np.max(np.abs(cc)):.5f}  截距 {mm.intercept_[0]:.4f}")
