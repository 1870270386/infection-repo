# -*- coding: utf-8 -*-
"""分支【ICU 锚点为主分析】：稳健性检验（进 manuscript/supplementary/）。

五组检验（照 plan/analysis_plan.md 的稳健性清单）：
  R1 Winsorize 阈值 0.5%/99.5%（基线 1%/99%）
  R2 不 Winsorize
  R3 特征数 20 / 50（基线 32）
  R4 留出切分种子更换 3 次
  R5 完全病例（0–6 h 生命体征与实验室齐全者）单独建模
  R6 结局窗口改为 ICUadm 后 24–72 h
LASSO 的 C 固定用基线 CV 选出的值，不在每个变体里重新调（避免多重比较）。
"""
import duckdb, json, os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, average_precision_score

BRANCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(BRANCH_ROOT, "results")
SUP = os.path.join(BRANCH_ROOT, "manuscript", "supplementary")
os.makedirs(SUP, exist_ok=True)
SEED = 20261105

con = duckdb.connect("D:/mimic_iv_branch_icu.duckdb", read_only=True)
df = con.execute("SELECT * FROM icu_feats").df()
con.close()

_race_u = df.race.astype(str).str.upper()
_at_u = df.admission_type.astype(str).str.upper()
df["race_BLACK"] = _race_u.str.startswith("BLACK").astype(int)
df["race_ASIAN"] = _race_u.str.startswith("ASIAN").astype(int)
df["race_HISPANIC"] = _race_u.str.startswith("HISPANIC").astype(int)
df["race_OTHER"] = (~_race_u.str.startswith(("WHITE", "BLACK", "ASIAN", "HISPANIC"))).astype(int)
df["atype_ELECTIVE"] = (_at_u == "ELECTIVE").astype(int)
df["atype_URGENT"] = (_at_u == "URGENT").astype(int)
df["atype_SURGICAL"] = _at_u.str.contains("SURGICAL").astype(int)
df["atype_OBSERVATION"] = _at_u.str.contains("OBSERVATION").astype(int)

with open(os.path.join(OUT, "icu_model_variables.json"), encoding="utf-8") as f:
    mv = json.load(f)
BASE_FEATS = mv["features"]
C_FIXED = mv["lasso_C"]
print("基线：特征数", len(BASE_FEATS), " LASSO C =", C_FIXED, flush=True)

CM = [c for c in df.columns if c.startswith("cm_")]
VIT = [c for c in df.columns if c.startswith("v_") and not c.endswith("_n")]
LAB = [c for c in df.columns if c.startswith("l_") and not c.endswith("_n")]
df["age"] = df.anchor_age.astype(float)
CAND = ["age"] + CM + VIT + LAB + ["race_BLACK", "race_ASIAN", "race_HISPANIC", "race_OTHER",
                                   "atype_ELECTIVE", "atype_URGENT", "atype_SURGICAL", "atype_OBSERVATION"]
CAND = [c for c in CAND if c in df.columns and df[c].dtype != object]

Y = df.broad_ge3.values.astype(int)


def make_split(seed, y):
    rng = np.random.default_rng(seed)
    d = np.zeros(len(y), bool)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        k = int(round(0.70 * len(idx)))
        d[rng.permutation(idx)[:k]] = True
    return d, ~d


def run(cand, y, dev, val, wlo=1.0, whi=99.0, n_feat=None, lasso_select=True, seed=SEED, dframe=None):
    d = df if dframe is None else dframe
    X = d[cand].astype(float).values
    keep = ~np.all(np.isnan(X), axis=0)
    X = X[:, keep]
    FN = [f for f, k in zip(cand, keep) if k]
    if wlo is not None:
        lo = np.nanpercentile(X[dev], wlo, axis=0)
        hi = np.nanpercentile(X[dev], whi, axis=0)
        X = np.clip(X, lo, hi)
    imp = SimpleImputer(strategy="median").fit(X[dev])
    Xi = imp.transform(X)
    Xi = StandardScaler().fit(Xi[dev]).transform(Xi)
    if lasso_select:
        ls = LogisticRegression(penalty="l1", solver="liblinear", C=C_FIXED, max_iter=1000,
                                tol=1e-3, random_state=seed).fit(Xi[dev], y[dev])
        lsel = {FN[i] for i in np.where(np.abs(ls.coef_[0]) > 1e-8)[0]}
        kk = min(50, len(FN))
        skb = SelectKBest(f_classif, k=kk).fit(Xi[dev], y[dev])
        fsel = {FN[i] for i in np.where(skb.get_support())[0]}
        inter = sorted(lsel & fsel)
        if len(inter) < 8:
            inter = sorted(lsel | fsel)[:40]
    else:
        inter = BASE_FEATS
    if n_feat is not None:
        skb = SelectKBest(f_classif, k=min(n_feat, len(FN))).fit(Xi[dev], y[dev])
        ftop = [FN[i] for i in np.where(skb.get_support())[0]]
        inter = sorted(set(inter) & set(ftop)) if set(inter) & set(ftop) else ftop[:n_feat]
    if not inter:
        inter = BASE_FEATS
    Xs = Xi[:, [FN.index(f) for f in inter]]
    m = LogisticRegression(solver="lbfgs", max_iter=3000, C=1.0).fit(Xs[dev], y[dev])
    p = m.predict_proba(Xs[val])[:, 1]
    return {"n_feat": len(inter), "auroc": roc_auc_score(y[val], p),
            "auprc": average_precision_score(y[val], p)}, inter


dev, val = make_split(SEED, Y)
rows = []
print("\n=== 稳健性检验 ===", flush=True)

r, sel = run(CAND, Y, dev, val)
rows.append({"analysis": "Baseline (winsorize 1–99%, LASSO∩F selection)", **r}); print("  baseline", r, flush=True)
r, _ = run(CAND, Y, dev, val, wlo=0.5, whi=99.5)
rows.append({"analysis": "R1 Winsorize 0.5–99.5%", **r}); print("  R1", r, flush=True)
r, _ = run(CAND, Y, dev, val, wlo=None, whi=None)
rows.append({"analysis": "R2 No winsorizing", **r}); print("  R2", r, flush=True)
for k in (20, 50):
    r, _ = run(CAND, Y, dev, val, n_feat=k)
    rows.append({"analysis": f"R3 Top-{k} features", **r}); print(f"  R3 {k}", r, flush=True)
for sd in (20261106, 20261107, 20261108):
    d2, v2 = make_split(sd, Y)
    r, _ = run(CAND, Y, d2, v2, seed=sd)
    rows.append({"analysis": f"R4 Split seed {sd}", **r}); print(f"  R4 {sd}", r, flush=True)

# R5 完全病例：0–6 h 生命体征与实验室都齐全者
lab_n = [c for c in df.columns if c.startswith("l_") and c.endswith("_n")]
vit_n = [c for c in df.columns if c.startswith("v_") and c.endswith("_n")]
cc = (df[lab_n].sum(axis=1) > 0).values & (df[vit_n].sum(axis=1) > 0).values
df_cc = df[cc].reset_index(drop=True)
Y_cc = df_cc.broad_ge3.values.astype(int)
dev_cc, val_cc = make_split(SEED, Y_cc)
r, _ = run(CAND, Y_cc, dev_cc, val_cc, dframe=df_cc)
rows.append({"analysis": f"R5 Complete cases only (both data types, n={int(cc.sum()):,})", **r})
print("  R5", r, flush=True)

# R6 结局窗口 ICUadm 后 24–72 h
Y72 = (df.n_cls_72h_icu >= 3).values.astype(int)
r, _ = run(CAND, Y72, dev, val)
rows.append({"analysis": "R6 Outcome window 24–72 h after ICU admission", **r}); print("  R6", r, flush=True)

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "icu_robustness.csv"), index=False)
print("\n=== 汇总 ===")
print(out.to_string(index=False))
out.to_csv(os.path.join(SUP, "tableS2_robustness.csv"), index=False)
print(f"\n已导出 {OUT}/icu_robustness.csv 与 {SUP}/tableS2_robustness.csv")
