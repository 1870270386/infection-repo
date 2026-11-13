# -*- coding: utf-8 -*-
"""MIMIC-IV：预测"入院后是否需要大包围（入院 24h 内 ≥3 类抗生素）"。

设计
  人群      入院 24h 内启动经验性抗生素的住院者（184,645）
  预测时点  入院后 6 小时（临床"初步评估"决策点）
  预测变量  入院后 0–6h 的生命体征与实验室统计量 + 人口学 + 入院诊断合并症
  标签      入院 24h 内 ≥3 类抗生素（broad_ge3）
  验证      ① 留出验证（按去标识化入院年份 70% 分位切分） ② bootstrap 1000 次
  报告      AUROC / AUPRC / Brier / 校准斜率与截距 / 决策曲线 / 亚组 / 多标签敏感性
  价值链    第一环：模型能否把"最终不需要大包围"的那部分人识别出来（低风险组结局）

切分的口径（重要）
  MIMIC-IV v3.0 的日期经去标识化随机平移到 2105–2214，**入院年份不含真实日历时间信息**，
  因此按年份切分只是**留出验证**，不是时间外验证（详见 plan/analysis_plan.md 第 5 节）。

泄漏控制
  预测窗口 0–6h 严格早于标签窗口 0–24h；不含任何用药、微生物学信息；
  不含 ICU 收治（其时刻中位在住院入院后 2.3h，晚于首剂抗生素，属决策后信息）。

三处必须说明的预处理决定
  1) **Winsorize（开发集 1%/99% 分位）**：原始数据存在录入伪影——
     v_map_last 最大 100,369 mmHg、v_hr_max 8,400、v_spo2_pls_mean 2,395。
     不处理时 L1-LASSO 只选中 1 个变量（极端值杠杆把解压塌）。
  2) **剔除 *_n 计数类变量**：检测次数属"观测过程特征"，是跨库 shortcut
     （加测量计数额外多掉 0.035 AUROC），且无法迁移到无连续监测的医院。
  3) **校准回归用无惩罚 logistic**（statsmodels Logit），sklearn 1.0.2 无 penalty=None。

脚本：05_model.py  |  输出：results/mimic_model_*.csv、mimic_model_variables.json
"""
import duckdb, json, os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import statsmodels.api as sm

OUT = "studies/infection-sepsis-2026-11/results"
SEED = 20261105
np.random.seed(SEED)
t_start = time.time()

con = duckdb.connect("D:/mimic_iv.duckdb", read_only=True)
has_f2 = con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name='feats2'").fetchone()[0]
src = "feats2" if has_f2 else "feats"
df = con.execute(f"SELECT * FROM {src} WHERE empiric_24h=1").df()
con.close()
print(f"数据源表: {src}")
print("分析人群:", len(df), " 事件:", int(df.broad_ge3.sum()),
      f"({100*df.broad_ge3.mean():.2f}%)", flush=True)

# ---------------------------------------------------------------- 1. 设计矩阵
CM  = [c for c in df.columns if c.startswith("cm_")]
VIT = [c for c in df.columns if c.startswith("v_") and not c.endswith("_n")
       and not any(k in c for k in ("gcseye", "gcsverbal", "gcsmotor"))]
LAB = [c for c in df.columns if c.startswith("l_") and not c.endswith("_n")]
df["age"] = df["anchor_age"].astype(float)
# MIMIC-IV v3 的 patients.gender 是字符串 'F'/'M'，须显式编码，否则会被 dtype 过滤掉
_g = df["gender"].astype(str).str.upper()
df["gender"] = np.where(_g == "M", 1, np.where(_g == "F", 0, np.nan)).astype(float)
# 急诊入院：MIMIC-IV 的 admission_type 没有字面值 'EMERGENCY'，急诊为 'EW EMER.' 与 'DIRECT EMER.'
df["is_emergency"] = df["admission_type"].astype(str).str.upper().isin(["EW EMER.", "DIRECT EMER."]).astype(int)
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
print("候选数值变量:", len(NUM), "= 人口学 2 + 合并症", len(CM),
      "+ 生命体征统计量", len(VIT), "+ 实验室统计量", len(LAB), flush=True)

# 数据可得性标记（仅用于分层描述，不进模型）
lab_n = [c for c in df.columns if c.startswith("l_") and c.endswith("_n")]
VIT_KEY = [c for c in ["v_hr_n", "v_rr_n", "v_sbp_n", "v_nibp_sbp_n",
                        "v_spo2_pls_n", "v_spo2_art_n"] if c in df.columns]
has_lab = (df[lab_n].sum(axis=1) > 0).values
has_vit = (df[VIT_KEY].sum(axis=1) > 0).values
df["has_lab_6h"] = has_lab.astype(int)
df["has_vit_6h"] = has_vit.astype(int)
df["data_avail"] = np.where(has_lab & has_vit, "Both",
                     np.where(has_lab, "Laboratory only",
                     np.where(has_vit, "Vital signs only", "Neither")))
print("\n=== 决策时点（入院 0–6h）的数据可得性 ===", flush=True)
for k, g in df.groupby("data_avail"):
    print(f"  {k:14s} n={len(g):>7,} ({100*len(g)/len(df):5.1f}%)  大包围率 {100*g.broad_ge3.mean():5.2f}%  "
          f"死亡 {100*g.hospital_expire_flag.mean():5.2f}%")
print(f"  → 有实验室 {has_lab.sum():,}（{100*has_lab.mean():.1f}%），"
      f"有生命体征 {has_vit.sum():,}（{100*has_vit.mean():.1f}%）", flush=True)

# 结局与留出切分
Y = df.broad_ge3.values.astype(int)
YEAR = df.adm_year.values
cut = int(np.quantile(YEAR, 0.70))
dev = YEAR <= cut
val = ~dev
print(f"\n留出切分（按去标识化入院年份 {cut}）: 开发 {dev.sum():,} 例 / 留出 {val.sum():,} 例", flush=True)

# ---------------------------------------------------------------- 2. Winsorize + 插补 + 标准化
keep = ~np.all(np.isnan(X := df[FEATS].astype(float).values), axis=0)
X = X[:, keep]
FN = [f for f, k in zip(FEATS, keep) if k]
print("剔除全缺失列后:", len(FN), flush=True)

lo = np.nanpercentile(X[dev], 1, axis=0)
hi = np.nanpercentile(X[dev], 99, axis=0)
Xw = np.clip(X, lo, hi)
print(f"Winsorize 完成（开发集 1%/99% 分位；截断单元格 "
      f"{int(((X[dev] < lo) | (X[dev] > hi)).sum()):,} 个）", flush=True)

imp = SimpleImputer(strategy="median")
imp.fit(Xw[dev])
X_imp = imp.transform(Xw)
scaler = StandardScaler().fit(X_imp[dev])
X_imp = scaler.transform(X_imp)
print("插补（中位数）+ 标准化完成（均在开发集上拟合）", flush=True)

# ---------------------------------------------------------------- 3. 变量筛选
print("\n=== 变量筛选（L1-LASSO，C 由 5 折 CV 选 ∩ 单变量 F 检验 top-50）===", flush=True)
dev_idx = np.where(dev)[0]
rng = np.random.default_rng(SEED)
sub = rng.choice(len(dev_idx), 30000, replace=False)
Xd_sub, Yd_sub = X_imp[dev_idx][sub], Y[dev_idx][sub]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
C_GRID = [0.005, 0.02, 0.1, 0.5, 2.0]
t0 = time.time()
gs = GridSearchCV(
    LogisticRegression(penalty="l1", solver="liblinear", max_iter=300,
                        tol=1e-3, random_state=SEED),
    {"C": C_GRID}, cv=cv, scoring="roc_auc", n_jobs=1, refit=True)
gs.fit(Xd_sub, Yd_sub)
cv_tab = pd.DataFrame([{"C": c, "cv_auroc": m, "std": s}
                       for c, m, s in zip(C_GRID, gs.cv_results_["mean_test_score"],
                                          gs.cv_results_["std_test_score"])])
print(cv_tab.to_string(index=False), flush=True)
best_C = gs.best_params_["C"]
print(f"  子样本 CV 最佳 C = {best_C}（CV AUROC {gs.best_score_:.4f}，{time.time()-t0:.0f}s）", flush=True)

t0 = time.time()
lasso = LogisticRegression(penalty="l1", solver="liblinear", C=best_C, max_iter=500,
                           tol=1e-3, random_state=SEED).fit(X_imp[dev], Y[dev])
print(f"  全开发集 L1 拟合完成（{time.time()-t0:.0f}s，n_iter={lasso.n_iter_[0]}）", flush=True)
# 注意：coef_ 形状为 (1, p)，必须取 [0] 后再 np.where，否则拿到的是行号（全 0）
lasso_sel = {FN[i] for i in np.where(np.abs(lasso.coef_[0]) > 1e-8)[0]}
skb = SelectKBest(f_classif, k=min(50, len(FN))).fit(X_imp[dev], Y[dev])
f_sel = {FN[i] for i in np.where(skb.get_support())[0]}
print(f"  LASSO 选中 {len(lasso_sel)}，F 检验 top-50 选中 {len(f_sel)}", flush=True)
inter = sorted(lasso_sel & f_sel)
if len(inter) < 8:
    inter = sorted(lasso_sel | f_sel)[:40]
    print("  交集过小，改用并集前 40 个", flush=True)
print("  最终变量数:", len(inter), flush=True)
print("  最终变量:", inter, flush=True)
XI = X_imp[:, [FN.index(f) for f in inter]]

# ---------------------------------------------------------------- 4. 模型评估
def calibration(p, y):
    """无惩罚 logistic 回归 y ~ logit(p)：斜率与截距（sklearn 1.0.2 无 penalty=None，用 statsmodels）"""
    eps = 1e-6
    pc = np.clip(p, eps, 1 - eps)
    lp = np.log(pc / (1 - pc))
    try:
        res = sm.Logit(np.asarray(y, float), sm.add_constant(lp.reshape(-1, 1))).fit(disp=0, maxiter=100)
        return float(res.params[1]), float(res.params[0])
    except Exception as e:
        print("   校准回归失败:", e)
        return np.nan, np.nan


def fit_eval(Xtr, ytr, Xte, yte, model, name):
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    res = {"model": name, "n_test": len(yte), "events": int(yte.sum()),
           "auroc": roc_auc_score(yte, p), "auprc": average_precision_score(yte, p),
           "brier": brier_score_loss(yte, p), "prevalence": float(yte.mean())}
    res["cal_slope"], res["cal_intercept"] = calibration(p, yte)
    return model, p, res


def make_lr():
    return LogisticRegression(solver="lbfgs", max_iter=3000, C=1.0)


def make_gb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                           max_leaf_nodes=31, min_samples_leaf=60,
                                           l2_regularization=1.0,
                                           early_stopping=False, random_state=SEED)


print("\n=== 主模型（开发集拟合 → 留出验证集评估）===", flush=True)
t0 = time.time()
lr, p_lr, r_lr = fit_eval(XI[dev], Y[dev], XI[val], Y[val], make_lr(), "Logistic")
print(f"  LR 拟合 {time.time()-t0:.0f}s", flush=True)
t0 = time.time()
gb, p_gb, r_gb = fit_eval(XI[dev], Y[dev], XI[val], Y[val], make_gb(), "HistGradientBoosting")
print(f"  HGB 拟合 {time.time()-t0:.0f}s", flush=True)
rows = [r_lr, r_gb]
for r in rows:
    print(f"  {r['model']:22s} AUROC {r['auroc']:.4f}  AUPRC {r['auprc']:.4f}  "
          f"Brier {r['brier']:.4f}  校准斜率 {r['cal_slope']:.3f}  截距 {r['cal_intercept']:.3f}",
          flush=True)

Yv = Y[val]          # 留出集结局（长度 = len(p_lr)）
vdf = df[val].copy()
vdf["p_lr"] = p_lr; vdf["p_gb"] = p_gb

# 校准表（留出集，按预测概率五分位）
print("\n=== 校准（留出集，按预测概率五分位）===", flush=True)
cal_rows = []
for m, col in [("Logistic", "p_lr"), ("HistGradientBoosting", "p_gb")]:
    q = pd.qcut(vdf[col], 5, labels=False, duplicates="drop")
    for k, g in vdf.groupby(q):
        cal_rows.append({"model": m, "quintile": int(k) + 1, "n": len(g),
                         "mean_pred": g[col].mean(), "observed": g.broad_ge3.mean()})
cal_tab = pd.DataFrame(cal_rows)
for m in cal_tab.model.unique():
    print(f"  {m}:")
    for _, r in cal_tab[cal_tab.model == m].iterrows():
        print(f"    Q{r.quintile}  n={int(r.n):>6,}  预测 {100*r.mean_pred:5.2f}%  实测 {100*r.observed:5.2f}%")

# ---------------------------------------------------------------- 5. 按数据可得性分层
print("\n=== 按决策时点数据可得性分层的性能（留出集）===", flush=True)
avail_rows = []
for k in ["Both", "Laboratory only", "Vital signs only", "Neither"]:
    m = (vdf.data_avail.values == k)
    if m.sum() < 200 or len(set(Yv[m])) < 2:
        continue
    r = {"data_avail": k, "n": int(m.sum()), "events": int(Yv[m].sum()),
         "prevalence": float(Yv[m].mean()),
         "auroc_lr": roc_auc_score(Yv[m], p_lr[m]),
         "auprc_lr": average_precision_score(Yv[m], p_lr[m]),
         "auroc_gb": roc_auc_score(Yv[m], p_gb[m]),
         "auprc_gb": average_precision_score(Yv[m], p_gb[m])}
    avail_rows.append(r)
    print(f"  {k:14s} n={r['n']:>7,}  事件率 {100*r['prevalence']:5.2f}%  "
          f"AUROC(LR) {r['auroc_lr']:.4f}  AUROC(HGB) {r['auroc_gb']:.4f}", flush=True)

# ---------------------------------------------------------------- 6. bootstrap 内部验证
N_BOOT = 1000
print(f"\n=== bootstrap 内部验证（{N_BOOT} 次，开发集内重抽样 → 留出集评估）===", flush=True)
rng = np.random.default_rng(SEED)
boot = []
t0 = time.time()
from joblib import Parallel, delayed
def _one_boot(seed_int):
    r2 = np.random.default_rng(seed_int)
    idx = dev_idx[r2.integers(0, len(dev_idx), len(dev_idx))]
    if Y[idx].sum() < 10 or len(set(Y[idx])) < 2:
        return None
    try:
        # 真实数据存在准分离，lbfgs 会一路迭代到 max_iter；bootstrap 里只要 AUROC/AUPRC，
        # 因此收紧 max_iter/tol 以控制耗时（判别力对正则化路径不敏感）
        m = LogisticRegression(solver="lbfgs", max_iter=200, tol=1e-3, C=1.0).fit(XI[idx], Y[idx])
        pp = m.predict_proba(XI[val])[:, 1]
        return (roc_auc_score(Yv, pp), average_precision_score(Yv, pp))
    except Exception:
        return None
_seeds = rng.integers(0, 2**31 - 1, N_BOOT).tolist()
_res = Parallel(n_jobs=8, backend="loky")(delayed(_one_boot)(int(s)) for s in _seeds)
for i, r in enumerate(_res):
    if r is not None:
        boot.append(r)
    if i % 200 == 0:
        print(f"    {i}/{N_BOOT} ({time.time()-t0:.0f}s)", flush=True)
boot = np.array(boot)
blo, bhi = np.percentile(boot[:, 0], [2.5, 97.5])
plo, phi = np.percentile(boot[:, 1], [2.5, 97.5])
print(f"  有效重复 {len(boot)}  AUROC {boot[:,0].mean():.4f} "
      f"(95%CI {blo:.4f}–{bhi:.4f})  AUPRC {boot[:,1].mean():.4f} "
      f"(95%CI {plo:.4f}–{phi:.4f})", flush=True)
boot_row = {"model": "Logistic", "reps": int(len(boot)),
             "auroc_mean": float(boot[:, 0].mean()), "auroc_lo": float(blo), "auroc_hi": float(bhi),
             "auprc_mean": float(boot[:, 1].mean()), "auprc_lo": float(plo), "auprc_hi": float(phi)}

# ---------------------------------------------------------------- 7. 亚组
print("\n=== 亚组性能（留出集）===", flush=True)
sub_rows = []
for name, mfull in [
    ("全体", val),
    ("年龄 <65 岁", val & (df.anchor_age < 65)),
    ("年龄 ≥65 岁", val & (df.anchor_age >= 65)),
    ("女性", val & (df.gender == 0)),
    ("男性", val & (df.gender == 1)),
    ("急诊入院", val & (df.is_emergency == 1)),
    ("非急诊入院", val & (df.is_emergency == 0)),
]:
    m = mfull[val]                    # 掩码转到留出集长度
    if m.sum() < 100 or len(set(Yv[m])) < 2:
        continue
    r = {"subgroup": name, "n": int(m.sum()), "events": int(Yv[m].sum()),
         "prevalence": float(Yv[m].mean()),
         "auroc_lr": roc_auc_score(Yv[m], p_lr[m]),
         "auprc_lr": average_precision_score(Yv[m], p_lr[m]),
         "auroc_gb": roc_auc_score(Yv[m], p_gb[m]),
         "auprc_gb": average_precision_score(Yv[m], p_gb[m])}
    sub_rows.append(r)
    print(f"  {name:12s} n={r['n']:>7,}  事件率 {100*r['prevalence']:5.2f}%  "
          f"AUROC(LR) {r['auroc_lr']:.4f}  AUROC(HGB) {r['auroc_gb']:.4f}", flush=True)

# ---------------------------------------------------------------- 8. 决策曲线
def dca(y, p, lo, hi):
    """净获益（Vickers 定义）：pred = p >= t；tp = Σ(pred & y=1)；fp = Σ(pred & y=0)。
    注意：不可写成 w=y/p 再与 t 比较——那等价于把全部阳性计为阈值以上阳性（t<1 时恒真）。"""
    out = []
    for t in np.linspace(lo, hi, 56):
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        nb = tp / len(y) - fp / len(y) * t / (1 - t)
        allm = y.sum() / len(y) - (len(y) - y.sum()) / len(y) * t / (1 - t)
        out.append((t, nb, allm))
    return np.array(out)


print("\n=== 决策曲线（留出集，阈值概率 0.05–0.60）===", flush=True)
dca_rows = []
for nm, p in [("Logistic", p_lr), ("HistGradientBoosting", p_gb)]:
    d = dca(Yv, p, 0.05, 0.60)
    mx = d[:, 1] - d[:, 2]
    i = int(np.argmax(mx))
    j = int(np.argmin(abs(d[:, 0] - 0.10)))
    print(f"  {nm:9s} 净获益超过 treat-all 的最大差 {mx[i]:.4f} @ 阈值 {d[i,0]:.3f}；"
          f"阈值 0.10 处 NB {d[j,1]:.4f} vs treat-all {d[j,2]:.4f}", flush=True)
    for t, nb, allm in d:
        dca_rows.append({"model": nm, "threshold": t, "net_benefit": nb,
                         "treat_all": allm, "treat_none": 0.0})

# ---------------------------------------------------------------- 9. 多标签敏感性
print("\n=== 多标签敏感性（同一模型，换『大包围』定义）===", flush=True)
sens_rows = []
for lab, col in [("≥3 类抗生素（主标签）", "broad_ge3"),
                 ("碳青霉烯", "broad_carbapenem"),
                 ("抗MRSA + 抗GNR广谱", "broad_mrsa_plus_gnr"),
                 ("碳青霉烯 + 万古/利奈唑胺", "broad_carb_vanc"),
                 ("≥3 类抗生素且含抗MRSA", "broad_ge3_mrsa")]:
    yv = df[col].values.astype(int)
    r = {"label": lab, "variable": col, "n": int(val.sum()),
         "events_val": int(yv[val].sum()), "prevalence_val": float(yv[val].mean()),
         "auroc_lr": roc_auc_score(yv[val], p_lr),
         "auprc_lr": average_precision_score(yv[val], p_lr),
         "auroc_gb": roc_auc_score(yv[val], p_gb),
         "auprc_gb": average_precision_score(yv[val], p_gb)}
    sens_rows.append(r)
    print(f"  {lab:28s} 留出集事件 {r['events_val']:>6,} ({100*r['prevalence_val']:5.2f}%)  "
          f"AUROC(LR) {r['auroc_lr']:.4f}  AUROC(HGB) {r['auroc_gb']:.4f}", flush=True)

# ---------------------------------------------------------------- 10. 价值链第一环
print("\n=== 价值链第一环：模型能否挑出『不需要大包围』的低风险人群 ===", flush=True)
thr = np.percentile(p_lr, 25)                 # 留出集风险最低四分位
lowrisk_v = (p_lr <= thr)                     # 留出集长度
lowrisk_full = np.zeros(len(df), bool); lowrisk_full[val] = lowrisk_v
vdf["lowrisk"] = lowrisk_v
print(f"  低风险阈值(留出集 P25) = {thr:.4f}")
chain_rows = []
for lab, mfull in [("低风险四分位", lowrisk_full), ("其余人群", val & ~lowrisk_full)]:
    sub_ = df[mfull]
    row = {"group": lab, "n": len(sub_),
           "broad_rate": float(sub_.broad_ge3.mean()),
           "mortality": float(sub_.hospital_expire_flag.mean()),
           "los_median_h": float(sub_.los_h.median())}
    chain_rows.append(row)
    print(f"    {lab:12s} n={len(sub_):>7,}  实际大包围率 {100*row['broad_rate']:5.2f}%  "
          f"死亡率 {100*row['mortality']:5.2f}%  LOS中位 {row['los_median_h']:.1f} h", flush=True)

low = df[lowrisk_full]
for lab, sub_ in [("低风险组内·实际用了大包围", low[low.broad_ge3 == 1]),
                  ("低风险组内·实际未用大包围", low[low.broad_ge3 == 0])]:
    if len(sub_) == 0:
        continue
    row = {"group": lab, "n": len(sub_),
           "broad_rate": float(sub_.broad_ge3.mean()),
           "mortality": float(sub_.hospital_expire_flag.mean()),
           "los_median_h": float(sub_.los_h.median())}
    chain_rows.append(row)
    print(f"    {lab:22s} n={len(sub_):>7,}  死亡 {100*row['mortality']:5.2f}%  "
          f"LOS中位 {row['los_median_h']:.1f} h", flush=True)

print("  按预测风险五分位（留出集）:", flush=True)
q = pd.qcut(p_lr, 5, labels=False, duplicates="drop")
vdf["risk_q"] = q
for k, g in vdf.groupby("risk_q"):
    row = {"group": f"风险 Q{int(k)+1}", "n": len(g),
           "broad_rate": float(g.broad_ge3.mean()),
           "mortality": float(g.hospital_expire_flag.mean()),
           "los_median_h": float(g.los_h.median())}
    chain_rows.append(row)
    print(f"    Q{int(k)+1}  n={len(g):>7,}  大包围率 {100*row['broad_rate']:5.2f}%  "
          f"死亡 {100*row['mortality']:5.2f}%  LOS中位 {row['los_median_h']:6.1f} h", flush=True)

# ---------------------------------------------------------------- 11. 保存
np.save(f"{OUT}/_p_lr.npy", p_lr); np.save(f"{OUT}/_p_gb.npy", p_gb)
with open(f"{OUT}/mimic_model_variables.json", "w", encoding="utf-8") as f:
    json.dump({"features": inter, "n_features": len(inter), "year_cut": cut,
               "split": "holdout_by_deidentified_admission_year_70_30",
               "split_note": "MIMIC-IV v3.0 的入院日期经去标识化随机平移，年份不含真实日历时间，故为留出验证而非时间外验证",
               "seed": SEED, "dev_n": int(dev.sum()), "val_n": int(val.sum()),
               "lasso_C": float(best_C),
               "lasso_cv_auroc_subsample": float(gs.best_score_),
               "lasso_subsample_n": int(len(sub_dev := Xd_sub)),
               "n_candidate_features": len(FN),
               "winsorize_pct": [1, 99], "count_features_excluded": True,
               "bootstrap_reps": int(len(boot))},
              f, ensure_ascii=False, indent=1)
pd.DataFrame(rows).to_csv(f"{OUT}/mimic_model_metrics.csv", index=False)
cv_tab.to_csv(f"{OUT}/mimic_lasso_cv.csv", index=False)
pd.DataFrame(cal_tab).to_csv(f"{OUT}/mimic_model_calibration.csv", index=False)
pd.DataFrame(avail_rows).to_csv(f"{OUT}/mimic_model_dataavailability.csv", index=False)
pd.DataFrame([boot_row]).to_csv(f"{OUT}/mimic_model_bootstrap.csv", index=False)
pd.DataFrame(sub_rows).to_csv(f"{OUT}/mimic_model_subgroups.csv", index=False)
pd.DataFrame(dca_rows).to_csv(f"{OUT}/mimic_model_dca.csv", index=False)
pd.DataFrame(sens_rows).to_csv(f"{OUT}/mimic_model_sensitivity.csv", index=False)
pd.DataFrame(chain_rows).to_csv(f"{OUT}/mimic_valuechain.csv", index=False)
df[["hadm_id", "subject_id", "admittime", "adm_year", "broad_ge3", "broad_carbapenem",
    "broad_mrsa_plus_gnr", "broad_ge3_mrsa", "hospital_expire_flag", "los_h",
    "n24", "n72", "micro_evidence_before_abx", "t_abx_first", "admission_type",
    "race", "gender", "anchor_age", "has_lab_6h", "has_vit_6h", "data_avail"]].to_csv(
    f"{OUT}/mimic_cohort_analysis.csv", index=False)
print("\n已导出模型指标、变量清单、校准/可得性分层/亚组/bootstrap/决策曲线/敏感性与分析队列")
print(f"总耗时 {time.time()-t_start:.0f}s")
