# -*- coding: utf-8 -*-
"""分支【ICU 锚点为主分析】：预测"ICU 入院后是否需要大包围"。

设计
  人群      住院后 24h 内收治入 ICU、且住院 24h 内启动经验性抗生素者（37,145）
  锚点      ICU 入院时刻（intime）；距住院入院中位 1.57 h
  预测变量  ICU 入院后 0–6h 的生命体征与实验室统计量 + 人口学 + 入院诊断合并症
  标签      ICUadm 后 24h 内使用 ≥3 类抗菌药物（broad_ge3，事件 26.59%）
  验证      ① 时间外切分（按 ICU 入院年份） ② bootstrap 1000 次 ③ 校准 ④ 决策曲线
            ⑤ 亚组 ⑥ 多标签敏感性 ⑦ 可迁移子模型（无生命体征）
  价值链    第一环：低风险组能否识别出"最终不需要大包围"的人，并用死亡/LOS 佐证

预处理三处硬决定（均写进自查记录）
  1) Winsorize（开发集 1%/99% 分位）：原始数据有录入伪影（MAP 100,369 mmHg、HR 8,400、
     SpO2 2,395），不处理时 L1-LASSO 只选中 1 个变量。
  2) 剔除 *_n 计数类变量：检测次数属"观测过程特征"，是跨库 shortcut，且无法迁移到
     无连续监测的医院（本院队列即无生命体征）。
  3) 校准回归用无惩罚 logistic（statsmodels Logit），sklearn 1.0.2 无 penalty=None。

输出：results/icu_model_*.csv、icu_model_variables.json
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

BRANCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(BRANCH_ROOT, "results")
os.makedirs(OUT, exist_ok=True)
SEED = 20261105
np.random.seed(SEED)
t_start = time.time()

con = duckdb.connect("D:/mimic_iv_branch_icu.duckdb", read_only=True)
df = con.execute("SELECT * FROM icu_feats").df()
con.close()
Y = df.broad_ge3.values.astype(int)
print("ICU 锚点队列:", len(df), " 事件:", int(Y.sum()), f"({100*Y.mean():.2f}%)", flush=True)

# ---------------------------------------------------------------- 变量定义
CM  = [c for c in df.columns if c.startswith("cm_")]
VIT = [c for c in df.columns if c.startswith("v_") and not c.endswith("_n")]
LAB = [c for c in df.columns if c.startswith("l_") and not c.endswith("_n")]
df["age"] = df.anchor_age.astype(float)
NUM = ["age", "gender"] + CM + VIT + LAB
NUM = [f for f in NUM if f in df.columns and df[f].dtype != object]
RACE_REF, ATYPE_REF = "WHITE", "EMERGENCY"
# 种族按 MIMIC-IV 原始字符串前缀归并（"BLACK/AFRICAN AMERICAN"、"ASIAN - CHINESE"、
# "HISPANIC/LATINO - PUERTO RICAN" 等；参照类 = White，其余归入 Other/unknown）
_race_u = df.race.astype(str).str.upper()
df["race_BLACK"] = _race_u.str.startswith("BLACK").astype(int)
df["race_ASIAN"] = _race_u.str.startswith("ASIAN").astype(int)
df["race_HISPANIC"] = _race_u.str.startswith("HISPANIC").astype(int)
df["race_OTHER"] = (~_race_u.str.startswith(("WHITE", "BLACK", "ASIAN", "HISPANIC"))).astype(int)
_at_u = df.admission_type.astype(str).str.upper()
df["atype_ELECTIVE"] = (_at_u == "ELECTIVE").astype(int)
df["atype_URGENT"] = (_at_u == "URGENT").astype(int)
df["atype_SURGICAL"] = _at_u.str.contains("SURGICAL").astype(int)
df["atype_OBSERVATION"] = _at_u.str.contains("OBSERVATION").astype(int)
DUM = ["race_BLACK", "race_ASIAN", "race_HISPANIC", "race_OTHER",
       "atype_ELECTIVE", "atype_URGENT", "atype_SURGICAL", "atype_OBSERVATION"]
FEATS = NUM + DUM
print("候选变量:", len(FEATS), "= 人口学 2 + 合并症", len(CM), "+ 生命体征统计量", len(VIT),
      "+ 实验室统计量", len(LAB), "+ 哑变量", len(DUM), flush=True)

# 数据可得性分层（描述用）
lab_n = [c for c in df.columns if c.startswith("l_") and c.endswith("_n")]
vit_n = [c for c in df.columns if c.startswith("v_") and c.endswith("_n")]
has_lab = (df[lab_n].sum(axis=1) > 0).values
has_vit = (df[vit_n].sum(axis=1) > 0).values
df["data_avail"] = np.where(has_lab & has_vit, "两者都有",
                     np.where(has_lab, "仅实验室", np.where(has_vit, "仅生命体征", "皆无")))
print("\n=== ICUadm 0–6h 数据可得性 ===", flush=True)
for k, g in df.groupby("data_avail"):
    print(f"  {k:10s} n={len(g):>7,} ({100*len(g)/len(df):5.1f}%)  大包围率 {100*g.broad_ge3.mean():5.2f}%  "
          f"死亡 {100*g.hospital_expire_flag.mean():5.2f}%")

# ---------------------------------------------------------------- 训练/验证切分
# 重要：MIMIC-IV 的日期经去标识化随机平移，admittime 的年份**不含真实时间信息**
#   （实测：同一患者多次住院的 hadm_id 顺序与 admittime 顺序一致率仅 23.8%，接近随机；
#     hadm_id 与 admittime 的 Spearman 相关 = 0.008）。
#   因此"按年份切分"并不是时间外验证。本分支改用**分层随机留出验证**（种子固定，可复现），
#   并在方法与局限中如实说明：MIMIC-IV 内部无法完成真正的时间外验证，
#   外部验证需依赖真实临床数据（本院队列）。
YEAR = df.adm_year.values
rng_split = np.random.default_rng(SEED)
dev = np.zeros(len(df), bool)
for cls in (0, 1):
    idx = np.where(Y == cls)[0]
    k = int(round(0.70 * len(idx)))
    dev[rng_split.permutation(idx)[:k]] = True
val = ~dev
vidx = np.where(val)[0]
print(f"\n分层随机留出: 开发 {dev.sum():,} 例 / 留出 {val.sum():,} 例"
      f"（开发集事件 {int(Y[dev].sum()):,}，留出集事件 {int(Y[val].sum()):,}）", flush=True)

X = df[FEATS].astype(float).values
keep = ~np.all(np.isnan(X), axis=0)
X = X[:, keep]
FN = [f for f, k in zip(FEATS, keep) if k]
lo = np.nanpercentile(X[dev], 1, axis=0)
hi = np.nanpercentile(X[dev], 99, axis=0)
Xw = np.clip(X, lo, hi)
print(f"Winsorize（开发集 1%/99%）：截断单元格 {int(((X[dev] < lo) | (X[dev] > hi)).sum()):,}", flush=True)
imp = SimpleImputer(strategy="median").fit(Xw[dev])
X_imp = imp.transform(Xw)
X_imp = StandardScaler().fit(X_imp[dev]).transform(X_imp)
print("插补（中位数）+ 标准化完成（均在开发集上拟合）", flush=True)

# ---------------------------------------------------------------- 变量筛选
print("\n=== 变量筛选（L1-LASSO，C 由 5 折 CV 选 ∩ 单变量 F 检验 top-50）===", flush=True)
dev_idx = np.where(dev)[0]
rng = np.random.default_rng(SEED)
sub = rng.choice(len(dev_idx), min(30000, len(dev_idx)), replace=False)
Xd_sub, Yd_sub = X_imp[dev_idx][sub], Y[dev_idx][sub]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
C_GRID = [0.005, 0.02, 0.1, 0.5, 2.0]
t0 = time.time()
gs = GridSearchCV(LogisticRegression(penalty="l1", solver="liblinear", max_iter=500,
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
lasso = LogisticRegression(penalty="l1", solver="liblinear", C=best_C, max_iter=1000,
                            tol=1e-3, random_state=SEED).fit(X_imp[dev], Y[dev])
print(f"  全开发集 L1 拟合（{time.time()-t0:.0f}s，n_iter={lasso.n_iter_[0]}）", flush=True)
lasso_sel = {FN[i] for i in np.where(np.abs(lasso.coef_[0]) > 1e-8)[0]}
skb = SelectKBest(f_classif, k=min(50, len(FN))).fit(X_imp[dev], Y[dev])
f_sel = {FN[i] for i in np.where(skb.get_support())[0]}
print(f"  LASSO 选中 {len(lasso_sel)}，F 检验 top-50 选中 {len(f_sel)}", flush=True)
inter = sorted(lasso_sel & f_sel)
if len(inter) < 8:
    inter = sorted(lasso_sel | f_sel)[:40]
    print("  交集过小，改用并集前 40 个", flush=True)
print(f"  最终变量数 {len(inter)}: {inter}", flush=True)
XI = X_imp[:, [FN.index(f) for f in inter]]

# ---------------------------------------------------------------- 模型评估
def calibration(p, y):
    eps = 1e-6
    pc = np.clip(p, eps, 1 - eps)
    lp = np.log(pc / (1 - pc))
    try:
        res = sm.Logit(np.asarray(y, float), sm.add_constant(lp.reshape(-1, 1))).fit(disp=0, maxiter=200)
        return float(res.params[1]), float(res.params[0])
    except Exception as e:
        print("   校准回归失败:", e)
        return np.nan, np.nan


def fit_eval(Xtr, ytr, Xte, yte, model, name):
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    slope, inter_ = calibration(p, yte)
    res = {"model": name, "n_test": len(yte), "events": int(yte.sum()),
           "auroc": roc_auc_score(yte, p), "auprc": average_precision_score(yte, p),
           "brier": brier_score_loss(yte, p), "prevalence": float(yte.mean()),
           "cal_slope": slope, "cal_intercept": inter_}
    return model, p, res


def make_lr():
    return LogisticRegression(solver="lbfgs", max_iter=3000, C=1.0)


def make_gb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                           min_samples_leaf=60, l2_regularization=1.0,
                                           early_stopping=False, random_state=SEED)


print("\n=== 主模型（开发集拟合 → 时间外验证集评估）===", flush=True)
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

# 可迁移子模型（无生命体征、无计数类；供无连续监测的医院使用）
PORT = [f for f in inter if not f.startswith("v_")]
if len(PORT) >= 5:
    Xi_p = X_imp[:, [FN.index(f) for f in PORT]]
    lr_p, p_lrp, r_lrp = fit_eval(Xi_p[dev], Y[dev], Xi_p[val], Y[val], make_lr(),
                                   "Logistic（可迁移子模型：不含生命体征）")
    rows.append(r_lrp)
    print(f"  {r_lrp['model']:22s} AUROC {r_lrp['auroc']:.4f}  AUPRC {r_lrp['auprc']:.4f}", flush=True)

# 校准表
print("\n=== 校准（验证集，按预测概率五分位）===", flush=True)
vdf = df[val].copy(); vdf["p_lr"] = p_lr; vdf["p_gb"] = p_gb
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

# ---------------------------------------------------------------- bootstrap
N_BOOT = 1000
print(f"\n=== bootstrap 内部验证（{N_BOOT} 次：开发集重抽样 → 时间外验证集评估）===", flush=True)
rng = np.random.default_rng(SEED)
boot = []
t0 = time.time()
for b in range(N_BOOT):
    idx = dev_idx[rng.integers(0, len(dev_idx), len(dev_idx))]
    if Y[idx].sum() < 10 or len(set(Y[idx])) < 2:
        continue
    try:
        m = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0).fit(XI[idx], Y[idx])
        pp = m.predict_proba(XI[val])[:, 1]
        boot.append((roc_auc_score(Y[val], pp), average_precision_score(Y[val], pp)))
    except Exception:
        continue
    if b % 250 == 0:
        print(f"    {b}/{N_BOOT} ({time.time()-t0:.0f}s)", flush=True)
boot = np.array(boot)
print(f"  有效重复 {len(boot)}  AUROC {boot[:,0].mean():.4f} "
      f"(95%CI {np.percentile(boot[:,0],2.5):.4f}–{np.percentile(boot[:,0],97.5):.4f})  "
      f"AUPRC {boot[:,1].mean():.4f} "
      f"(95%CI {np.percentile(boot[:,1],2.5):.4f}–{np.percentile(boot[:,1],97.5):.4f})", flush=True)
pd.DataFrame([{"model": "Logistic", "auroc_lo": float(np.percentile(boot[:,0],2.5)), "auroc_hi": float(np.percentile(boot[:,0],97.5)), "auroc_mean": float(boot[:,0].mean()), "auprc_lo": float(np.percentile(boot[:,1],2.5)), "auprc_hi": float(np.percentile(boot[:,1],97.5)), "auprc_mean": float(boot[:,1].mean()), "reps": int(len(boot))}]).to_csv(
    "results/icu_model_bootstrap.csv", index=False)

# ---------------------------------------------------------------- 亚组
print("\n=== 亚组性能（时间外验证集）===", flush=True)
sub_rows = []
for name, mask in [
    ("全体", val),
    ("年龄 <65 岁", val & (df.anchor_age < 65)),
    ("年龄 ≥65 岁", val & (df.anchor_age >= 65)),
    ("女性", val & (df.gender.astype(str).str.upper() == "F")),
    ("男性", val & (df.gender.astype(str).str.upper() == "M")),
    ("住院 6h 内入 ICU", val & (df.adm_to_icu_h <= 6)),
    ("住院 6h 后入 ICU", val & (df.adm_to_icu_h > 6)),
    ("急诊入院", val & (df.admission_type.astype(str).str.upper().isin(["EW EMER.", "DIRECT EMER."]))),
    ("非急诊入院", val & (~df.admission_type.astype(str).str.upper().isin(["EW EMER.", "DIRECT EMER."]))),
    ("白种人", val & (_race_u.str.startswith("WHITE"))),
    ("黑人", val & (_race_u.str.startswith("BLACK"))),
    ("亚裔", val & (_race_u.str.startswith("ASIAN"))),
    ("西语裔", val & (_race_u.str.startswith("HISPANIC"))),
    ("有阳性培养", val & (df.micro_evidence_before_abx == 1)),
    ("无阳性培养", val & (df.micro_evidence_before_abx == 0)),
    ("0–6h 数据齐全", val & (df.data_avail == "两者都有")),
    ("0–6h 仅生命体征", val & (df.data_avail == "仅生命体征")),
]:
    if mask[val].sum() < 100 or len(set(Y[val][mask[val]])) < 2:
        continue
    mv_ = mask[val]
    Yv_ = Y[val]
    r = {"subgroup": name, "n": int(mv_.sum()), "events": int(Yv_[mv_].sum()),
         "prevalence": float(Yv_[mv_].mean()),
         "auroc_lr": roc_auc_score(Yv_[mv_], p_lr[mv_]),
         "auprc_lr": average_precision_score(Yv_[mv_], p_lr[mv_]),
         "auroc_gb": roc_auc_score(Yv_[mv_], p_gb[mv_]),
         "auprc_gb": average_precision_score(Yv_[mv_], p_gb[mv_])}
    sub_rows.append(r)
    print(f"  {name:16s} n={r['n']:>6,}  事件率 {100*r['prevalence']:5.2f}%  "
          f"AUROC(LR) {r['auroc_lr']:.4f}  AUROC(HGB) {r['auroc_gb']:.4f}", flush=True)

# ---------------------------------------------------------------- 决策曲线
def dca(y, p, lo_, hi_):
    out = []
    for t in np.linspace(lo_, hi_, 56):
        w = y / p
        tp = (w >= t).sum(); fp = ((w < t) & (p >= t)).sum()
        nb = (tp - fp * t / (1 - t)) / len(y)
        allm = (y.sum() - (len(y) - y.sum()) * t / (1 - t)) / len(y)
        out.append((t, nb, allm))
    return np.array(out)


print("\n=== 决策曲线（验证集，阈值概率 0.05–0.60）===", flush=True)
dca_rows = []
for nm, p in [("Logistic", p_lr), ("HGB", p_gb)]:
    d = dca(Y[val], p, 0.05, 0.60)
    mx = d[:, 1] - d[:, 2]
    i = int(np.argmax(mx)); j = int(np.argmin(abs(d[:, 0] - 0.10)))
    print(f"  {nm:9s} 净获益超 treat-all 的最大差 {mx[i]:.4f} @ 阈值 {d[i,0]:.3f}；"
          f"阈值 0.10 处 NB {d[j,1]:.4f} vs treat-all {d[j,2]:.4f}", flush=True)
    for t, nb, allm in d:
        dca_rows.append({"model": nm, "threshold": t, "net_benefit": nb,
                         "treat_all": allm, "treat_none": 0.0})

# ---------------------------------------------------------------- 多标签敏感性
print("\n=== 多标签敏感性（同一模型，换『大包围』定义）===", flush=True)
sens_rows = []
for lab, col in [("≥3 类抗菌药物（主标签）", "broad_ge3"),
                 ("≥2 类抗菌药物", "broad_ge2"),
                 ("碳青霉烯", "broad_carbapenem"),
                 ("抗MRSA + 抗GNR广谱", "broad_mrsa_plus_gnr"),
                 ("碳青霉烯 + 万古/利奈唑胺", "broad_carb_vanc"),
                 ("≥3 类 + 含抗MRSA", "broad_ge3_mrsa")]:
    yv = df[col].values.astype(int)
    r = {"label": lab, "variable": col, "n": int(val.sum()),
         "events_val": int(yv[val].sum()), "prevalence_val": float(yv[val].mean()),
         "auroc_lr": roc_auc_score(yv[val], p_lr), "auprc_lr": average_precision_score(yv[val], p_lr),
         "auroc_gb": roc_auc_score(yv[val], p_gb), "auprc_gb": average_precision_score(yv[val], p_gb)}
    sens_rows.append(r)
    print(f"  {lab:26s} 验证集事件 {r['events_val']:>5,} ({100*r['prevalence_val']:5.2f}%)  "
          f"AUROC(LR) {r['auroc_lr']:.4f}  AUROC(HGB) {r['auroc_gb']:.4f}", flush=True)

# ---------------------------------------------------------------- 价值链第一环
print("\n=== 价值链第一环：模型能否挑出『不需要大包围』的低风险人群 ===", flush=True)
thr = np.percentile(p_lr, 25)
lowrisk_val = (p_lr <= thr)               # 留出集内（p_lr 已与留出集对齐）
lowrisk = np.zeros(len(df), bool)
lowrisk[vidx] = lowrisk_val
vdf["lowrisk"] = lowrisk_val
print(f"  低风险阈值（验证集 P25）= {thr:.4f}")
chain_rows = []
for lab, m in [("低风险四分位", lowrisk), ("其余人群", val & ~lowrisk)]:
    sub_ = df[m]
    row = {"group": lab, "n": len(sub_), "broad_rate": float(sub_.broad_ge3.mean()),
           "mortality": float(sub_.hospital_expire_flag.mean()),
           "los_median_h": float(sub_.los_h.median())}
    chain_rows.append(row)
    print(f"    {lab:12s} n={len(sub_):>6,}  实际大包围率 {100*row['broad_rate']:5.2f}%  "
          f"死亡率 {100*row['mortality']:5.2f}%  LOS中位 {row['los_median_h']:.1f} h", flush=True)
low = df[lowrisk]
for lab, s_ in [("低风险组内·实际用了大包围", low[low.broad_ge3 == 1]),
                ("低风险组内·实际未用大包围", low[low.broad_ge3 == 0])]:
    if len(s_) == 0:
        continue
    row = {"group": lab, "n": len(s_), "broad_rate": float(s_.broad_ge3.mean()),
           "mortality": float(s_.hospital_expire_flag.mean()),
           "los_median_h": float(s_.los_h.median())}
    chain_rows.append(row)
    print(f"    {lab:22s} n={len(s_):>6,}  死亡 {100*row['mortality']:5.2f}%  "
          f"LOS中位 {row['los_median_h']:.1f} h", flush=True)
print("  按预测风险五分位（验证集）:", flush=True)
q = pd.qcut(p_lr, 5, labels=False, duplicates="drop")
vdf["risk_q"] = q
for k, g in vdf.groupby("risk_q"):
    row = {"group": f"风险 Q{int(k)+1}", "n": len(g), "broad_rate": float(g.broad_ge3.mean()),
           "mortality": float(g.hospital_expire_flag.mean()), "los_median_h": float(g.los_h.median())}
    chain_rows.append(row)
    print(f"    Q{int(k)+1}  n={len(g):>6,}  大包围率 {100*row['broad_rate']:5.2f}%  "
          f"死亡 {100*row['mortality']:5.2f}%  LOS中位 {row['los_median_h']:6.1f} h", flush=True)

# ---------------------------------------------------------------- 保存
np.save(os.path.join(OUT, "_icu_p_lr.npy"), p_lr)
np.save(os.path.join(OUT, "_icu_p_gb.npy"), p_gb)
np.save(os.path.join(OUT, "_icu_p_portable.npy"), p_lrp)
with open(os.path.join(OUT, "icu_model_variables.json"), "w", encoding="utf-8") as f:
    json.dump({"features": inter, "n_features": len(inter), "portable_features": PORT,
               "split": "stratified_random_holdout_70_30", "split_note":
               "MIMIC-IV 去标识化使 admittime 年份不含真实时间序（hadm_id 与 admittime "
               "Spearman=0.008；同一患者多次住院顺序一致率 23.8%），故非时间外验证",
               "seed": SEED, "dev_n": int(dev.sum()), "val_n": int(val.sum()),
               "events_dev": int(Y[dev].sum()), "events_val": int(Y[val].sum()),
               "lasso_C": float(best_C), "lasso_cv_auroc": float(gs.best_score_),
               "n_candidate": len(FN), "bootstrap_reps": int(len(boot)),
               "boot_auroc_ci": [float(np.percentile(boot[:, 0], 2.5)),
                                 float(np.percentile(boot[:, 0], 97.5))],
               "boot_auprc_ci": [float(np.percentile(boot[:, 1], 2.5)),
                                 float(np.percentile(boot[:, 1], 97.5))]}, f, ensure_ascii=False, indent=1)
pd.DataFrame(rows).to_csv(os.path.join(OUT, "icu_model_metrics.csv"), index=False)
cv_tab.to_csv(os.path.join(OUT, "icu_lasso_cv.csv"), index=False)
pd.DataFrame(cal_tab).to_csv(os.path.join(OUT, "icu_model_calibration.csv"), index=False)
pd.DataFrame(sub_rows).to_csv(os.path.join(OUT, "icu_model_subgroups.csv"), index=False)
pd.DataFrame(dca_rows).to_csv(os.path.join(OUT, "icu_model_dca.csv"), index=False)
pd.DataFrame(sens_rows).to_csv(os.path.join(OUT, "icu_model_sensitivity.csv"), index=False)
pd.DataFrame(chain_rows).to_csv(os.path.join(OUT, "icu_valuechain.csv"), index=False)
df.to_csv(os.path.join(OUT, "icu_cohort_analysis.csv"), index=False)
print("\n已导出 ICU 模型全部结果")
print(f"总耗时 {time.time()-t_start:.0f}s")
