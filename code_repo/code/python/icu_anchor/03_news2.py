# -*- coding: utf-8 -*-
"""分支【ICU 锚点为主分析】：与 NEWS2 / qSOFA 头对头比较。

为什么必须做
  查新硬约束：AI 模型相对传统评分的增量有限（PMID 42506073），NEWS2 常胜过商业模型
  （PMID 39405061），医师经验判断 0.90 也常胜过 LASSO 0.84（PMID 38530675）。
  只报"模型 AUROC 0.74"无法回答"这个模型值得进临床吗"。

计算口径（RCP 2017 NEWS2，SpO2 scale 1）
  · 呼吸频率取 ICUadm 0–6 h 最高值；SpO2 取最低（脉氧与动脉血氧合并取低）
  · 收缩压取最低值；心率取最高值；GCS 取最低值（床旁三分量求和，覆盖 94.8%）
  · 体温统一换算为摄氏度（华氏列换算后与摄氏列合并取最差）
  · 吸氧：0–6 h 内 O2 Flow>0 或 Inspired O2 Fraction>0.21 或 O2 Delivery Device 记录
  · 意识：GCS<15 计 +3（以 GCS 替代 NEWS2 的 ACVPU "新发意识模糊"）
  · 缺失分量按 0 分计入，并报告各分量可得率

输出：results/icu_news2_compare.csv、icu_news2_delta.csv、icu_news2_bands.csv
"""
import duckdb, json, os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

BRANCH_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(BRANCH_ROOT, "results")
SEED = 20261105
np.random.seed(SEED)

con = duckdb.connect("D:/mimic_iv_branch_icu.duckdb", read_only=True)
df = con.execute("SELECT * FROM icu_feats").df()
con.close()
print("ICU 锚点队列:", len(df), " 事件:", int(df.broad_ge3.sum()),
      f"({100*df.broad_ge3.mean():.2f}%)", flush=True)

# ---------------------------------------------------------------- 打分
def tmin_max(r):
    """体温统一为摄氏度：华氏列换算后与摄氏列合并取最差值"""
    cands_min, cands_max = [], []
    if pd.notna(r.v_tempC_min):
        cands_min.append(float(r.v_tempC_min)); cands_max.append(float(r.v_tempC_max))
    if pd.notna(r.v_tempF_min):
        cands_min.append((float(r.v_tempF_min) - 32) * 5 / 9)
        cands_max.append((float(r.v_tempF_max) - 32) * 5 / 9)
    tmin = min(cands_min) if cands_min else np.nan
    tmax = max(cands_max) if cands_max else np.nan
    return tmin, tmax


def news2(r):
    pts, comps = 0, 0
    rr = r.v_rr_max
    if pd.notna(rr):
        comps += 1
        pts += 3 if rr <= 8 else 1 if rr <= 11 else 0 if rr <= 20 else 2 if rr <= 24 else 3
    s1, s2 = r.v_spo2_pulse_min, r.v_spo2_arterial_min
    if pd.notna(s1) or pd.notna(s2):
        spo2 = np.nanmin([x for x in (s1, s2) if pd.notna(x)])
        comps += 1
        pts += 3 if spo2 <= 91 else 2 if spo2 <= 93 else 0
    on_o2 = any(bool(x) if pd.notna(x) else False
                for x in (r.on_o2_flow, r.on_o2_fio2, r.on_o2_device))
    if on_o2:
        comps += 1; pts += 2
    sbp = r.v_sbp_min
    if pd.notna(sbp):
        comps += 1
        pts += 3 if sbp <= 90 else 2 if sbp <= 100 else 1 if sbp <= 110 else 0 if sbp <= 219 else 3
    hr = r.v_hr_max
    if pd.notna(hr):
        comps += 1
        pts += 3 if hr <= 40 else 1 if hr <= 50 else 0 if hr <= 90 else 1 if hr <= 110 else 2 if hr <= 130 else 3
    gcs = r.v_gcs_total
    if pd.notna(gcs):
        comps += 1
        pts += 3 if gcs < 15 else 0
    tmin, tmax = tmin_max(r)
    if pd.notna(tmin):
        comps += 1
        pts += 3 if tmin <= 35.0 else 1 if tmin <= 36.0 else 0 if tmin <= 38.0 else 1 if tmin <= 39.0 else 2
    if pd.notna(tmax):
        comps += 1
        pts += 3 if tmax <= 35.0 else 1 if tmax <= 36.0 else 0 if tmax <= 38.0 else 1 if tmax <= 39.0 else 2
    return pts, comps


def qsofa(r):
    p = 0
    if pd.notna(r.v_rr_max) and r.v_rr_max >= 22: p += 1
    if pd.notna(r.v_sbp_min) and r.v_sbp_min <= 100: p += 1
    if pd.notna(r.v_gcs_total) and r.v_gcs_total < 15: p += 1
    return p


print("\n=== 计算 NEWS2 / qSOFA ===", flush=True)
sc = df.apply(news2, axis=1, result_type="expand")
df["news2"] = sc[0].astype(float)
df["news2_comps"] = sc[1].astype(int)
df["qsofa"] = df.apply(qsofa, axis=1).astype(float)
print(f"  NEWS2 分量齐全(8/8，含体温最小与最大两个分量): {(df.news2_comps==8).sum():,} / {len(df):,} "
      f"({100*(df.news2_comps==8).mean():.1f}%)")
print("  各分量可得率:")
for lab, col in [("呼吸频率", "v_rr_max"), ("SpO2", "v_spo2_pulse_min"),
                 ("收缩压", "v_sbp_min"), ("心率", "v_hr_max"),
                 ("GCS 总分", "v_gcs_total")]:
    k = df[col].notna().sum()
    print(f"     {lab:8s} {k:>7,}  ({100*k/len(df):5.1f}%)")
k = ((df.on_o2_flow.fillna(0) + df.on_o2_fio2.fillna(0) + df.on_o2_device.fillna(0)) > 0).sum()
print(f"     {'吸氧':8s} {k:>7,}  ({100*k/len(df):5.1f}%)")
kt = (df.v_tempC_min.notna() | df.v_tempF_min.notna()).sum()
print(f"     {'体温(C或F)':10s} {kt:>7,}  ({100*kt/len(df):5.1f}%)")
print(f"  NEWS2 中位 {df.news2.median():.0f}（IQR {df.news2.quantile(.25):.0f}–{df.news2.quantile(.75):.0f}），"
      f"qSOFA 中位 {df.qsofa.median():.0f}")
df[["hadm_id", "news2", "news2_comps", "qsofa", "broad_ge3", "hospital_expire_flag",
    "los_h", "adm_year"]].to_csv(os.path.join(OUT, "icu_news2_scores.csv"), index=False)

# ---------------------------------------------------------------- 切分（与 02_model.py 完全一致）
Y = df.broad_ge3.values.astype(int)
# 与 02_model.py 一致的哑变量（这些列在 02 里临时生成，这里重建）
for lv in ["BLACK", "ASIAN", "HISPANIC", "OTHER"]:
    df[f"race_{lv}"] = (df.race.astype(str).str.upper() == lv).astype(int)
for lv in ["ELECTIVE", "URGENT", "NEWBORN"]:
    df[f"atype_{lv}"] = (df.admission_type.astype(str).str.upper() == lv).astype(int)
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
rng_split = np.random.default_rng(SEED)
dev = np.zeros(len(df), bool)
for cls in (0, 1):
    idx = np.where(Y == cls)[0]
    k = int(round(0.70 * len(idx)))
    dev[rng_split.permutation(idx)[:k]] = True
val = ~dev
vidx = np.where(val)[0]
print(f"\n留出集 {len(vidx):,} 例，事件 {int(Y[val].sum()):,}", flush=True)

with open(os.path.join(OUT, "icu_model_variables.json"), encoding="utf-8") as f:
    mvars = json.load(f)
FEATS = mvars["features"]
PORT = mvars["portable_features"]
X = df[FEATS].astype(float).values
keep = ~np.all(np.isnan(X), axis=0)
X = X[:, keep]
FN = [f for f, k in zip(FEATS, keep) if k]
lo = np.nanpercentile(X[dev], 1, axis=0)
hi = np.nanpercentile(X[dev], 99, axis=0)
X = np.clip(X, lo, hi)
imp = SimpleImputer(strategy="median").fit(X[dev])
X_imp = imp.transform(X)
X_imp = StandardScaler().fit(X_imp[dev]).transform(X_imp)
XI = X_imp[:, [FN.index(f) for f in FEATS]]
Xi_p = X_imp[:, [FN.index(f) for f in PORT]]

Yv = Y[val]
rows = []


def add(name, p):
    rows.append({"model": name, "auroc": roc_auc_score(Yv, p),
                 "auprc": average_precision_score(Yv, p)})
    print(f"  {name:34s} AUROC {rows[-1]['auroc']:.4f}  AUPRC {rows[-1]['auprc']:.4f}", flush=True)


print("\n=== 留出集上各模型对比（同一标签、同一指标）===", flush=True)
m = LogisticRegression(max_iter=3000).fit(df[["news2"]].values[dev], Y[dev])
p_news2 = m.predict_proba(df[["news2"]].values[val])[:, 1]
add("NEWS2（连续, 单独）", p_news2)

df["male"] = (df.gender.astype(str).str.upper() == "M").astype(int)
Xs = df[["news2", "anchor_age", "male"]].astype(float).values
Xs = StandardScaler().fit(Xs[dev]).transform(Xs)
m = LogisticRegression(max_iter=3000).fit(Xs[dev], Y[dev])
p_s = m.predict_proba(Xs[val])[:, 1]
add("NEWS2 + 年龄 + 性别", p_s)

m = LogisticRegression(max_iter=3000).fit(df[["qsofa"]].values[dev], Y[dev])
p_q = m.predict_proba(df[["qsofa"]].values[val])[:, 1]
add("qSOFA（连续, 单独）", p_q)

# ML 模型的预测直接取自 02_model.py（唯一真源，避免两次拟合出现数值差）
p_lr = np.load(os.path.join(OUT, "_icu_p_lr.npy"))
p_gb = np.load(os.path.join(OUT, "_icu_p_gb.npy"))
p_lrp = np.load(os.path.join(OUT, "_icu_p_portable.npy"))
add("完整 ML（Logistic）", p_lr)
add("完整 ML（HistGradientBoosting）", p_gb)
add("可迁移子模型（不含生命体征）", p_lrp)

# 公平对照：体温分量只有 8.6% 的患者有记录（NEWS2 体温分量在 91.4% 的人中按 0 分计，
# 会系统性低估 NEWS2）。因此在"体温完整"子集上再做一次对照，把缺失影响降到最低。
tc = df.v_tempC_min.notna() | df.v_tempF_min.notna()
print(f"\n=== 公平对照：体温完整的子集（n={int(tc.sum()):,}，占 {100*tc.mean():.1f}%）===", flush=True)
tcm = val & tc.values
Ytc = Y[tcm]
m = LogisticRegression(max_iter=3000).fit(df[["news2"]].values[dev], Y[dev])
p_n_tc = m.predict_proba(df[["news2"]].values[tcm])[:, 1]
m2 = LogisticRegression(max_iter=3000, C=1.0).fit(XI[dev], Y[dev])
p_m_tc = m2.predict_proba(XI[tcm])[:, 1]
a1 = roc_auc_score(Ytc, p_n_tc); a2 = roc_auc_score(Ytc, p_m_tc)
print(f"  体温完整子集 n={int(tcm.sum()):,}，事件 {int(Ytc.sum()):,}")
print(f"    NEWS2 AUROC {a1:.4f}   完整ML AUROC {a2:.4f}   ΔAUROC {a2-a1:+.4f}", flush=True)
rng_tc = np.random.default_rng(SEED)
d_tc = []
for _ in range(2000):
    s_ = rng_tc.integers(0, len(Ytc), len(Ytc))
    if len(set(Ytc[s_])) < 2:
        continue
    d_tc.append(roc_auc_score(Ytc[s_], p_m_tc[s_]) - roc_auc_score(Ytc[s_], p_n_tc[s_]))
d_tc = np.array(d_tc)
print(f"    配对 bootstrap 95%CI {np.percentile(d_tc,2.5):+.4f} – {np.percentile(d_tc,97.5):+.4f}",
      flush=True)

# ---------------------------------------------------------------- 配对 bootstrap ΔAUROC
print("\n=== 配对 bootstrap（留出集重抽样 2000 次，ΔAUROC 的 95%CI）===", flush=True)
rng = np.random.default_rng(SEED)
defs = [("完整ML(LR) − NEWS2", p_lr, p_news2),
        ("完整ML(HGB) − NEWS2", p_gb, p_news2),
        ("完整ML(LR) − qSOFA", p_lr, p_q),
        ("完整ML(LR) − (NEWS2+人口学)", p_lr, p_s)]
if len(PORT) >= 5:
    defs.append(("可迁移子模型 − NEWS2", p_lrp, p_news2))
diff_rows = []
for nm, pa, pb in defs:
    d = []
    for _ in range(2000):
        s_ = rng.integers(0, len(Yv), len(Yv))
        if len(set(Yv[s_])) < 2:
            continue
        d.append(roc_auc_score(Yv[s_], pa[s_]) - roc_auc_score(Yv[s_], pb[s_]))
    d = np.array(d)
    obs = roc_auc_score(Yv, pa) - roc_auc_score(Yv, pb)
    lo, hi = np.percentile(d, [2.5, 97.5])
    diff_rows.append({"comparison": nm, "delta_auroc": obs, "ci_low": lo, "ci_high": hi,
                      "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean()))})
    print(f"  {nm:28s} ΔAUROC {obs:+.4f}  95%CI {lo:+.4f} – {hi:+.4f}  "
          f"P={diff_rows[-1]['p_two_sided']:.4f}", flush=True)

# ---------------------------------------------------------------- 分层的标签率
print("\n=== 按 NEWS2 / qSOFA 分层的『大包围』阳性率（留出集）===", flush=True)
vd = df[val].copy()
bands = []
for lo_, hi_, lab in [(0, 2, "NEWS2 0–2"), (3, 4, "NEWS2 3–4"), (5, 6, "NEWS2 5–6"), (7, 99, "NEWS2 ≥7")]:
    g = vd[(vd.news2 >= lo_) & (vd.news2 <= hi_)]
    if len(g) == 0:
        continue
    bands.append({"stratum": lab, "n": len(g), "events": int(g.broad_ge3.sum()),
                  "rate": float(g.broad_ge3.mean()), "mortality": float(g.hospital_expire_flag.mean()),
                  "los_median_h": float(g.los_h.median())})
for k in sorted(vd.qsofa.unique()):
    g = vd[vd.qsofa == k]
    bands.append({"stratum": f"qSOFA {int(k)}", "n": len(g), "events": int(g.broad_ge3.sum()),
                  "rate": float(g.broad_ge3.mean()), "mortality": float(g.hospital_expire_flag.mean()),
                  "los_median_h": float(g.los_h.median())})
bands = pd.DataFrame(bands)
for _, r in bands.iterrows():
    print(f"  {r.stratum:12s} n={int(r.n):>6,}  大包围率 {100*r.rate:5.2f}%  "
          f"死亡 {100*r.mortality:5.2f}%  LOS中位 {r.los_median_h:.1f} h")

rows.append({"model": "【公平对照】体温完整子集 · NEWS2",
             "auroc": a1, "auprc": average_precision_score(Ytc, p_n_tc)})
rows.append({"model": "【公平对照】体温完整子集 · 完整ML",
             "auroc": a2, "auprc": average_precision_score(Ytc, p_m_tc)})
diff_rows.append({"comparison": "完整ML − NEWS2（体温完整子集）",
                  "delta_auroc": a2 - a1,
                  "ci_low": float(np.percentile(d_tc, 2.5)),
                  "ci_high": float(np.percentile(d_tc, 97.5)),
                  "p_two_sided": float(2 * min((d_tc <= 0).mean(), (d_tc >= 0).mean()))})
pd.DataFrame(rows).to_csv(os.path.join(OUT, "icu_news2_compare.csv"), index=False)
pd.DataFrame(diff_rows).to_csv(os.path.join(OUT, "icu_news2_delta.csv"), index=False)
bands.to_csv(os.path.join(OUT, "icu_news2_bands.csv"), index=False)
np.save(os.path.join(OUT, "_icu_p_news2.npy"), p_news2)
np.save(os.path.join(OUT, "_icu_p_qsofa.npy"), p_q)
print("\n已导出 NEWS2 头对头结果")
