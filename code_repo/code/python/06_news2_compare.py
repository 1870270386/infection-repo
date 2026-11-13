# -*- coding: utf-8 -*-
"""MIMIC-IV：与 NEWS2 / qSOFA 头对头比较——完整模型是否比标准严重度评分更能预测"需要大包围"。

为什么必须做
  查新硬约束：AI 模型相对传统评分的增量有限（PMID 42506073），NEWS2 甚至常胜过商业模型
  （PMID 39405061），医师 gestalt 0.90 也能打赢 LASSO 0.84（PMID 38530675）。
  只报"模型 AUROC 0.7x"无法回答"这个模型值得进临床吗"。

比较口径（严格同标签、同验证集、同指标）
  · 结局：入院 24 h 内 ≥3 类抗生素（与 05_model.py 完全一致）
  · 验证集：按去标识化年份分层的留出集（与 05_model.py 完全一致）
  · 对照：NEWS2（RCP 2017，SpO2 scale 1）、qSOFA（2016）
  · 差异显著性：配对 bootstrap 2000 次（对验证集重抽样），给 ΔAUROC 的 95%CI

NEWS2 的映射（全部在入院后 0–6 h 窗口内取最差值）
  · 呼吸频率 = v_rr_max；SpO2 = min(脉氧, 动脉血氧)；收缩压 = min(有创, 无创)；心率 = v_hr_max
  · 体温统一换算为摄氏度（华氏单独换算后与摄氏取最差值）
  · "吸氧" = on_o2（O2 Flow>0 或 FiO2>0.21 或有吸氧装置记录）
  · 意识 = GCS 总分 < 15 计 +3（以 GCS 替代 NEWS2 的 ACVPU"新发意识模糊"）
  · 缺失分量按 0 分计入（"未记录异常"），并记录分量齐全比例，另做完全病例敏感性分析

重要口径说明（必须写进手稿）
  MIMIC-IV 的生命体征只存在于 ICU，故 NEWS2 仅能在"有生命体征可得"的子集上计算
  （约占目标人群 15%）。因此头对头比较在**同一子集**内进行，并在正文如实标注。

脚本：06_news2_compare.py  |  输出：results/mimic_news2_*.csv
"""
import duckdb, json, os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = "studies/infection-sepsis-2026-11/results"
SEED = 20261105
np.random.seed(SEED)
t_start = time.time()

con = duckdb.connect("D:/mimic_iv.duckdb", read_only=True)
df = con.execute("SELECT * FROM feats2 WHERE empiric_24h=1").df()
con.close()
print("分析人群:", len(df), flush=True)

# ---------------------------------------------------------------- 1. NEWS2 / qSOFA
def c_to_f(x):
    return (x - 32) * 5 / 9

def temp_c(min_c, max_c, min_f, max_f):
    """华氏换算成摄氏后与摄氏列取最差值；全缺返回 (nan, nan)"""
    lo = [v for v in [min_c, c_to_f(min_f) if pd.notna(min_f) else np.nan] if pd.notna(v)]
    hi = [v for v in [max_c, c_to_f(max_f) if pd.notna(max_f) else np.nan] if pd.notna(v)]
    return (np.min(lo) if lo else np.nan, np.max(hi) if hi else np.nan)


def news2_row(r):
    """返回 (总分, 有效分量数, 核心四分量齐全)"""
    pts, comps = 0, 0
    # 呼吸频率
    if pd.notna(r.v_rr_max):
        comps += 1
        pts += 3 if r.v_rr_max <= 8 else 1 if r.v_rr_max <= 11 else 0 if r.v_rr_max <= 20 else 2 if r.v_rr_max <= 24 else 3
    # SpO2
    s = [v for v in [r.v_spo2_pls_min, r.v_spo2_art_min] if pd.notna(v)]
    if s:
        comps += 1
        spo2 = min(s)
        pts += 3 if spo2 <= 91 else 2 if spo2 <= 93 else 0
    # 吸氧
    if int(r.on_o2) == 1:
        comps += 1; pts += 2
    # 收缩压（有创 + 无创取低）
    b = [v for v in [r.v_sbp_min, r.v_nibp_sbp_min] if pd.notna(v)]
    if b:
        comps += 1
        sbp = min(b)
        pts += 3 if sbp <= 90 else 2 if sbp <= 100 else 1 if sbp <= 110 else 0 if sbp <= 219 else 3
    # 心率
    if pd.notna(r.v_hr_max):
        comps += 1
        hr = r.v_hr_max
        pts += 3 if hr <= 40 else 1 if hr <= 50 else 0 if hr <= 90 else 1 if hr <= 110 else 2 if hr <= 130 else 3
    # 意识（GCS）
    if pd.notna(r.v_gcs_total_min):
        comps += 1
        pts += 3 if r.v_gcs_total_min < 15 else 0
    # 体温（最高与最低各计一次）
    tmin, tmax = temp_c(r.v_tempC_min, r.v_tempC_max, r.v_tempF_min, r.v_tempF_max)
    if pd.notna(tmin):
        comps += 1
        pts += 3 if tmin <= 35.0 else 1 if tmin <= 36.0 else 0 if tmin <= 38.0 else 1 if tmin <= 39.0 else 2
    if pd.notna(tmax):
        comps += 1
        pts += 3 if tmax <= 35.0 else 1 if tmax <= 36.0 else 0 if tmax <= 38.0 else 1 if tmax <= 39.0 else 2
    core = int(pd.notna(r.v_rr_max) and pd.notna(r.v_hr_max) and bool(b) and bool(s))
    return pts, comps, core


def qsofa_row(r):
    p = 0
    if pd.notna(r.v_rr_max) and r.v_rr_max >= 22: p += 1
    b = [v for v in [r.v_sbp_min, r.v_nibp_sbp_min] if pd.notna(v)]
    if b and min(b) <= 100: p += 1
    if pd.notna(r.v_gcs_total_min) and r.v_gcs_total_min < 15: p += 1
    return p


print("\n=== 计算 NEWS2 / qSOFA ===", flush=True)
sc = df.apply(news2_row, axis=1, result_type="expand")
df["news2"] = sc[0].astype(float)
df["news2_comps"] = sc[1].astype(int)
df["news2_core_ok"] = sc[2].astype(int)
df["qsofa"] = df.apply(qsofa_row, axis=1).astype(float)
core = df.news2_core_ok.values == 1
print(f"  NEWS2 核心四分量（RR/SpO2/血压/心率）齐全: {core.sum():,} / {len(df):,} ({100*core.mean():.1f}%)")
print(f"  NEWS2 七分量齐全: {(df.news2_comps == 7).sum():,}（{100*(df.news2_comps == 7).mean():.1f}%）")
print(f"  NEWS2 中位 {df.news2.median():.0f}（IQR {df.news2.quantile(.25):.0f}–{df.news2.quantile(.75):.0f}）")
df[["hadm_id", "news2", "news2_comps", "news2_core_ok", "qsofa", "on_o2",
    "broad_ge3", "hospital_expire_flag", "los_h", "adm_year"]].to_csv(
    f"{OUT}/mimic_news2_scores.csv", index=False)

# ---------------------------------------------------------------- 2. 与 05 的验证集对齐
Y = df.broad_ge3.values.astype(int)
cut = int(np.quantile(df.adm_year.values, 0.70))
dev = df.adm_year.values <= cut
val = ~dev
p_lr = np.load(f"{OUT}/_p_lr.npy"); p_gb = np.load(f"{OUT}/_p_gb.npy")
assert len(p_lr) == int(val.sum()), "预测向量与验证集长度不符"
vdf = df[val].copy(); vdf["p_lr"] = p_lr; vdf["p_gb"] = p_gb
Yv = Y[val]
print(f"  验证集 n={len(Yv):,}  事件率 {100*Yv.mean():.2f}%", flush=True)

# 头对头子集：NEWS2 核心分量齐全者
sub = vdf.news2_core_ok.values == 1
devmask_core = (df.news2_core_ok.values == 1) & dev
df["_gender_num"] = df.gender.astype(str).str.upper().map({"M": 1, "F": 0})
print(f"  头对头子集（NEWS2 可算）n={int(sub.sum()):,}  事件率 {100*Yv[sub].mean():.2f}%", flush=True)

# ---------------------------------------------------------------- 3. 各模型 AUROC
rows = []
def add(name, p_sub, y_sub):
    a = roc_auc_score(y_sub, p_sub); b = average_precision_score(y_sub, p_sub)
    rows.append({"model": name, "n": int(len(y_sub)), "events": int(y_sub.sum()),
                 "prevalence": float(y_sub.mean()), "auroc": a, "auprc": b})
    print(f"  {name:34s} n={len(y_sub):>7,}  事件率 {100*y_sub.mean():5.2f}%  "
          f"AUROC {a:.4f}  AUPRC {b:.4f}", flush=True)
    return p_sub

print("\n=== 头对头比较（同一子集：NEWS2 核心分量齐全者）===", flush=True)
ys = Yv[sub]
# A. NEWS2 单独（在开发集同子集上拟合）
m = LogisticRegression(max_iter=2000).fit(df.loc[devmask_core, ["news2"]].values, Y[devmask_core])
p_news2 = m.predict_proba(vdf.loc[sub, ["news2"]].values)[:, 1]
add("NEWS2（连续, 仅评分）", p_news2, ys)
# B. NEWS2 + 年龄 + 性别
cols = ["news2", "anchor_age", "_gender_num"]
Xs = df[cols].astype(float).values
from sklearn.preprocessing import StandardScaler
Xs = StandardScaler().fit(Xs[devmask_core]).transform(Xs)
m = LogisticRegression(max_iter=2000).fit(Xs[devmask_core], Y[devmask_core])
p_ns = m.predict_proba(Xs[val][sub])[:, 1]
add("NEWS2 + 年龄 + 性别", p_ns, ys)
# C. qSOFA
m = LogisticRegression(max_iter=2000).fit(df.loc[devmask_core, ["qsofa"]].values, Y[devmask_core])
p_q = m.predict_proba(vdf.loc[sub, ["qsofa"]].values)[:, 1]
add("qSOFA（连续, 仅评分）", p_q, ys)
# D/E. 完整 ML 模型（05 的留出验证集预测）
p_lr_s = p_lr[sub]; p_gb_s = p_gb[sub]
add("完整 ML (Logistic)", p_lr_s, ys)
add("完整 ML (HistGradientBoosting)", p_gb_s, ys)

print("\n=== 参照：完整模型在全部留出验证集上的表现 ===", flush=True)
for nm, p in [("完整 ML (Logistic)", p_lr), ("完整 ML (HGB)", p_gb)]:
    a = roc_auc_score(Yv, p); b = average_precision_score(Yv, p)
    print(f"  {nm:34s} n={len(Yv):>7,}  AUROC {a:.4f}  AUPRC {b:.4f}")

# ---------------------------------------------------------------- 4. 配对 bootstrap
print("\n=== 配对 bootstrap（子集内重抽样 2000 次，ΔAUROC 的 95%CI）===", flush=True)
rng = np.random.default_rng(SEED)
defs = [("完整ML(LR) − NEWS2", p_lr_s, p_news2),
        ("完整ML(HGB) − NEWS2", p_gb_s, p_news2),
        ("完整ML(LR) − qSOFA", p_lr_s, p_q),
        ("完整ML(LR) − (NEWS2+人口学)", p_lr_s, p_ns)]
diff_rows = []
for nm, pa, pb in defs:
    d = []
    for _ in range(2000):
        s = rng.integers(0, len(ys), len(ys))
        if len(set(ys[s])) < 2:
            continue
        d.append(roc_auc_score(ys[s], pa[s]) - roc_auc_score(ys[s], pb[s]))
    d = np.array(d)
    obs = roc_auc_score(ys, pa) - roc_auc_score(ys, pb)
    lo, hi = np.percentile(d, [2.5, 97.5])
    pval = float(2 * min((d <= 0).mean(), (d >= 0).mean()))
    diff_rows.append({"comparison": nm, "delta_auroc": obs, "ci_low": lo, "ci_high": hi,
                      "p_two_sided": pval})
    print(f"  {nm:28s} ΔAUROC {obs:+.4f}  95%CI {lo:+.4f} – {hi:+.4f}  P={pval:.4f}", flush=True)

# ---------------------------------------------------------------- 5. 评分分层的标签率
print("\n=== 按 NEWS2 / qSOFA 分层的『大包围』阳性率（头对头子集）===", flush=True)
sd = vdf[sub].copy()
bands = []
for lab, m_ in [("NEWS2 0–2", sd.news2 <= 2), ("NEWS2 3–4", (sd.news2 >= 3) & (sd.news2 <= 4)),
                ("NEWS2 5–6", (sd.news2 >= 5) & (sd.news2 <= 6)),
                ("NEWS2 ≥7", sd.news2 >= 7)]:
    g = sd[m_]
    bands.append({"stratum": lab, "n": len(g), "events": int(g.broad_ge3.sum()),
                  "rate": float(g.broad_ge3.mean()),
                  "mortality": float(g.hospital_expire_flag.mean()),
                  "los_median_h": float(g.los_h.median())})
for k in sorted(sd.qsofa.unique()):
    g = sd[sd.qsofa == k]
    bands.append({"stratum": f"qSOFA {int(k)}", "n": len(g), "events": int(g.broad_ge3.sum()),
                  "rate": float(g.broad_ge3.mean()),
                  "mortality": float(g.hospital_expire_flag.mean()),
                  "los_median_h": float(g.los_h.median())})
bands = pd.DataFrame(bands)
for _, r in bands.iterrows():
    print(f"  {r.stratum:12s} n={int(r.n):>7,}  大包围率 {100*r.rate:5.2f}%  "
          f"死亡 {100*r.mortality:5.2f}%  LOS中位 {r.los_median_h:.1f} h")

# ---------------------------------------------------------------- 6. 保存
pd.DataFrame(rows).to_csv(f"{OUT}/mimic_news2_compare.csv", index=False)
pd.DataFrame(diff_rows).to_csv(f"{OUT}/mimic_news2_delta.csv", index=False)
bands.to_csv(f"{OUT}/mimic_news2_bands.csv", index=False)
print(f"\n总耗时 {time.time()-t_start:.0f}s")
