# -*- coding: utf-8 -*-
"""从 05_model.py 的运行日志里取 bootstrap 的 95%CI，落成结果表。

（05_model.py 的 bootstrap 只打印不落盘；本脚本解析日志中的那一行，数字原样搬运，不重算。）
脚本：07e_bootstrap_csv.py
"""
import os, re, sys
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

LOG = "studies/infection-sepsis-2026-11/results/05_model_run.log"
OUT = "studies/infection-sepsis-2026-11/results/mimic_model_bootstrap.csv"

txt = open(LOG, encoding="utf-8").read()
m = re.search(r"有效重复\s+(\d+)\s+AUROC\s+([0-9.]+)\s+\(95%CI\s+([0-9.]+)–([0-9.]+)\)\s+"
               r"AUPRC\s+([0-9.]+)\s+\(95%CI\s+([0-9.]+)–([0-9.]+)\)", txt)
if not m:
    sys.exit(f"日志里没找到 bootstrap 结果行，请先跑完 05_model.py：{LOG}")
reps, auroc, alo, ahi, auprc, plo, phi = m.groups()
row = {"model": "Logistic", "reps": int(reps),
       "auroc_mean": float(auroc), "auroc_lo": float(alo), "auroc_hi": float(ahi),
       "auprc_mean": float(auprc), "auprc_lo": float(plo), "auprc_hi": float(phi)}
pd.DataFrame([row]).to_csv(OUT, index=False)
print("bootstrap 95%CI：")
print(f"  有效重复 {reps}")
print(f"  AUROC {auroc}（95%CI {alo}–{ahi}）")
print(f"  AUPRC {auprc}（95%CI {plo}–{phi}）")
print("已导出", OUT)
