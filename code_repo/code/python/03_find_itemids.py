# -*- coding: utf-8 -*-
"""查 MIMIC-IV 的 itemid（生命体征与实验室），用于窄表抽取。"""
import duckdb, sys
sys.stdout.reconfigure(encoding="utf-8")
con = duckdb.connect("D:/mimic_iv.duckdb")
con.execute("PRAGMA threads=4")

MIMIC = "D:/mimic-iv-3"
if not con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name='d_items'").fetchone()[0]:
    con.execute(f"CREATE TABLE d_items AS SELECT * FROM read_csv_auto('{MIMIC}/icu/d_items.csv.gz', header=true)")
if not con.execute("SELECT count(*) FROM duckdb_tables() WHERE table_name='d_labitems'").fetchone()[0]:
    con.execute(f"CREATE TABLE d_labitems AS SELECT * FROM read_csv_auto('{MIMIC}/hosp/d_labitems.csv.gz', header=true)")

VIT = ["heart rate", "arterial blood pressure", "arterial blood pressure [systolic]",
       "arterial blood pressure [diastolic]", "arterial blood pressure mean",
       "temperature", "respiratory rate", "o2 saturation", "oxygen saturation",
       "glasgow coma scale", "gcs", "weight", "height"]
print("=== d_items 候选（生命体征/意识）===")
for kw in VIT:
    rows = con.execute("""SELECT itemid, label, linksto FROM d_items
                           WHERE LOWER(label) LIKE ? ORDER BY itemid LIMIT 12""", ["%" + kw + "%"]).fetchall()
    for r in rows:
        print("  ", r)

LAB = ["creatinine", "lactate", "bicarbonate", "ph", "pco2", "po2", "sodium", "chloride",
       "potassium", "glucose", "urea nitrogen", "bun", "bilirubin, total", "albumin",
       "platelet count", "white blood cell", "neutrophil", "hematocrit", "hemoglobin",
       "aspartate", "alanine", "anion gap", "troponin", "c-reactive protein", "procalcitonin",
       "magnesium", "calcium, total", "phosphate"]
print("\n=== d_labitems 候选（实验室）===")
for kw in LAB:
    rows = con.execute("""SELECT itemid, label, fluid FROM d_labitems
                           WHERE LOWER(label) LIKE ? ORDER BY itemid LIMIT 8""", ["%" + kw + "%"]).fetchall()
    for r in rows:
        print("  ", r)
con.close()
