# -*- coding: utf-8 -*-
## MIMIC-IV：三线表（投稿级 Word）
## 用法：Rscript 09_tables.R <输出目录>
## 表格：Table 1 队列特征；Table 2 模型性能；Table 3 稳健性与对照
## 三线表规范：仅顶线、表头下横线、底线三条横线；无竖线、无多余横线；表题在上、表注在下；Arial 8 pt
suppressPackageStartupMessages({
  library(flextable); library(officer); library(data.table)
})

BASE <- "studies/infection-sepsis-2026-11"
RES  <- file.path(BASE, "results")
args <- commandArgs(trailingOnly = TRUE)
TAB  <- if (length(args) >= 1) args[1] else file.path(BASE, "manuscript", "tables")
dir.create(TAB, showWarnings = FALSE, recursive = TRUE)

## 中文键 → 英文显示（稿件与表格一律英文）
TR <- c(
  "全体" = "Overall", "年龄 <65 岁" = "Age <65 years", "年龄 ≥65 岁" = "Age ≥65 years",
  "女性" = "Female", "男性" = "Male", "急诊入院" = "Emergency admission",
  "非急诊入院" = "Non-emergency admission",
  "Both" = "Laboratory and vital signs", "Laboratory only" = "Laboratory only",
  "Vital signs only" = "Vital signs only", "Neither" = "Neither laboratory nor vital signs",
  "≥3 类抗生素（主标签）" = "Three or more classes (primary)",
  "碳青霉烯" = "Any carbapenem",
  "抗MRSA + 抗GNR广谱" = "Anti-MRSA plus anti-gram-negative",
  "碳青霉烯 + 万古/利奈唑胺" = "Carbapenem plus vancomycin or linezolid",
  "≥3 类抗生素且含抗MRSA" = "Three or more classes including anti-MRSA",
  "NEWS2（连续, 仅评分）" = "NEWS2 (continuous score only)",
  "NEWS2 + 年龄 + 性别" = "NEWS2 plus age and sex",
  "qSOFA（连续, 仅评分）" = "qSOFA (continuous score only)",
  "完整 ML (Logistic)" = "Full model (logistic)",
  "完整 ML (HistGradientBoosting)" = "Full model (boosted trees)",
  "完整ML(LR) − NEWS2" = "full model (logistic) minus NEWS2",
  "完整ML(HGB) − NEWS2" = "full model (boosted trees) minus NEWS2",
  "完整ML(LR) − qSOFA" = "full model (logistic) minus qSOFA",
  "完整ML(LR) − (NEWS2+人口学)" = "full model (logistic) minus NEWS2 with age and sex")
tr <- function(x) { y <- unname(TR[as.character(x)]); ifelse(is.na(y), as.character(x), y) }

read_csv <- function(f) as.data.table(read.csv(file.path(RES, f), stringsAsFactors = FALSE,
                                                check.names = FALSE, encoding = "UTF-8"))
have <- function(f) file.exists(file.path(RES, f))

## ---------- 三线表构造器 ----------
three_line <- function(ft, n_header = 1) {
  ft |>
    border_remove() |>
    hline_top(border = fp_border_default(width = 1.5, color = "black")) |>
    hline(i = n_header, part = "body",
          border = fp_border_default(width = 0.75, color = "black")) |>
    hline_bottom(border = fp_border_default(width = 1.5, color = "black")) |>
    font(fontname = "Arial") |>
    fontsize(size = 8) |>
    padding(padding = 1.6) |>
    align(align = "left", part = "all") |>
    valign(valign = "top")
}

mk_doc <- function(title, ft, notes, file) {
  doc <- read_docx()
  doc <- body_add_par(doc, title, style = "Normal")
  doc <- body_add_flextable(doc, ft)
  doc <- body_add_par(doc, notes, style = "Normal")
  print(doc, target = file.path(TAB, file))
  message("已生成 ", file.path(TAB, file))
}

## ================================================================ Table 1
t1 <- read_csv("desc_table1.csv")
d <- copy(t1)
d$variable[is.na(d$variable)] <- ""
d$item <- ifelse(d$type == "categorical",
                 ifelse(d$variable == "", paste0("    ", d$level), paste0(d$variable, ": ", d$level)),
                 d$variable)
tab1 <- data.frame(
  Characteristic = d$item,
  `Overall (n = 184,645)` = d$overall,
  `Broad-spectrum (n = 32,431)` = d$broad_spectrum,
  `Narrower (n = 152,214)` = d$no_broad_spectrum,
  `Missing, %` = as.character(d$missing_pct),
  check.names = FALSE)
ft1 <- flextable(tab1, col_keys = names(tab1)) |>
  
  three_line() |>
  align(align = "right", j = 2:5) |> align(align = "right", j = 2:5, part = "header") |>
  bold(part = "header") |>
  fontsize(size = 8, part = "header")

mk_doc(
  paste0("Table 1. Characteristics of the 184,645 admissions with empiric antibiotics started within ",
         "24 h of admission, grouped by broad-spectrum therapy."),
  ft1,
  paste0("Notes: Continuous variables are median (IQR); categorical variables are n (%). Broad-spectrum ",
         "therapy was defined as three or more distinct antibiotic classes within the first 24 h of ",
         "admission. Objective data availability refers to whether laboratory results or vital signs were ",
         "charted within 0-6 h of admission. Vital signs and laboratory values are the maximum, or the ",
         "minimum where indicated, recorded within that window; the proportion of admissions with no ",
         "measurement of the variable is given in the last column. GCS, Glasgow Coma Scale; SpO2, oxygen ",
         "saturation by pulse oximetry; ICU, intensive care unit."),
  "table1_cohort_characteristics.docx")

## ================================================================ Table 2
if (have("mimic_model_metrics.csv") && have("mimic_model_subgroups.csv")) {
  met   <- read_csv("mimic_model_metrics.csv")
  subg  <- read_csv("mimic_model_subgroups.csv")
  avm   <- read_csv("mimic_model_dataavailability.csv")
  chain <- read_csv("mimic_valuechain_v2.csv")
  chainq <- read_csv("mimic_valuechain_quintiles.csv")
  boot  <- if (have("mimic_model_bootstrap_holdout.csv")) read_csv("mimic_model_bootstrap_holdout.csv") else NULL
  rows <- list()
  for (i in seq_len(nrow(met))) {
    r <- met[i, ]
    rows[[length(rows) + 1]] <- data.frame(
      Section = "Overall (hold-out validation set)", Item = r$model,
      `AUROC` = sprintf("%.3f", r$auroc), `AUPRC` = sprintf("%.3f", r$auprc),
      `Brier` = sprintf("%.4f", r$brier),
      `Calibration slope` = sprintf("%.3f", r$cal_slope),
      `Calibration intercept` = sprintf("%.3f", r$cal_intercept), check.names = FALSE)
  }
  if (!is.null(boot)) {
    for (i in seq_len(nrow(boot))) {
      r <- boot[i, ]
      rows[[length(rows) + 1]] <- data.frame(
        Section = "Overall (hold-out validation set)",
        Item = paste0("Hold-out bootstrap 95% CI, ", r$model),
        `AUROC` = sprintf("%.3f-%.3f", r$auroc_lo, r$auroc_hi),
        `AUPRC` = sprintf("%.3f-%.3f", r$auprc_lo, r$auprc_hi),
        `Brier` = "", `Calibration slope` = "", `Calibration intercept` = "", check.names = FALSE)
    }
  }
  for (i in seq_len(nrow(subg))) {
    r <- subg[i, ]
    rows[[length(rows) + 1]] <- data.frame(
      Section = paste0("Subgroup: ", tr(r$subgroup)),
      Item = sprintf("n = %s; events = %s (%.1f%%)", format(r$n, big.mark = ","),
                     format(r$events, big.mark = ","), 100 * r$prevalence),
      `AUROC` = sprintf("%.3f", r$auroc_lr), `AUPRC` = sprintf("%.3f", r$auprc_lr),
      `Brier` = "", `Calibration slope` = "", `Calibration intercept` = "", check.names = FALSE)
  }
  for (i in seq_len(nrow(avm))) {
    r <- avm[i, ]
    rows[[length(rows) + 1]] <- data.frame(
      Section = paste0("Data available at 0-6 h: ", tr(r$data_avail)),
      Item = sprintf("n = %s; events = %s (%.1f%%)", format(r$n, big.mark = ","),
                     format(r$events, big.mark = ","), 100 * r$prevalence),
      `AUROC` = sprintf("%.3f", r$auroc_lr), `AUPRC` = sprintf("%.3f", r$auprc_lr),
      `Brier` = "", `Calibration slope` = "", `Calibration intercept` = "", check.names = FALSE)
  }
  tie <- chain[grepl("^A ", chain$group), ]
  gv <- rbind(
    data.frame(nm = "No objective data at 0-6 h (single prediction)", n = tie$n,
               rate = tie$broad_rate, mort = tie$mortality, los = tie$los_median_h),
    data.frame(nm = paste0("Predicted-risk quintile ", chainq$quintile), n = chainq$n,
               rate = chainq$broad_rate, mort = chainq$mortality, los = chainq$los_median_h))
  for (i in seq_len(nrow(gv))) {
    r <- gv[i, ]
    rows[[length(rows) + 1]] <- data.frame(
      Section = "Value-chain first link (validation set)",
      Item = sprintf("%s; n = %s", r$nm, format(r$n, big.mark = ",")),
      `AUROC` = "", `AUPRC` = "",
      `Brier` = sprintf("Broad-spectrum %.1f%%", 100 * r$rate),
      `Calibration slope` = sprintf("Mortality %.2f%%", 100 * r$mort),
      `Calibration intercept` = sprintf("LOS %.0f h", r$los), check.names = FALSE)
  }
  tab2 <- do.call(rbind, rows); tab2$Section[is.na(tab2$Section)] <- ""
  ft2 <- flextable(tab2, col_keys = names(tab2)) |>
     three_line() |>
    align(align = "right", j = 3:7) |> align(align = "right", j = 3:7, part = "header") |> bold(part = "header") |> fontsize(size = 8, part = "header")
  mk_doc(
    "Table 2. Performance of the prediction models in the hold-out validation set, by subgroup and by the objective data available at the decision point.",
    ft2,
    paste0("Notes: The outcome was three or more distinct antibiotic classes within the first 24 h of ",
           "admission. Predictions were made from data available in the first 6 h of admission. AUROC, area ",
           "under the receiver operating characteristic curve; AUPRC, area under the precision-recall curve; ",
           "Brier score, the mean squared error of the predicted probabilities; LOS, length of stay. ",
           "Calibration slope and intercept were estimated by an unpenalised logistic regression of the outcome ",
           "on the logit of the predicted probability in the validation set; a slope of 1 and an intercept of 0 ",
           "indicate perfect calibration. The value-chain section reports observed outcomes in groups defined ",
           "by the model: the first group comprises the 56.0% of admissions that shared a single predicted ",
           "probability because no laboratory or vital-sign data were available in the first 6 h, and the ",
           "remaining groups are quintiles of predicted risk among the admissions the model was able to rank."),
    "table2_model_performance.docx")
}

## ================================================================ Table 3
if (have("mimic_model_sensitivity.csv") && have("mimic_news2_compare.csv")) {
  sens <- read_csv("mimic_model_sensitivity.csv")
  n2c  <- read_csv("mimic_news2_compare.csv")
  n2d  <- read_csv("mimic_news2_delta.csv")
  sub  <- if (have("mimic_model_subset.csv")) read_csv("mimic_model_subset.csv") else NULL
  rows <- list()
  for (i in seq_len(nrow(sens))) {
    r <- sens[i, ]
    rows[[length(rows) + 1]] <- data.frame(
      Section = "Outcome definition (same model, alternative labels)", Item = tr(r$label),
      `AUROC` = sprintf("%.3f", r$auroc_lr), `AUPRC` = sprintf("%.3f", r$auprc_lr),
      `Events, n` = format(r$events_val, big.mark = ","),
      `Prevalence, %` = sprintf("%.2f", 100 * r$prevalence_val), check.names = FALSE)
  }
  for (i in seq_len(nrow(n2c))) {
    r <- n2c[i, ]
    rows[[length(rows) + 1]] <- data.frame(
      Section = "Head-to-head with standard severity scores (subset with computable NEWS2)",
      Item = tr(r$model), `AUROC` = sprintf("%.3f", r$auroc), `AUPRC` = sprintf("%.3f", r$auprc),
      `Events, n` = format(r$events, big.mark = ","),
      `Prevalence, %` = sprintf("%.2f", 100 * r$prevalence), check.names = FALSE)
  }
  for (i in seq_len(nrow(n2d))) {
    r <- n2d[i, ]
    rows[[length(rows) + 1]] <- data.frame(
      Section = "Head-to-head with standard severity scores (subset with computable NEWS2)",
      Item = paste0("Difference in AUROC: ", tr(r$comparison)),
      `AUROC` = sprintf("%+.3f", r$delta_auroc),
      `AUPRC` = sprintf("95%% CI %+.3f to %+.3f", r$ci_low, r$ci_high),
      `Events, n` = "", `Prevalence, %` = ifelse(r$p_two_sided < 0.001, "P < 0.001", sprintf("P = %.3f", r$p_two_sided)), check.names = FALSE)
  }
  if (!is.null(sub)) {
    for (i in seq_len(nrow(sub))) {
      r <- sub[i, ]
      rows[[length(rows) + 1]] <- data.frame(
        Section = "Predictor set (same validation set)", Item = r$model,
        `AUROC` = sprintf("%.3f", r$auroc), `AUPRC` = sprintf("%.3f", r$auprc),
        `Events, n` = "", `Prevalence, %` = "", check.names = FALSE)
    }
  }
  tab3 <- do.call(rbind, rows); tab3$Section[is.na(tab3$Section)] <- ""
  ft3 <- flextable(tab3, col_keys = names(tab3)) |>
     three_line() |>
    align(align = "right", j = 3:5) |> align(align = "right", j = 3:5, part = "header") |> bold(part = "header") |> fontsize(size = 8, part = "header")
  mk_doc(
    "Table 3. Robustness of the model and comparison with standard severity scores.",
    ft3,
    paste0("Notes: AUROC, area under the receiver operating characteristic curve; AUPRC, area under the ",
           "precision-recall curve; NEWS2, National Early Warning Score 2 (RCP 2017); qSOFA, quick ",
           "sepsis-related organ failure assessment. Confidence intervals for the difference in AUROC were ",
           "obtained from 2000 paired bootstrap resamples of the validation set. The head-to-head comparison ",
           "was restricted to the subset in which NEWS2 could be computed, because vital signs are recorded only ",
           "for patients admitted to an intensive care unit in MIMIC-IV."),
    "table3_robustness_and_comparison.docx")
}
message("表格脚本完成")
