# -*- coding: utf-8 -*-
## 补充材料三线表（manuscript/supplementary/）
##   Table S1  决策时点数据可得性分层：样本量、结局率、模型判别力
##   Table S2  ICU 入院的敏感性分析（另起锚点重建队列）
suppressPackageStartupMessages({ library(flextable); library(officer); library(data.table) })

BASE <- "studies/infection-sepsis-2026-11"
RES  <- file.path(BASE, "results")
ICU  <- ".branches/ICU 锚点为主分析（推荐）/results"
SUP  <- file.path(BASE, "manuscript", "supplementary")
dir.create(SUP, showWarnings = FALSE, recursive = TRUE)

read_csv <- function(f, dir = RES) as.data.table(read.csv(file.path(dir, f), stringsAsFactors = FALSE,
                                                          check.names = FALSE, encoding = "UTF-8"))

three_line <- function(ft, n_header = 1) {
  ft |>
    border_remove() |>
    hline_top(border = fp_border_default(width = 1.5, color = "black")) |>
    hline(i = n_header, part = "body", border = fp_border_default(width = 0.75, color = "black")) |>
    hline_bottom(border = fp_border_default(width = 1.5, color = "black")) |>
    font(fontname = "Arial") |> fontsize(size = 8) |> padding(padding = 1.6) |>
    align(align = "left", part = "all") |> valign(valign = "top")
}
mk_doc <- function(title, ft, notes, file) {
  doc <- read_docx()
  doc <- body_add_par(doc, title, style = "Normal")
  doc <- body_add_flextable(doc, ft)
  doc <- body_add_par(doc, notes, style = "Normal")
  print(doc, target = file.path(SUP, file))
  message("已生成 ", file.path(SUP, file))
}

## ---------------------------------------------------------------- Table S1
av  <- read_csv("desc_availability.csv")
mav <- read_csv("mimic_model_dataavailability.csv")
mapk <- c("Both laboratory and vital signs" = "Both",
          "Laboratory only" = "Laboratory only",
          "Vital signs only" = "Vital signs only",
          "Neither" = "Neither")
av$key <- unname(mapk[av$stratum])
av <- merge(av, mav, by.x = "key", by.y = "data_avail", all.x = TRUE)
av <- av[match(c("Both", "Laboratory only", "Vital signs only", "Neither"), av$key), ]
ts1 <- data.frame(
  `Objective data in the first 6 h` = c("Laboratory and vital signs", "Laboratory only",
                                        "Vital signs only", "Neither"),
  `Admissions, n (%)` = sprintf("%s (%.1f)", format(av$n_all, big.mark = ","), av$pct_all),
  `Broad-spectrum therapy, %` = sprintf("%.1f", av$broad_rate_all),
  `In-hospital mortality, %` = sprintf("%.2f", av$mortality_all),
  `AUROC, logistic` = sprintf("%.3f", av$auroc_lr),
  `AUROC, boosted trees` = sprintf("%.3f", av$auroc_gb),
  check.names = FALSE)
ft1 <- flextable(ts1) |> three_line() |>
  align(align = "right", j = 2:6) |> align(align = "right", j = 2:6, part = "header") |> bold(part = "header")
mk_doc("Table S1. Clinical outcomes and model discrimination by the objective data available in the first 6 h of admission (hold-out validation set).",
       ft1,
       paste0("Notes: Broad-spectrum therapy was defined as three or more distinct antibiotic classes within the ",
              "first 24 h of admission. AUROC was computed within each stratum of the hold-out validation set. ",
              "Laboratory data and vital signs were both absent in 60.0% of admissions, and the model could not ",
              "discriminate within that stratum (AUROC close to chance)."),
       "tableS1_data_availability.docx")

## ---------------------------------------------------------------- Table S2
hl  <- read_csv("icu_headline_numbers.csv", ICU)
met <- read_csv("icu_model_metrics.csv", ICU)
bst <- read_csv("icu_model_bootstrap.csv", ICU)
n2  <- read_csv("icu_news2_compare.csv", ICU)
n2d <- read_csv("icu_news2_delta.csv", ICU)
vc  <- read_csv("icu_valuechain.csv", ICU)
gs  <- function(q) hl[hl$quantity == q, ]
rows <- list()
add <- function(sec, item, a = "", b = "", c = "", d = "") {
  rows[[length(rows) + 1]] <<- data.frame(Section = sec, Item = item, `Value 1` = a, `Value 2` = b,
                                          `Value 3` = c, `Value 4` = d, check.names = FALSE)
}
add("Cohort anchored on ICU admission", "Admissions with empiric antibiotics within 24 h of hospital admission",
    format(gs("Admissions with empiric antibiotics started within 24 h of hospital admission")$n, big.mark = ","))
add("Cohort anchored on ICU admission", "Analytic cohort (ICU admission within 24 h)",
    format(gs("Analytic cohort (ICU admission within 24 h of hospital admission)")$n, big.mark = ","))
add("Cohort anchored on ICU admission", "Hold-out validation set",
    format(gs("Hold-out set")$n, big.mark = ","),
    sprintf("%d events (%.1f%%)", met$events[1], 100 * met$prevalence[1]))
add("Cohort anchored on ICU admission", "Laboratory data available in the first 6 h after ICU admission",
    sprintf("%.1f%%", gs("Patients with laboratory data available in the first 6 h after ICU admission")$percent))
add("Cohort anchored on ICU admission", "Vital signs available in the first 6 h after ICU admission",
    sprintf("%.1f%%", gs("Patients with vital signs available in the first 6 h after ICU admission")$percent))
for (i in seq_len(nrow(met))) {
  r <- met[i, ]
  nm <- if (grepl("可迁移", r$model)) "Portable model (no vital signs)" else
        if (r$model == "Logistic") "Full model (logistic)" else "Full model (boosted trees)"
  ci <- if (bst$model[1] == "Logistic" && r$model == "Logistic")
    sprintf("%.3f-%.3f", bst$auroc_lo[1], bst$auroc_hi[1]) else ""
  add("Model performance", nm,
      sprintf("AUROC %.3f", r$auroc), sprintf("AUPRC %.3f", r$auprc),
      sprintf("Brier %.3f; slope %.2f; intercept %.2f", r$brier, r$cal_slope, r$cal_intercept), ci)
}
for (i in seq_len(nrow(n2))) {
  r <- n2[i, ]
  nm <- c("NEWS2（连续, 单独）" = "NEWS2", "NEWS2 + 年龄 + 性别" = "NEWS2 plus age and sex",
          "qSOFA（连续, 单独）" = "qSOFA", "完整 ML（Logistic）" = "Full model (logistic)",
          "完整 ML（HistGradientBoosting）" = "Full model (boosted trees)",
          "可迁移子模型（不含生命体征）" = "Portable model (no vital signs)" )[r$model]
  if (is.na(nm)) next
  add("Head-to-head with severity scores", nm, sprintf("AUROC %.3f", r$auroc), sprintf("AUPRC %.3f", r$auprc))
}
for (i in seq_len(nrow(n2d))) {
  r <- n2d[i, ]
  nm <- c("完整ML(LR) − NEWS2" = "Full model (logistic) minus NEWS2",
          "完整ML(HGB) − NEWS2" = "Full model (boosted trees) minus NEWS2",
          "完整ML(LR) − qSOFA" = "Full model (logistic) minus qSOFA",
          "完整ML(LR) − (NEWS2+人口学)" = "Full model (logistic) minus NEWS2 with age and sex",
          "可迁移子模型 − NEWS2" = "Portable model minus NEWS2" )[r$comparison]
  if (is.na(nm)) next
  add("Difference in AUROC", nm, sprintf("%+.3f", r$delta_auroc),
      sprintf("95%% CI %+.3f to %+.3f", r$ci_low, r$ci_high),
      ifelse(r$p_two_sided < 0.001, "P < 0.001", sprintf("P = %.3f", r$p_two_sided)))
}
for (g in c("低风险四分位", "其余人群")) {
  r <- vc[vc$group == g, ]
  add("Value-chain first link", ifelse(g == "低风险四分位", "Lowest-risk quartile", "Remaining three quartiles"),
      sprintf("n = %s", format(r$n, big.mark = ",")),
      sprintf("Broad-spectrum %.1f%%", 100 * r$broad_rate),
      sprintf("Mortality %.2f%%", 100 * r$mortality), sprintf("LOS %.0f h", r$los_median_h))
}
add("Decision curve", "Net benefit at a 10% threshold",
    sprintf("Model %.4f", gs("Net benefit of the model at a 10% threshold")$percent),
    sprintf("Treat all %.4f", gs("Net benefit of treating all patients at a 10% threshold")$percent))
ts2 <- do.call(rbind, rows)
ft2 <- flextable(ts2) |> three_line() |>
  fontsize(size = 7.5, part = "all") |>
  align(align = "right", j = 3:6) |> align(align = "right", j = 3:6, part = "header") |> bold(part = "header")
mk_doc("Table S2. Sensitivity analysis with the cohort anchored on intensive-care admission.",
       ft2,
       paste0("Notes: In this sensitivity analysis the clock was started at ICU admission rather than at hospital ",
              "admission, and only admissions with empiric antibiotics started within 24 h of hospital admission ",
              "were retained. Because vital signs are recorded almost exclusively in the ICU in MIMIC-IV, this ",
              "anchor makes objective data available for most patients and therefore represents the upper bound ",
              "of the model's performance. AUROC, area under the receiver operating characteristic curve; AUPRC, ",
              "area under the precision-recall curve; LOS, length of stay. The hold-out set was defined by the 70th percentile of ",
              "de-identified year, that is a non-temporal rather than a temporal split."),
       "tableS2_icu_anchor.docx")
message("补充表完成")

## ---------------------------------------------------------------- Table S3
if (file.exists(file.path(RES, "mimic_riskgroup_profile.csv"))) {
  prof <- read_csv("mimic_riskgroup_profile.csv")
  keep <- c("group", "n", "broad_rate_pct", "mortality_pct", "los_median_h",
            "l_lactate_max", "l_bicarbonate_min", "v_rr_max", "v_spo2_pls_min",
            "v_gcs_total_min", "l_albumin_min", "l_bun_max")
  keep <- keep[keep %in% names(prof)]
  p3 <- prof[, keep, with = FALSE]
  names(p3) <- c("Predicted-risk group", "n", "Broad-spectrum, %", "Mortality, %",
                 "LOS, h", "Lactate max, mmol/L", "Bicarbonate min, mmol/L",
                 "Respiratory rate max", "SpO2 min, %", "GCS min",
                 "Albumin min, g/dL", "Urea nitrogen max, mg/dL")[seq_along(keep)]
  p3[["n"]] <- format(as.integer(p3[["n"]]), big.mark = ",")
  ft3 <- flextable(p3) |> three_line() |> fontsize(size = 7.5, part = "all") |>
    align(align = "right", j = 2:ncol(p3)) |> bold(part = "header")
  mk_doc("Table S3. Clinical profile of the model-defined risk groups in the hold-out validation set.",
         ft3,
         paste0("Notes: Values are medians (interquartile range) unless stated otherwise. The first row ",
                "comprises admissions that shared a single predicted probability because no laboratory ",
                "result or vital sign was recorded in the first 6 h; the remaining rows are quintiles of ",
                "predicted risk among the admissions the model could rank. Medians are shown only where ",
                "at least 30 measurements were available in the group. These profiles describe who did not ",
                "ultimately receive broad-spectrum therapy; they are not evidence that withholding it ",
                "would be safe."),
         "tableS3_risk_group_profile.docx")
}

## ---------------------------------------------------------------- Table S4（按种族）
if (file.exists(file.path(RES, "mimic_subgroup_race.csv"))) {
  rc <- read_csv("mimic_subgroup_race.csv")
  p4 <- data.frame(
    `Race and ethnicity` = rc$race,
    `Admissions, n` = format(rc$n, big.mark = ","),
    `Events, n` = format(rc$events, big.mark = ","),
    `Event rate, %` = sprintf("%.1f", 100 * rc$prevalence),
    `AUROC, logistic` = ifelse(is.na(rc$auroc_lr), "NE", sprintf("%.3f", rc$auroc_lr)),
    `AUROC, boosted trees` = ifelse(is.na(rc$auroc_gb), "NE", sprintf("%.3f", rc$auroc_gb)),
    check.names = FALSE)
  ft4 <- flextable(p4) |> three_line() |>
    align(align = "right", j = 2:6) |> align(align = "right", j = 2:6, part = "header") |> bold(part = "header")
  mk_doc("Table S4. Model discrimination by race and ethnicity in the hold-out validation set.",
         ft4,
         paste0("Notes: Race and ethnicity were collapsed to the six categories used in MIMIC-IV analyses. ",
                "NE, not estimated because the group had fewer than 30 events. Race was available as a ",
                "covariate in the candidate predictor set but was not selected into the final model. ",
                "The 'Unknown' group had substantially higher discrimination than the other groups; this ",
                "most plausibly reflects a higher proportion of intensively monitored patients in that group ",
                "rather than a property of the algorithm, and is reported here for transparency."),
         "tableS4_race_subgroups.docx")
}

## ---------------------------------------------------------------- Table S5（模型系数）
if (file.exists(file.path(RES, "mimic_model_coefficients.csv"))) {
  cf <- read_csv("mimic_model_coefficients.csv")
  ic <- read_csv("mimic_model_intercept.csv")
  p5 <- data.frame(
    Rank = cf$rank_by_abs_beta, Variable = cf$variable,
    `beta (per SD)` = sprintf("%+.3f", cf$beta_per_sd),
    `OR per SD` = sprintf("%.3f", cf$odds_ratio_per_sd),
    `Imputed median` = sprintf("%.3f", cf$imputed_value_median),
    `Dev mean` = sprintf("%.3f", cf$development_mean),
    `Dev SD` = sprintf("%.3f", cf$development_sd),
    `Winsor low` = sprintf("%.3f", cf$winsor_lower),
    `Winsor high` = sprintf("%.3f", cf$winsor_upper),
    check.names = FALSE)
  ft5 <- flextable(p5) |> three_line() |> fontsize(size = 7, part = "all") |>
    align(align = "right", j = 3:9) |> align(align = "right", j = 3:9, part = "header") |> bold(part = "header")
  mk_doc(sprintf("Table S5. Full specification of the logistic model (%d predictors; intercept beta0 = %.4f).",
                 nrow(p5), ic$value[1]),
         ft5,
         paste0("Notes: beta is the coefficient on the standardised scale (one development-set standard ",
                "deviation). To reproduce the linear predictor for a new patient: winzorise each variable at ",
                "the stated bounds, replace missing values with the imputed median, standardise using the ",
                "development mean and SD, multiply by beta, sum, and add the intercept; the predicted ",
                "probability is the logistic transform of that sum. Variables are ordered by absolute beta. ",
                "The corresponding pipeline is in scripts/mimic/05_model.py, with the same seed and parameters."),
         "tableS5_model_coefficients.docx")
}
