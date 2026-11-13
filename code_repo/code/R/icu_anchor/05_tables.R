# 05_tables.R — 分支【ICU 锚点为主分析】：生成投稿级三线表 Word
# 输入：results/ 下的结果 CSV（全部来自已跑通的脚本，不在本脚本重算模型）
# 输出：manuscript/tables/table1_*.docx / table2_*.docx / table3_*.docx
# 样式：三线表（顶线、表头下横线、底线），无竖线；英文标签；P 值规范书写
suppressPackageStartupMessages({library(flextable); library(officer)})

BRANCH <- normalizePath(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value=TRUE)[1])), "..", ".."))
RES <- file.path(BRANCH, "results")
TAB <- file.path(BRANCH, "manuscript", "tables")
dir.create(TAB, showWarnings = FALSE, recursive = TRUE)

fmt_p <- function(p) {
  if (length(p) == 0 || is.na(p)) return("—")
  if (p < 0.001) return("P<0.001")
  if (p > 0.995) return("P>0.99")
  paste0("P=", formatC(p, format = "f", digits = 2))
}
med_iqr <- function(x) {
  x <- x[is.finite(x)]
  q <- quantile(x, c(0.25, 0.5, 0.75), na.rm = TRUE)
  sprintf("%.1f (%.1f–%.1f)", q[2], q[1], q[3])
}
mean_sd <- function(x) {
  x <- x[is.finite(x)]
  sprintf("%.1f (%.1f)", mean(x), sd(x))
}
n_pct <- function(k, n) sprintf("%d (%.1f)", k, 100 * k / n)
wilson <- function(k, n) {
  if (n == 0) return(c(NA, NA))
  p <- k / n; z <- 1.96
  c(100*(p + z^2/(2*n) - z*sqrt(p*(1-p)/n + z^2/(4*n^2)))/(1 + z^2/n),
    100*(p + z^2/(2*n) + z*sqrt(p*(1-p)/n + z^2/(4*n^2)))/(1 + z^2/n))
}

coh <- read.csv(file.path(RES, "icu_cohort_analysis.csv"), stringsAsFactors = FALSE)
coh$race <- toupper(trimws(coh$race))
coh$rgrp <- ifelse(grepl("^WHITE", coh$race), "White",
            ifelse(grepl("^BLACK", coh$race), "Black",
            ifelse(grepl("^ASIAN", coh$race), "Asian",
            ifelse(grepl("^HISPANIC", coh$race), "Hispanic/Latino", "Other or unknown"))))
coh$male <- as.integer(toupper(coh$gender) == "M")
coh$emergency <- as.integer(toupper(coh$admission_type) %in% c("EW EMER.", "DIRECT EMER."))
a <- coh[coh$broad_ge3 == 0, ]; b <- coh[coh$broad_ge3 == 1, ]
N1 <- nrow(a); N2 <- nrow(b)

fmt_num <- function(x, d = 1) formatC(x, format = "f", digits = d)
p_wilcox <- function(x, y) {
  x <- x[is.finite(x)]; y <- y[is.finite(y)]
  if (length(x) < 5 || length(y) < 5) return(NA)
  tryCatch(wilcox.test(x, y)$p.value, error = function(e) NA)
}
p_chisq <- function(ka, na, kb, nb) {
  tab <- rbind(c(ka, na - ka), c(kb, nb - kb))
  tryCatch({
    if (any(chisq.test(tab)$expected < 5)) fisher.test(tab)$p.value else chisq.test(tab)$p.value
  }, error = function(e) NA)
}

rows <- list()
add <- function(variable, v1, v2, p) rows[[length(rows) + 1]] <<- data.frame(Variable = variable, g1 = v1, g2 = v2, P = p, stringsAsFactors = FALSE)
add("Age, years", med_iqr(a$anchor_age), med_iqr(b$anchor_age), fmt_p(p_wilcox(a$anchor_age, b$anchor_age)))
add("Male sex", n_pct(sum(a$male), N1), n_pct(sum(b$male), N2), fmt_p(p_chisq(sum(a$male), N1, sum(b$male), N2)))
add("Age ≥65 years", n_pct(sum(a$anchor_age >= 65), N1), n_pct(sum(b$anchor_age >= 65), N2),
    fmt_p(p_chisq(sum(a$anchor_age >= 65), N1, sum(b$anchor_age >= 65), N2)))
add("Emergency admission", n_pct(sum(a$emergency), N1), n_pct(sum(b$emergency), N2),
    fmt_p(p_chisq(sum(a$emergency), N1, sum(b$emergency), N2)))
for (rc in c("White", "Black", "Asian", "Hispanic/Latino", "Other or unknown")) {
  add(paste0("Race/ethnicity: ", rc), n_pct(sum(a$rgrp == rc), N1), n_pct(sum(b$rgrp == rc), N2),
      fmt_p(p_chisq(sum(a$rgrp == rc), N1, sum(b$rgrp == rc), N2)))
}
add("Hours from hospital admission to ICU admission", med_iqr(a$adm_to_icu_h), med_iqr(b$adm_to_icu_h),
    fmt_p(p_wilcox(a$adm_to_icu_h, b$adm_to_icu_h)))

CM <- c("cm_hf" = "Heart failure", "cm_ckd" = "Chronic kidney disease", "cm_dm" = "Diabetes mellitus",
        "cm_copd" = "COPD", "cm_cirrhosis" = "Cirrhosis", "cm_cancer" = "Malignancy",
        "cm_pneumonia" = "Pneumonia", "cm_sepsis_adm" = "Sepsis at admission", "cm_uti" = "Urinary tract infection",
        "cm_bacteremia" = "Bacteremia", "cm_immunocomp" = "Immunocompromised state")
for (k in names(CM)) {
  if (!k %in% names(coh)) next
  add(paste0("Comorbidity: ", CM[[k]]), n_pct(sum(a[[k]] == 1), N1), n_pct(sum(b[[k]] == 1), N2),
      fmt_p(p_chisq(sum(a[[k]] == 1), N1, sum(b[[k]] == 1), N2)))
}

VI <- c("v_hr_max" = "Heart rate, max (beats/min)", "v_sbp_min" = "Systolic blood pressure, min (mmHg)",
        "v_rr_max" = "Respiratory rate, max (breaths/min)", "v_spo2_pulse_min" = "SpO2, min (%)",
        "v_gcs_total" = "GCS, min", "v_tempF_max" = "Temperature, max (°F)")
for (k in names(VI)) {
  if (!k %in% names(coh)) next
  add(paste0("Vital sign (0–6 h): ", VI[[k]]), med_iqr(a[[k]]), med_iqr(b[[k]]), fmt_p(p_wilcox(a[[k]], b[[k]])))
}
LB <- c("l_creatinine_max" = "Creatinine, max (mg/dL)", "l_lactate_max" = "Lactate, max (mmol/L)",
        "l_wbc_min" = "White cell count, min (×10⁹/L)", "l_bicarbonate_min" = "Bicarbonate, min (mmol/L)",
        "l_hgb_min" = "Haemoglobin, min (g/dL)", "l_platelet_min" = "Platelet count, min (×10⁹/L)",
        "l_sodium_min" = "Sodium, min (mmol/L)")
for (k in names(LB)) {
  if (!k %in% names(coh)) next
  add(paste0("Laboratory (0–6 h): ", LB[[k]]), med_iqr(a[[k]]), med_iqr(b[[k]]), fmt_p(p_wilcox(a[[k]], b[[k]])))
}
add("Positive culture before first antibiotic", n_pct(sum(a$micro_evidence_before_abx == 1), N1),
    n_pct(sum(b$micro_evidence_before_abx == 1), N2),
    fmt_p(p_chisq(sum(a$micro_evidence_before_abx == 1), N1, sum(b$micro_evidence_before_abx == 1), N2)))
add("In-hospital mortality", n_pct(sum(a$hospital_expire_flag == 1), N1), n_pct(sum(b$hospital_expire_flag == 1), N2),
    fmt_p(p_chisq(sum(a$hospital_expire_flag == 1), N1, sum(b$hospital_expire_flag == 1), N2)))
add("Length of stay, h", med_iqr(a$los_h), med_iqr(b$los_h), fmt_p(p_wilcox(a$los_h, b$los_h)))

t1 <- do.call(rbind, rows)
t1out <- data.frame(Variable = t1$Variable, `No broad-spectrum therapy\n(n = 37,269)` = t1$g1,
                    `Broad-spectrum therapy (≥3 classes)\n(n = 9,876)` = t1$g2, P = t1$P, check.names = FALSE)

ft1 <- flextable(t1out) |>
  theme_booktabs() |>
  set_header_labels(Variable = "Characteristic") |>
  fontsize(size = 8, part = "all") |>
  bold(i = 1, part = "header") |>
  align(j = 2:3, align = "center") |>
  align(j = 4, align = "center") |>
  width(j = 1, width = 2.9) |> width(j = 2:4, width = 1.35) |>
  padding(padding = 2, part = "all") |>
  hline_top(border = fp_border(color = "black", width = 1.2), part = "all") |>
  hline_bottom(border = fp_border(color = "black", width = 1.2), part = "all")
doc1 <- read_docx()
doc1 <- body_add_par(doc1, "Table 1. Characteristics of the study cohort according to receipt of broad-spectrum antimicrobial therapy (>=3 classes) within 24 h of ICU admission.",
                     style = "Normal")
doc1 <- body_add_flextable(doc1, ft1)
doc1 <- body_add_par(doc1, "Values are median (interquartile range) for continuous variables and n (%) for categorical variables. P values are from Wilcoxon rank-sum or Fisher's exact/chi-square tests as appropriate. ICU, intensive care unit; GCS, Glasgow Coma Scale; BP, blood pressure.",
                     style = "Normal")
print(doc1, target = file.path(TAB, "table1_cohort_characteristics.docx"))
cat("Table 1 written:", file.path(TAB, "table1_cohort_characteristics.docx"), "\n")

# ---------------------------------------------------------------- Table 2
met <- read.csv(file.path(RES, "icu_model_metrics.csv"), stringsAsFactors = FALSE)
sub <- read.csv(file.path(RES, "icu_model_subgroups.csv"), stringsAsFactors = FALSE)
bsp <- read.csv(file.path(RES, "icu_model_bootstrap.csv"), stringsAsFactors = FALSE)

mk <- function(name, auroc, auprc, brier, slope, inter, n, ev) {
  data.frame(Model = name, `AUROC` = sprintf("%.3f", auroc), `AUPRC` = sprintf("%.3f", auprc),
             `Brier score` = sprintf("%.3f", brier), `Calibration slope` = sprintf("%.2f", slope),
             `Calibration intercept` = sprintf("%.2f", inter),
             `Hold-out n` = format(n, big.mark = ","), `Events` = format(ev, big.mark = ","),
             check.names = FALSE, stringsAsFactors = FALSE)
}
t2a <- do.call(rbind, lapply(seq_len(nrow(met)), function(i) {
  mk(met$model[i], met$auroc[i], met$auprc[i], met$brier[i], met$cal_slope[i], met$cal_intercept[i],
     met$n_test[i], met$events[i])
}))
ft2a <- flextable(t2a) |> theme_booktabs() |> fontsize(size = 8, part = "all") |>
  bold(i = 1, part = "header") |> set_header_labels(Model = "Model") |>
  width(j = 1, width = 3.2) |> width(j = 2:8, width = 1.0) |> padding(padding = 2, part = "all") |>
  hline_top(border = fp_border(color = "black", width = 1.2), part = "all") |>
  hline_bottom(border = fp_border(color = "black", width = 1.2), part = "all")

sub$subgroup_en <- sub$subgroup
sub$subgroup_en[sub$subgroup_en == "全体"] <- "Overall"
sub$subgroup_en[sub$subgroup_en == "年龄 <65 岁"] <- "Age <65 years"
sub$subgroup_en[sub$subgroup_en == "年龄 ≥65 岁"] <- "Age ≥65 years"
sub$subgroup_en[sub$subgroup_en == "女性"] <- "Female"
sub$subgroup_en[sub$subgroup_en == "男性"] <- "Male"
sub$subgroup_en[sub$subgroup_en == "住院 6h 内入 ICU"] <- "ICU admission within 6 h of hospital admission"
sub$subgroup_en[sub$subgroup_en == "住院 6h 后入 ICU"] <- "ICU admission >6 h after hospital admission"
sub$subgroup_en[sub$subgroup_en == "急诊入院"] <- "Emergency admission"
sub$subgroup_en[sub$subgroup_en == "非急诊入院"] <- "Non-emergency admission"
sub$subgroup_en[sub$subgroup_en == "有阳性培养"] <- "Positive culture before first antibiotic"
sub$subgroup_en[sub$subgroup_en == "无阳性培养"] <- "No positive culture before first antibiotic"
sub$subgroup_en[sub$subgroup_en == "0–6h 数据齐全"] <- "Both vital signs and laboratory data available"
sub$subgroup_en[sub$subgroup_en == "0–6h 仅生命体征"] <- "Vital signs only"
t2b <- data.frame(Subgroup = sub$subgroup_en, n = format(sub$n, big.mark = ","),
                  Events = format(sub$events, big.mark = ","),
                  `Event rate (%)` = sprintf("%.1f", 100 * sub$prevalence),
                  `AUROC, logistic` = sprintf("%.3f", sub$auroc_lr),
                  `AUROC, boosting` = sprintf("%.3f", sub$auroc_gb), check.names = FALSE, stringsAsFactors = FALSE)
ft2b <- flextable(t2b) |> theme_booktabs() |> fontsize(size = 8, part = "all") |>
  bold(i = 1, part = "header") |>
  width(j = 1, width = 3.4) |> width(j = 2:6, width = 1.15) |> padding(padding = 2, part = "all") |>
  hline_top(border = fp_border(color = "black", width = 1.2), part = "all") |>
  hline_bottom(border = fp_border(color = "black", width = 1.2), part = "all")

boot_line <- sprintf("Internal validation (bootstrap, %d resamples): AUROC %.3f (95%% CI %.3f–%.3f); AUPRC %.3f (95%% CI %.3f–%.3f)",
                      bsp$reps[1], bsp$auroc_mean[1], bsp$auroc_lo[1], bsp$auroc_hi[1],
                      bsp$auprc_mean[1], bsp$auprc_lo[1], bsp$auprc_hi[1])
doc2 <- read_docx()
doc2 <- body_add_par(doc2, "Table 2. Discrimination, calibration and subgroup performance of the prediction models for broad-spectrum antimicrobial therapy within 24 h of ICU admission.",
                     style = "Normal")
doc2 <- body_add_flextable(doc2, ft2a)
doc2 <- body_add_par(doc2, "Subgroup performance (hold-out set)", style = "Normal")
doc2 <- body_add_flextable(doc2, ft2b)
doc2 <- body_add_par(doc2, paste0("AUROC, area under the receiver operating characteristic curve; AUPRC, area under the precision-recall curve; Brier score, mean squared error of the predicted probabilities (lower is better); calibration slope and intercept from an unpenalised logistic regression of the outcome on the logit of the predicted probability. ", boot_line),
                     style = "Normal")
print(doc2, target = file.path(TAB, "table2_model_performance.docx"))
cat("Table 2 written\n")
cat(boot_line, "\n")
writeLines(boot_line, file.path(RES, "table2_bootstrap_note.txt"))

# ---------------------------------------------------------------- Table 3
cmp <- read.csv(file.path(RES, "icu_news2_compare.csv"), stringsAsFactors = FALSE)
dlt <- read.csv(file.path(RES, "icu_news2_delta.csv"), stringsAsFactors = FALSE)
sens <- read.csv(file.path(RES, "icu_model_sensitivity.csv"), stringsAsFactors = FALSE)
sens$label_en <- sens$label
sens$label_en[sens$label_en == "≥3 类抗菌药物（主标签）"] <- "≥3 antimicrobial classes (primary outcome)"
sens$label_en[sens$label_en == "≥2 类抗菌药物"] <- "≥2 antimicrobial classes"
sens$label_en[sens$label_en == "碳青霉烯"] <- "Carbapenem"
sens$label_en[sens$label_en == "抗MRSA + 抗GNR广谱"] <- "Anti-MRSA plus anti-GNR broad spectrum"
sens$label_en[sens$label_en == "碳青霉烯 + 万古/利奈唑胺"] <- "Carbapenem plus vancomycin/linezolid"
sens$label_en[sens$label_en == "≥3 类 + 含抗MRSA"] <- "≥3 classes including anti-MRSA"
cmp$model_en <- cmp$model
cmp$model_en[cmp$model_en == "NEWS2（连续, 单独）"] <- "NEWS2 (continuous)"
cmp$model_en[cmp$model_en == "NEWS2 + 年龄 + 性别"] <- "NEWS2 plus age and sex"
cmp$model_en[cmp$model_en == "qSOFA（连续, 单独）"] <- "qSOFA (continuous)"
cmp$model_en[cmp$model_en == "完整 ML（Logistic）"] <- "Full model, logistic regression"
cmp$model_en[cmp$model_en == "完整 ML（HistGradientBoosting）"] <- "Full model, gradient boosting"
cmp$model_en[cmp$model_en == "可迁移子模型（不含生命体征）"] <- "Portable model (no vital signs)"
t3a <- data.frame(Model = cmp$model_en, `AUROC` = sprintf("%.3f", cmp$auroc),
                  `AUPRC` = sprintf("%.3f", cmp$auprc), check.names = FALSE, stringsAsFactors = FALSE)
ft3a <- flextable(t3a) |> theme_booktabs() |> fontsize(size = 8, part = "all") |>
  bold(i = 1, part = "header") |> set_header_labels(Model = "Predictor (hold-out set, n = 11,144)") |>
  width(j = 1, width = 4.2) |> width(j = 2:3, width = 1.1) |> padding(padding = 2, part = "all") |>
  hline_top(border = fp_border(color = "black", width = 1.2), part = "all") |>
  hline_bottom(border = fp_border(color = "black", width = 1.2), part = "all")

dlt$comparison_en <- dlt$comparison
dlt$comparison_en[dlt$comparison_en == "完整ML(LR) − NEWS2"] <- "Full model (logistic) minus NEWS2"
dlt$comparison_en[dlt$comparison_en == "完整ML(HGB) − NEWS2"] <- "Full model (boosting) minus NEWS2"
dlt$comparison_en[dlt$comparison_en == "完整ML(LR) − qSOFA"] <- "Full model (logistic) minus qSOFA"
dlt$comparison_en[dlt$comparison_en == "完整ML(LR) − (NEWS2+人口学)"] <- "Full model (logistic) minus NEWS2+age+sex"
dlt$comparison_en[dlt$comparison_en == "可迁移子模型 − NEWS2"] <- "Portable model minus NEWS2"
dlt$comparison_en[dlt$comparison_en == "完整ML − NEWS2（体温完整子集）"] <- "Full model minus NEWS2, temperature-complete subset"
t3b <- data.frame(Comparison = dlt$comparison_en, `ΔAUROC` = sprintf("%+.3f", dlt$delta_auroc),
                  `95% CI` = sprintf("%+.3f to %+.3f", dlt$ci_low, dlt$ci_high),
                  P = vapply(dlt$p_two_sided, fmt_p, character(1)), check.names = FALSE, stringsAsFactors = FALSE)
ft3b <- flextable(t3b) |> theme_booktabs() |> fontsize(size = 8, part = "all") |>
  bold(i = 1, part = "header") |> width(j = 1, width = 4.2) |> width(j = 2:4, width = 1.35) |>
  padding(padding = 2, part = "all") |>
  hline_top(border = fp_border(color = "black", width = 1.2), part = "all") |>
  hline_bottom(border = fp_border(color = "black", width = 1.2), part = "all")

t3c <- data.frame(`Alternative outcome definition` = sens$label_en,
                  Events = format(sens$events_val, big.mark = ","),
                  `Event rate (%)` = sprintf("%.1f", 100 * sens$prevalence_val),
                  `AUROC, logistic` = sprintf("%.3f", sens$auroc_lr),
                  `AUPRC, logistic` = sprintf("%.3f", sens$auprc_lr),
                  `AUROC, boosting` = sprintf("%.3f", sens$auroc_gb),
                  check.names = FALSE, stringsAsFactors = FALSE)
ft3c <- flextable(t3c) |> theme_booktabs() |> fontsize(size = 8, part = "all") |>
  bold(i = 1, part = "header") |> width(j = 1, width = 3.4) |> width(j = 2:6, width = 1.15) |>
  padding(padding = 2, part = "all") |>
  hline_top(border = fp_border(color = "black", width = 1.2), part = "all") |>
  hline_bottom(border = fp_border(color = "black", width = 1.2), part = "all")

doc3 <- read_docx()
doc3 <- body_add_par(doc3, "Table 3. Comparison with standard severity scores and performance across alternative definitions of broad-spectrum therapy.",
                     style = "Normal")
doc3 <- body_add_flextable(doc3, ft3a)
doc3 <- body_add_par(doc3, "Paired comparison of discrimination (hold-out set)", style = "Normal")
doc3 <- body_add_flextable(doc3, ft3b)
doc3 <- body_add_par(doc3, "Performance across alternative outcome definitions", style = "Normal")
doc3 <- body_add_flextable(doc3, ft3c)
doc3 <- body_add_par(doc3, "All estimates are from the same hold-out set (n = 11,144; 2,963 events). Delta AUROC is the difference from the comparator; 95% confidence intervals and two-sided P values from 2,000 paired bootstrap resamples of the hold-out set. NEWS2, National Early Warning Score 2; qSOFA, quick Sequential Organ Failure Assessment.",
                     style = "Normal")
print(doc3, target = file.path(TAB, "table3_headtohead_sensitivity.docx"))
cat("Table 3 written\n")
cat("All tables done.\n")
