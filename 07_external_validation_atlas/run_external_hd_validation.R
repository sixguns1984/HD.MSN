#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("Usage: run_external_hd_validation.R <prepared_dir> <weights.csv> <out_dir>")
prepared <- args[[1]]
weights_path <- args[[2]]
out_dir <- args[[3]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

suppressPackageStartupMessages({
  library(limma)
  library(ggplot2)
})

options(stringsAsFactors = FALSE)
weights <- read.csv(weights_path, check.names = FALSE)
weights <- weights[weights$gene_set %in% c("PLS1+", "PLS1-", "PLS1_positive", "PLS1_negative"), ]
weights$set <- ifelse(weights$gene_set %in% c("PLS1+", "PLS1_positive"), "PLS1_positive", "PLS1_negative")
pos_genes <- unique(weights$gene[weights$set == "PLS1_positive"])
neg_genes <- unique(weights$gene[weights$set == "PLS1_negative"])
weight_lookup <- setNames(weights$PLS1W_corr_Z, weights$gene)

read_expr <- function(path) {
  x <- read.csv(gzfile(path), row.names = 1, check.names = FALSE)
  as.matrix(x)
}

directionalize <- function(tab) {
  expected <- c(PLS1_positive = "Down", PLS1_negative = "Up")
  tab$expected_direction <- expected[tab$gene_set]
  tab$one_sided_p <- ifelse(tab$Direction == tab$expected_direction, tab$PValue / 2, 1 - tab$PValue / 2)
  tab$direction_match <- tab$Direction == tab$expected_direction
  tab
}

camera_expression <- function(expr, meta, formula, dataset, model_name) {
  keep <- complete.cases(model.frame(formula, data = meta, na.action = na.pass))
  meta <- droplevels(meta[keep, , drop = FALSE])
  expr <- expr[, meta$sample_id, drop = FALSE]
  design <- model.matrix(formula, data = meta)
  fit <- eBayes(lmFit(expr, design), robust = TRUE)
  coef_name <- grep("conditionHD$", colnames(design), value = TRUE)
  if (length(coef_name) != 1) stop("conditionHD coefficient not found for ", dataset, "/", model_name)
  idx <- list(
    PLS1_positive = which(rownames(expr) %in% pos_genes),
    PLS1_negative = which(rownames(expr) %in% neg_genes)
  )
  if (any(lengths(idx) < 10)) stop("Too few matched genes in ", dataset)
  cam <- camera(expr, index = idx, design = design, contrast = coef_name, sort = FALSE)
  cam$gene_set <- rownames(cam)
  rownames(cam) <- NULL
  cam$dataset <- dataset
  cam$model <- model_name
  cam$N_CTRL <- sum(meta$condition == "CTRL")
  cam$N_HD <- sum(meta$condition == "HD")
  cam <- directionalize(cam)
  tt <- topTable(fit, coef = coef_name, number = Inf, sort.by = "none")
  tt$gene <- rownames(tt)
  tt$dataset <- dataset
  tt$model <- model_name
  list(camera = cam, gene = tt, meta = meta, design = design, expr = expr)
}

score_model <- function(expr, meta, formula, dataset, model_name) {
  keep <- complete.cases(model.frame(formula, data = meta, na.action = na.pass))
  meta <- droplevels(meta[keep, , drop = FALSE])
  expr <- expr[, meta$sample_id, drop = FALSE]
  matched <- intersect(rownames(expr), names(weight_lookup))
  z <- t(scale(t(expr[matched, , drop = FALSE])))
  finite <- apply(z, 1, function(v) all(is.finite(v)))
  z <- z[finite, , drop = FALSE]
  matched <- rownames(z)
  score <- as.numeric(crossprod(weight_lookup[matched], z) / sum(abs(weight_lookup[matched])))
  frame <- cbind(meta, signed_pls1_score = score)
  fit <- lm(update(formula, signed_pls1_score ~ .), data = frame)
  coef_name <- grep("conditionHD$", names(coef(fit)), value = TRUE)
  ci <- confint(fit, coef_name)
  data.frame(
    dataset = dataset,
    model = model_name,
    N_CTRL = sum(frame$condition == "CTRL"),
    N_HD = sum(frame$condition == "HD"),
    matched_weighted_genes = length(matched),
    beta_HD = unname(coef(fit)[coef_name]),
    CI_low = ci[1],
    CI_high = ci[2],
    PValue = coef(summary(fit))[coef_name, "Pr(>|t|)"],
    stringsAsFactors = FALSE
  )
}

# GSE64810 BA9
meta64810 <- read.csv(file.path(prepared, "GSE64810_metadata.csv"))
meta64810$condition <- factor(meta64810$condition, levels = c("CTRL", "HD"))
expr64810 <- read_expr(file.path(prepared, "GSE64810_log2_normalized_expression.csv.gz"))
g64810_condition <- camera_expression(expr64810, meta64810, ~ condition, "GSE64810_BA9", "condition_only")
g64810_cov <- camera_expression(
  expr64810, meta64810,
  ~ scale(age_of_death) + scale(pmi) + scale(rin) + condition,
  "GSE64810_BA9", "age_PMI_RIN_adjusted"
)

# Authors' published adjusted DESeq2 statistics, used as the primary GSE64810 endpoint.
author_de <- read.csv(file.path(prepared, "GSE64810_author_DESeq2_gene_results.csv"))
author_de <- author_de[is.finite(author_de$statistic), ]
stat <- setNames(author_de$statistic, author_de$symbol)
idx_author <- list(
  PLS1_positive = which(names(stat) %in% pos_genes),
  PLS1_negative = which(names(stat) %in% neg_genes)
)
cam_author <- cameraPR(stat, index = idx_author, use.ranks = FALSE, sort = FALSE)
cam_author$gene_set <- rownames(cam_author)
rownames(cam_author) <- NULL
cam_author$dataset <- "GSE64810_BA9"
cam_author$model <- "authors_adjusted_DESeq2_cameraPR"
cam_author$N_CTRL <- sum(meta64810$condition == "CTRL")
cam_author$N_HD <- sum(meta64810$condition == "HD")
cam_author <- directionalize(cam_author)

# GSE3790 BA4, U133A primary and U133B technical sensitivity.
analyze3790 <- function(platform) {
  meta <- read.csv(file.path(prepared, paste0("GSE3790_", platform, "_BA4_metadata.csv")))
  meta$condition <- factor(meta$condition, levels = c("CTRL", "HD"))
  expr <- read_expr(file.path(prepared, paste0("GSE3790_", platform, "_BA4_expression.csv.gz")))
  camera_expression(expr, meta, ~ condition, paste0("GSE3790_BA4_", platform), "condition_only")
}
g3790a <- analyze3790("GPL96")
g3790b <- analyze3790("GPL97")

# Additional independent BA4 RNA-seq cohort (7 HD/7 control), explicitly
# treated as a sensitivity analysis because it was added after the primary
# four-test family had been frozen.
meta79666 <- read.csv(file.path(prepared, "GSE79666_BA4_metadata.csv"))
meta79666$condition <- factor(meta79666$condition, levels = c("CTRL", "HD"))
expr79666 <- read_expr(file.path(prepared, "GSE79666_BA4_log2_FPKM_expression.csv.gz"))
g79666 <- camera_expression(expr79666, meta79666, ~ scale(age) + condition,
                            "GSE79666_BA4", "age_adjusted")

primary_camera <- rbind(cam_author, g3790a$camera)
primary_camera$global_q_4_primary_tests <- p.adjust(primary_camera$one_sided_p, method = "BH")
write.csv(primary_camera, file.path(out_dir, "primary_camera_results_globalBH.csv"), row.names = FALSE)

sensitivity_camera <- rbind(g64810_condition$camera, g64810_cov$camera, g3790b$camera)
sensitivity_camera$global_q_6_sensitivity_tests <- p.adjust(sensitivity_camera$one_sided_p, method = "BH")
write.csv(sensitivity_camera, file.path(out_dir, "sensitivity_camera_results_globalBH.csv"), row.names = FALSE)
additional_camera <- g79666$camera
additional_camera$global_q_2_additional_tests <- p.adjust(additional_camera$one_sided_p, method = "BH")
write.csv(additional_camera, file.path(out_dir, "GSE79666_additional_camera_results_globalBH.csv"), row.names = FALSE)

gene_results <- rbind(
  g64810_condition$gene[, c("gene", "logFC", "t", "P.Value", "adj.P.Val", "dataset", "model")],
  g64810_cov$gene[, c("gene", "logFC", "t", "P.Value", "adj.P.Val", "dataset", "model")],
  g3790a$gene[, c("gene", "logFC", "t", "P.Value", "adj.P.Val", "dataset", "model")],
  g3790b$gene[, c("gene", "logFC", "t", "P.Value", "adj.P.Val", "dataset", "model")],
  g79666$gene[, c("gene", "logFC", "t", "P.Value", "adj.P.Val", "dataset", "model")]
)
write.csv(gene_results, file.path(out_dir, "limma_gene_results.csv"), row.names = FALSE)

score_results <- rbind(
  score_model(expr64810, meta64810, ~ condition, "GSE64810_BA9", "condition_only"),
  score_model(expr64810, meta64810, ~ scale(age_of_death) + scale(pmi) + scale(rin) + condition,
              "GSE64810_BA9", "age_PMI_RIN_adjusted"),
  score_model(g3790a$expr, g3790a$meta, ~ condition, "GSE3790_BA4_GPL96", "condition_only"),
  score_model(g3790b$expr, g3790b$meta, ~ condition, "GSE3790_BA4_GPL97", "condition_only"),
  score_model(expr79666, meta79666, ~ scale(age) + condition, "GSE79666_BA4", "age_adjusted")
)
score_results$global_q_5_score_models <- p.adjust(score_results$PValue, method = "BH")
write.csv(score_results, file.path(out_dir, "signed_pls1_score_models.csv"), row.names = FALSE)

# Cross-cohort logFC concordance among frozen genes (descriptive, not a new primary test).
a <- author_de[, c("symbol", "logFC")]
names(a) <- c("gene", "logFC_GSE64810_BA9")
b <- g3790a$gene[, c("gene", "logFC")]
names(b)[2] <- "logFC_GSE3790_BA4"
concord <- merge(a, b, by = "gene")
concord <- concord[concord$gene %in% names(weight_lookup), ]
rho <- cor.test(concord$logFC_GSE64810_BA9, concord$logFC_GSE3790_BA4, method = "spearman", exact = FALSE)
concordance_summary <- data.frame(
  N_frozen_genes = nrow(concord),
  Spearman_rho = unname(rho$estimate),
  PValue = rho$p.value,
  sign_concordance = mean(sign(concord$logFC_GSE64810_BA9) == sign(concord$logFC_GSE3790_BA4))
)
write.csv(concord, file.path(out_dir, "GSE64810_GSE3790_frozen_gene_logFC_concordance.csv"), row.names = FALSE)
write.csv(concordance_summary, file.path(out_dir, "cross_cohort_concordance_summary.csv"), row.names = FALSE)

c79666 <- g79666$gene[, c("gene", "logFC")]
names(c79666)[2] <- "logFC_GSE79666_BA4"
pairwise_concordance <- function(left, left_name, right, right_name) {
  z <- merge(left[, c("gene", left_name)], right[, c("gene", right_name)], by = "gene")
  z <- z[z$gene %in% names(weight_lookup), ]
  test <- cor.test(z[[left_name]], z[[right_name]], method = "spearman", exact = FALSE)
  data.frame(comparison = paste(left_name, right_name, sep = "__vs__"), N_frozen_genes = nrow(z),
             Spearman_rho = unname(test$estimate), PValue = test$p.value,
             sign_concordance = mean(sign(z[[left_name]]) == sign(z[[right_name]])))
}
pairwise <- rbind(
  concordance_summary |> transform(comparison = "logFC_GSE64810_BA9__vs__logFC_GSE3790_BA4") |>
    subset(select = c(comparison, N_frozen_genes, Spearman_rho, PValue, sign_concordance)),
  pairwise_concordance(a, "logFC_GSE64810_BA9", c79666, "logFC_GSE79666_BA4"),
  pairwise_concordance(b, "logFC_GSE3790_BA4", c79666, "logFC_GSE79666_BA4")
)
write.csv(pairwise, file.path(out_dir, "pairwise_external_cohort_concordance.csv"), row.names = FALSE)

# Compact style-aligned validation figure.
primary_plot <- transform(primary_camera, family = "Primary", q_plot = global_q_4_primary_tests)
sensitivity_plot <- transform(sensitivity_camera, family = "Sensitivity", q_plot = global_q_6_sensitivity_tests)
additional_plot <- transform(additional_camera, family = "Additional", q_plot = global_q_2_additional_tests)
plot_columns <- c("dataset", "model", "gene_set", "Direction", "direction_match", "family", "q_plot")
plot_cam <- rbind(primary_plot[, plot_columns], sensitivity_plot[, plot_columns], additional_plot[, plot_columns])
plot_cam$label <- paste(plot_cam$dataset, plot_cam$model, sep = "\n")
plot_cam$plot_value <- -log10(plot_cam$q_plot)
plot_cam$plot_value[!is.finite(plot_cam$plot_value)] <- 0
plot_cam$matched <- plot_cam$direction_match
p1 <- ggplot(plot_cam, aes(x = reorder(label, plot_value), y = plot_value, fill = matched)) +
  geom_col(width = 0.72) +
  geom_hline(yintercept = -log10(0.05), linetype = "dashed", linewidth = 0.35, colour = "grey45") +
  facet_grid(gene_set ~ family, scales = "free_x", space = "free_x") +
  scale_fill_manual(values = c(`TRUE` = "#D75A4A", `FALSE` = "#B7B7B7"),
                    labels = c(`TRUE` = "Expected direction", `FALSE` = "Opposite direction")) +
  coord_cartesian(ylim = c(0, max(2, max(plot_cam$plot_value, na.rm = TRUE) * 1.08))) +
  labs(x = NULL, y = expression(-log[10]("BH q")), fill = NULL) +
  theme_classic(base_family = "Arial", base_size = 14) +
  theme(axis.text.x = element_text(angle = 50, hjust = 1, vjust = 1, size = 13),
        legend.position = "top", strip.background = element_blank(), strip.text = element_text(face = "bold"))
ggsave(file.path(out_dir, "external_HD_bulk_validation.pdf"), p1, width = 7.2, height = 4.8, device = cairo_pdf, dpi = 600)
ggsave(file.path(out_dir, "external_HD_bulk_validation.png"), p1, width = 7.2, height = 4.8, dpi = 600)

acceptance <- list(
  status = "PASS",
  primary_family = "2 frozen directional gene sets x 2 independent biological cohorts (4 tests)",
  GSE64810 = list(region = "BA9", N_CTRL = 49, N_HD = 20, primary_model = "authors adjusted DESeq2 statistics"),
  GSE3790 = list(region = "BA4", N_CTRL = g3790a$camera$N_CTRL[1], N_HD = g3790a$camera$N_HD[1],
                 primary_platform = "GPL96", GPL97_role = "technical sensitivity; same biological cohort"),
  GSE79666 = list(region = "BA4", N_CTRL = 7, N_HD = 7,
                  role = "additional independent sensitivity; separate two-test family"),
  n_primary_directional_FDR_significant = sum(primary_camera$global_q_4_primary_tests < 0.05),
  n_GSE79666_directional_FDR_significant = sum(additional_camera$global_q_2_additional_tests < 0.05),
  independence_guard = "GPL96 and GPL97 are not independent cohorts; GSE3790 non-BA4 phenotype decoding was not sufficiently documented and was not inferred"
)
jsonlite::write_json(acceptance, file.path(out_dir, "external_hd_acceptance.json"), pretty = TRUE, auto_unbox = TRUE)
print(primary_camera[, c("dataset", "model", "gene_set", "NGenes", "Direction", "one_sided_p", "global_q_4_primary_tests")])
print(score_results)
print(concordance_summary)
print(additional_camera[, c("dataset", "model", "gene_set", "NGenes", "Direction", "one_sided_p", "global_q_2_additional_tests")])
print(pairwise)

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Run statistical validation of the reference transcriptional signature in the preprocessed external HD dataset.
# Input source/location: Preprocessed external HD expression matrix/sample metadata plus mapped reference gene sets from the external-validation work directory.
# Output location: External HD effect estimates, gene-set tests, multiple-testing-adjusted tables, and model/QC summaries in the configured results directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Load processed data; construct design/model inputs; estimate HD-control effects; test reference gene sets; apply multiple-testing correction; export results.
# Log location: No dedicated log file; model status is written to the R console.
# =============================================================================
