#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(Matrix)
  library(edgeR)
  library(limma)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) stop("Usage: Rscript analyze_gse233387_hd_control.R <prepared_dir> <weights.csv> <output_dir>")
prepared_dir <- args[[1]]
weights_path <- args[[2]]
output_dir <- args[[3]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

safe_z <- function(x) {
  s <- sd(x)
  if (!is.finite(s) || s <= 1e-12) return(rep(0, length(x)))
  as.numeric((x - mean(x)) / s)
}

read_mtx_gz <- function(path) {
  con <- gzfile(path, open = "rt")
  on.exit(close(con), add = TRUE)
  as(readMM(con), "dgCMatrix")
}

read_lines_gz <- function(path) {
  con <- gzfile(path, open = "rt")
  on.exit(close(con), add = TRUE)
  readLines(con, warn = FALSE)
}

weights <- read.csv(weights_path, stringsAsFactors = FALSE, check.names = FALSE)
if (anyDuplicated(toupper(weights$gene))) stop("PLS weights contain duplicate gene symbols")
plus_genes <- weights$gene[weights$gene_set == "PLS1+"]
minus_genes <- weights$gene[weights$gene_set == "PLS1-"]
if (length(plus_genes) != 350L || length(minus_genes) != 154L) stop("Unexpected predefined PLS1 gene-set sizes")

counts_all <- read_mtx_gz(file.path(prepared_dir, "GSE233387_counts_genes_by_samples.mtx.gz"))
genes <- read_lines_gz(file.path(prepared_dir, "GSE233387_genes.tsv.gz"))
meta_all <- read.csv(file.path(prepared_dir, "GSE233387_metadata.csv"), stringsAsFactors = FALSE)
if (nrow(counts_all) != length(genes) || ncol(counts_all) != nrow(meta_all)) stop("Count dimensions do not match genes/metadata")
if (anyDuplicated(genes)) stop("Duplicate gene symbols remain")
rownames(counts_all) <- genes
if (any(counts_all@x < 0) || any(counts_all@x != floor(counts_all@x))) stop("Counts are not non-negative integers")
if (anyDuplicated(meta_all[c("donor_id", "cell_type")])) stop("Duplicate donor by cell-type pseudobulk")
if (any(table(meta_all$cell_type, meta_all$condition)[, c("Control", "HD")] != matrix(c(2L, 3L), nrow = 14L, ncol = 2L, byrow = TRUE))) {
  stop("Expected exactly 2 controls and 3 HD donors per cell type")
}

cell_order <- c(
  "Layer 2", "Layer 4", "Betz cells", "Layer 5a", "Layer 6a", "Layer 6b",
  "VIP", "RELN", "LAMP5", "PVALB", "Astrocytes", "Microglia",
  "Oligodendrocytes", "OPCs"
)

score_rows <- list()
effect_rows <- list()
camera_rows <- list()
match_rows <- list()

for (cell in cell_order) {
  idx <- which(meta_all$cell_type == cell)
  meta <- meta_all[idx, , drop = FALSE]
  counts <- counts_all[, idx, drop = FALSE]
  meta$condition <- relevel(factor(meta$condition), ref = "Control")

  dge <- DGEList(counts = counts)
  dge <- calcNormFactors(dge, method = "TMM")
  logcpm <- cpm(dge, log = TRUE, prior.count = 0.5)
  gene_sd <- apply(logcpm, 1L, sd)
  valid <- is.finite(gene_sd) & gene_sd > 1e-12
  z <- matrix(0, nrow = nrow(logcpm), ncol = ncol(logcpm), dimnames = dimnames(logcpm))
  z[valid, ] <- t(scale(t(logcpm[valid, , drop = FALSE])))
  plus_idx <- which(rownames(z) %in% plus_genes)
  minus_idx <- which(rownames(z) %in% minus_genes)
  if (length(plus_idx) < 10L || length(minus_idx) < 10L) stop(cell, ": too few matched PLS genes")
  meta$PLS1_positive_score <- colMeans(z[plus_idx, , drop = FALSE])
  meta$PLS1_positive_score_z <- safe_z(meta$PLS1_positive_score)
  score_rows[[length(score_rows) + 1L]] <- meta

  fit <- lm(PLS1_positive_score_z ~ condition, data = meta)
  co <- coef(summary(fit))["conditionHD", ]
  ci <- confint(fit, "conditionHD", level = 0.95)
  effect_rows[[length(effect_rows) + 1L]] <- data.frame(
    cell_type = cell,
    standardized_HD_control_difference = unname(co[["Estimate"]]),
    standard_error = unname(co[["Std. Error"]]),
    ci_low = unname(ci[[1]]),
    ci_high = unname(ci[[2]]),
    score_p_value = unname(co[["Pr(>|t|)"]]),
    residual_df = df.residual(fit),
    n_control = sum(meta$condition == "Control"),
    n_hd = sum(meta$condition == "HD"),
    PLS1_positive_matched_before_expression_filter = length(plus_idx),
    stringsAsFactors = FALSE
  )

  design <- model.matrix(~ condition, meta)
  if (qr(design)$rank != ncol(design) || nrow(design) - qr(design)$rank < 3L) stop(cell, ": invalid CAMERA design")
  dge2 <- DGEList(counts = counts)
  keep <- filterByExpr(dge2, design = design)
  dge2 <- calcNormFactors(dge2[keep, , keep.lib.sizes = FALSE])
  v <- voom(dge2, design = design, plot = FALSE)
  indices <- list(
    PLS1_positive = which(rownames(v$E) %in% plus_genes),
    PLS1_negative = which(rownames(v$E) %in% minus_genes)
  )
  cam <- camera(
    v, index = indices, design = design,
    contrast = which(colnames(design) == "conditionHD"),
    inter.gene.cor = 0.01, use.ranks = FALSE, sort = FALSE
  )
  cam$gene_set <- rownames(cam)
  rownames(cam) <- NULL
  cam$cell_type <- cell
  cam$background_genes <- nrow(v$E)
  cam$matched_genes <- lengths(indices)[cam$gene_set]
  camera_rows[[length(camera_rows) + 1L]] <- cam
  match_rows[[length(match_rows) + 1L]] <- data.frame(
    cell_type = cell,
    PLS1_positive_matched_before_expression_filter = length(plus_idx),
    PLS1_negative_matched_before_expression_filter = length(minus_idx),
    PLS1_positive_matched_after_expression_filter = length(indices$PLS1_positive),
    PLS1_negative_matched_after_expression_filter = length(indices$PLS1_negative),
    background_genes = nrow(v$E)
  )
}

scores <- do.call(rbind, score_rows)
effects <- do.call(rbind, effect_rows)
camera_results <- do.call(rbind, camera_rows)
matches <- do.call(rbind, match_rows)
rownames(scores) <- NULL
rownames(effects) <- NULL
rownames(camera_results) <- NULL

effects$score_BH_q_value <- p.adjust(effects$score_p_value, method = "BH")
camera_results$BH_q_value <- NA_real_
for (set_name in c("PLS1_positive", "PLS1_negative")) {
  use <- which(camera_results$gene_set == set_name)
  camera_results$BH_q_value[use] <- p.adjust(camera_results$PValue[use], method = "BH")
}
plus_camera <- camera_results[camera_results$gene_set == "PLS1_positive", c("cell_type", "Direction", "PValue", "BH_q_value", "matched_genes", "background_genes")]
names(plus_camera)[names(plus_camera) == "PValue"] <- "camera_p_value"
names(plus_camera)[names(plus_camera) == "BH_q_value"] <- "camera_BH_q_value"
effects <- merge(effects, plus_camera, by = "cell_type", all.x = TRUE, sort = FALSE)
effects$cell_type <- factor(effects$cell_type, levels = cell_order)
effects <- effects[order(effects$cell_type), ]
effects$cell_type <- as.character(effects$cell_type)

# Repeated-measures fixed-donor interaction test: does the HD-control score difference vary by cell type?
scores$cell_type <- factor(scores$cell_type, levels = cell_order)
scores$donor_id <- factor(scores$donor_id)
reduced <- lm(PLS1_positive_score_z ~ donor_id + cell_type, data = scores)
full <- lm(PLS1_positive_score_z ~ donor_id + cell_type + condition:cell_type, data = scores)
omnibus <- anova(reduced, full)
omnibus_result <- data.frame(
  test = "HD-by-cell-type interaction with donor fixed effects",
  numerator_df = omnibus$Df[2],
  denominator_df = df.residual(full),
  F_value = omnibus$F[2],
  p_value = omnibus$`Pr(>F)`[2],
  stringsAsFactors = FALSE
)

write.csv(scores, file.path(output_dir, "GSE233387_donor_celltype_scores.csv"), row.names = FALSE)
write.csv(effects, file.path(output_dir, "GSE233387_HD_control_effects.csv"), row.names = FALSE)
write.csv(camera_results, file.path(output_dir, "GSE233387_camera_results.csv"), row.names = FALSE)
write.csv(matches, file.path(output_dir, "GSE233387_gene_matching.csv"), row.names = FALSE)
write.csv(omnibus_result, file.path(output_dir, "GSE233387_omnibus_interaction.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(output_dir, "R_sessionInfo.txt"))

# Function: estimate GSE233387 PLS1-positive gene-set HD-control differences for 14 BA4 cell types.
# Inputs: full-transcriptome pseudobulk counts, donor metadata, and the predefined PLS1 weights table.
# Outputs: donor scores, standardized differences with 95% CIs, CAMERA results, omnibus test, and QC.
# Main steps: TMM/logCPM, gene-wise standardization, gene-set scoring, linear models, CAMERA, BH correction.
# Logs: standard output/error should be redirected by the calling shell when required.

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Estimate HD-control transcriptional effects and gene-set enrichment from GSE233387 pseudobulk profiles.
# Input source/location: Aggregated pseudobulk counts/sample metadata plus reference PLS gene sets from the local analysis input/output tree.
# Output location: Cell-type effect estimates, gene-set tests, multiple-testing results, and model summaries in the local results directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Apply TMM/logCPM normalization; standardize gene expression; construct gene-set scores; fit linear models; run CAMERA; apply BH correction; export results.
# Log location: No dedicated log file; model status is written to the R console.
# =============================================================================
