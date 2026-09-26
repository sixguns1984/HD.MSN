#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) {
  stop("Usage: run_gse233408_real_regions.R <manifest.csv> <counts_dir> <frozen_weights.csv> <out_dir>")
}
manifest_path <- args[[1]]
counts_dir <- args[[2]]
weights_path <- args[[3]]
out_dir <- args[[4]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(out_dir, "gene_level"), recursive = TRUE, showWarnings = FALSE)

atlas_library <- Sys.getenv("HD_MSN_R_LIBRARY", unset = file.path(getwd(), "env", "R", "library"))
.libPaths(c(atlas_library, .libPaths()))
suppressPackageStartupMessages({
  library(edgeR)
  library(limma)
  library(data.table)
  library(ggplot2)
})

bh <- function(p) p.adjust(p, method = "BH")
safe_name <- function(x) gsub("[^A-Za-z0-9]+", "_", x)

manifest <- read.csv(manifest_path, check.names = FALSE, stringsAsFactors = FALSE, fileEncoding = "UTF-8-BOM")
manifest <- manifest[nzchar(manifest$count_url), , drop = FALSE]
manifest$count_file <- basename(manifest$count_url)
manifest$count_path <- file.path(counts_dir, manifest$count_file)
if (nrow(manifest) != 163L) stop(sprintf("Expected 163 count-bearing samples, found %d", nrow(manifest)))
if (any(!file.exists(manifest$count_path))) stop("One or more GSE233408 count files are missing")
if (length(unique(manifest$donor)) != 19L) stop("Expected 19 unique biological donors")

read_count <- function(path) {
  frame <- fread(cmd = paste("gzip -dc", shQuote(path)), sep = "\t", header = TRUE, showProgress = FALSE)
  if (ncol(frame) != 3L) stop(paste("Unexpected count file format:", path))
  setnames(frame, c("gene_id", "gene_symbol", "count"))
  frame
}

first <- read_count(manifest$count_path[[1]])
gene_id <- as.character(first$gene_id)
gene_symbol <- as.character(first$gene_symbol)
counts <- matrix(0, nrow = nrow(first), ncol = nrow(manifest),
                 dimnames = list(gene_id, manifest$gsm))
counts[, 1] <- first$count
for (index in 2:nrow(manifest)) {
  frame <- read_count(manifest$count_path[[index]])
  if (!identical(as.character(frame$gene_id), gene_id) || !identical(as.character(frame$gene_symbol), gene_symbol)) {
    stop(paste("Gene rows differ in", manifest$count_file[[index]]))
  }
  counts[, index] <- frame$count
}
storage.mode(counts) <- "integer"
if (any(counts < 0L)) stop("Negative raw count detected")

group_key <- interaction(manifest$region, manifest$donor, manifest$cell_type, drop = TRUE, sep = "||")
membership <- model.matrix(~ 0 + group_key)
aggregated <- counts %*% membership
group_levels <- levels(group_key)
colnames(aggregated) <- group_levels
group_rows <- lapply(group_levels, function(key) {
  idx <- which(as.character(group_key) == key)
  unique_value <- function(column) {
    value <- unique(manifest[idx, column])
    if (length(value) != 1L) stop(paste("Conflicting", column, "within", key))
    value[[1]]
  }
  data.frame(
    library_id = key,
    region = unique_value("region"), donor = unique_value("donor"),
    cell_type = unique_value("cell_type"), condition = unique_value("condition"),
    age = as.numeric(unique_value("age")), sex = unique_value("sex"),
    pmi_hours = as.numeric(unique_value("pmi_hours")), n_input_libraries = length(idx),
    stringsAsFactors = FALSE
  )
})
group_meta <- do.call(rbind, group_rows)
rownames(group_meta) <- group_meta$library_id
if (!identical(colnames(aggregated), rownames(group_meta))) stop("Aggregated metadata ordering failure")
if (sum(aggregated) != sum(counts)) stop("Count conservation failure during donor-region-cell-type aggregation")

write.csv(group_meta, file.path(out_dir, "donor_region_celltype_metadata.csv"), row.names = FALSE)
saveRDS(list(counts = aggregated, gene_id = gene_id, gene_symbol = gene_symbol, metadata = group_meta),
        file.path(out_dir, "donor_region_celltype_pseudobulk.rds"), compress = "xz")

weights <- read.csv(weights_path, stringsAsFactors = FALSE)
pos_genes <- unique(weights$gene[weights$gene_set %in% c("PLS1+", "PLS1_positive")])
neg_genes <- unique(weights$gene[weights$gene_set %in% c("PLS1-", "PLS1_negative")])
if (length(pos_genes) != 350L || length(neg_genes) != 154L) {
  stop(sprintf("Frozen gene-set cardinality mismatch: PLS1+=%d, PLS1-=%d", length(pos_genes), length(neg_genes)))
}
weight_lookup <- setNames(weights$PLS1W_corr_Z, weights$gene)

coverage <- do.call(rbind, lapply(split(group_meta, list(group_meta$region, group_meta$cell_type), drop = TRUE), function(frame) {
  data.frame(
    region = frame$region[[1]], cell_type = frame$cell_type[[1]],
    control_donors = length(unique(frame$donor[frame$condition == "CTRL"])),
    hd_donors = length(unique(frame$donor[frame$condition == "HD"])),
    n_pseudobulk_libraries = nrow(frame),
    eligible = length(unique(frame$donor[frame$condition == "CTRL"])) >= 3L &&
      length(unique(frame$donor[frame$condition == "HD"])) >= 3L,
    stringsAsFactors = FALSE
  )
}))
coverage <- coverage[order(coverage$region, coverage$cell_type), ]
write.csv(coverage, file.path(out_dir, "region_celltype_donor_coverage.csv"), row.names = FALSE)

camera_rows <- list()
sensitivity_camera_rows <- list()
model_rows <- list()
score_rows <- list()
model_index <- 0L
for (row_index in which(coverage$eligible)) {
  region <- coverage$region[[row_index]]
  cell_type <- coverage$cell_type[[row_index]]
  selected <- which(group_meta$region == region & group_meta$cell_type == cell_type)
  meta <- group_meta[selected, , drop = FALSE]
  condition <- factor(meta$condition, levels = c("CTRL", "HD"))
  design <- model.matrix(~ condition)
  if (qr(design)$rank != ncol(design)) stop(paste("Rank-deficient primary design:", region, cell_type))
  dge <- DGEList(counts = aggregated[, selected, drop = FALSE],
                 genes = data.frame(gene_id = gene_id, gene_symbol = gene_symbol, stringsAsFactors = FALSE))
  keep <- filterByExpr(dge, design = design)
  dge <- dge[keep, , keep.lib.sizes = FALSE]
  dge <- calcNormFactors(dge, method = "TMM")
  dge <- estimateDisp(dge, design, robust = TRUE)
  fit <- glmQLFit(dge, design, robust = TRUE)
  qlf <- glmQLFTest(fit, coef = "conditionHD")
  table <- topTags(qlf, n = Inf, sort.by = "none")$table
  table$gene_id <- rownames(table)
  table$gene_symbol <- dge$genes$gene_symbol
  table$region <- region
  table$cell_type <- cell_type
  fwrite(table, file.path(out_dir, "gene_level", paste0(safe_name(region), "__", safe_name(cell_type), "__primary.csv.gz")))

  voomed <- voom(dge, design, plot = FALSE)
  gene_sets <- list(
    PLS1_positive = which(voomed$genes$gene_symbol %in% pos_genes),
    PLS1_negative = which(voomed$genes$gene_symbol %in% neg_genes)
  )
  camera_result <- camera(voomed$E, index = gene_sets, design = design, contrast = "conditionHD",
                          inter.gene.cor = 0.01, use.ranks = FALSE)
  camera_result$gene_set <- rownames(camera_result)
  camera_result$expected_direction <- c(PLS1_positive = "Down", PLS1_negative = "Up")[camera_result$gene_set]
  camera_result$one_sided_p <- ifelse(
    camera_result$Direction == camera_result$expected_direction,
    camera_result$PValue / 2,
    1 - camera_result$PValue / 2
  )
  camera_result$region <- region
  camera_result$cell_type <- cell_type
  camera_result$control_donors <- coverage$control_donors[[row_index]]
  camera_result$hd_donors <- coverage$hd_donors[[row_index]]
  camera_rows[[length(camera_rows) + 1L]] <- camera_result

  logcpm <- cpm(dge, log = TRUE, prior.count = 0.5)
  matched <- which(voomed$genes$gene_symbol %in% names(weight_lookup))
  matched_symbols <- voomed$genes$gene_symbol[matched]
  z_expression <- t(scale(t(logcpm[matched, , drop = FALSE])))
  z_expression[!is.finite(z_expression)] <- 0
  model_weights <- unname(weight_lookup[matched_symbols])
  scores <- as.numeric(crossprod(model_weights, z_expression) / sum(abs(model_weights)))
  score_frame <- data.frame(meta, signed_pls1_score = scores, stringsAsFactors = FALSE)
  score_rows[[length(score_rows) + 1L]] <- score_frame
  score_fit <- lm(signed_pls1_score ~ condition, data = transform(score_frame, condition = factor(condition, levels = c("CTRL", "HD"))))
  coef_table <- summary(score_fit)$coefficients

  sensitivity_status <- "not_run"
  sensitivity_beta <- sensitivity_p <- NA_real_
  complete_covariates <- all(is.finite(meta$age)) && all(nzchar(meta$sex))
  if (complete_covariates) {
    sensitivity_design <- model.matrix(~ scale(age) + factor(sex) + condition, data = meta)
    if (qr(sensitivity_design)$rank == ncol(sensitivity_design) && nrow(sensitivity_design) - ncol(sensitivity_design) >= 3L) {
      sensitivity_status <- "full_rank"
      sensitivity_dge <- estimateDisp(dge, sensitivity_design, robust = TRUE)
      sensitivity_fit <- glmQLFit(sensitivity_dge, sensitivity_design, robust = TRUE)
      condition_column <- grep("conditionHD$", colnames(sensitivity_design), value = TRUE)
      sensitivity_qlf <- glmQLFTest(sensitivity_fit, coef = condition_column)
      sensitivity_table <- topTags(sensitivity_qlf, n = Inf, sort.by = "none")$table
      sensitivity_beta <- cor(table$logFC, sensitivity_table$logFC, method = "spearman", use = "complete.obs")
      sensitivity_p <- cor(table$logFC, sensitivity_table$logFC, method = "pearson", use = "complete.obs")
      sensitivity_voom <- voom(sensitivity_dge, sensitivity_design, plot = FALSE)
      sensitivity_camera <- camera(
        sensitivity_voom$E, index = gene_sets, design = sensitivity_design,
        contrast = condition_column, inter.gene.cor = 0.01, use.ranks = FALSE
      )
      sensitivity_camera$gene_set <- rownames(sensitivity_camera)
      sensitivity_camera$expected_direction <- c(PLS1_positive = "Down", PLS1_negative = "Up")[sensitivity_camera$gene_set]
      sensitivity_camera$one_sided_p <- ifelse(
        sensitivity_camera$Direction == sensitivity_camera$expected_direction,
        sensitivity_camera$PValue / 2,
        1 - sensitivity_camera$PValue / 2
      )
      sensitivity_camera$region <- region
      sensitivity_camera$cell_type <- cell_type
      sensitivity_camera$control_donors <- coverage$control_donors[[row_index]]
      sensitivity_camera$hd_donors <- coverage$hd_donors[[row_index]]
      sensitivity_camera_rows[[length(sensitivity_camera_rows) + 1L]] <- sensitivity_camera
    } else {
      sensitivity_status <- "rank_or_df_guard"
    }
  } else {
    sensitivity_status <- "missing_covariates"
  }

  model_index <- model_index + 1L
  model_rows[[model_index]] <- data.frame(
    region = region, cell_type = cell_type,
    control_donors = coverage$control_donors[[row_index]], hd_donors = coverage$hd_donors[[row_index]],
    tested_genes = nrow(dge), design_rank = qr(design)$rank, design_columns = ncol(design),
    residual_df = fit$df.residual[1], n_gene_fdr_lt_0_05 = sum(table$FDR < 0.05, na.rm = TRUE),
    signed_score_hd_minus_ctrl = coef_table["conditionHD", "Estimate"],
    signed_score_p = coef_table["conditionHD", "Pr(>|t|)"],
    sensitivity_status = sensitivity_status,
    primary_vs_covariate_logfc_spearman = sensitivity_beta,
    primary_vs_covariate_logfc_pearson = sensitivity_p,
    stringsAsFactors = FALSE
  )
}

models <- do.call(rbind, model_rows)
camera_all <- do.call(rbind, camera_rows)
scores_all <- do.call(rbind, score_rows)
camera_all$global_q_all_eligible_region_celltype_sets <- bh(camera_all$one_sided_p)
camera_all$primary_ba4_q <- NA_real_
ba4 <- camera_all$region == "BA4"
camera_all$primary_ba4_q[ba4] <- bh(camera_all$one_sided_p[ba4])
write.csv(models, file.path(out_dir, "edger_model_summary.csv"), row.names = FALSE)
write.csv(camera_all, file.path(out_dir, "camera_directional_results.csv"), row.names = FALSE)
write.csv(scores_all, file.path(out_dir, "donor_signed_pls1_scores.csv"), row.names = FALSE)
if (length(sensitivity_camera_rows) > 0L) {
  sensitivity_camera_all <- do.call(rbind, sensitivity_camera_rows)
  sensitivity_camera_all$global_q_all_covariate_eligible_tests <- bh(sensitivity_camera_all$one_sided_p)
  write.csv(sensitivity_camera_all, file.path(out_dir, "camera_covariate_sensitivity_results.csv"), row.names = FALSE)
} else {
  sensitivity_camera_all <- data.frame()
}

plot_frame <- camera_all
plot_frame$directional_z <- qnorm(pmax(pmin(plot_frame$one_sided_p, 1 - 1e-15), 1e-300), lower.tail = FALSE)
plot_frame$label <- ifelse(plot_frame$global_q_all_eligible_region_celltype_sets < 0.05,
                           sprintf("q=%.2g", plot_frame$global_q_all_eligible_region_celltype_sets), "")
p <- ggplot(plot_frame, aes(x = region, y = cell_type, fill = directional_z)) +
  geom_tile(color = "white", linewidth = 0.35) +
  geom_text(aes(label = label), size = 4.6) +
  facet_grid(gene_set ~ ., scales = "free_y", space = "free_y") +
  scale_fill_gradient2(low = "#D6604D", mid = "white", high = "#2166AC", midpoint = 0,
                       limits = c(-4, 6), oob = scales::squish, name = "Directional z") +
  labs(x = NULL, y = NULL) +
  theme_classic(base_family = "Arial", base_size = 14) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1), strip.background = element_blank(),
        strip.text = element_text(face = "bold"), legend.position = "right")
ggsave(file.path(out_dir, "GSE233408_real_region_camera_globalBH.pdf"), p, width = 7.2, height = 6.4, dpi = 600)
ggsave(file.path(out_dir, "GSE233408_real_region_camera_globalBH.png"), p, width = 7.2, height = 6.4, dpi = 600)

significant <- camera_all[which(!is.na(camera_all$global_q_all_eligible_region_celltype_sets) &
                                      camera_all$global_q_all_eligible_region_celltype_sets < 0.05),
                          c("region", "cell_type", "gene_set", "Direction", "expected_direction", "one_sided_p", "global_q_all_eligible_region_celltype_sets")]
write.csv(significant, file.path(out_dir, "camera_globalBH_significant.csv"), row.names = FALSE)

acceptance <- list(
  status = if (nrow(models) > 0 && all(models$design_rank == models$design_columns) &&
                 all(c("L5a pyramidal neuron", "L6a pyramidal neuron", "L6b pyramidal neuron") %in%
                     models$cell_type[models$region == "BA4"])) "PASS" else "FAIL",
  source_samples = nrow(manifest),
  unique_donors = length(unique(manifest$donor)),
  count_conservation = as.character(sum(aggregated)) == as.character(sum(counts)),
  biological_replicate = "donor",
  technical_replication_handling = "raw counts summed within donor x region x cell type before inference",
  primary_region = "BA4",
  secondary_regions = sort(setdiff(unique(models$region), "BA4")),
  eligible_models = nrow(models),
  primary_ba4_models = sum(models$region == "BA4"),
  global_camera_tests = nrow(camera_all),
  global_camera_significant = nrow(significant),
  covariate_sensitivity_camera_tests = nrow(sensitivity_camera_all),
  covariate_sensitivity_camera_significant = if (nrow(sensitivity_camera_all)) sum(sensitivity_camera_all$global_q_all_covariate_eligible_tests < 0.05) else 0L,
  model_summary = models,
  significant_camera = significant
)
jsonlite::write_json(acceptance, file.path(out_dir, "gse233408_acceptance.json"), pretty = TRUE, auto_unbox = TRUE, na = "null")
print(acceptance[c("status", "source_samples", "unique_donors", "eligible_models", "primary_ba4_models", "global_camera_tests", "global_camera_significant")])

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Analyze GSE233408 sampled cortical regions/cell populations using the reference transcriptional gene sets.
# Input source/location: GSE233408 expression/count data, sample metadata, region labels, and reference PLS gene sets from configured input paths/R library.
# Output location: Region/cell-population effect estimates, gene-set tests, and validation summaries in the configured results directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Load and QC expression/metadata; construct analyzable groups; normalize/model expression; evaluate reference gene-set effects; correct P values; export results.
# Log location: No dedicated log file; status is written to the R console.
# =============================================================================
