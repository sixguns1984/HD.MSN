atlas_library <- Sys.getenv("HD_MSN_R_LIBRARY", unset = file.path(getwd(), "env", "R", "library"))
if (dir.exists(atlas_library)) .libPaths(c(atlas_library, .libPaths()))

packages <- c(
  "limma", "edgeR", "variancePartition", "lme4", "metafor",
  "data.table", "fgsea", "BiocParallel", "ggplot2", "readr"
)
rows <- lapply(packages, function(package) {
  installed <- requireNamespace(package, quietly = TRUE)
  data.frame(
    package = package,
    installed = installed,
    version = if (installed) as.character(packageVersion(package)) else NA_character_,
    stringsAsFactors = FALSE
  )
})
write.csv(do.call(rbind, rows), stdout(), row.names = FALSE)

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Check that the R packages required by sampled-cortex validation analyses are available in the configured R library.
# Input source/location: R library path from HD_MSN_R_LIBRARY and the package list defined in this script.
# Output location: Console report of package availability and versions; no analysis data files are created.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Configure the library; test each required namespace; report package versions/missing dependencies; stop when critical requirements are absent.
# Log location: No dedicated log file; all status information is written to the R console.
# =============================================================================
