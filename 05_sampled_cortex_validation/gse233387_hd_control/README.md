# GSE233387 donor-by-cell-type HD-control analysis

This workflow generates the 14 BA4 cell-type results used in the sampled-cortex validation.

1. `aggregate_gse233387_full_transcriptome.py` maps the released annotations to the 14 manuscript cell types and aggregates raw nuclear counts into donor-by-cell-type pseudobulks. It checks exact count conservation.
2. `analyze_gse233387_hd_control.R` performs TMM normalization, logCPM transformation, PLS1-positive gene-set scoring, donor-adjusted HD-control contrasts, CAMERA competitive gene-set tests, Benjamini-Hochberg correction, and the omnibus HD-by-cell-type interaction test.
3. `plot_gse233387_hd_control_effects.py` plots the standardized HD-control estimates and 95% confidence intervals from the R output.

The raw H5AD file and donor metadata are not redistributed. Supply them through the command-line arguments of the aggregation script. The R script accepts the prepared-data directory, PLS1-weight table, and output directory as positional arguments.

The plotting script expects the statistical output tables in a sibling `results` directory. Adjust `ROOT` or preserve that directory arrangement when running the script outside the original release structure.
