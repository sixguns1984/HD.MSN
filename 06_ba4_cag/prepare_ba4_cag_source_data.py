#!/usr/bin/env python3
"""Prepare panel-specific source-data tables for the BA4 CAG analysis and supplements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--donor-scores", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out = root / "figure_source_data"
    out.mkdir(parents=True, exist_ok=True)

    cag = pd.read_csv(root / "source_data" / "Pressl_Table1_BA4_CAG.csv")
    cag.to_csv(out / "Pressl_BA4_CAG_by_cell_population.csv", index=False)

    fs = pd.read_csv(root / "analysis_results" / "family_specific_celltype_scores.csv")
    test = pd.read_csv(root / "analysis_results" / "score_cag_primary_tests.csv")
    panel_b = fs[
        fs.analysis.eq("Six cortical excitatory populations")
        & fs.score.eq("PLS1-positive expression score")
    ].merge(cag, on="target_cell_type", validate="one_to_one")
    row = test[
        test.analysis.eq("Six cortical excitatory populations")
        & test.score.eq("PLS1-positive expression score")
    ].iloc[0]
    for col in ["spearman_rho", "permutation_p_two_sided", "q_across_nine_score_tests",
                "permutations", "permutation_method"]:
        panel_b[col] = row[col]
    panel_b.to_csv(out / "GSE233387_BA4_excitatory_PLS1positive_vs_CAG.csv", index=False)

    alignment = pd.read_csv(root / "gse233408_analysis" / "gse233408_ba4_four_population_alignment.csv")
    tests408 = pd.read_csv(root / "gse233408_analysis" / "gse233408_ba4_four_population_tests.csv")
    alignment.to_csv(out / "GSE233408_BA4_deep_neuron_alignment.csv", index=False)
    tests408.to_csv(out / "GSE233408_BA4_exact_permutation_tests.csv", index=False)

    donor = pd.read_csv(args.donor_scores)
    label_map = {
        "Betz cell": "Betz cells",
        "L5a pyramidal neuron": "Layer 5a",
        "L6a pyramidal neuron": "Layer 6a",
        "L6b pyramidal neuron": "Layer 6b",
    }
    donor = donor[donor.region.eq("BA4") & donor.cell_type.isin(label_map)].copy()
    donor["target_cell_type"] = donor.cell_type.map(label_map)
    donor = donor.merge(
        alignment[["target_cell_type", "cag_mean_somatic_length_gain", "directional_z",
                   "global_q_all_eligible_region_celltype_sets"]],
        on="target_cell_type", validate="many_to_one"
    )
    donor.to_csv(out / "GSE233408_BA4_donor_signed_PLS1_scores.csv", index=False)

    meta = pd.read_csv(root / "gse233387_remote_results" / "donor_celltype_metadata.csv")
    mapping = pd.read_csv(root / "source_data" / "GSE233387_Pressl_exclusive_mapping.csv")
    meta.to_csv(out / "GSE233387_BA4_donor_cell_counts.csv", index=False)
    mapping.to_csv(out / "GSE233387_Pressl_population_mapping.csv", index=False)

    test.to_csv(out / "nine_score_CAG_permutation_tests.csv", index=False)
    nulls = np.load(root / "analysis_results" / "score_cag_null_distributions.npz")
    null_rows = []
    for key in nulls.files:
        values = nulls[key]
        for i, value in enumerate(values):
            null_rows.append({"test_key": key, "permutation_index": i + 1, "null_spearman_rho": value})
    pd.DataFrame(null_rows).to_csv(
        out / "score_CAG_null_distributions.csv.gz", index=False, compression="gzip"
    )

    sensitivity = pd.read_csv(root / "analysis_results" / "score_cag_sensitivity.csv")
    sensitivity.to_csv(out / "score_CAG_sensitivity.csv", index=False)

    genes = pd.read_csv(root / "analysis_results" / "gene_level_cag_permutation_globalBH.csv.gz")
    genes.to_csv(out / "gene_level_CAG_permutation_results.csv.gz", index=False, compression="gzip")

    enrich = root / "analysis_results" / "neuronal10_17gene_gprofiler_enrichment.csv"
    if enrich.exists():
        pd.read_csv(enrich).to_csv(out / "gProfiler_significant_enrichment.csv", index=False)

    null408 = np.load(root / "gse233408_analysis" / "gse233408_ba4_four_population_nulls.npz")
    rows408 = []
    for key in null408.files:
        for i, value in enumerate(null408[key]):
            rows408.append({"measure": key, "permutation_index": i + 1, "null_spearman_rho": value})
    pd.DataFrame(rows408).to_csv(out / "GSE233408_exact_null_distributions.csv", index=False)

    manifest = {
        "main_figure": {
            "A": "Pressl_BA4_CAG_by_cell_population.csv",
            "B": "GSE233387_BA4_excitatory_PLS1positive_vs_CAG.csv",
            "C": ["GSE233408_BA4_deep_neuron_alignment.csv",
                  "GSE233408_BA4_exact_permutation_tests.csv"],
            "D": "GSE233408_BA4_donor_signed_PLS1_scores.csv",
        },
        "supplementary_figures": {
            "mapping_and_coverage": ["GSE233387_BA4_donor_cell_counts.csv",
                    "GSE233387_Pressl_population_mapping.csv"],
            "score_permutation_tests": ["nine_score_CAG_permutation_tests.csv",
                    "score_CAG_null_distributions.csv.gz"],
            "score_sensitivity": "score_CAG_sensitivity.csv",
            "gene_level_sensitivity": "gene_level_CAG_permutation_results.csv.gz",
            "pathway_enrichment": "gProfiler_significant_enrichment.csv",
            "gse233408_exact_tests": "GSE233408_exact_null_distributions.csv",
        },
    }
    (out / "source_data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output": str(out), "files": len(list(out.glob('*')))}, indent=2))


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Assemble clean source-data tables for the BA4 transcriptional/CAG composite visualization.
# Input source/location: BA4 CAG association, permutation, subtype-validation, and enrichment result tables from the configured analysis directories.
# Output location: Descriptively named source-data CSV files in the configured BA4 figure/source-data directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Load component results; select/rename presentation fields; verify keys and units; merge where required; export reproducible source-data tables.
# Log location: No dedicated log file; validation failures are raised to the caller.
# =============================================================================
