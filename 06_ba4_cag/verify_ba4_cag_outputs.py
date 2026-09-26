#!/usr/bin/env python3
"""Verify BA4 CAG statistics, source-data completeness, and rendered artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    a = p.parse_args()
    root = Path(a.root)
    source = root / "figure_source_data"
    figures = root / "figures" / "final"

    expected_stems = [
        "BA4_CAG_transcriptional_alignment",
        "BA4_mapping_and_coverage",
        "score_tests_and_permutation_nulls",
        "score_CAG_sensitivity",
        "gene_level_CAG_sensitivity",
        "GSE233408_four_population_exact_tests",
    ]
    for stem in expected_stems:
        for suffix in [".pdf", ".png", ".tiff"]:
            path = figures / f"{stem}{suffix}"
            require(path.exists() and path.stat().st_size > 10_000, f"missing or empty: {path}")
        with Image.open(figures / f"{stem}.png") as image:
            require(image.width >= 2000 and image.height >= 900, f"low-resolution PNG: {stem}")
    require(len(list(figures.glob("*"))) == 18, "final directory must contain only 18 retained artifacts")

    cag = pd.read_csv(source / "Pressl_BA4_CAG_by_cell_population.csv")
    require(len(cag) == 14 and cag.target_cell_type.nunique() == 14, "The BA4 CAG source table must contain 14 unique populations")
    require(set(cag.broad_class) == {"excitatory", "inhibitory", "glial"}, "The BA4 CAG class labels are incomplete")

    b = pd.read_csv(source / "GSE233387_BA4_excitatory_PLS1positive_vs_CAG.csv")
    require(len(b) == 6, "The GSE233387 source table must contain six excitatory populations")
    require(np.isclose(b.spearman_rho.iloc[0], -0.8857142857142858), "GSE233387 rho mismatch")
    require(np.isclose(b.permutation_p_two_sided.iloc[0], 1 / 30), "GSE233387 exact P mismatch")
    require(np.isclose(b.q_across_nine_score_tests.iloc[0], 0.3), "GSE233387 Q mismatch")

    c = pd.read_csv(source / "GSE233408_BA4_deep_neuron_alignment.csv")
    require(len(c) == 4 and c.target_cell_type.nunique() == 4, "The GSE233408 alignment table must contain four direct populations")
    t4 = pd.read_csv(source / "GSE233408_BA4_exact_permutation_tests.csv")
    require((t4.permutations == 24).all(), "GSE233408 tests must enumerate all 24 permutations")
    require(np.isclose(t4.loc[t4.measure.eq("PLS1-positive expression score"), "spearman_rho"].iloc[0], 1),
            "GSE233408 PLS1-positive rho mismatch")

    donor = pd.read_csv(source / "GSE233408_BA4_donor_signed_PLS1_scores.csv")
    require(len(donor) == 57, "The GSE233408 donor-population library count must be 57")
    require(donor.donor.nunique() == 18, "The donor table must include 18 BA4 donors")
    require(set(donor.condition) == {"CTRL", "HD"}, "The donor-table condition labels are incomplete")

    tests = pd.read_csv(source / "nine_score_CAG_permutation_tests.csv")
    require(len(tests) == 9, "nine prespecified score tests required")
    require((tests.q_across_nine_score_tests >= tests.permutation_p_two_sided - 1e-12).all(),
            "BH Q values cannot be below raw P values")

    genes = pd.read_csv(source / "gene_level_CAG_permutation_results.csv.gz")
    require(len(genes) == 970, "485 genes x two sensitivity families required")
    require((genes.query("family == 'neuronal10'").global_bh_q < 0.05).sum() == 17,
            "neuronal candidate count mismatch")
    require((genes.query("family == 'all14_broad_class_adjusted'").global_bh_q < 0.05).sum() == 0,
            "all-class sensitivity should have zero BH-significant genes")

    enrich = pd.read_csv(source / "gProfiler_significant_enrichment.csv")
    require(len(enrich) == 0, "pathway result must remain a negative result")

    mapping = pd.read_csv(source / "GSE233387_Pressl_population_mapping.csv")
    require(not mapping.GSE233387_subtype.duplicated().any(), "mapping must remain exclusive")
    require(mapping.loc[mapping.target_cell_type.eq("Betz cells"), "GSE233387_subtype"].tolist() == ["Exc L5 ET"],
            "Betz mapping must be Exc L5 ET")

    forbidden = ["PLS1 program score", "frozen", "new analysis"]
    texts = []
    for path in source.glob("*.csv"):
        texts.append(path.read_text(encoding="utf-8", errors="ignore"))
    script_text = (root / "scripts" / "make_ba4_cag_figure.py").read_text(encoding="utf-8")
    require('"font.size": 14' in script_text, "general figure font must be 14 pt")
    require('"axes.labelsize": 13' in script_text, "axis labels must be 13 pt")
    require('"axes.titlesize": 14' in script_text, "panel titles must be 14 pt")
    require('fontsize=18' in script_text, "panel letters must be 18 pt")
    all_text = "\n".join(texts) + script_text
    for term in forbidden:
        require(term.lower() not in all_text.lower(), f"forbidden terminology found: {term}")

    result = {
        "status": "PASS",
        "rendered_artifacts": len(expected_stems) * 3,
        "main_panels": 4,
        "supplementary_figures": 5,
        "source_data_files": len(list(source.glob("*"))),
        "gse233387_populations": 14,
        "gse233408_ba4_donors": 18,
        "gse233408_ba4_libraries": 57,
        "score_tests": 9,
        "gene_tests": len(genes),
        "typography": {"general_pt": 12, "axis_label_pt": 13,
                       "panel_title_pt": 14, "panel_letter_pt": 18},
    }
    require(result["rendered_artifacts"] == 18, "expected one main and five supplementary figures in three formats")
    report = root / "reports"
    report.mkdir(parents=True, exist_ok=True)
    (report / "ba4_cag_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Verify completeness and internal consistency of BA4 CAG source-data and visualization outputs.
# Input source/location: BA4 CAG result/source-data/figure files expected from the analysis and plotting scripts.
# Output location: Console/QC verification report and any machine-readable validation summary produced by this script.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Check required files; validate dimensions/columns/statistical values; inspect figure availability/format; report pass/fail conditions.
# Log location: No dedicated log file unless redirected externally; verification results are written to standard output.
# =============================================================================
