#!/usr/bin/env python3
"""Compact numerical verification for the CAG sensitivity outputs."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "cag_sensitivity"


def main() -> None:
    score = pd.read_csv(ROOT / "CAG_score_definition_sensitivity.csv")
    residual = pd.read_csv(ROOT / "CAG_residualization_sensitivity.csv")
    scope = pd.read_csv(ROOT / "CAG_prespecified_cell_scope_sensitivity.csv")
    unadjusted = pd.read_csv(ROOT / "CAG_unadjusted_diagnostic.csv")
    assert (len(score), len(residual), len(scope), len(unadjusted)) == (18, 9, 30, 3)
    for table in [score, residual, scope]:
        assert np.isfinite(table[["rho", "exact_two_sided_p"]]).all().all()
        assert table.exact_two_sided_p.between(0, 1).all()
    primary = score[
        score.donor_group_used_for_expression.eq("CTRL")
        & score.gene_standardization.eq("zscore")
        & score.score.eq("PLS1-positive expression score")
    ].iloc[0]
    assert np.isclose(primary.rho, -0.4068255843980004)
    assert np.isclose(primary.exact_two_sided_p, 0.1927806712962963)
    excitatory = scope[
        scope.score_scaling.eq("within_scope_gene_standardization")
        & scope.cell_type_scope.eq("excitatory6")
        & scope.score.eq("PLS1-positive expression score")
    ].iloc[0]
    assert np.isclose(excitatory.rho, -0.8857142857142857)
    assert np.isclose(excitatory.exact_two_sided_p, 1 / 30)
    print({
        "verified": True,
        "minimum_score_definition_p": float(score.exact_two_sided_p.min()),
        "minimum_residualization_p": float(residual.exact_two_sided_p.min()),
        "minimum_scope_p": float(scope.exact_two_sided_p.min()),
        "minimum_scope_q_across_30": float(scope.BH_q_across_30_scope_and_scaling_tests.min()),
        "unadjusted_positive_score_monte_carlo_p": float(
            unadjusted.loc[
                unadjusted.score.eq("PLS1-positive expression score"),
                "monte_carlo_two_sided_p",
            ].iloc[0]
        ),
    })


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Verify the expected robustness-analysis outputs for the BA4 CAG/transcriptional association.
# Input source/location: CAG sensitivity result tables and null-distribution files under the configured BA4 CAG results root.
# Output location: Verification status/QC summaries; no primary scientific results are recomputed.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Check required files and schemas; compare key statistics against expected relationships; flag missing/inconsistent products; report status.
# Log location: No dedicated log file; verification status is written to standard output.
# =============================================================================
