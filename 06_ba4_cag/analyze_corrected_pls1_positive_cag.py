#!/usr/bin/env python3
"""Re-run BA4 CAG analyses using the user-supplied corrected PLS1+ scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from run_cag_sensitivity import (
    bh,
    exact_test,
    permutation_indices,
    transformed_vectors,
    unrestricted_monte_carlo_test,
)


ROOT = Path(os.environ.get("HD_MSN_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
CAG_ROOT = Path(os.environ.get("HD_MSN_BA4_CAG_ROOT", ROOT / "results" / "ba4_cag_reanalysis")).resolve()
PRESSL_CAG = CAG_ROOT / "source_data" / "Pressl_Table1_BA4_CAG.csv"
OLD_SCORES = CAG_ROOT / "analysis_results" / "celltype_pls1_expression_scores.csv"
OUT = CAG_ROOT / "corrected_score_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-file", required=True)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--unadjusted-permutations", type=int, default=1_000_000)
    return parser.parse_args()


def correlation_from_transformed(x: np.ndarray, y: np.ndarray, classes: np.ndarray, method: str) -> float:
    xt, yt = transformed_vectors(x, y, classes, method)
    return float((xt @ yt) / np.sqrt(float(xt @ xt) * float(yt @ yt)))


def validate_and_prepare(score_file: Path) -> tuple[pd.DataFrame, dict]:
    supplied = pd.read_csv(score_file)
    required = {
        "target_cell_type",
        "broad_class",
        "pls1_positive_expression_score",
        "cag_mean_somatic_length_gain",
    }
    missing = required.difference(supplied.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {sorted(missing)}")
    if len(supplied) != 14 or supplied.target_cell_type.nunique() != 14:
        raise RuntimeError("Corrected score file must contain 14 unique cell types")
    if supplied[list(required)].isna().any().any():
        raise RuntimeError("Corrected score file contains missing required values")
    expected_classes = {"excitatory": 6, "inhibitory": 4, "glial": 4}
    if supplied.broad_class.value_counts().to_dict() != expected_classes:
        raise RuntimeError("Corrected score file does not have the expected 6/4/4 class structure")

    reference = pd.read_csv(PRESSL_CAG)
    merged = reference[[
        "target_cell_type", "broad_class", "cag_mean_somatic_length_gain", "cag_sd", "analysis_set", "source"
    ]].merge(
        supplied[["target_cell_type", "broad_class", "pls1_positive_expression_score", "cag_mean_somatic_length_gain"]],
        on="target_cell_type",
        suffixes=("_reference", "_supplied"),
        validate="one_to_one",
    )
    if len(merged) != 14:
        raise RuntimeError("Corrected score cell types do not match the Pressl CAG table")
    if not (merged.broad_class_reference == merged.broad_class_supplied).all():
        raise RuntimeError("Broad-class labels disagree with the Pressl CAG table")
    if not np.allclose(
        merged.cag_mean_somatic_length_gain_reference,
        merged.cag_mean_somatic_length_gain_supplied,
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError("CAG values disagree with the Pressl CAG table")

    data = merged.rename(columns={
        "broad_class_reference": "broad_class",
        "cag_mean_somatic_length_gain_reference": "cag_mean_somatic_length_gain",
    })[[
        "target_cell_type", "broad_class", "pls1_positive_expression_score",
        "cag_mean_somatic_length_gain", "cag_sd", "analysis_set", "source",
    ]]
    data["PLS1_positive_score_rank_low_to_high"] = data.pls1_positive_expression_score.rank(
        method="average", ascending=True
    ).astype(int)
    data["CAG_gain_rank_low_to_high"] = data.cag_mean_somatic_length_gain.rank(
        method="average", ascending=True
    ).astype(int)
    data["rank_difference_expression_minus_CAG"] = (
        data.PLS1_positive_score_rank_low_to_high - data.CAG_gain_rank_low_to_high
    )

    if "PLS1_positive_score_rank_low_to_high" in supplied:
        supplied_rank = supplied.set_index("target_cell_type").loc[data.target_cell_type, "PLS1_positive_score_rank_low_to_high"]
        if not np.array_equal(supplied_rank.to_numpy(int), data.PLS1_positive_score_rank_low_to_high.to_numpy(int)):
            raise RuntimeError("Supplied expression ranks do not match recalculated ranks")
    if "CAG_gain_rank_low_to_high" in supplied:
        supplied_rank = supplied.set_index("target_cell_type").loc[data.target_cell_type, "CAG_gain_rank_low_to_high"]
        if not np.array_equal(supplied_rank.to_numpy(int), data.CAG_gain_rank_low_to_high.to_numpy(int)):
            raise RuntimeError("Supplied CAG ranks do not match recalculated ranks")

    audit = {
        "input_file": str(score_file),
        "input_sha256": hashlib.sha256(score_file.read_bytes()).hexdigest(),
        "rows": len(data),
        "unique_cell_types": int(data.target_cell_type.nunique()),
        "broad_class_counts": data.broad_class.value_counts().to_dict(),
        "CAG_values_match_Pressl_source": True,
        "supplied_ranks_recalculated_and_verified": True,
    }
    return data, audit


def main() -> None:
    args = parse_args()
    score_file = Path(args.score_file).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    data, audit = validate_and_prepare(score_file)
    data.sort_values("PLS1_positive_score_rank_low_to_high").to_csv(
        OUT / "corrected_GSE233387_14celltype_PLS1_positive_and_CAG_ranks.csv", index=False
    )
    # Preserve an exact normalized snapshot of the user-supplied scoring input.
    pd.read_csv(score_file).to_csv(OUT / "input_snapshot_corrected_PLS1_positive_scores.csv", index=False)

    x = data.pls1_positive_expression_score.to_numpy(float)
    y = data.cag_mean_somatic_length_gain.to_numpy(float)
    classes = data.broad_class.to_numpy(str)

    unadjusted = unrestricted_monte_carlo_test(
        x, y, seed=args.seed, n_permutations=args.unadjusted_permutations
    )
    pd.DataFrame([{
        "analysis": "All 14 cell types; no broad-class adjustment",
        "n_cell_types": 14,
        **unadjusted,
    }]).to_csv(OUT / "corrected_CAG_unadjusted_14celltype_test.csv", index=False)

    all_indices = permutation_indices(classes)
    primary = exact_test(x, y, classes, "rank_then_class_residual", all_indices)
    pd.DataFrame([{
        "analysis": "All 14 cell types; ranks residualized for excitatory/inhibitory/glial class",
        "n_cell_types": 14,
        **primary,
    }]).to_csv(OUT / "corrected_CAG_class_adjusted_14celltype_test.csv", index=False)

    residual_rows = []
    for method in [
        "rank_then_class_residual",
        "raw_class_residual_then_rank",
        "within_class_standardized_ranks",
    ]:
        result = exact_test(x, y, classes, method, all_indices)
        residual_rows.append({"residualization_method": method, "n_cell_types": 14, **result})
    residual = pd.DataFrame(residual_rows)
    residual["BH_q_across_3_residualization_methods"] = bh(residual.exact_two_sided_p)
    residual.to_csv(OUT / "corrected_CAG_residualization_sensitivity.csv", index=False)

    scopes = {
        "all14": np.ones(len(data), dtype=bool),
        "neuronal10": data.broad_class.isin(["excitatory", "inhibitory"]).to_numpy(),
        "excitatory6": data.broad_class.eq("excitatory").to_numpy(),
        "inhibitory4": data.broad_class.eq("inhibitory").to_numpy(),
        "glial4": data.broad_class.eq("glial").to_numpy(),
    }
    scope_rows = []
    for scope, mask in scopes.items():
        subset = data.loc[mask].reset_index(drop=True)
        if scope == "all14":
            subset_classes = subset.broad_class.to_numpy(str)
            method = "rank_then_class_residual"
            adjustment = "excitatory/inhibitory/glial indicators"
        elif scope == "neuronal10":
            subset_classes = subset.broad_class.to_numpy(str)
            method = "rank_then_class_residual"
            adjustment = "excitatory/inhibitory indicator"
        else:
            subset_classes = np.repeat(scope, len(subset)).astype(str)
            method = "spearman_no_covariate"
            adjustment = "not applicable; one broad class"
        result = exact_test(
            subset.pls1_positive_expression_score.to_numpy(float),
            subset.cag_mean_somatic_length_gain.to_numpy(float),
            subset_classes,
            method,
            permutation_indices(subset_classes),
        )
        scope_rows.append({
            "cell_type_scope": scope,
            "included_cell_types": "; ".join(subset.target_cell_type),
            "n_cell_types": len(subset),
            "class_adjustment": adjustment,
            **result,
        })
    scope = pd.DataFrame(scope_rows)
    scope["BH_q_across_5_prespecified_scopes"] = bh(scope.exact_two_sided_p)
    scope.to_csv(OUT / "corrected_CAG_prespecified_cell_scope_tests.csv", index=False)

    overall_two = pd.DataFrame([
        {
            "analysis": "all14_unadjusted",
            "rho": unadjusted["rho"],
            "two_sided_p": unadjusted["monte_carlo_two_sided_p"],
            "permutation_method": "1,000,000 unrestricted Monte Carlo permutations",
        },
        {
            "analysis": "all14_broad_class_adjusted",
            "rho": primary["rho"],
            "two_sided_p": primary["exact_two_sided_p"],
            "permutation_method": "414,720 complete within-class permutations",
        },
    ])
    overall_two["BH_q_across_2_overall_14celltype_tests"] = bh(overall_two.two_sided_p)
    overall_two.to_csv(OUT / "corrected_CAG_two_overall_14celltype_tests.csv", index=False)

    all_reported = pd.concat([
        pd.DataFrame([{
            "analysis": "all14_unadjusted",
            "rho": unadjusted["rho"],
            "two_sided_p": unadjusted["monte_carlo_two_sided_p"],
        }]),
        scope[["cell_type_scope", "rho", "exact_two_sided_p"]].rename(
            columns={"cell_type_scope": "analysis", "exact_two_sided_p": "two_sided_p"}
        ),
    ], ignore_index=True)
    all_reported["BH_q_across_6_reported_association_tests"] = bh(all_reported.two_sided_p)
    all_reported.to_csv(OUT / "corrected_CAG_all_6_reported_association_tests.csv", index=False)

    lodo_rows = []
    for omitted in data.target_cell_type:
        subset = data[data.target_cell_type.ne(omitted)].reset_index(drop=True)
        sx = subset.pls1_positive_expression_score.to_numpy(float)
        sy = subset.cag_mean_somatic_length_gain.to_numpy(float)
        sc = subset.broad_class.to_numpy(str)
        lodo_rows.append({
            "omitted_cell_type": omitted,
            "unadjusted_spearman_rho": correlation_from_transformed(sx, sy, sc, "spearman_no_covariate"),
            "class_adjusted_partial_spearman_rho": correlation_from_transformed(
                sx, sy, sc, "rank_then_class_residual"
            ),
        })
    lodo = pd.DataFrame(lodo_rows)
    lodo.to_csv(OUT / "corrected_CAG_leave_one_cell_type_out.csv", index=False)

    old = pd.read_csv(OLD_SCORES)
    old = old[
        old.donor_group.eq("CTRL") & old.expression_standardization.eq("zscore")
    ][["target_cell_type", "pls1_positive_expression_score"]].rename(
        columns={"pls1_positive_expression_score": "old_pls1_positive_expression_score"}
    )
    comparison = data[["target_cell_type", "pls1_positive_expression_score"]].merge(
        old, on="target_cell_type", validate="one_to_one"
    )
    comparison["corrected_rank"] = comparison.pls1_positive_expression_score.rank().astype(int)
    comparison["old_rank"] = comparison.old_pls1_positive_expression_score.rank().astype(int)
    comparison["rank_change_corrected_minus_old"] = comparison.corrected_rank - comparison.old_rank
    comparison.to_csv(OUT / "corrected_vs_old_PLS1_positive_scores.csv", index=False)

    audit.update({
        "unadjusted_permutations": int(args.unadjusted_permutations),
        "unadjusted_seed": int(args.seed),
        "restricted_permutations_all14": int(len(all_indices)),
        "nonfinite_output_statistics": int(
            residual[["rho", "exact_two_sided_p"]].isna().sum().sum()
            + scope[["rho", "exact_two_sided_p"]].isna().sum().sum()
            + lodo[["unadjusted_spearman_rho", "class_adjusted_partial_spearman_rho"]].isna().sum().sum()
        ),
        "output_files": [
            "corrected_GSE233387_14celltype_PLS1_positive_and_CAG_ranks.csv",
            "corrected_CAG_unadjusted_14celltype_test.csv",
            "corrected_CAG_class_adjusted_14celltype_test.csv",
            "corrected_CAG_residualization_sensitivity.csv",
            "corrected_CAG_prespecified_cell_scope_tests.csv",
            "corrected_CAG_two_overall_14celltype_tests.csv",
            "corrected_CAG_all_6_reported_association_tests.csv",
            "corrected_CAG_leave_one_cell_type_out.csv",
            "corrected_vs_old_PLS1_positive_scores.csv",
        ],
    })
    (OUT / "corrected_CAG_analysis_QC.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Analyze the corrected PLS1-positive expression score against somatic CAG expansion across the predefined BA4 cell-type aggregate units.
# Input source/location: BA4 aggregate CAG/transcriptional table plus reference PLS1-positive gene-set inputs under HD_MSN_BA4_CAG_ROOT or results/ba4_cag_reanalysis.
# Output location: Correlation/permutation statistics, sensitivity tables, and supporting plots in the configured BA4 CAG results tree.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Validate aggregate units; compute corrected expression scores; estimate Spearman association; run permutation/sensitivity analyses; export tables and figures.
# Log location: No dedicated log file unless the script redirects output; progress is written to standard output.
# =============================================================================
