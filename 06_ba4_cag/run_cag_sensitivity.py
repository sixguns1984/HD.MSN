#!/usr/bin/env python3
"""Sensitivity analyses for the GSE233387 14-cell-type CAG association.

The statistical unit is the published BA4 cell type (n=14).  The primary
analysis removes mean differences among excitatory, inhibitory, and glial
classes from the ranks of both variables and uses an exact permutation test
that only reorders CAG values within those three classes.

Sensitivity analyses vary one feature at a time:
1. expression-score definition, gene standardization, and donor group;
2. the order/form of broad-class residualization;
3. prespecified cell-type ranges and whether gene standardization is fixed
   across all 14 types or recomputed inside the selected range.
"""

from __future__ import annotations

import itertools
import hashlib
import json
import os
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(os.environ.get("HD_MSN_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
CAG_ROOT = Path(os.environ.get("HD_MSN_BA4_CAG_ROOT", ROOT / "results" / "ba4_cag_reanalysis")).resolve()
INPUT_RESULTS = CAG_ROOT / "analysis_results"
AGG = CAG_ROOT / "gse233387_remote_results"
SOURCE = CAG_ROOT / "source_data" / "Pressl_Table1_BA4_CAG.csv"
OUT = CAG_ROOT / "cag_sensitivity"

SCORE_COLUMNS = {
    "Signed PLS1 expression score": "signed_pls1_expression_score",
    "PLS1-positive expression score": "pls1_positive_expression_score",
    "PLS1-negative expression score": "pls1_negative_expression_score",
}


def bh(values: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    pv = p[valid]
    if len(pv) == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    qv = np.empty_like(pv)
    qv[order] = np.minimum(adjusted, 1.0)
    q[valid] = qv
    return q


def class_residual(values: np.ndarray, classes: np.ndarray) -> np.ndarray:
    design = pd.get_dummies(pd.Series(classes), drop_first=True, dtype=float)
    design.insert(0, "intercept", 1.0)
    matrix = design.to_numpy(float)
    return values - matrix @ np.linalg.lstsq(matrix, values, rcond=None)[0]


def transformed_vectors(
    x: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "rank_then_class_residual":
        xt = class_residual(stats.rankdata(x, method="average"), classes)
        yt = class_residual(stats.rankdata(y, method="average"), classes)
    elif method == "raw_class_residual_then_rank":
        xr = class_residual(np.asarray(x, dtype=float), classes)
        yr = class_residual(np.asarray(y, dtype=float), classes)
        xt = stats.rankdata(xr, method="average")
        yt = stats.rankdata(yr, method="average")
    elif method == "within_class_standardized_ranks":
        xt = np.zeros(len(x), dtype=float)
        yt = np.zeros(len(y), dtype=float)
        for label in pd.unique(classes):
            idx = np.flatnonzero(classes == label)
            xr = stats.rankdata(x[idx], method="average")
            yr = stats.rankdata(y[idx], method="average")
            xt[idx] = (xr - xr.mean()) / xr.std(ddof=0)
            yt[idx] = (yr - yr.mean()) / yr.std(ddof=0)
    elif method == "spearman_no_covariate":
        xt = stats.rankdata(x, method="average")
        yt = stats.rankdata(y, method="average")
    else:
        raise ValueError(f"Unknown method: {method}")
    return xt - np.mean(xt), yt - np.mean(yt)


def permutation_indices(classes: np.ndarray) -> np.ndarray:
    blocks = [np.flatnonzero(classes == label) for label in pd.unique(classes)]
    block_perms = [list(itertools.permutations(block.tolist())) for block in blocks]
    total = math.prod(len(p) for p in block_perms)
    result = np.tile(np.arange(len(classes), dtype=np.int16), (total, 1))
    row = 0
    for combination in itertools.product(*block_perms):
        for block, perm in zip(blocks, combination):
            result[row, block] = perm
        row += 1
    return result


def exact_test(
    x: np.ndarray,
    y: np.ndarray,
    classes: np.ndarray,
    method: str,
    indices: np.ndarray,
) -> dict[str, float | int]:
    xt, yt = transformed_vectors(x, y, classes, method)
    denominator = math.sqrt(float(xt @ xt) * float(yt @ yt))
    observed = float((xt @ yt) / denominator)
    extreme = 0
    batch = 50_000
    for start in range(0, len(indices), batch):
        current = indices[start : start + batch]
        null = yt[current] @ xt / denominator
        extreme += int(np.sum(np.abs(null) >= abs(observed) - 1e-12))
    return {
        "rho": observed,
        "exact_two_sided_p": extreme / len(indices),
        "extreme_permutations": extreme,
        "complete_permutations": len(indices),
    }


def unrestricted_monte_carlo_test(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    n_permutations: int = 1_000_000,
) -> dict[str, float | int]:
    """Unadjusted diagnostic; it does not control broad cell class."""
    xt, yt = transformed_vectors(
        x,
        y,
        np.repeat("all", len(x)),
        "spearman_no_covariate",
    )
    denominator = math.sqrt(float(xt @ xt) * float(yt @ yt))
    observed = float((xt @ yt) / denominator)
    rng = np.random.default_rng(seed)
    extreme = 0
    batch = 50_000
    for start in range(0, n_permutations, batch):
        current_n = min(batch, n_permutations - start)
        keys = rng.random((current_n, len(y)))
        indices = np.argsort(keys, axis=1)
        null = yt[indices] @ xt / denominator
        extreme += int(np.sum(np.abs(null) >= abs(observed) - 1e-12))
    return {
        "rho": observed,
        "asymptotic_two_sided_p": float(stats.spearmanr(x, y).pvalue),
        "monte_carlo_two_sided_p": (extreme + 1) / (n_permutations + 1),
        "extreme_permutations": extreme,
        "monte_carlo_permutations": n_permutations,
        "seed": seed,
    }


def rank_columns(x: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(stats.rankdata, 0, x)


def weighted_score(expr: np.ndarray, weights: np.ndarray, standardization: str) -> np.ndarray:
    if standardization == "zscore":
        transformed = np.asarray(expr, dtype=float)
    elif standardization == "rank":
        transformed = rank_columns(expr)
    else:
        raise ValueError(standardization)
    center = np.nanmean(transformed, axis=0)
    spread = np.nanstd(transformed, axis=0, ddof=0)
    z = np.zeros_like(transformed, dtype=float)
    valid = np.isfinite(spread) & (spread > 1e-12)
    z[:, valid] = (transformed[:, valid] - center[valid]) / spread[valid]
    z = np.nan_to_num(z)
    return z @ weights / np.sum(np.abs(weights))


def celltype_expression(meta: pd.DataFrame, logcpm: np.ndarray, condition: str) -> tuple[np.ndarray, list[str]]:
    if condition == "CTRL":
        eligible = meta.condition.eq("CTRL") & meta.passes_min_cells
    elif condition == "HD":
        eligible = meta.condition.eq("HD") & meta.passes_min_cells
    elif condition == "ALL":
        eligible = meta.passes_min_cells
    else:
        raise ValueError(condition)
    targets = meta.target_cell_type.drop_duplicates().tolist()
    values = []
    for target in targets:
        idx = np.flatnonzero((eligible & meta.target_cell_type.eq(target)).to_numpy())
        values.append(np.nanmean(logcpm[idx, :], axis=0))
    return np.vstack(values), targets


def scope_classes(data: pd.DataFrame, scope: str) -> np.ndarray:
    if scope in {"all14", "neuronal10"}:
        return data.broad_class.to_numpy(str)
    return np.repeat(scope, len(data)).astype(str)


def scope_method(scope: str) -> str:
    return "rank_then_class_residual" if scope in {"all14", "neuronal10"} else "spearman_no_covariate"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cag = pd.read_csv(SOURCE)
    expected = {"excitatory": 6, "inhibitory": 4, "glial": 4}
    if cag.broad_class.value_counts().to_dict() != expected:
        raise RuntimeError("Unexpected 6/4/4 broad-class composition")
    if cag.target_cell_type.duplicated().any():
        raise RuntimeError("Duplicated target cell type in CAG table")

    order = cag.target_cell_type.tolist()
    # Cache complete permutation matrices for the prespecified class structures.
    index_cache: dict[tuple[str, ...], np.ndarray] = {}

    def indices_for(classes: np.ndarray) -> np.ndarray:
        key = tuple(classes.tolist())
        if key not in index_cache:
            index_cache[key] = permutation_indices(classes)
        return index_cache[key]

    # A. Score definition, gene standardization, and donor-group sensitivity.
    score_rows = []
    all_classes = cag.broad_class.to_numpy(str)
    all_indices = indices_for(all_classes)
    for donor_group in ["CTRL", "HD", "ALL"]:
        for normalization in ["zscore", "rank"]:
            selected = pd.read_csv(INPUT_RESULTS / "celltype_pls1_expression_scores.csv")
            selected = selected[
                selected.donor_group.eq(donor_group)
                & selected.expression_standardization.eq(normalization)
            ].set_index("target_cell_type").loc[order].reset_index()
            for score_label, column in SCORE_COLUMNS.items():
                result = exact_test(
                    selected[column].to_numpy(float),
                    cag.cag_mean_somatic_length_gain.to_numpy(float),
                    all_classes,
                    "rank_then_class_residual",
                    all_indices,
                )
                score_rows.append({
                    "donor_group_used_for_expression": donor_group,
                    "gene_standardization": normalization,
                    "score": score_label,
                    "n_cell_types": 14,
                    "class_adjustment": "excitatory/inhibitory/glial indicators",
                    **result,
                })
    score_sensitivity = pd.DataFrame(score_rows)
    score_sensitivity["BH_q_across_18_sensitivity_tests"] = bh(score_sensitivity.exact_two_sided_p)
    score_sensitivity["BH_q_within_donor_group_and_standardization"] = (
        score_sensitivity.groupby(
            ["donor_group_used_for_expression", "gene_standardization"], observed=True
        ).exact_two_sided_p.transform(bh)
    )
    score_sensitivity.to_csv(OUT / "CAG_score_definition_sensitivity.csv", index=False)

    # B. Alternative ways of removing broad-class differences.
    primary = pd.read_csv(INPUT_RESULTS / "celltype_pls1_expression_scores.csv")
    primary = primary[
        primary.donor_group.eq("CTRL") & primary.expression_standardization.eq("zscore")
    ].set_index("target_cell_type").loc[order].reset_index()
    ranking = primary[[
        "target_cell_type",
        "broad_class",
        "pls1_positive_expression_score",
        "cag_mean_somatic_length_gain",
    ]].copy()
    ranking["PLS1_positive_score_rank_low_to_high"] = ranking[
        "pls1_positive_expression_score"
    ].rank(method="average", ascending=True).astype(int)
    ranking["CAG_gain_rank_low_to_high"] = ranking[
        "cag_mean_somatic_length_gain"
    ].rank(method="average", ascending=True).astype(int)
    ranking["rank_difference_expression_minus_CAG"] = (
        ranking.PLS1_positive_score_rank_low_to_high - ranking.CAG_gain_rank_low_to_high
    )
    ranking = ranking.sort_values("PLS1_positive_score_rank_low_to_high")
    ranking.to_csv(OUT / "GSE233387_14celltype_PLS1_positive_score_ranking.csv", index=False)
    residual_rows = []
    for method in [
        "rank_then_class_residual",
        "raw_class_residual_then_rank",
        "within_class_standardized_ranks",
    ]:
        for score_label, column in SCORE_COLUMNS.items():
            result = exact_test(
                primary[column].to_numpy(float),
                cag.cag_mean_somatic_length_gain.to_numpy(float),
                all_classes,
                method,
                all_indices,
            )
            residual_rows.append({
                "residualization_method": method,
                "score": score_label,
                "n_cell_types": 14,
                **result,
            })
    residualization = pd.DataFrame(residual_rows)
    residualization["BH_q_across_9_residualization_tests"] = bh(residualization.exact_two_sided_p)
    residualization["BH_q_within_method"] = residualization.groupby(
        "residualization_method", observed=True
    ).exact_two_sided_p.transform(bh)
    residualization.to_csv(OUT / "CAG_residualization_sensitivity.csv", index=False)

    # Diagnostic comparison without broad-class adjustment. This answers a
    # different question and is kept separate from the restricted primary test.
    unadjusted_rows = []
    for score_index, (score_label, column) in enumerate(SCORE_COLUMNS.items()):
        result = unrestricted_monte_carlo_test(
            primary[column].to_numpy(float),
            cag.cag_mean_somatic_length_gain.to_numpy(float),
            seed=20260827 + score_index,
        )
        unadjusted_rows.append({
            "score": score_label,
            "n_cell_types": 14,
            "broad_class_adjustment": "none",
            **result,
        })
    unadjusted = pd.DataFrame(unadjusted_rows)
    unadjusted["BH_q_across_3_unadjusted_diagnostics"] = bh(unadjusted.monte_carlo_two_sided_p)
    unadjusted.to_csv(OUT / "CAG_unadjusted_diagnostic.csv", index=False)

    # C. Prespecified cell-type scopes, evaluated with fixed all-14 scores and
    # with scores recalculated after gene standardization inside each scope.
    archive = np.load(AGG / "donor_celltype_logcpm_pls_genes.npz", allow_pickle=True)
    logcpm = archive["logcpm"]
    genes = archive["genes"].astype(str)
    meta = pd.read_csv(AGG / "donor_celltype_metadata.csv")
    meta["passes_min_cells"] = meta.passes_min_cells.astype(bool)
    weights_table = pd.read_csv(AGG / "matched_pls1_weights.csv").set_index("gene").loc[genes]
    weights = weights_table.PLS1W_corr_Z.to_numpy(float)
    gene_set = weights_table.gene_set.fillna("").to_numpy(str)
    ctrl_expr, expression_order = celltype_expression(meta, logcpm, "CTRL")
    position = {name: i for i, name in enumerate(expression_order)}
    ctrl_expr = ctrl_expr[[position[name] for name in order], :]

    scope_masks = {
        "all14": np.ones(len(cag), dtype=bool),
        "neuronal10": cag.broad_class.isin(["excitatory", "inhibitory"]).to_numpy(),
        "excitatory6": cag.broad_class.eq("excitatory").to_numpy(),
        "inhibitory4": cag.broad_class.eq("inhibitory").to_numpy(),
        "glial4": cag.broad_class.eq("glial").to_numpy(),
    }
    scope_rows = []
    for scaling in ["fixed_all14_gene_standardization", "within_scope_gene_standardization"]:
        for scope, mask in scope_masks.items():
            subset_cag = cag.loc[mask].reset_index(drop=True)
            classes = scope_classes(subset_cag, scope)
            perm_indices = indices_for(classes)
            if scaling == "fixed_all14_gene_standardization":
                score_values = {
                    label: primary.loc[mask, column].to_numpy(float)
                    for label, column in SCORE_COLUMNS.items()
                }
            else:
                expr = ctrl_expr[mask, :]
                score_values = {
                    "Signed PLS1 expression score": weighted_score(expr, weights, "zscore"),
                    "PLS1-positive expression score": weighted_score(
                        expr[:, gene_set == "PLS1+"], np.ones(np.sum(gene_set == "PLS1+")), "zscore"
                    ),
                    "PLS1-negative expression score": weighted_score(
                        expr[:, gene_set == "PLS1-"], np.ones(np.sum(gene_set == "PLS1-")), "zscore"
                    ),
                }
            for score_label, values in score_values.items():
                result = exact_test(
                    values,
                    subset_cag.cag_mean_somatic_length_gain.to_numpy(float),
                    classes,
                    scope_method(scope),
                    perm_indices,
                )
                scope_rows.append({
                    "score_scaling": scaling,
                    "cell_type_scope": scope,
                    "included_cell_types": "; ".join(subset_cag.target_cell_type),
                    "score": score_label,
                    "n_cell_types": int(mask.sum()),
                    "class_adjustment": (
                        "excitatory/inhibitory/glial indicators"
                        if scope == "all14"
                        else "excitatory/inhibitory indicator"
                        if scope == "neuronal10"
                        else "not applicable; one broad class"
                    ),
                    **result,
                })
    scope_results = pd.DataFrame(scope_rows)
    scope_results["BH_q_across_30_scope_and_scaling_tests"] = bh(scope_results.exact_two_sided_p)
    scope_results["BH_q_within_scaling_across_15_tests"] = scope_results.groupby(
        "score_scaling", observed=True
    ).exact_two_sided_p.transform(bh)
    scope_results["BH_q_within_scope_across_3_scores"] = scope_results.groupby(
        ["score_scaling", "cell_type_scope"], observed=True
    ).exact_two_sided_p.transform(bh)
    scope_results.to_csv(OUT / "CAG_prespecified_cell_scope_sensitivity.csv", index=False)

    # Machine-readable audit and a compact table containing all primary-score rows.
    primary_score = "PLS1-positive expression score"
    compact = pd.concat([
        score_sensitivity[score_sensitivity.score.eq(primary_score)].assign(analysis_block="score_definition"),
        residualization[residualization.score.eq(primary_score)].assign(analysis_block="residualization"),
        scope_results[scope_results.score.eq(primary_score)].assign(analysis_block="cell_type_scope"),
    ], ignore_index=True, sort=False)
    compact.to_csv(OUT / "CAG_PLS1_positive_sensitivity_summary.csv", index=False)

    qc = {
        "input_cell_types": len(cag),
        "broad_class_counts": cag.broad_class.value_counts().to_dict(),
        "complete_permutation_counts": {
            "all14_6x4x4": int(len(all_indices)),
            "neuronal10_6x4": int(len(indices_for(np.array(["excitatory"] * 6 + ["inhibitory"] * 4)))),
            "single_class_6": int(math.factorial(6)),
            "single_class_4": int(math.factorial(4)),
        },
        "score_definition_tests": len(score_sensitivity),
        "residualization_tests": len(residualization),
        "cell_scope_tests": len(scope_results),
        "unadjusted_diagnostic_tests": len(unadjusted),
        "nonfinite_statistics": int(
            score_sensitivity[["rho", "exact_two_sided_p"]].isna().sum().sum()
            + residualization[["rho", "exact_two_sided_p"]].isna().sum().sum()
            + scope_results[["rho", "exact_two_sided_p"]].isna().sum().sum()
        ),
        "minimum_exact_p_by_permutation_count": {
            "414720": 2 / 414720,
            "17280": 2 / 17280,
            "720": 2 / 720,
            "24": 2 / 24,
        },
        "input_sha256": {
            str(SOURCE.relative_to(ROOT)): hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            str((INPUT_RESULTS / "celltype_pls1_expression_scores.csv").relative_to(ROOT)):
                hashlib.sha256((INPUT_RESULTS / "celltype_pls1_expression_scores.csv").read_bytes()).hexdigest(),
            str((AGG / "donor_celltype_logcpm_pls_genes.npz").relative_to(ROOT)):
                hashlib.sha256((AGG / "donor_celltype_logcpm_pls_genes.npz").read_bytes()).hexdigest(),
            str((AGG / "matched_pls1_weights.csv").relative_to(ROOT)):
                hashlib.sha256((AGG / "matched_pls1_weights.csv").read_bytes()).hexdigest(),
        },
    }
    (OUT / "CAG_sensitivity_QC.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Run prespecified robustness analyses for the BA4 cell-type aggregate CAG/transcriptional association.
# Input source/location: Primary BA4 cell-type aggregate analysis table and alternative score/aggregation inputs under HD_MSN_BA4_CAG_ROOT or results/ba4_cag_reanalysis.
# Output location: Sensitivity association tables, null distributions, and diagnostic outputs in the configured BA4 CAG results tree.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Load primary data; construct alternative specifications; recompute correlations/null tests; compare effect stability; export sensitivity results.
# Log location: No dedicated log file unless redirected externally; progress is written to standard output.
# =============================================================================
