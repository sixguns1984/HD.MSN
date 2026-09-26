#!/usr/bin/env python3
"""Analyze BA4 cell-population expression versus Pressl Table 1 CAG gain.

Primary expression summaries use equally weighted control donors to describe
baseline cell-population expression. All-donor and HD-only summaries, alternate
normalizations, leave-one-population-out analyses, and an alternative Betz
mapping are sensitivity analyses. The statistical unit for CAG association is
the cell population; donors stabilize expression estimates but do not increase
the CAG association sample size.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PINK = "#FEA3A2"
PURPLE = "#8E8BFE"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--aggregation", required=True)
    p.add_argument("--cag", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=20260814)
    p.add_argument("--score-permutations", type=int, default=1_000_000)
    p.add_argument("--gene-permutations", type=int, default=100_000)
    return p.parse_args()


def rank_columns(x: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(stats.rankdata, 0, x)


def corr_columns(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc = x - np.nanmean(x, axis=0, keepdims=True)
    yc = y - np.nanmean(y)
    denom = np.sqrt(np.nansum(xc * xc, axis=0) * np.nansum(yc * yc))
    out = np.full(x.shape[1], np.nan, dtype=float)
    valid = np.isfinite(denom) & (denom > 1e-12)
    out[valid] = np.nansum(xc[:, valid] * yc[:, None], axis=0) / denom[valid]
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(stats.spearmanr(x, y).statistic)


def exact_p(x: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    obs = abs(spearman(x, y))
    n = len(y)
    exceed = 0
    total = 0
    for perm in itertools.permutations(range(n)):
        total += 1
        if abs(spearman(x, y[list(perm)])) >= obs - 1e-12:
            exceed += 1
    return exceed / total, total


def permutation_index_matrix(n: int, n_perm: int, rng: np.random.Generator,
                             blocks: np.ndarray | None = None) -> np.ndarray:
    keys = rng.random((n_perm, n))
    if blocks is None:
        return np.argsort(keys, axis=1)
    index = np.tile(np.arange(n), (n_perm, 1))
    for block in np.unique(blocks):
        cols = np.flatnonzero(blocks == block)
        index[:, cols] = cols[np.argsort(keys[:, cols], axis=1)]
    return index


def monte_carlo_p(x: np.ndarray, y: np.ndarray, n_perm: int, rng: np.random.Generator,
                  blocks: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    obs = abs(spearman(x, y))
    xr = stats.zscore(stats.rankdata(x), ddof=0)
    yr = stats.zscore(stats.rankdata(y), ddof=0)
    null = np.empty(n_perm, dtype=np.float32)
    batch = 100_000
    for start in range(0, n_perm, batch):
        stop = min(start + batch, n_perm)
        idx = permutation_index_matrix(len(y), stop - start, rng, blocks)
        null[start:stop] = (yr[idx] @ xr / len(y)).astype(np.float32)
    return (1 + int(np.sum(np.abs(null) >= obs - 1e-12))) / (n_perm + 1), null


def weighted_score(expr: np.ndarray, weights: np.ndarray, standardization: str) -> np.ndarray:
    # expr: populations x genes. Standardization is performed for each gene
    # across populations so the score captures relative cell-population expression.
    if standardization == "zscore":
        center = np.nanmean(expr, axis=0)
        spread = np.nanstd(expr, axis=0, ddof=0)
        z = np.zeros_like(expr, dtype=float)
        valid = np.isfinite(spread) & (spread > 1e-12)
        z[:, valid] = (expr[:, valid] - center[valid]) / spread[valid]
    elif standardization == "rank":
        r = rank_columns(expr)
        center = np.nanmean(r, axis=0)
        spread = np.nanstd(r, axis=0, ddof=0)
        z = np.zeros_like(r, dtype=float)
        valid = np.isfinite(spread) & (spread > 1e-12)
        z[:, valid] = (r[:, valid] - center[valid]) / spread[valid]
    else:
        raise ValueError(standardization)
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


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    valid = np.isfinite(p)
    pv = p[valid]
    order = np.argsort(pv)
    qv = np.empty_like(pv)
    qv[order] = np.minimum.accumulate((pv[order] * len(pv) / np.arange(1, len(pv)+1))[::-1])[::-1]
    q[valid] = np.minimum(qv, 1)
    return q


def main() -> None:
    a = parse_args()
    agg = Path(a.aggregation)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(agg / "donor_celltype_metadata.csv")
    meta["passes_min_cells"] = meta.passes_min_cells.astype(bool)
    archive = np.load(agg / "donor_celltype_logcpm_pls_genes.npz", allow_pickle=True)
    logcpm = archive["logcpm"]
    genes = archive["genes"].astype(str)
    weights = pd.read_csv(agg / "matched_pls1_weights.csv").set_index("gene").loc[genes]
    w = weights.PLS1W_corr_Z.to_numpy(float)
    gene_set = weights.gene_set.fillna("").to_numpy(str)
    cag = pd.read_csv(a.cag)
    target_order = meta.target_cell_type.drop_duplicates().tolist()
    cag = cag.set_index("target_cell_type").loc[target_order].reset_index()
    y = cag.cag_mean_somatic_length_gain.to_numpy(float)
    broad = cag.broad_class.to_numpy(str)

    score_rows = []
    expr_lookup: dict[str, np.ndarray] = {}
    for donor_group in ["CTRL", "HD", "ALL"]:
        expr, targets = celltype_expression(meta, logcpm, donor_group)
        expr_lookup[donor_group] = expr
        for normalization in ["zscore", "rank"]:
            signed = weighted_score(expr, w, normalization)
            plus = weighted_score(expr[:, gene_set == "PLS1+"], np.ones(np.sum(gene_set == "PLS1+")), normalization)
            minus = weighted_score(expr[:, gene_set == "PLS1-"], np.ones(np.sum(gene_set == "PLS1-")), normalization)
            for i, target in enumerate(targets):
                score_rows.append({
                    "target_cell_type": target,
                    "donor_group": donor_group,
                    "expression_standardization": normalization,
                    "signed_pls1_expression_score": signed[i],
                    "pls1_positive_expression_score": plus[i],
                    "pls1_negative_expression_score": minus[i],
                })
    scores = pd.DataFrame(score_rows).merge(cag, on="target_cell_type", how="left", validate="many_to_one")
    scores.to_csv(out / "celltype_pls1_expression_scores.csv", index=False)

    rng = np.random.default_rng(a.seed)
    tests = []
    nulls = {}
    family_score_rows = []
    families = {
        "Six cortical excitatory populations": broad == "excitatory",
        "Ten neuronal populations": np.isin(broad, ["excitatory", "inhibitory"]),
        "All 14 populations, broad-class restricted permutation": np.ones(len(y), dtype=bool),
    }
    score_columns = {
        "Signed PLS1 expression score": "signed_pls1_expression_score",
        "PLS1-positive expression score": "pls1_positive_expression_score",
        "PLS1-negative expression score": "pls1_negative_expression_score",
    }
    for label, mask in families.items():
        family_expr = expr_lookup["CTRL"][mask, :]
        family_scores = {
            "Signed PLS1 expression score": weighted_score(family_expr, w, "zscore"),
            "PLS1-positive expression score": weighted_score(family_expr[:, gene_set == "PLS1+"], np.ones(np.sum(gene_set == "PLS1+")), "zscore"),
            "PLS1-negative expression score": weighted_score(family_expr[:, gene_set == "PLS1-"], np.ones(np.sum(gene_set == "PLS1-")), "zscore"),
        }
        for local_i, global_i in enumerate(np.flatnonzero(mask)):
            for score_label, values in family_scores.items():
                family_score_rows.append({"analysis": label, "target_cell_type": target_order[global_i],
                                          "score": score_label, "value": values[local_i],
                                          "donor_group": "CTRL", "expression_standardization": "zscore"})
        for score_label, x0 in family_scores.items():
            y0 = y[mask]
            rho = spearman(x0, y0)
            if mask.sum() <= 8:
                pval, nperm = exact_p(x0, y0)
                null = np.array([spearman(x0, y0[list(p)]) for p in itertools.permutations(range(mask.sum()))])
                method = "complete enumeration"
            else:
                blocks = broad[mask] if label.startswith("All 14") else None
                pval, null = monte_carlo_p(x0, y0, a.score_permutations, rng, blocks=blocks)
                nperm = a.score_permutations
                method = "broad-class-restricted Monte Carlo" if blocks is not None else "Monte Carlo"
            null_key = f"{label} | {score_label}"
            nulls[null_key] = null
            tests.append({"analysis": label, "score": score_label, "n_cell_types": int(mask.sum()), "spearman_rho": rho,
                          "permutation_p_two_sided": pval, "permutations": nperm, "permutation_method": method})
    tests = pd.DataFrame(tests)
    tests["q_across_nine_score_tests"] = bh(tests.permutation_p_two_sided.to_numpy())
    tests.to_csv(out / "score_cag_primary_tests.csv", index=False)
    pd.DataFrame(family_score_rows).to_csv(out / "family_specific_celltype_scores.csv", index=False)
    np.savez_compressed(out / "score_cag_null_distributions.npz",
                        **{k.replace(" ", "_").replace(",", ""): v for k, v in nulls.items()})

    sensitivity_rows = []
    masks = {
        "excitatory6": broad == "excitatory",
        "neuronal10": np.isin(broad, ["excitatory", "inhibitory"]),
        "all14": np.ones(len(y), dtype=bool),
    }
    for donor_group in ["CTRL", "HD", "ALL"]:
        for normalization in ["zscore", "rank"]:
            for family, mask in masks.items():
                family_expr = expr_lookup[donor_group][mask, :]
                family_score_values = {
                    "Signed PLS1 expression score": weighted_score(family_expr, w, normalization),
                    "PLS1-positive expression score": weighted_score(family_expr[:, gene_set == "PLS1+"], np.ones(np.sum(gene_set == "PLS1+")), normalization),
                    "PLS1-negative expression score": weighted_score(family_expr[:, gene_set == "PLS1-"], np.ones(np.sum(gene_set == "PLS1-")), normalization),
                }
                for score_label, score in family_score_values.items():
                    sensitivity_rows.append({"sensitivity": f"{donor_group}_{normalization}", "score": score_label, "family": family,
                                             "n_cell_types": int(mask.sum()), "spearman_rho": spearman(score, y[mask])})
    # Leave-one-population-out for the six-population primary family.
    exc = np.flatnonzero(broad == "excitatory")
    for omitted in exc:
        keep = exc[exc != omitted]
        loo_expr = expr_lookup["CTRL"][keep, :]
        loo_scores = {
            "Signed PLS1 expression score": weighted_score(loo_expr, w, "zscore"),
            "PLS1-positive expression score": weighted_score(
                loo_expr[:, gene_set == "PLS1+"], np.ones(np.sum(gene_set == "PLS1+")), "zscore"
            ),
            "PLS1-negative expression score": weighted_score(
                loo_expr[:, gene_set == "PLS1-"], np.ones(np.sum(gene_set == "PLS1-")), "zscore"
            ),
        }
        for score_label, loo_score in loo_scores.items():
            sensitivity_rows.append({"sensitivity": f"omit_{cag.target_cell_type.iloc[omitted]}",
                                     "score": score_label, "family": "excitatory6_LOO",
                                     "n_cell_types": len(keep),
                                     "spearman_rho": spearman(loo_score, y[keep])})
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(out / "score_cag_sensitivity.csv", index=False)

    # Gene-level strict sensitivity: the prespecified 504-gene PLS1+/PLS1-
    # family only, not all background genes used to form the weighted score.
    ctrl_expr, _ = celltype_expression(meta, logcpm, "CTRL")
    selected_genes = np.isin(gene_set, ["PLS1+", "PLS1-"])
    tested_genes = genes[selected_genes]
    tested_sets = gene_set[selected_genes]
    gene_rows = []
    for family, mask in [("neuronal10", masks["neuronal10"]), ("all14_broad_class_adjusted", masks["all14"])]:
        # Rank within the analysis family itself. Ranking all 14 populations and
        # then subsetting would not equal a Spearman correlation within neurons.
        xmat = rank_columns(ctrl_expr[mask, :][:, selected_genes])
        yy = stats.rankdata(y[mask])
        obs = corr_columns(xmat, yy)
        exceed = np.zeros(len(tested_genes), dtype=np.int64)
        blocks = broad[mask]
        # Use a common permutation family across genes.
        x_center = xmat - xmat.mean(axis=0, keepdims=True)
        x_norm = np.sqrt(np.sum(x_center * x_center, axis=0))
        valid_gene = np.isfinite(x_norm) & (x_norm > 1e-12)
        y_rank = stats.rankdata(y[mask])
        y_center = y_rank - y_rank.mean()
        y_norm = np.sqrt(np.sum(y_center * y_center))
        batch = 10_000
        for start in range(0, a.gene_permutations, batch):
            stop = min(start + batch, a.gene_permutations)
            block_arg = None if family == "neuronal10" else blocks
            perm_idx = permutation_index_matrix(mask.sum(), stop - start, rng, block_arg)
            rp = np.full((stop - start, len(tested_genes)), np.nan)
            rp[:, valid_gene] = (y_center[perm_idx] @ x_center[:, valid_gene]) / (y_norm * x_norm[valid_gene])
            exceed[valid_gene] += np.sum(np.abs(rp[:, valid_gene]) >= np.abs(obs[valid_gene])[None, :] - 1e-12, axis=0)
        pvals = np.full(len(tested_genes), np.nan)
        finite = np.isfinite(obs) & valid_gene
        pvals[finite] = (exceed[finite] + 1) / (a.gene_permutations + 1)
        qvals = bh(pvals)
        for i, gene in enumerate(tested_genes):
            gene_rows.append({"gene": gene, "gene_set": tested_sets[i], "family": family,
                              "spearman_rho": obs[i], "permutation_p_two_sided": pvals[i],
                              "global_bh_q": qvals[i], "permutations": a.gene_permutations})
    gene_results = pd.DataFrame(gene_rows)
    gene_results.to_csv(out / "gene_level_cag_permutation_globalBH.csv.gz", index=False, compression="gzip")

    summary = {
        "score_name": "MSN-associated signed PLS1 gene-expression score",
        "axis_name": "Signed PLS1 expression score",
        "primary_expression_summary": "equal-weight mean of eligible control-donor log2 CPM profiles",
        "primary_cell_types": "six motor-cortex excitatory populations",
        "primary_statistical_unit": "cell population",
        "cag_source": "Pressl et al. Neuron 2024 Table 1, motor cortex (BA4)",
        "matched_pls_genes": int(len(genes)),
        "matched_pls1_positive": int(np.sum(gene_set == "PLS1+")),
        "matched_pls1_negative": int(np.sum(gene_set == "PLS1-")),
        "gene_level_tests": int(len(tested_genes)),
        "gene_level_finite_tests_neuronal10": int(gene_results.query("family == 'neuronal10'").permutation_p_two_sided.notna().sum()),
        "gene_level_finite_tests_class_adjusted14": int(gene_results.query("family == 'all14_broad_class_adjusted'").permutation_p_two_sided.notna().sum()),
        "gene_level_global_bh_hits_neuronal10": int((gene_results.query("family == 'neuronal10'").global_bh_q < 0.05).sum()),
        "gene_level_global_bh_hits_class_adjusted14": int((gene_results.query("family == 'all14_broad_class_adjusted'").global_bh_q < 0.05).sum()),
        "score_permutations": a.score_permutations,
        "gene_permutations": a.gene_permutations,
        "seed": a.seed,
    }
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(tests.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Quantify cell-type aggregate associations between GSE233387 transcriptional scores and somatic CAG expansion.
# Input source/location: GSE233387 BA4 aggregate expression/metadata, somatic CAG summaries, and reference gene-set definitions.
# Output location: Cell-type score/CAG association tables, permutation results, and related visualization data in the configured BA4 CAG results directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Harmonize aggregate units; compute transcriptional scores; merge CAG summaries; estimate association statistics; run null tests; export results.
# Log location: No dedicated log file; progress is written to standard output.
# =============================================================================
