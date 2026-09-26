#!/usr/bin/env python3
"""Direct four-population BA4 alignment for GSE233408 and CAG summaries."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--aggregation", required=True)
    p.add_argument("--cag", required=True)
    p.add_argument("--disease-scores", required=True)
    p.add_argument("--camera", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def score(expr: np.ndarray, weights: np.ndarray) -> np.ndarray:
    center = expr.mean(axis=0)
    spread = expr.std(axis=0)
    z = np.zeros_like(expr)
    valid = spread > 1e-12
    z[:, valid] = (expr[:, valid] - center[valid]) / spread[valid]
    return z @ weights / np.sum(np.abs(weights))


def exact_spearman(x: np.ndarray, y: np.ndarray):
    obs = float(stats.spearmanr(x, y).statistic)
    null = np.array([stats.spearmanr(x, y[list(p)]).statistic for p in itertools.permutations(range(len(y)))])
    p = float(np.mean(np.abs(null) >= abs(obs) - 1e-12))
    return obs, p, null


def main():
    a = parse_args()
    agg = Path(a.aggregation)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(agg / "donor_celltype_metadata.csv")
    arc = np.load(agg / "donor_celltype_logcpm_pls_genes.npz", allow_pickle=True)
    expr = arc["logcpm"]; genes = arc["genes"].astype(str)
    weights = pd.read_csv(agg / "matched_pls1_weights.csv").set_index("gene").loc[genes]
    w = weights.PLS1W_corr_Z.to_numpy(float)
    target_order = ["Betz cells", "Layer 5a", "Layer 6a", "Layer 6b"]
    baseline = []
    for target in target_order:
        idx = np.flatnonzero((meta.condition.eq("CTRL") & meta.target_cell_type.eq(target)).to_numpy())
        baseline.append(expr[idx].mean(axis=0))
    baseline = np.vstack(baseline)
    baseline_score = score(baseline, w)
    plus_score = score(baseline[:, weights.gene_set.eq("PLS1+").to_numpy()],
                       np.ones(weights.gene_set.eq("PLS1+").sum()))
    minus_score = score(baseline[:, weights.gene_set.eq("PLS1-").to_numpy()],
                        np.ones(weights.gene_set.eq("PLS1-").sum()))
    cag = pd.read_csv(a.cag).set_index("target_cell_type").loc[target_order].reset_index()

    table = cag.copy()
    table["signed_pls1_expression_score"] = baseline_score
    table["pls1_positive_expression_score"] = plus_score
    table["pls1_negative_expression_score"] = minus_score

    disease = pd.read_csv(a.disease_scores)
    disease = disease[disease.region.eq("BA4")].copy()
    disease["target_cell_type"] = disease.cell_type.map({
        "Betz cell": "Betz cells", "L5a pyramidal neuron": "Layer 5a",
        "L6a pyramidal neuron": "Layer 6a", "L6b pyramidal neuron": "Layer 6b"})
    disease_summary = disease.groupby(["target_cell_type", "condition"]).signed_pls1_score.mean().unstack()
    disease_summary["hd_minus_control_signed_score"] = disease_summary["HD"] - disease_summary["CTRL"]
    table = table.merge(disease_summary[["hd_minus_control_signed_score"]], left_on="target_cell_type", right_index=True)

    camera = pd.read_csv(a.camera)
    camera = camera[(camera.region.eq("BA4")) & camera.gene_set.eq("PLS1_positive")].copy()
    camera["target_cell_type"] = camera.cell_type.map({
        "Betz cell": "Betz cells", "L5a pyramidal neuron": "Layer 5a",
        "L6a pyramidal neuron": "Layer 6a", "L6b pyramidal neuron": "Layer 6b"})
    camera["directional_z"] = stats.norm.isf(camera.one_sided_p)  # positive supports expected downregulation
    table = table.merge(camera[["target_cell_type", "directional_z", "global_q_all_eligible_region_celltype_sets",
                                "control_donors", "hd_donors"]], on="target_cell_type", validate="one_to_one")
    table.to_csv(out / "gse233408_ba4_four_population_alignment.csv", index=False)

    tests = []
    nulls = {}
    for label, col in {
        "Signed PLS1 expression score": "signed_pls1_expression_score",
        "PLS1-positive expression score": "pls1_positive_expression_score",
        "PLS1-negative expression score": "pls1_negative_expression_score",
        "HD-control signed-score difference": "hd_minus_control_signed_score",
        "PLS1-positive directional camera z": "directional_z",
    }.items():
        rho, p, null = exact_spearman(table[col].to_numpy(float), table.cag_mean_somatic_length_gain.to_numpy(float))
        tests.append({"measure": label, "n_cell_types": 4, "spearman_rho": rho,
                      "complete_permutation_p_two_sided": p, "permutations": 24,
                      "interpretation": "descriptive due to four cell populations"})
        nulls[label.replace(" ", "_")] = null
    pd.DataFrame(tests).to_csv(out / "gse233408_ba4_four_population_tests.csv", index=False)
    np.savez_compressed(out / "gse233408_ba4_four_population_nulls.npz", **nulls)
    summary = {"source": "GSE233408", "region": "BA4", "direct_cell_population_match": True,
               "n_cell_types": 4, "inference_role": "descriptive auxiliary alignment; not a powered spatial correlation"}
    (out / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.DataFrame(tests).to_string(index=False))


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Evaluate the reference transcriptional signature across GSE233408 BA4 cell subtypes/populations.
# Input source/location: GSE233408 BA4 aggregate data and reference PLS gene sets/weights from the configured project inputs.
# Output location: Subtype-level effect/enrichment tables and validation summaries in the configured BA4 results directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Load aggregates; align genes/subtypes; calculate signature scores/effects; run statistical tests and multiple-testing correction; export results.
# Log location: No dedicated log file; progress is written to standard output.
# =============================================================================
