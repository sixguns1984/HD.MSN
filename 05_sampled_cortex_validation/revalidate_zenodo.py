#!/usr/bin/env python3
"""Read-only statistical revalidation and style-aligned visualization for the Zenodo atlas."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm


PROJECT_ROOT = Path(os.environ.get("HD_MSN_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
ATLAS = Path(os.environ.get("HD_MSN_ATLAS_ROOT", PROJECT_ROOT / "data" / "atlas_validation"))
AHBA_RANK = Path(os.environ.get("HD_MSN_AHBA_RANK", PROJECT_ROOT / "results" / "ahba_pls" / "PLS1_gene_rank_HD_vs_HC_tvalues_with_names.csv"))


def bh(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=float)
    order = np.argsort(array)
    ranked = array[order]
    adjusted = np.minimum.accumulate((ranked * len(array) / np.arange(1, len(array) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return pd.Series(output, index=values.index)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def compare_frozen_weights() -> dict:
    frozen_path = ATLAS / "inputs" / "pls" / "reference_pls1_gene_weights.csv"
    frozen = pd.read_csv(frozen_path).set_index("gene")
    source = pd.read_csv(AHBA_RANK).set_index("gene")
    common = frozen.index.intersection(source.index)
    fields = ["PLS1_weight", "PLS1W_BSE", "PLS1W_corr_Z", "p_raw", "p_Bonferroni", "significant_Bonferroni"]
    checks = {}
    for field in fields:
        left = frozen.loc[common, field]
        right = source.loc[common, field]
        if field == "significant_Bonferroni":
            equal = bool((left.astype(bool).to_numpy() == right.astype(bool).to_numpy()).all())
            maximum = 0.0 if equal else 1.0
        else:
            diff = np.abs(left.astype(float).to_numpy() - right.astype(float).to_numpy())
            maximum = float(np.nanmax(diff))
            equal = bool(np.allclose(left.astype(float), right.astype(float), rtol=0, atol=1e-12, equal_nan=True))
        checks[field] = {"equal_at_1e-12": equal, "max_abs_diff": maximum}
    return {
        "frozen_sha256": sha256(frozen_path),
        "ahba_rank_sha256": sha256(AHBA_RANK),
        "n_frozen_genes": int(len(frozen)),
        "n_source_genes": int(len(source)),
        "n_common_genes": int(len(common)),
        "fields": checks,
        "all_fields_equal": all(item["equal_at_1e-12"] for item in checks.values()),
    }


def add_panel_label(ax, label: str) -> None:
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="bottom")


def make_figure(camera: pd.DataFrame, out: Path) -> None:
    scores = pd.read_csv(ATLAS / "03_pls_signature_tests/donor_signed_pls1_scores.csv")
    interactions = pd.read_csv(ATLAS / "04_celltype_specificity/mixedlm_interactions.csv")
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 14, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "axes.linewidth": 0.8,
    })
    sns.set_style("white")
    fig = plt.figure(figsize=(8.2, 7.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.78, 1.25], hspace=0.48, wspace=0.42)
    ax_a = fig.add_subplot(grid[0, 0]); ax_c = fig.add_subplot(grid[0, 1]); ax_b = fig.add_subplot(grid[1, :])

    frame = scores[scores["cell_type"] == "Ex_neuron"].copy()
    order = [("frontal", "Control"), ("frontal", "HD"), ("cingulate", "Control"), ("cingulate", "HD")]
    positions = {key: index for index, key in enumerate(order)}
    colors = {"Control": "#0072B2", "HD": "#D55E00"}
    rng = np.random.default_rng(20260803)
    for key, group in frame.groupby(["region", "condition"], observed=True):
        position = positions[key]
        jitter = rng.uniform(-0.09, 0.09, len(group))
        ax_a.scatter(position + jitter, group["signed_pls1_score"], s=20, color=colors[key[1]], alpha=0.88)
        mean = group["signed_pls1_score"].mean(); sem = group["signed_pls1_score"].std(ddof=1) / np.sqrt(len(group))
        ax_a.errorbar(position, mean, yerr=1.96 * sem, fmt="_", markersize=13, color="black", capsize=3, linewidth=1)
    ax_a.axhline(0, color="#777777", linewidth=0.7, linestyle="--")
    ax_a.set_xticks(range(4), ["Control", "HD", "Control", "HD"])
    ax_a.text(0.5, -0.20, "Frontal", ha="center", transform=ax_a.get_xaxis_transform())
    ax_a.text(2.5, -0.20, "Cingulate", ha="center", transform=ax_a.get_xaxis_transform())
    ax_a.set_ylabel("Donor signed PLS1 score"); ax_a.set_title("Excitatory-neuron pseudobulk")

    interactions = interactions.sort_values("comparison_cell_type").reset_index(drop=True)
    y = np.arange(len(interactions))[::-1]
    ax_c.errorbar(interactions.beta, y,
                  xerr=[interactions.beta - interactions.ci_low, interactions.ci_high - interactions.beta],
                  fmt="o", markersize=4, color="#7A3E9D", capsize=2, linewidth=1)
    ax_c.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax_c.set_yticks(y, interactions["comparison_cell_type"])
    ax_c.set_xlabel("Difference in standardized HD effect")
    ax_c.set_title("Condition × cell-type interactions")

    heat = camera[camera["gene_set"] == "PLS1_positive"].copy()
    heat["directional_z"] = norm.isf(heat["one_sided_p"].clip(1e-300, 1 - 1e-16))
    cell_order = ["Ex_neuron", "In_neuron", "SST/NPY+ IN", "VIP+ IN", "Astrocyte", "Microglia", "OLG", "NFO", "OPC"]
    present = [item for item in cell_order if item in set(heat.cell_type)]
    values = heat.pivot(index="cell_type", columns="region", values="directional_z").reindex(present).reindex(columns=["frontal", "cingulate"])
    qvalues = heat.pivot(index="cell_type", columns="region", values="global_q_all_directional_tests").reindex(present).reindex(columns=["frontal", "cingulate"])
    annotations = qvalues.map(lambda value: "" if pd.isna(value) else f"q={value:.2g}")
    display = {"Ex_neuron": "Excitatory neuron", "In_neuron": "Inhibitory neuron", "SST/NPY+ IN": "SST/NPY+ interneuron", "VIP+ IN": "VIP+ interneuron"}
    values.index = [display.get(value, value) for value in values.index]; annotations.index = values.index
    sns.heatmap(values, cmap="RdBu", center=0, vmin=-4, vmax=6, linewidths=0.4,
                linecolor="white", annot=annotations, fmt="", annot_kws={"fontsize": 8},
                cbar_kws={"label": "Directional z (PLS1+ expected down)"}, ax=ax_b)
    ax_b.set_xlabel(""); ax_b.set_ylabel(""); ax_b.set_xticklabels(["Frontal", "Cingulate"], rotation=0)
    ax_b.set_title("Competitive PLS1+ gene-set tests (global BH family)")
    for axis, label in [(ax_a, "A"), (ax_b, "B"), (ax_c, "C")]: add_panel_label(axis, label)
    sns.despine(ax=ax_a); sns.despine(ax=ax_c)
    fig.savefig(out / "zenodo_globalBH_style_aligned.png", dpi=600, bbox_inches="tight")
    fig.savefig(out / "zenodo_globalBH_style_aligned.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    camera = pd.read_csv(ATLAS / "03_pls_signature_tests/directional_gene_sets/camera_directional_results.csv")
    camera["global_q_all_directional_tests"] = bh(camera["one_sided_p"])
    camera.to_csv(args.out / "camera_directional_results_globalBH.csv", index=False)
    primary = camera[(camera.region.isin(["frontal", "cingulate"])) & (camera.cell_type == "Ex_neuron")]
    combined = {}
    for gene_set, group in primary.groupby("gene_set"):
        z = norm.isf(group["one_sided_p"].to_numpy(dtype=float))
        combined[gene_set] = float(norm.sf(z.sum() / np.sqrt(len(z))))
    weights = compare_frozen_weights()
    significant = camera[camera["global_q_all_directional_tests"] < 0.05][
        ["region", "cell_type", "gene_set", "Direction", "one_sided_p", "global_q_all_directional_tests"]
    ]
    significant.to_csv(args.out / "camera_globalBH_significant.csv", index=False)
    acceptance = {
        "status": "PASS" if weights["all_fields_equal"] and len(primary) == 4 else "FAIL",
        "global_bh_family_size": int(len(camera)),
        "global_bh_significant_n": int((camera["global_q_all_directional_tests"] < 0.05).sum()),
        "excitatory_cross_region_stouffer_one_sided": combined,
        "frozen_weight_audit": weights,
        "primary_rows": primary[["region", "gene_set", "one_sided_p", "one_sided_q", "global_q_all_directional_tests"]].to_dict("records"),
    }
    (args.out / "zenodo_revalidation_acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")
    make_figure(camera, args.out)
    print(json.dumps(acceptance, indent=2))


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Revalidate an external Zenodo-derived cortical dataset against the reference transcriptional signature using project-neutral paths.
# Input source/location: External validation effect/enrichment files, atlas mapping, and reference AHBA/PLS gene-rank resources from configurable paths.
# Output location: Revalidation statistics, global-BH-aligned tables, and supporting plots in the configured validation results directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Load external results; harmonize region/gene identifiers; reconstruct directional tests; apply global multiple-testing alignment; export reproducibility checks.
# Log location: No dedicated log file; progress is written to standard output.
# =============================================================================
