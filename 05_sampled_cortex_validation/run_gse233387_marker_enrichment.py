#!/usr/bin/env python3
"""PLS1 enrichment in GSE233387 reference marker sets (annotation evidence only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact, hypergeom


def bh(values: pd.Series) -> pd.Series:
    array = values.to_numpy(float)
    order = np.argsort(array)
    ranked = array[order]
    adjusted = np.minimum.accumulate((ranked * len(array) / np.arange(1, len(array) + 1))[::-1])[::-1]
    output = np.empty_like(adjusted); output[order] = np.minimum(adjusted, 1)
    return pd.Series(output, index=values.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markers", type=Path, required=True)
    parser.add_argument("--background", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--geo-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)

    markers = pd.read_csv(args.markers)
    background = {line.strip() for line in args.background.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()}
    weights = pd.read_csv(args.weights)
    geo = pd.read_csv(args.geo_manifest, encoding="utf-8-sig")
    gene_sets = {
        "PLS1_positive": set(weights.loc[weights.gene_set.isin(["PLS1+", "PLS1_positive"]), "gene"]) & background,
        "PLS1_negative": set(weights.loc[weights.gene_set.isin(["PLS1-", "PLS1_negative"]), "gene"]) & background,
    }
    if not all(gene_sets.values()):
        raise RuntimeError("Frozen PLS1 gene sets are empty after background intersection")
    rows = []
    for cell_type, frame in markers.groupby("celltype"):
        marker_set = set(frame.gene.dropna().astype(str)) & background
        for name, genes in gene_sets.items():
            overlap = marker_set & genes
            universe_n = len(background); set_n = len(genes); marker_n = len(marker_set); observed = len(overlap)
            p = hypergeom.sf(observed - 1, universe_n, set_n, marker_n)
            a = observed; b = marker_n - observed; c = set_n - observed; d = universe_n - a - b - c
            odds_ratio = fisher_exact([[a, b], [c, d]], alternative="greater").statistic
            rows.append({
                "cell_type": cell_type, "gene_set": name, "background_n": universe_n,
                "gene_set_n": set_n, "marker_n": marker_n, "overlap_n": observed,
                "odds_ratio": odds_ratio, "p_value": p, "overlap_genes": ";".join(sorted(overlap)),
            })
    result = pd.DataFrame(rows)
    result["global_q_28_tests"] = bh(result.p_value)
    result.to_csv(args.out / "gse233387_pls1_marker_enrichment.csv", index=False)
    result[result.global_q_28_tests < 0.05].to_csv(args.out / "gse233387_pls1_marker_enrichment_significant.csv", index=False)

    plot = result.copy()
    plot["minus_log10_q"] = -np.log10(plot.global_q_28_tests.clip(lower=1e-300))
    values = plot.pivot(index="cell_type", columns="gene_set", values="minus_log10_q")
    annotations = plot.pivot(index="cell_type", columns="gene_set", values="global_q_28_tests").map(lambda q: f"{q:.2g}")
    sns.set_style("white")
    plt.rcParams.update({"font.family": "Arial", "font.size": 14})
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    sns.heatmap(values, cmap="Blues", linewidths=0.4, linecolor="white", annot=annotations, fmt="",
                annot_kws={"fontsize": 7}, cbar_kws={"label": "−log10(global BH q)"}, ax=ax)
    ax.set_xlabel(""); ax.set_ylabel("")
    fig.savefig(args.out / "GSE233387_marker_enrichment_globalBH.png", dpi=600, bbox_inches="tight")
    fig.savefig(args.out / "GSE233387_marker_enrichment_globalBH.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)

    donors = geo[["donor", "condition"]].drop_duplicates()
    acceptance = {
        "status": "PASS" if len(result) == 28 and geo.donor.nunique() == 5 else "FAIL",
        "source_samples": int(len(geo)), "unique_donors": int(geo.donor.nunique()),
        "donors_by_condition": donors.groupby("condition").size().astype(int).to_dict(),
        "region": sorted(geo.region.unique().tolist()), "tests": int(len(result)),
        "global_significant": int((result.global_q_28_tests < 0.05).sum()),
        "interpretation_guard": "Reference marker enrichment only; the 18 sorted samples are not treated as independent disease replicates.",
        "significant_results": result.loc[result.global_q_28_tests < 0.05,
            ["cell_type", "gene_set", "overlap_n", "odds_ratio", "p_value", "global_q_28_tests"]].to_dict("records"),
    }
    (args.out / "gse233387_acceptance.json").write_text(json.dumps(acceptance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Test reference PLS gene sets for enrichment among GSE233387 cell-type marker genes.
# Input source/location: GSE233387 marker/statistical tables and reference PLS-ranked gene lists from configured project inputs.
# Output location: Cell-type marker-enrichment statistics and multiple-testing-adjusted result tables in the configured output directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Load markers and reference gene sets; harmonize symbols; compute enrichment/directional statistics; correct for multiple testing; export results.
# Log location: No dedicated log file; progress is written to standard output.
# =============================================================================
