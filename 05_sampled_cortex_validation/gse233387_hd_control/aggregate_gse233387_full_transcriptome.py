#!/usr/bin/env python3
"""Aggregate GSE233387 raw nuclear counts by donor and the 14 mapped BA4 cell types."""

from __future__ import annotations

import argparse
import gzip
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import mmwrite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    mapping = pd.read_csv(args.mapping)
    if mapping["GSE233387_subtype"].duplicated().any():
        raise RuntimeError("Each released subtype must map to one target cell type")
    subtype_to_target = dict(zip(mapping["GSE233387_subtype"], mapping["target_cell_type"]))

    manifest = pd.read_csv(args.manifest, encoding="utf-8-sig")
    manifest["sample_key"] = manifest["gsm"].astype(str) + "_ID_" + manifest["sample_id"].astype(str)
    sample_to_donor = dict(zip(manifest["sample_key"], manifest["donor"]))
    sample_to_condition = dict(zip(manifest["sample_key"], manifest["condition"]))

    a = ad.read_h5ad(args.h5ad)
    obs = a.obs.copy()
    obs["target_cell_type"] = obs["subtype"].astype(str).map(subtype_to_target)
    # The released atlas includes vascular, immune, and neuronal subtypes outside
    # the 14 prespecified BA4 categories. Retain only nuclei with an explicit map.
    released_nuclei = int(a.n_obs)
    mapped = obs["target_cell_type"].notna().to_numpy()
    a = a[mapped, :].copy()
    obs = obs.loc[mapped].copy()
    obs["donor_id"] = obs["sample"].astype(str).map(sample_to_donor)
    obs["condition"] = obs["sample"].astype(str).map(sample_to_condition)
    if obs[["donor_id", "condition"]].isna().any().any():
        missing = sorted(obs.loc[obs["donor_id"].isna(), "sample"].astype(str).unique())
        raise RuntimeError(f"Samples absent from manifest: {missing}")

    condition_map = {"CTRL": "Control", "HD": "HD"}
    obs["condition"] = obs["condition"].map(condition_map)
    if obs["condition"].isna().any():
        raise RuntimeError("Unexpected condition in manifest")

    cell_order = [
        "Layer 2", "Layer 4", "Betz cells", "Layer 5a", "Layer 6a", "Layer 6b",
        "VIP", "RELN", "LAMP5", "PVALB", "Astrocytes", "Microglia",
        "Oligodendrocytes", "OPCs",
    ]
    donor_order = ["BEB18135", "HCTSU", "HSB2116", "HSB2341", "HSB2385"]
    obs["group_key"] = obs["donor_id"].astype(str) + "||" + obs["target_cell_type"].astype(str)
    groups = [f"{donor}||{cell}" for cell in cell_order for donor in donor_order]
    group_index = {key: i for i, key in enumerate(groups)}
    group_codes = obs["group_key"].map(group_index)
    if group_codes.isna().any():
        raise RuntimeError("Unexpected donor/cell-type group")

    membership = sparse.csr_matrix(
        (np.ones(a.n_obs, dtype=np.int64), (group_codes.to_numpy(), np.arange(a.n_obs))),
        shape=(len(groups), a.n_obs),
    )
    raw = a.layers["counts"]
    if not sparse.issparse(raw):
        raw = sparse.csr_matrix(raw)
    raw = raw.tocsr()
    if raw.min() < 0 or np.any(raw.data != np.floor(raw.data)):
        raise RuntimeError("Raw counts are not non-negative integers")
    grouped = (membership @ raw).tocsr()

    gene_symbols = a.var["gene_name"].astype(str).str.strip()
    keep = gene_symbols.notna() & gene_symbols.ne("") & gene_symbols.ne("nan")
    grouped = grouped[:, keep.to_numpy()]
    symbols = gene_symbols[keep].to_numpy()
    unique_symbols, inverse = np.unique(symbols, return_inverse=True)
    collapse = sparse.csr_matrix(
        (np.ones(len(symbols), dtype=np.int64), (np.arange(len(symbols)), inverse)),
        shape=(len(symbols), len(unique_symbols)),
    )
    grouped = (grouped @ collapse).tocsr()

    metadata_rows = []
    counts_per_group = obs.groupby("group_key", observed=True).size().to_dict()
    for key in groups:
        donor, cell = key.split("||", 1)
        cond = obs.loc[obs["donor_id"].eq(donor), "condition"].iloc[0]
        metadata_rows.append({
            "sample_id": key,
            "donor_id": donor,
            "condition": cond,
            "cell_type": cell,
            "n_nuclei": int(counts_per_group.get(key, 0)),
            "library_size": int(grouped[group_index[key], :].sum()),
        })
    metadata = pd.DataFrame(metadata_rows)
    if (metadata["n_nuclei"] == 0).any():
        raise RuntimeError("At least one donor by cell-type pseudobulk is empty")

    # R expects genes by samples.
    with gzip.open(out / "GSE233387_counts_genes_by_samples.mtx.gz", "wb") as handle:
        mmwrite(handle, grouped.T)
    with gzip.open(out / "GSE233387_genes.tsv.gz", "wt", encoding="utf-8", newline="") as handle:
        for gene in unique_symbols:
            handle.write(f"{gene}\n")
    metadata.to_csv(out / "GSE233387_metadata.csv", index=False)

    qc = pd.DataFrame({
        "metric": [
            "released_nuclei", "mapped_nuclei", "input_features", "unique_gene_symbols", "pseudobulks",
            "input_count_sum", "aggregated_count_sum", "minimum_nuclei_per_pseudobulk",
        ],
        "value": [
            released_nuclei, a.n_obs, a.n_vars, len(unique_symbols), len(groups), int(raw.sum()),
            int(grouped.sum()), int(metadata["n_nuclei"].min()),
        ],
    })
    qc.to_csv(out / "aggregation_qc.csv", index=False)
    if int(raw.sum()) != int(grouped.sum()):
        raise RuntimeError("Count conservation failed")


if __name__ == "__main__":
    main()

# Function: aggregate raw GSE233387 counts into donor-by-cell-type pseudobulks.
# Inputs: annotated H5AD, released-subtype mapping, and GSE233387 sample manifest.
# Outputs: compressed Matrix Market counts, genes, metadata, and count-conservation QC.
# Main steps: map released annotations, sum nuclei, collapse duplicate gene symbols, verify counts.
# Logs: standard output/error should be redirected by the calling shell when required.

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Aggregate GSE233387 nucleus-level counts into donor/cell-type full-transcriptome pseudobulk matrices for HD-control comparisons.
# Input source/location: Released GSE233387 count matrix and cell/nucleus annotations from the configured dataset directory.
# Output location: Pseudobulk count matrices, sample annotations, and aggregation QC tables in the local analysis output directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Map released annotations; group nuclei by donor and cell type; sum counts; collapse duplicate gene symbols; verify totals/dimensions; export pseudobulk data.
# Log location: No dedicated log file; progress and QC summaries are written to standard output.
# =============================================================================
