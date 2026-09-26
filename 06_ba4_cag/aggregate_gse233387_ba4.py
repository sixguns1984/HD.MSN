#!/usr/bin/env python3
"""Aggregate released GSE233387 BA4 nuclei into donor-by-Pressl-population profiles.

The mapping is mutually exclusive: each released transcriptomic subtype can enter
at most one of the 14 Pressl Table 1 populations. Raw counts are summed within
donor and target population. No additional cell filtering or re-annotation is
performed; the released, annotated object is used as supplied by the authors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--mapping", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min-cells", type=int, default=20)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(a.manifest)
    mapping = pd.read_csv(a.mapping)
    weights = pd.read_csv(a.weights)
    if mapping["GSE233387_subtype"].duplicated().any():
        dup = mapping.loc[mapping["GSE233387_subtype"].duplicated(False), "GSE233387_subtype"].tolist()
        raise ValueError(f"Non-exclusive subtype mapping: {dup}")
    subtype_to_target = dict(zip(mapping.GSE233387_subtype, mapping.target_cell_type))

    print("Reading released H5AD into memory", flush=True)
    obj = ad.read_h5ad(a.h5ad)
    if "counts" not in obj.layers:
        raise ValueError("Released object has no counts layer")
    if not sparse.issparse(obj.layers["counts"]):
        raise ValueError("Expected sparse raw-count layer")

    sample_to_donor = dict(zip(manifest["gsm"], manifest["donor"]))
    sample_to_condition = dict(zip(manifest["gsm"], manifest["condition"]))
    gsm = obj.obs["sample"].astype(str).str.extract(r"(GSM\d+)", expand=False)
    obj.obs["gsm"] = gsm.to_numpy()
    obj.obs["donor"] = obj.obs["gsm"].map(sample_to_donor).to_numpy()
    obj.obs["condition_manifest"] = obj.obs["gsm"].map(sample_to_condition).to_numpy()
    if obj.obs[["donor", "condition_manifest"]].isna().any().any():
        raise ValueError("One or more released samples did not map to the GEO manifest")
    case_norm = obj.obs["case"].astype(str).str.lower().map({"case": "HD", "control": "CTRL"})
    if not np.array_equal(case_norm.to_numpy(), obj.obs["condition_manifest"].to_numpy()):
        raise ValueError("Condition mismatch between H5AD and GEO manifest")

    obj.obs["target_cell_type"] = obj.obs["subtype"].astype(str).map(subtype_to_target).to_numpy()
    selected = obj.obs["target_cell_type"].notna().to_numpy()
    observed_targets = set(obj.obs.loc[selected, "target_cell_type"])
    expected_targets = set(mapping.target_cell_type)
    if observed_targets != expected_targets:
        raise ValueError(f"Target coverage mismatch: observed={observed_targets}, expected={expected_targets}")

    weight_genes = weights["gene"].astype(str).tolist()
    var_lookup = {str(g).upper(): i for i, g in enumerate(obj.var_names)}
    matched_weights = weights.loc[[g.upper() in var_lookup for g in weight_genes]].copy()
    matched_indices = np.array([var_lookup[g.upper()] for g in matched_weights.gene], dtype=int)
    matched_weights["h5ad_var_name"] = obj.var_names[matched_indices].astype(str)
    matched_weights.to_csv(out / "matched_pls1_weights.csv", index=False)

    counts = obj.layers["counts"][:, matched_indices].tocsr()
    donors = manifest[["donor", "condition"]].drop_duplicates().sort_values(["condition", "donor"])
    targets = mapping[["target_cell_type", "broad_class", "analysis_set"]].drop_duplicates()
    target_order = targets.target_cell_type.tolist()

    rows = []
    matrices = []
    obs = obj.obs
    for donor, condition in donors.itertuples(index=False):
        for target in target_order:
            mask = ((obs["donor"] == donor) & (obs["target_cell_type"] == target)).to_numpy()
            n_cells = int(mask.sum())
            if n_cells:
                summed = np.asarray(counts[mask].sum(axis=0)).ravel().astype(np.int64)
            else:
                summed = np.zeros(counts.shape[1], dtype=np.int64)
            rows.append({
                "donor": donor,
                "condition": condition,
                "target_cell_type": target,
                "n_cells": n_cells,
                "passes_min_cells": n_cells >= a.min_cells,
                "library_size": int(summed.sum()),
            })
            matrices.append(summed)

    meta = pd.DataFrame(rows).merge(targets, on="target_cell_type", how="left", validate="many_to_one")
    matrix = np.vstack(matrices)
    meta.to_csv(out / "donor_celltype_metadata.csv", index=False)
    np.savez_compressed(out / "donor_celltype_counts_pls_genes.npz", counts=matrix,
                        genes=matched_weights.gene.astype(str).to_numpy())

    lib = matrix.sum(axis=1).astype(float)
    logcpm = np.log2(matrix / np.maximum(lib[:, None], 1.0) * 1e6 + 0.5)
    logcpm[~meta.passes_min_cells.to_numpy(), :] = np.nan
    np.savez_compressed(out / "donor_celltype_logcpm_pls_genes.npz", logcpm=logcpm,
                        genes=matched_weights.gene.astype(str).to_numpy())

    audit = {
        "source": "GSE233387 released annotated BA4 H5AD",
        "h5ad_shape": [int(obj.n_obs), int(obj.n_vars)],
        "released_nuclei": int(obj.n_obs),
        "mapped_nuclei": int(selected.sum()),
        "donors": int(donors.donor.nunique()),
        "control_donors": int(donors.query("condition == 'CTRL'").donor.nunique()),
        "hd_donors": int(donors.query("condition == 'HD'").donor.nunique()),
        "target_populations": int(len(target_order)),
        "mapping_exclusive": True,
        "min_cells_per_donor_population": int(a.min_cells),
        "pls_genes_requested": int(len(weights)),
        "pls_genes_matched": int(len(matched_weights)),
        "no_additional_cell_filtering": True,
    }
    (out / "aggregation_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Aggregate GSE233387 BA4 data to the predefined cell-type units used for somatic CAG/transcriptional alignment analyses.
# Input source/location: GSE233387 BA4 expression/count data and released cell/donor annotations from configurable project inputs.
# Output location: BA4 aggregate expression/metadata tables and QC summaries in the configured BA4 results directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Filter BA4 observations; map cells to analysis cell types; aggregate expression/metadata; validate unit counts; export analysis-ready tables.
# Log location: No dedicated log file; progress is written to standard output.
# =============================================================================
