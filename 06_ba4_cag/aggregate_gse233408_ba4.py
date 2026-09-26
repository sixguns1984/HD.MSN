#!/usr/bin/env python3
"""Build BA4 donor-by-sFANS-population profiles from GSE233408 counts.

GSE233408 contains four BA4 populations that directly match Pressl Table 1:
Betz cells, L5a, L6a, and L6b. Technical libraries from the same donor and
population are summed before normalization. No raw counts are pooled across
cell populations or regions.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


NAME_MAP = {
    "Betz cell": "Betz cells",
    "L5a pyramidal neuron": "Layer 5a",
    "L6a pyramidal neuron": "Layer 6a",
    "L6b pyramidal neuron": "Layer 6b",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--counts", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    return p.parse_args()


def read_count(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt") as handle:
        frame = pd.read_csv(handle, sep="\t")
    if frame.shape[1] != 3:
        raise ValueError(f"Unexpected count format: {path}")
    frame.columns = ["gene_id", "gene", "count"]
    return frame


def main() -> None:
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(a.manifest)
    manifest = manifest[manifest.region.eq("BA4") & manifest.cell_type.isin(NAME_MAP)].copy()
    manifest["target_cell_type"] = manifest.cell_type.map(NAME_MAP)
    manifest["count_path"] = manifest.count_url.map(lambda x: str(Path(a.counts) / Path(x).name))
    if len(manifest) != 57 or manifest.donor.nunique() != 18:
        raise ValueError(f"Unexpected BA4 manifest: {len(manifest)} rows, {manifest.donor.nunique()} donors")
    if not manifest.count_path.map(lambda x: Path(x).exists()).all():
        raise FileNotFoundError("One or more BA4 count files are missing")

    first = read_count(Path(manifest.count_path.iloc[0]))
    genes = first.gene.astype(str).to_numpy()
    gene_ids = first.gene_id.astype(str).to_numpy()
    raw = []
    for row in manifest.itertuples(index=False):
        frame = read_count(Path(row.count_path))
        if not np.array_equal(frame.gene.astype(str).to_numpy(), genes):
            raise ValueError(f"Gene order mismatch: {row.gsm}")
        raw.append(frame["count"].to_numpy(np.int64))
    raw = np.column_stack(raw)

    groups = manifest.groupby(["donor", "condition", "target_cell_type"], sort=True).indices
    group_rows = []
    aggregated = []
    for (donor, condition, target), indices in groups.items():
        aggregated.append(raw[:, indices].sum(axis=1))
        group_rows.append({"donor": donor, "condition": condition, "target_cell_type": target,
                           "n_input_libraries": len(indices)})
    aggregated = np.column_stack(aggregated)
    group_meta = pd.DataFrame(group_rows)
    if aggregated.sum() != raw.sum():
        raise ValueError("Count conservation failed")

    weights = pd.read_csv(a.weights)
    lookup = {g.upper(): i for i, g in enumerate(genes)}
    matched = weights.loc[weights.gene.astype(str).str.upper().isin(lookup)].copy()
    idx = np.array([lookup[g.upper()] for g in matched.gene.astype(str)], dtype=int)
    counts = aggregated[idx, :].T
    lib = aggregated.sum(axis=0).astype(float)
    logcpm = np.log2(counts / np.maximum(lib[:, None], 1) * 1e6 + 0.5)

    group_meta["library_size"] = lib.astype(np.int64)
    group_meta.to_csv(out / "donor_celltype_metadata.csv", index=False)
    matched.to_csv(out / "matched_pls1_weights.csv", index=False)
    np.savez_compressed(out / "donor_celltype_logcpm_pls_genes.npz", logcpm=logcpm,
                        genes=matched.gene.astype(str).to_numpy())
    audit = {
        "source": "GSE233408 released sFANS-seq gene counts",
        "region": "BA4 motor cortex",
        "released_ba4_libraries": int(len(manifest)),
        "biological_donors": int(manifest.donor.nunique()),
        "control_donors": int(manifest.query("condition == 'CTRL'").donor.nunique()),
        "hd_donors": int(manifest.query("condition == 'HD'").donor.nunique()),
        "directly_matched_populations": sorted(NAME_MAP.values()),
        "donor_population_pseudobulks": int(len(group_meta)),
        "pls_genes_matched": int(len(matched)),
        "count_conservation": True,
    }
    (out / "aggregation_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Aggregate GSE233408 BA4 observations into analysis-ready subtype/cell-population summaries.
# Input source/location: GSE233408 BA4 expression/count data and released sample/cell annotations from configurable project inputs.
# Output location: BA4 subtype aggregate tables and QC summaries in the configured BA4 results directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Select BA4 samples; harmonize subtype labels; aggregate expression/effect information; validate coverage; export summary tables.
# Log location: No dedicated log file; progress is written to standard output.
# =============================================================================
