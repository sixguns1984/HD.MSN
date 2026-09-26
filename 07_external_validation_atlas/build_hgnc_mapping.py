from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pandas as pd


ROOT = Path(
    os.environ.get(
        "EXTERNAL_VALIDATION_ROOT",
        str(Path(__file__).resolve().parent / "work"),
    )
)
SOURCE = ROOT / "adult_cortex_atlas" / "source"
HGNC = SOURCE / "hgnc_complete_set.txt"
XLSX = SOURCE / "41467_2025_62793_MOESM5_ESM.xlsx"
OUT_CSV = SOURCE / "ensembl_gene_symbol_mapping.csv"
OUT_META = SOURCE / "ensembl_gene_symbol_mapping_metadata.json"

regional = pd.read_excel(XLSX, sheet_name="Marker gene-subclass-region-EXC", usecols=["gene"])
requested = sorted(
    {
        value.split(".")[0]
        for value in regional["gene"].dropna().astype(str)
        if value.startswith("ENSG")
    }
)
hgnc = pd.read_csv(HGNC, sep="\t", dtype=str, low_memory=False)
hgnc = hgnc[["ensembl_gene_id", "symbol", "locus_type", "status"]].dropna(subset=["ensembl_gene_id", "symbol"])
hgnc["ensembl_gene_id"] = hgnc["ensembl_gene_id"].str.split(".").str[0]
hgnc = hgnc[hgnc["ensembl_gene_id"].isin(requested)].drop_duplicates("ensembl_gene_id")
hgnc = hgnc.rename(columns={"symbol": "gene_symbol", "locus_type": "biotype"})
hgnc["species"] = "homo_sapiens"
hgnc.to_csv(OUT_CSV, index=False)

sha256 = hashlib.sha256(HGNC.read_bytes()).hexdigest()
metadata = {
    "mapping_authority": "HGNC",
    "source_url": "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
    "source_sha256": sha256,
    "requested_unique_ensembl_ids": len(requested),
    "mapped_unique_ensembl_ids": int(hgnc["ensembl_gene_id"].nunique()),
    "mapped_unique_symbols": int(hgnc["gene_symbol"].nunique()),
    "mapping_rate": float(hgnc["ensembl_gene_id"].nunique() / len(requested)) if requested else None,
}
OUT_META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps(metadata, indent=2))

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Build a reproducible HGNC gene-identifier mapping table for external validation analyses.
# Input source/location: Public HGNC complete-set table downloaded from the recorded HGNC source URL or supplied local cache.
# Output location: HGNC symbol/alias/identifier crosswalk in the configured mapping/output directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Load HGNC records; select approved symbols/aliases/IDs; normalize mapping fields; resolve duplicates according to documented rules; export crosswalk.
# Log location: No dedicated log file; status is written to standard output.
# =============================================================================
