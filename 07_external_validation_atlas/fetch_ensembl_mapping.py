from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(os.environ["EXTERNAL_VALIDATION_ROOT"])
SOURCE = ROOT / "adult_cortex_atlas" / "source"
XLSX = SOURCE / "41467_2025_62793_MOESM5_ESM.xlsx"
OUT_CSV = SOURCE / "ensembl_gene_symbol_mapping.csv"
OUT_META = SOURCE / "ensembl_gene_symbol_mapping_metadata.json"

tab = pd.read_excel(XLSX, sheet_name="Marker gene-subclass-region-EXC", usecols=["gene"])
ids = sorted({x.split(".")[0] for x in tab["gene"].dropna().astype(str) if x.startswith("ENSG")})

rows = []
url = "https://rest.ensembl.org/lookup/id"
headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "User-Agent": "HD-MSN-external-validation/1.0",
}
for start in range(0, len(ids), 500):
    batch = ids[start : start + 500]
    last_error = None
    for attempt in range(5):
        try:
            response = requests.post(url, headers=headers, json={"ids": batch}, timeout=90)
            response.raise_for_status()
            payload = response.json()
            for ensembl_id in batch:
                item = payload.get(ensembl_id) or {}
                rows.append(
                    {
                        "ensembl_gene_id": ensembl_id,
                        "gene_symbol": item.get("display_name"),
                        "biotype": item.get("biotype"),
                        "species": item.get("species"),
                    }
                )
            break
        except Exception as error:  # network endpoint can transiently rate-limit
            last_error = repr(error)
            time.sleep(2**attempt)
    else:
        raise RuntimeError(f"Ensembl REST batch failed at {start}: {last_error}")
    time.sleep(0.2)

mapping = pd.DataFrame(rows)
mapping.to_csv(OUT_CSV, index=False)
metadata = {
    "endpoint": url,
    "requested_ids": len(ids),
    "returned_rows": len(mapping),
    "mapped_symbols": int(mapping["gene_symbol"].notna().sum()),
    "unique_mapped_symbols": int(mapping["gene_symbol"].dropna().nunique()),
}
OUT_META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(json.dumps(metadata, indent=2))

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Retrieve/build Ensembl-to-gene-symbol mapping resources needed for external transcriptomic datasets.
# Input source/location: Dataset gene identifiers and public Ensembl mapping service/cache settings from configurable inputs.
# Output location: Ensembl identifier crosswalk and mapping QC summaries in the configured mapping directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Collect unique identifiers; query/read mapping resources; normalize versions; resolve one-to-many mappings conservatively; export crosswalk/QC.
# Log location: No dedicated log file; fetch/mapping status is written to standard output.
# =============================================================================
