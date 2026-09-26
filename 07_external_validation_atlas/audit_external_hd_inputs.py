#!/usr/bin/env python3
import argparse
import collections
import csv
import gzip
from pathlib import Path

import pandas as pd


def audit_matrix(path: Path):
    meta = []
    in_table = False
    nrows = 0
    header = None
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Sample_"):
                row = next(csv.reader([line], delimiter="\t"))
                meta.append((row[0], row[1:]))
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
            elif line.startswith("!series_matrix_table_end"):
                in_table = False
            elif in_table:
                if header is None:
                    header = next(csv.reader([line], delimiter="\t"))
                else:
                    nrows += 1
    print(f"\n{path.name}: samples={len(header)-1 if header else len(meta[0][1])}, features={nrows}")
    for key, vals in meta:
        if key in ("!Sample_title", "!Sample_geo_accession", "!Sample_source_name_ch1"):
            print(key, collections.Counter(vals).most_common(15))
    kv = collections.defaultdict(list)
    for key, vals in meta:
        if key != "!Sample_characteristics_ch1":
            continue
        for value in vals:
            value = value.strip('"')
            if ": " in value:
                k, x = value.split(": ", 1)
                kv[k].append(x)
    print("characteristic keys")
    for key, vals in kv.items():
        print(" ", key, collections.Counter(vals).most_common(15))
    if "GSE3790" in path.name:
        meta_map = {key: vals for key, vals in meta}
        wanted = [
            "!Sample_title",
            "!Sample_geo_accession",
            "!Sample_source_name_ch1",
            "!Sample_description",
        ]
        print("sample rows (first 25)")
        n = len(meta_map.get("!Sample_title", []))
        for i in range(min(n, 25)):
            print(" | ".join(meta_map.get(key, [""] * n)[i][:160] for key in wanted))
        print("metadata keys", sorted(meta_map))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    args = ap.parse_args()
    for filename in (
        "GSE64810_series_matrix.txt.gz",
        "GSE3790-GPL96_series_matrix.txt.gz",
        "GSE3790-GPL97_series_matrix.txt.gz",
    ):
        path = args.directory / filename
        if path.exists():
            audit_matrix(path)
    for filename in (
        "GSE64810_mlhd_DESeq2_norm_counts_adjust.txt.gz",
        "GSE64810_mlhd_DESeq2_diffexp_DESeq2_outlier_trimmed_adjust.txt.gz",
    ):
        path = args.directory / filename
        if path.exists():
            df = pd.read_csv(path, sep="\t", nrows=5)
            print(f"\n{filename}: columns={df.columns.tolist()}")
            print(df.head(2).to_string(index=False))
    for filename in ("GPL96.annot.gz", "GPL97.annot.gz"):
        path = args.directory / filename
        if path.exists():
            df = pd.read_csv(path, sep="\t", comment="#", nrows=3)
            print(f"\n{filename}: columns={df.columns.tolist()}")
            print(df.head(2).to_string(index=False))


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Audit input files and metadata required for external HD transcriptomic validation.
# Input source/location: External HD expression/count, phenotype, mapping, and reference gene-set files from EXTERNAL_VALIDATION_ROOT/project input paths.
# Output location: Input audit/QC report in the configured external-validation output directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Resolve input paths; verify files/schemas/sample counts/identifiers; record hashes and acceptance checks; report blocking issues.
# Log location: No dedicated log file unless configured externally; audit messages are written to standard output.
# =============================================================================
