#!/usr/bin/env python3
"""Prepare official GEO processed matrices for external-HD validation validation."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_series(path: Path):
    metadata_rows = []
    table_lines = []
    in_table = False
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!Sample_"):
                row = next(csv.reader([line], delimiter="\t"))
                metadata_rows.append((row[0], row[1:]))
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
            elif line.startswith("!series_matrix_table_end"):
                in_table = False
            elif in_table:
                table_lines.append(line)
    meta_map = {}
    characteristics = []
    for key, values in metadata_rows:
        if key == "!Sample_characteristics_ch1":
            characteristics.append(values)
        else:
            meta_map[key] = values
    titles = [x.strip('"') for x in meta_map["!Sample_title"]]
    meta = pd.DataFrame(
        {
            "sample_id": titles,
            "geo_accession": [x.strip('"') for x in meta_map["!Sample_geo_accession"]],
            "source": [x.strip('"') for x in meta_map.get("!Sample_source_name_ch1", [""] * len(titles))],
        }
    )
    for values in characteristics:
        for idx, raw in enumerate(values):
            value = raw.strip('"')
            if ": " in value:
                key, item = value.split(": ", 1)
                meta.loc[idx, key.strip().lower().replace(" ", "_")] = item.strip()
    expression = None
    if table_lines:
        expression = pd.read_csv(io.StringIO("".join(table_lines)), sep="\t", index_col=0)
        expression.columns = [str(x).strip('"') for x in expression.columns]
        expression.index = expression.index.astype(str).str.strip('"')
    return meta, expression


def read_annotation(path: Path):
    with gzip.open(path, "rt", errors="replace") as handle:
        for line in handle:
            if line.startswith("!platform_table_begin"):
                break
        annot = pd.read_csv(handle, sep="\t", dtype=str)
    annot = annot[["ID", "Gene symbol"]].dropna()
    # Ambiguous multi-gene probes are excluded rather than assigned arbitrarily.
    annot = annot[~annot["Gene symbol"].str.contains("///", regex=False)].copy()
    annot["Gene symbol"] = annot["Gene symbol"].str.strip()
    annot = annot[(annot["Gene symbol"] != "") & (annot["Gene symbol"] != "---")]
    return annot.drop_duplicates("ID")


def prepare_gse64810(download: Path, output: Path):
    meta, _ = parse_series(download / "GSE64810_series_matrix.txt.gz")
    meta["condition"] = meta["diagnosis"].map({"Neurologically normal": "CTRL", "Huntington's Disease": "HD"})
    for col in ("pmi", "age_of_death", "rin", "cag"):
        meta[col] = pd.to_numeric(meta.get(col), errors="coerce")
    counts = pd.read_csv(download / "GSE64810_mlhd_DESeq2_norm_counts_adjust.txt.gz", sep="\t")
    counts = counts.rename(columns={counts.columns[0]: "ensembl_id"}).set_index("ensembl_id")
    de = pd.read_csv(download / "GSE64810_mlhd_DESeq2_diffexp_DESeq2_outlier_trimmed_adjust.txt.gz", sep="\t")
    de = de.rename(columns={de.columns[0]: "ensembl_id"})
    mapping = de[["ensembl_id", "symbol"]].dropna().drop_duplicates("ensembl_id")
    counts = counts.reset_index().merge(mapping, on="ensembl_id", how="inner")
    sample_cols = [c for c in meta.sample_id if c in counts.columns]
    logexpr = np.log2(counts[sample_cols].astype(float) + 0.5)
    logexpr.insert(0, "gene", counts["symbol"].astype(str))
    logexpr = logexpr.groupby("gene", sort=True, as_index=True)[sample_cols].mean()
    de = de.dropna(subset=["symbol", "stat", "log2FoldChange"]).copy()
    de = de.groupby("symbol", as_index=False).agg(
        logFC=("log2FoldChange", "mean"),
        statistic=("stat", "mean"),
        raw_p=("pvalue", "min"),
        author_padj=("padj", "min"),
    )
    meta.to_csv(output / "GSE64810_metadata.csv", index=False)
    logexpr.to_csv(output / "GSE64810_log2_normalized_expression.csv.gz", compression="gzip")
    de.to_csv(output / "GSE64810_author_DESeq2_gene_results.csv", index=False)
    return {
        "samples": len(meta),
        "CTRL": int(meta.condition.eq("CTRL").sum()),
        "HD": int(meta.condition.eq("HD").sum()),
        "genes_expression": len(logexpr),
        "genes_author_de": len(de),
        "missing_PMI": int(meta.pmi.isna().sum()),
    }


def classify_ba4(title: str):
    token = title.split()[0]
    if token.startswith("HC"):
        return "HD"
    if token.startswith("H"):
        return "CTRL"
    return np.nan


def prepare_gse3790_platform(download: Path, output: Path, platform: str):
    meta, probes = parse_series(download / f"GSE3790-{platform}_series_matrix.txt.gz")
    meta = meta[meta.sample_id.str.contains("BA4", regex=False)].copy()
    meta["condition"] = meta.sample_id.map(classify_ba4)
    meta["donor_id"] = meta.sample_id.str.split().str[0]
    if meta.condition.isna().any():
        raise ValueError(f"Unclassified BA4 titles in {platform}: {meta.loc[meta.condition.isna(), 'sample_id'].tolist()}")
    annot = read_annotation(download / f"{platform}.annot.gz")
    geo_cols = meta.geo_accession.tolist()
    table = probes.loc[probes.index.intersection(annot.ID), geo_cols].copy()
    table.columns = meta.sample_id.tolist()
    sample_cols = meta.sample_id.tolist()
    table.insert(0, "ID", table.index)
    table = table.merge(annot, on="ID", how="inner")
    table = table.drop(columns="ID").groupby("Gene symbol", sort=True)[sample_cols].mean()
    # GEO series values are on a positive linear intensity scale (max > 5,000),
    # so the gene-level matrix is log2 transformed before limma modeling.
    if float(np.nanmax(table.to_numpy(float))) > 100:
        table = np.log2(table.astype(float))
    table.index.name = "gene"
    meta.to_csv(output / f"GSE3790_{platform}_BA4_metadata.csv", index=False)
    table.to_csv(output / f"GSE3790_{platform}_BA4_expression.csv.gz", compression="gzip")
    return {
        "samples": len(meta),
        "CTRL": int(meta.condition.eq("CTRL").sum()),
        "HD": int(meta.condition.eq("HD").sum()),
        "genes": len(table),
        "log2_expression_min": float(np.nanmin(table.to_numpy(float))),
        "log2_expression_max": float(np.nanmax(table.to_numpy(float))),
    }


def prepare_gse79666(download: Path, output: Path):
    meta, _ = parse_series(download / "GSE79666_series_matrix.txt.gz")
    meta["condition"] = np.where(meta.sample_id.str.startswith("HD-"), "HD", "CTRL")
    meta["age"] = pd.to_numeric(meta.get("age"), errors="coerce")
    raw = pd.read_excel(download / "GSE79666_FPKMs.HD.xlsx")
    fpkm_cols = [c for c in raw.columns if str(c).startswith("FPKM_")]
    rename = {c: str(c).replace("FPKM_", "") for c in fpkm_cols}
    expr = raw[["geneSymbol"] + fpkm_cols].dropna(subset=["geneSymbol"]).rename(columns=rename)
    expr["geneSymbol"] = expr["geneSymbol"].astype(str).str.strip()
    expr = expr[(expr.geneSymbol != "") & (expr.geneSymbol != "---")]
    sample_cols = meta.sample_id.tolist()
    expr = expr.groupby("geneSymbol", sort=True)[sample_cols].mean()
    expr = np.log2(expr.astype(float) + 0.1)
    expr.index.name = "gene"
    meta.to_csv(output / "GSE79666_BA4_metadata.csv", index=False)
    expr.to_csv(output / "GSE79666_BA4_log2_FPKM_expression.csv.gz", compression="gzip")
    return {
        "samples": len(meta),
        "CTRL": int(meta.condition.eq("CTRL").sum()),
        "HD": int(meta.condition.eq("HD").sum()),
        "genes": len(expr),
        "role": "additional independent BA4 sensitivity; not added post hoc to the four-test primary family",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    audit = {
        "GSE64810": prepare_gse64810(args.download, args.output),
        "GSE3790_GPL96_BA4": prepare_gse3790_platform(args.download, args.output, "GPL96"),
        "GSE3790_GPL97_BA4": prepare_gse3790_platform(args.download, args.output, "GPL97"),
        "GSE79666_BA4": prepare_gse79666(args.download, args.output),
        "GSE3790_condition_rule": "BA4 title prefix HC=HD case; H=control; reproduces the independently documented 19-case/16-control GPL96 motor-cortex sample count",
    }
    (args.output / "preprocessing_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Preprocess external HD transcriptomic data into harmonized analysis-ready matrices and metadata.
# Input source/location: Raw/processed external HD expression data, phenotypes, and gene mappings under EXTERNAL_VALIDATION_ROOT or configured project inputs.
# Output location: Harmonized expression matrices, sample metadata, and preprocessing QC tables in the external-validation work directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Load source data; map genes; harmonize samples/phenotypes; apply dataset-appropriate normalization/filtering; validate matrices; export processed inputs.
# Log location: No dedicated log file unless configured externally; progress is written to standard output.
# =============================================================================
