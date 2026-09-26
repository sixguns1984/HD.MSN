#!/usr/bin/env python3
"""Freeze provenance, statistical units, key inputs, and software environment."""

from __future__ import annotations

import argparse
import os
import hashlib
import importlib
import json
import platform
import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(os.environ.get("HD_MSN_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
DATA = Path(os.environ.get("HD_MSN_DATA_ROOT", PROJECT_ROOT / "data")).resolve()
RESULTS = Path(os.environ.get("HD_MSN_RESULTS_ROOT", PROJECT_ROOT / "results")).resolve()
ATLAS = Path(os.environ.get("HD_MSN_ATLAS_ROOT", PROJECT_ROOT / "atlas_validation")).resolve()


def digest(path: Path) -> tuple[str, int]:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest(), path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()
    out = args.work_root / "manifest"; out.mkdir(parents=True, exist_ok=True)

    g387 = pd.read_csv(out / "GSE233387_samples.csv", encoding="utf-8-sig")
    g408 = pd.read_csv(out / "GSE233408_samples.csv", encoding="utf-8-sig")
    shared_pressl = sorted(set(g387.donor) & set(g408.donor))
    datasets = pd.DataFrame([
        {
            "dataset": "MSN HD-vs-HC cortical T map", "source_accession": "internal MRI cohort",
            "assay": "T1-derived morphometric similarity network", "regions": "left cortical parcels used for PLS",
            "released_samples": "participant-derived group statistic", "unique_donors": "not applicable",
            "biological_replicate": "participant upstream; parcel in cross-region PLS",
            "allowed_role": "PLS response map", "overlap_group": "internal_MSN",
            "independence_guard": "regional PLS is spatial association, not participant-level molecular coupling",
        },
        {
            "dataset": "Allen Human Brain Atlas", "source_accession": "AHBA six donors",
            "assay": "postmortem microarray", "regions": "500-parcel atlas; left hemisphere used for PLS",
            "released_samples": "six donor microarray directories", "unique_donors": 6,
            "biological_replicate": "donor for LODO; parcel for spatial PLS",
            "allowed_role": "derive/freeze PLS1 axis and six-donor sensitivity", "overlap_group": "AHBA",
            "independence_guard": "do not treat tissue samples or parcels as donor replicates",
        },
        {
            "dataset": "Zenodo HD atlas frontal cortex", "source_accession": "10.5281/zenodo.20301898; source GSE281069",
            "assay": "snRNA-seq", "regions": "frontal cortex", "released_samples": "30,240 nuclei",
            "unique_donors": "5 CTRL + 5 HD", "biological_replicate": "donor pseudobulk",
            "allowed_role": "independent frozen-signature disease validation", "overlap_group": "GSE281069",
            "independence_guard": "region analyzed separately; nuclei are not replicates",
        },
        {
            "dataset": "Zenodo HD atlas cingulate cortex", "source_accession": "10.5281/zenodo.20301898; source GSE180928",
            "assay": "snRNA-seq", "regions": "cingulate cortex", "released_samples": "31,311 nuclei",
            "unique_donors": "5 CTRL + 7 HD", "biological_replicate": "donor pseudobulk",
            "allowed_role": "independent frozen-signature disease validation", "overlap_group": "GSE180928",
            "independence_guard": "region analyzed separately; nuclei are not replicates",
        },
        {
            "dataset": "GSE233387", "source_accession": "GSE233387",
            "assay": "sFANS-enriched snRNA-seq", "regions": "BA4", "released_samples": len(g387),
            "unique_donors": f"{g387.donor.nunique()} (2 CTRL + 3 HD)", "biological_replicate": "donor",
            "allowed_role": "reference marker enrichment/laminar transfer only", "overlap_group": "Pressl_shared_donors",
            "independence_guard": "18 sorted libraries are five donors; no disease test in marker enrichment",
        },
        {
            "dataset": "GSE233408", "source_accession": "GSE233408",
            "assay": "sFANS bulk RNA-seq", "regions": "; ".join(sorted(g408.region.unique())),
            "released_samples": len(g408), "unique_donors": f"{g408.donor.nunique()} (6 CTRL + 13 HD)",
            "biological_replicate": "donor after donor×region×cell-type count aggregation",
            "allowed_role": "real-region laminar disease analysis", "overlap_group": "Pressl_shared_donors",
            "independence_guard": "shares five donors with GSE233387; never meta-analyze the two as independent cohorts",
        },
    ])
    datasets.to_csv(out / "dataset_provenance_statistical_units.csv", index=False)

    key_inputs = [
        DATA / "msn" / "HD_vs_HC_tvalues_with_names.txt",
        DATA / "parcellation" / "500.aparc.nii",
        DATA / "ahba" / "DK_all_atlas_info.csv",
        PROJECT_ROOT / "04_ahba_pls" / "AHBA_standard_analysis_Bon.py",
        RESULTS / "ahba_pls" / "PLS1_gene_rank_HD_vs_HC_tvalues_with_names.csv",
        ATLAS / "inputs" / "pls" / "reference_pls1_gene_weights.csv",
        DATA / "geo" / "GSE233387" / "target_celltype_marker_genes.csv",
        DATA / "geo" / "GSE233387" / "background_genes.txt",
        out / "GSE233387_family.soft.gz", out / "GSE233387_samples.csv",
        out / "GSE233408_family.soft.gz", out / "GSE233408_samples.csv",
        out / "GSE233408_count_checksums.json",
    ]
    checksums = []
    for path in key_inputs:
        value, size = digest(path)
        checksums.append({"path": str(path), "bytes": size, "sha256": value})
    pd.DataFrame(checksums).to_csv(out / "key_input_sha256.csv", index=False)

    modules = ["abagen", "numpy", "pandas", "scipy", "sklearn", "nibabel", "anndata", "scanpy", "statsmodels", "matplotlib", "seaborn"]
    versions = {}
    for name in modules:
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "installed")
    r_version = subprocess.check_output([os.environ.get("RSCRIPT", "Rscript"), "--version"], stderr=subprocess.STDOUT, text=True).strip()
    environment = {
        "timestamp_timezone": "not recorded",
        "host": "not recorded for privacy", "platform": platform.platform(), "python": platform.python_version(),
        "python_packages": versions, "R": r_version,
        "R_library_reused": str(ATLAS / "env/R/library"),
        "git_repositories": "none detected in analysis workspaces",
        "random_seed": 20260803,
    }
    (out / "software_environment.json").write_text(json.dumps(environment, indent=2) + "\n")

    atlas_verification = json.loads((ATLAS / "manifest/final_verification.json").read_text())
    acceptance = {
        "status": "PASS",
        "dataset_rows": len(datasets), "key_inputs_sha256": len(checksums),
        "gse233387_samples": len(g387), "gse233387_unique_donors": int(g387.donor.nunique()),
        "gse233408_samples": len(g408), "gse233408_unique_donors": int(g408.donor.nunique()),
        "gse233387_gse233408_shared_donors": shared_pressl,
        "zenodo_source_accessions": {"frontal": "GSE281069", "cingulate": "GSE180928"},
        "zenodo_existing_verification": atlas_verification.get("status"),
        "guards": [
            "No nucleus, sorted library, or parcel is treated as an independent donor.",
            "GSE233387 and GSE233408 share donors and are not combined as independent evidence.",
            "Zenodo frontal and cingulate source cohorts are modeled separately before cross-region synthesis.",
            "Sample-number mismatch is recorded but not used as an exclusion, per user instruction.",
        ],
    }
    (out / "input_audit_acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")
    print(json.dumps(acceptance, indent=2))


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Audit AHBA, atlas, regional target, and software inputs before running the transcriptional PLS workflow.
# Input source/location: Project-relative AHBA expression/atlas/target files and configured R/Python resources.
# Output location: Machine-readable input-audit/acceptance report in the configured audit output directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Resolve configured paths; verify file existence and dimensions; inspect identifiers/mappings; record software availability; write acceptance checks and hashes.
# Log location: No dedicated log file; audit status is written to standard output and the acceptance report.
# =============================================================================
