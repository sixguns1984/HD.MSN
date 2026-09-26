# HD morphometric similarity network analysis code

This repository contains the analysis scripts for the study **Neuron transcriptional dysregulation is associated with somatic CAG expansion and cortical network disintegration in Huntington's disease**.

The code follows the order of the manuscript analyses. Participant MRI and clinical data are not redistributed. Public molecular datasets are referenced by accession number or DOI.

## Analysis workflow

1. `01\_msn\_construction` constructs individual DK308 morphometric similarity networks and regional morphometric similarity measures.
2. `02\_group\_statistics` fits HD-control, disease-stage, covariate, parcellation, inherited-CAG, and clinical sensitivity models.
3. `03\_clinical\_associations` generates regional clinical-association analyses and publication-ready visualizations.
4. `04\_ahba\_pls` assigns Allen Human Brain Atlas samples to DK308 parcels, fits the PLS model, and performs donor-omission analyses.
5. `05\_sampled\_cortex\_validation` evaluates PLS1 gene sets in GSE233387, GSE233408, GSE281069, and GSE180928 within their sampled cortical regions and dataset-provided cell types. The `gse233387\_hd\_control` subdirectory contains the full-transcriptome donor-by-cell-type pseudobulk workflow underlying the standardized HD-control contrasts.
6. `06\_ba4\_cag` calculates the GSE233387 BA4 cell-type expression scores, performs the 14-cell-type expression-CAG permutation test, runs neuronal and GSE233408 sensitivity analyses, and prepares BA4 CAG source data and visualizations.
7. `07\_external\_validation\_atlas` analyzes the GSE64810, GSE3790, and GSE79666 bulk-cortex datasets and the STDS0000242 neurotypical cortical reference.
8. `08\_regional\_limitation\_sensitivities` contains the prespecified left-precentral-parcel exclusion analysis, the 14-cell-type CAG bootstrap and power analysis, the BA4 spatial anchoring analysis, the cross-dataset directional-concordance test, and the laminar-reference comparison.

## Public data sources

* Allen Human Brain Atlas
* Zenodo DOI `10.5281/zenodo.20301898`, containing frontal-cortex GSE281069 and cingulate-cortex GSE180928 data
* GEO `GSE233387`
* GEO `GSE233408`
* GEO `GSE64810`
* GEO `GSE3790`, platforms `GPL96` and `GPL97`
* GEO `GSE79666`
* Spatial transcriptomic reference `STDS0000242`

## MSN covariate policy

Individual morphometric similarity networks are constructed from within-subject morphometric feature similarity; participant-level demographic variables are not inserted into the within-subject network formula. All cross-subject MSN inference and HC-normative MSN residualization in this release use the same nuisance-covariate set: `age`, `sex`, `education\_years`, and `eTIV`. Age and sex are therefore retained as nuisance covariates but are not treated as clinical outcomes. Sex coding is harmonized to a common binary representation before modeling, and education is represented as numeric years.

## Private clinical schema

Concrete clinical-indicator field names are intentionally not stored in the public code tree. Clinical analyses load an external JSON schema from the path specified by `HD\_MSN\_PRIVATE\_CLINICAL\_SCHEMA`. Keep that JSON file outside the repository and outside public release archives. Public code and primary association tables use anonymous clinical IDs.

## Statistical units

* MRI analyses use one regional vector per participant.
* The AHBA PLS analysis uses the 152 left-hemisphere DK308 parcels with valid expression coverage.
* GSE281069 and GSE180928 disease-tissue analyses use donor-level pseudobulk expression profiles within each sampled region and dataset-provided cell type.
* GSE233408 analyses use donor-level RNA-sequencing profiles after technical replicates from the same donor, region, and sorted cell type are combined.
* The primary GSE233387 expression-CAG test uses the 14 matched BA4 cell types as the statistical units.
* External bulk-expression datasets are analyzed separately; raw expression matrices from different studies are not pooled.

## Software environments

Python dependencies are listed in `environment/requirements.txt`. R and Bioconductor versions are recorded in `environment/R\_environment.md`. Raw or controlled-access inputs are not included. Scripts that use the original project hierarchy document the expected input files in their final comment block. For the regional-limitation workflow, set `HD\_MSN\_PROJECT\_ROOT` and, optionally, `HD\_MSN\_REGIONAL\_WORK\_DIR` before execution.

## Repository contents

`MANIFEST.csv` records file size and SHA-256 for every distributed file. `SHA256SUMS.txt` supports integrity verification after download. Dataset access and reuse remain subject to the terms of the original data providers.

## 

