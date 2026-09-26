# Regional-limitation and CAG uncertainty analyses

The scripts in this directory reproduce the analyses used to evaluate whether the whole-cortex PLS1 result depends on motor-cortex parcels and to quantify uncertainty in the 14-cell-type expression-CAG association.

- `leave_precentral_out_pls.py`: excludes the nine left precentral DK308 parcels, refits the one-component PLS model, and compares gene weights with the full 152-parcel model.
- `cag_bootstrap_power.py`: reproduces the 14-cell-type Spearman association, performs 1,000,000 label permutations, calculates percentile and BCa bootstrap confidence intervals, and estimates power across effect sizes.
- `regional_anchoring_directional_concordance_laminar.py`: calculates the precentral spatial anchor, performs the cross-dataset exact sign test, and compares the normal-adult laminar reference with BA4 cell-type CAG summaries.
- `validate_regional_sensitivity_figures.py`: verifies output completeness, editable PDF/SVG text, embedded font types, and raster resolution.

The first two scripts use input directories relative to their script locations. For the combined workflow, set:

```text
HD_MSN_PROJECT_ROOT=<local project-data root>
HD_MSN_REGIONAL_WORK_DIR=<optional output directory>
```

Input datasets are not redistributed. The expected file names and analysis steps are documented in the final comment block of each script.
