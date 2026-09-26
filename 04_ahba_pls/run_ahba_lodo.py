#!/usr/bin/env python3
"""Six-donor AHBA leave-one-donor-out stability audit for the frozen HD PLS1 axis."""

from __future__ import annotations

import argparse
import os
import json
from pathlib import Path

import abagen
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.cross_decomposition import PLSRegression


DONORS = ["9861", "10021", "12876", "14380", "15496", "15697"]

# abagen 0.1.3 predates pandas 2.0 and still calls DataFrame.append internally.
# This compatibility shim is exactly the former append behavior for the call
# pattern used by abagen and leaves the numerical pipeline unchanged.
if not hasattr(pd.DataFrame, "append"):
    def _dataframe_append(self, other, ignore_index=False, verify_integrity=False, sort=False):
        return pd.concat(
            [self, other], ignore_index=ignore_index,
            verify_integrity=verify_integrity, sort=sort,
        )
    pd.DataFrame.append = _dataframe_append


def fit_pls(x: pd.DataFrame, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    model = PLSRegression(n_components=1)
    model.fit(x.to_numpy(dtype=float), y)
    return model.x_scores_[:, 0].copy(), model.x_weights_[:, 0].copy()


def align(scores: np.ndarray, weights: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    common = np.isfinite(weights) & np.isfinite(reference)
    if spearmanr(weights[common], reference[common]).statistic < 0:
        return -scores, -weights
    return scores, weights


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else float("nan")


def aggregate(donor_frames: list[pd.DataFrame]) -> pd.DataFrame:
    stacked = pd.concat(donor_frames, keys=range(len(donor_frames)), names=["donor", "region"])
    return stacked.groupby(level="region").mean()


def prepare(expression: pd.DataFrame, atlas_labels: pd.Series, tmap: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    frame = expression.copy()
    frame.index = frame.index.astype(int)
    frame["label"] = frame.index.map(atlas_labels)
    left = tmap[tmap["Region_Name"].str.startswith("lh_")].copy()
    merged = frame.reset_index(names="id").merge(left, left_on="label", right_on="Region_Name", how="inner")
    genes = [column for column in expression.columns if column in merged.columns]
    matrix = merged[genes]
    complete = np.isfinite(matrix.to_numpy(dtype=float)).all(axis=1)
    matrix = matrix.loc[complete].reset_index(drop=True)
    y = merged.loc[complete, "T_Value"].to_numpy(dtype=float)
    regions = merged.loc[complete, "Region_Name"].tolist()
    return matrix, y, regions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    root = Path(os.environ.get("HD_MSN_DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
    ahba = Path(os.environ.get("HD_MSN_AHBA_ROOT", root / "ahba")).resolve()
    atlas_img = Path(os.environ.get("HD_MSN_ATLAS_IMAGE", root / "parcellation" / "500.aparc.nii")).resolve()
    atlas_info_path = ahba / "data/DK_all_atlas_info.csv"
    tmap_path = Path(os.environ.get("HD_MSN_REFERENCE_TMAP", root / "msn" / "HD_vs_HC_tvalues_with_names.txt")).resolve()
    original_path = ahba / "results_Bonferroni/expression_with_label.csv"
    frozen_path = Path(os.environ.get("HD_MSN_PLS_WEIGHTS", root / "pls" / "final_pls1_gene_weights.csv"))

    atlas_info = pd.read_csv(atlas_info_path)
    labels = atlas_info.set_index("id")["label"]
    tmap = pd.read_csv(tmap_path, sep="\t")
    frozen = pd.read_csv(frozen_path).set_index("gene")

    cache_paths = [args.out / f"donor_{donor}_expression.pkl.gz" for donor in DONORS]
    if all(path.exists() for path in cache_paths):
        donor_frames = [pd.read_pickle(path, compression="gzip") for path in cache_paths]
        print("Loaded six cached donor-expression matrices")
    else:
        donor_frames = abagen.get_expression_data(
            str(atlas_img),
            atlas_info=str(atlas_info_path),
            data_dir=str(ahba / "AHBA_raw"),
            ibf_threshold=0.5,
            probe_selection="diff_stability",
            missing="centroids",
            tolerance=2,
            norm_structures=True,
            norm_matched=True,
            donor_probes="aggregate",
            lr_mirror="bidirectional",
            corrected_mni=True,
            reannotated=True,
            return_counts=False,
            return_donors=True,
            donors=DONORS,
            n_proc=6,
            verbose=1,
        )
        if isinstance(donor_frames, dict):
            donor_map = {str(key): value for key, value in donor_frames.items()}
            missing = [donor for donor in DONORS if donor not in donor_map]
            if missing:
                raise RuntimeError(f"Donor-expression dictionary lacks expected donors: {missing}; keys={list(donor_map)}")
            donor_frames = [donor_map[donor] for donor in DONORS]
        if len(donor_frames) != len(DONORS):
            raise RuntimeError(f"Expected six donor frames, received {len(donor_frames)}")
        for donor, frame in zip(DONORS, donor_frames):
            frame.index = frame.index.astype(int)
            frame.to_pickle(args.out / f"donor_{donor}_expression.pkl.gz", compression="gzip")

    original = pd.read_csv(original_path).set_index("id")
    original_expression = original.drop(columns=["label"])
    common_genes = original_expression.columns.intersection(donor_frames[0].columns).intersection(frozen.index)
    original_expression = original_expression[common_genes]
    donor_frames = [frame[common_genes] for frame in donor_frames]

    all_donor_mean = aggregate(donor_frames)
    common_regions = original_expression.index.intersection(all_donor_mean.index)
    reproduced = all_donor_mean.loc[common_regions, common_genes]
    stored = original_expression.loc[common_regions, common_genes]
    flat_mask = np.isfinite(reproduced.to_numpy()) & np.isfinite(stored.to_numpy())
    reproduction_r = pearsonr(reproduced.to_numpy()[flat_mask], stored.to_numpy()[flat_mask]).statistic
    reproduction_mae = float(np.nanmean(np.abs(reproduced.to_numpy() - stored.to_numpy())))

    x_full, y_full, regions_full = prepare(original_expression, labels, tmap)
    full_scores, full_weights = fit_pls(x_full, y_full)
    frozen_raw = frozen.loc[x_full.columns, "PLS1_weight"].to_numpy(dtype=float)
    full_scores, full_weights = align(full_scores, full_weights, frozen_raw)
    full_spatial_r = pearsonr(full_scores, y_full).statistic
    full_weight_frozen_rho = spearmanr(full_weights, frozen_raw).statistic

    frozen_pos = set(frozen.index[frozen.get("gene_set", pd.Series(index=frozen.index, dtype=object)) == "PLS1_positive"])
    frozen_neg = set(frozen.index[frozen.get("gene_set", pd.Series(index=frozen.index, dtype=object)) == "PLS1_negative"])
    if not frozen_pos or not frozen_neg:
        significant = frozen["significant_Bonferroni"].astype(bool)
        frozen_pos = set(frozen.index[significant & (frozen["PLS1W_corr_Z"] > 0)])
        frozen_neg = set(frozen.index[significant & (frozen["PLS1W_corr_Z"] < 0)])
    genes = np.asarray(x_full.columns)
    full_pos_rank = set(genes[np.argsort(full_weights)[-len(frozen_pos):]])
    full_neg_rank = set(genes[np.argsort(full_weights)[:len(frozen_neg)]])

    full_score_series = pd.Series(full_scores, index=regions_full)
    rows: list[dict] = []
    for omitted, omitted_frame in zip(DONORS, donor_frames):
        retained = [frame for donor, frame in zip(DONORS, donor_frames) if donor != omitted]
        lodo_expression = aggregate(retained)
        x_lodo, y_lodo, regions_lodo = prepare(lodo_expression, labels, tmap)
        x_lodo = x_lodo[x_full.columns]
        scores, weights = fit_pls(x_lodo, y_lodo)
        scores, weights = align(scores, weights, full_weights)
        spatial_r = pearsonr(scores, y_lodo).statistic
        score_stability = pearsonr(scores, full_score_series.loc[regions_lodo].to_numpy()).statistic
        weight_rho = spearmanr(weights, full_weights).statistic
        pos_rank = set(genes[np.argsort(weights)[-len(frozen_pos):]])
        neg_rank = set(genes[np.argsort(weights)[:len(frozen_neg)]])
        rows.append(
            {
                "omitted_donor": omitted,
                "n_retained_donors": 5,
                "n_regions": len(regions_lodo),
                "n_genes": len(genes),
                "spatial_r_pls1_vs_msn_t": spatial_r,
                "spatial_direction_consistent": np.sign(spatial_r) == np.sign(full_spatial_r),
                "region_score_r_vs_full": score_stability,
                "gene_weight_spearman_vs_full": weight_rho,
                "pls1_positive_jaccard_vs_full_rank_matched": jaccard(pos_rank, full_pos_rank),
                "pls1_negative_jaccard_vs_full_rank_matched": jaccard(neg_rank, full_neg_rank),
                "pls1_positive_jaccard_vs_frozen_bonferroni": jaccard(pos_rank, frozen_pos),
                "pls1_negative_jaccard_vs_frozen_bonferroni": jaccard(neg_rank, frozen_neg),
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(args.out / "ahba_six_donor_lodo_summary.csv", index=False)
    acceptance = {
        "status": "PASS" if int(result["spatial_direction_consistent"].sum()) >= 5 and float(result["gene_weight_spearman_vs_full"].median()) >= 0.70 else "FAIL",
        "criteria": {
            "spatial_direction_consistent_min": "5/6",
            "median_gene_weight_spearman_min": 0.70,
            "jaccard": "reported as rank-tail stability; no hard gate",
        },
        "observed": {
            "full_spatial_r": full_spatial_r,
            "full_weight_spearman_vs_frozen_raw_weight": full_weight_frozen_rho,
            "stored_vs_recomputed_six_donor_expression_r": reproduction_r,
            "stored_vs_recomputed_six_donor_expression_mae": reproduction_mae,
            "spatial_direction_consistent_n": int(result["spatial_direction_consistent"].sum()),
            "median_gene_weight_spearman": float(result["gene_weight_spearman_vs_full"].median()),
            "median_region_score_r": float(result["region_score_r_vs_full"].median()),
            "median_positive_jaccard_vs_full_rank": float(result["pls1_positive_jaccard_vs_full_rank_matched"].median()),
            "median_negative_jaccard_vs_full_rank": float(result["pls1_negative_jaccard_vs_full_rank_matched"].median()),
            "frozen_positive_n": len(frozen_pos),
            "frozen_negative_n": len(frozen_neg),
        },
        "note": "LODO Jaccard uses same-cardinality raw-weight tails; it is a rank-stability sensitivity analysis and does not relabel genes as Bonferroni-significant without a new bootstrap.",
    }
    (args.out / "ahba_lodo_acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")
    print(json.dumps(acceptance, indent=2))


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Run leave-one-donor-out AHBA PLS sensitivity analysis to quantify donor dependence of the transcriptional signature.
# Input source/location: Donor-resolved AHBA expression data, parcel labels, and the regional MSN target map from configurable project paths.
# Output location: LODO PLS weights/scores, donor-wise concordance statistics, and summary tables in the configured output directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Identify donors; iteratively omit one donor; rebuild the parcel-level expression input; refit PLS; compare weights/scores with the reference fit; export stability metrics.
# Log location: No dedicated log file; progress is written to standard output.
# =============================================================================
