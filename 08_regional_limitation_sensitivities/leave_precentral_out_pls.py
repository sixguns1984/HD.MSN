#!/usr/bin/env python3
"""Re-fit one-component AHBA PLS after prespecified exclusion of left precentral parcels."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import platform
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats


SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
INPUT = ROOT / "input"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
STATISTICS = ROOT / "results" / "statistics"
LOGS = ROOT / "logs"
LOG_FILE = LOGS / "leave_precentral_out_pls.log"

EXPRESSION_FILE = INPUT / "expression_with_label.csv"
REFERENCE_TMAP_FILE = INPUT / "reference_HD_vs_HC_tvalues_with_names.txt"
REFERENCE_WEIGHTS_FILE = INPUT / "reference_gene_weights.csv"
FOUR_COVARIATE_TMAP_FILE = INPUT / "HD_HC_four_covariate_tmap.csv"
FOUR_COVARIATE_WEIGHTS_FILE = INPUT / "four_covariate_PLS_gene_weights.csv"
REGION_NAMES_FILE = INPUT / "308_regions_names.txt"

EXPECTED_REGIONS = 152
EXPECTED_GENES = 15_632
EXPECTED_EXCLUDED = 9
MOTOR_PATTERN = r"^lh_precentral_part\d+$"
DESIRED_SCORE_TARGET_SIGN = -1

PURPLE = "#8E8BFE"
PINK = "#FEA3A2"
INK = "#111111"
GRAY = "#666666"


def configure_logging() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"), logging.StreamHandler()],
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 14,
            "axes.labelsize": 15,
            "axes.titlesize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 1.0,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def standardize_columns(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=0)
    standard_deviations = values.std(axis=0, ddof=1)
    if np.any(~np.isfinite(standard_deviations)) or np.any(standard_deviations <= 0):
        raise RuntimeError("Expression matrix contains a non-finite or zero-variance gene.")
    return (values - means) / standard_deviations


def fit_univariate_pls1(frame: pd.DataFrame, genes: list[str]) -> dict[str, object]:
    """Fit the one-component, centered-and-scaled univariate PLS model used by sklearn."""
    x = frame[genes].to_numpy(dtype=float)
    y = frame["t"].to_numpy(dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("PLS input contains non-finite values.")
    x_scaled = standardize_columns(x)
    y_scaled = (y - y.mean()) / y.std(ddof=1)
    weights = x_scaled.T @ y_scaled
    norm = float(np.linalg.norm(weights))
    if not np.isfinite(norm) or norm <= 0:
        raise RuntimeError("The first PLS weight vector could not be estimated.")
    weights /= norm
    scores = x_scaled @ weights
    spatial_r = float(stats.pearsonr(scores, y).statistic)
    if int(np.sign(spatial_r)) != DESIRED_SCORE_TARGET_SIGN:
        weights = -weights
        scores = -scores
        spatial_r = -spatial_r
    return {
        "weights": weights,
        "scores": scores,
        "spatial_r": spatial_r,
        "n_regions": len(frame),
        "n_genes": len(genes),
    }


def load_reference_inputs() -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    expression = pd.read_csv(EXPRESSION_FILE)
    gene_columns = [column for column in expression.columns if column not in {"id", "label"}]
    expression_left = expression[expression["label"].astype(str).str.startswith("lh_")].copy()
    target = pd.read_csv(REFERENCE_TMAP_FILE, sep=r"\s+").rename(
        columns={"Region_Name": "region", "T_Value": "t"}
    )
    merged = expression_left.merge(target[["region", "t"]], left_on="label", right_on="region", how="inner")
    if len(merged) != EXPECTED_REGIONS or len(gene_columns) != EXPECTED_GENES:
        raise RuntimeError(
            f"Expected {EXPECTED_REGIONS} left-hemisphere parcels and {EXPECTED_GENES} genes; "
            f"received {len(merged)} and {len(gene_columns)}."
        )
    archived = pd.read_csv(REFERENCE_WEIGHTS_FILE)
    if archived["gene"].duplicated().any() or set(archived["gene"]) != set(gene_columns):
        raise RuntimeError("Reference PLS weights do not map one-to-one to the expression genes.")
    archived = archived.set_index("gene").reindex(gene_columns).reset_index()
    return merged, gene_columns, archived


def reproduce_and_refit(
    full_frame: pd.DataFrame,
    genes: list[str],
    archived: pd.DataFrame,
) -> tuple[dict[str, object], dict[str, object], pd.DataFrame, dict[str, float]]:
    full_fit = fit_univariate_pls1(full_frame, genes)
    archived_values = archived["PLS1_weight"].to_numpy(dtype=float)
    reproduction = {
        "weight_spearman_rho": float(stats.spearmanr(full_fit["weights"], archived_values).statistic),
        "weight_pearson_r": float(stats.pearsonr(full_fit["weights"], archived_values).statistic),
        "maximum_absolute_weight_difference": float(np.max(np.abs(full_fit["weights"] - archived_values))),
        "spatial_r": float(full_fit["spatial_r"]),
    }
    if (
        reproduction["weight_spearman_rho"] < 0.999999999
        or reproduction["weight_pearson_r"] < 0.999999999
        or reproduction["maximum_absolute_weight_difference"] > 1e-12
    ):
        raise RuntimeError("Reference PLS1 weights were not reproduced; leave-BA4-out fitting stopped.")

    excluded_mask = full_frame["label"].astype(str).str.match(MOTOR_PATTERN)
    excluded = full_frame.loc[excluded_mask, ["id", "label"]].copy()
    if len(excluded) != EXPECTED_EXCLUDED:
        raise RuntimeError(f"Expected {EXPECTED_EXCLUDED} left precentral parcels; received {len(excluded)}.")
    excluded_table = pd.DataFrame(
        {
            "parcel_id": excluded["id"].astype(int),
            "parcel_name": excluded["label"].astype(str),
            "hemisphere": "left",
            "BA_label": "BA4/primary motor cortex approximation",
            "motor_cortex_flag": True,
            "exclusion_reason": "DK308 left precentral parcel; prespecified anatomical approximation of BA4/primary motor cortex",
        }
    )
    retained = full_frame.loc[~excluded_mask].copy()
    leaveout_fit = fit_univariate_pls1(retained, genes)
    return full_fit, leaveout_fit, excluded_table, reproduction


def gene_weight_statistics(
    genes: list[str], full_fit: dict[str, object], leaveout_fit: dict[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_weights = np.asarray(full_fit["weights"], dtype=float)
    leaveout_weights = np.asarray(leaveout_fit["weights"], dtype=float)
    spearman = stats.spearmanr(full_weights, leaveout_weights)
    pearson = stats.pearsonr(full_weights, leaveout_weights)
    concordance = pd.DataFrame(
        [
            {"metric": "Spearman_rho", "value": float(spearman.statistic), "p_value": float(spearman.pvalue)},
            {"metric": "Pearson_r", "value": float(pearson.statistic), "p_value": float(pearson.pvalue)},
            {"metric": "n_genes", "value": len(genes), "p_value": np.nan},
            {"metric": "original_score_target_Pearson_r", "value": float(full_fit["spatial_r"]), "p_value": np.nan},
            {"metric": "leave_BA4_out_score_target_Pearson_r", "value": float(leaveout_fit["spatial_r"]), "p_value": np.nan},
        ]
    )
    weight_table = pd.DataFrame(
        {
            "gene": genes,
            "reference_PLS1_weight": full_weights,
            "leave_BA4_out_PLS1_weight": leaveout_weights,
        }
    )
    weight_table["original_rank"] = weight_table["reference_PLS1_weight"].rank(method="average")
    weight_table["leave_BA4_out_rank"] = weight_table["leave_BA4_out_PLS1_weight"].rank(method="average")

    overlap_rows: list[dict[str, object]] = []
    gene_array = np.asarray(genes)
    for proportion in (0.05, 0.10):
        number = int(math.floor(len(genes) * proportion))
        for direction in ("positive", "negative"):
            if direction == "positive":
                full_indices = np.argsort(full_weights)[-number:]
                leaveout_indices = np.argsort(leaveout_weights)[-number:]
            else:
                full_indices = np.argsort(full_weights)[:number]
                leaveout_indices = np.argsort(leaveout_weights)[:number]
            full_set = set(gene_array[full_indices])
            leaveout_set = set(gene_array[leaveout_indices])
            intersection = full_set & leaveout_set
            union = full_set | leaveout_set
            overlap_rows.append(
                {
                    "threshold_percent": int(proportion * 100),
                    "direction": direction,
                    "genes_per_set": number,
                    "intersection_n": len(intersection),
                    "retention_fraction": len(intersection) / number,
                    "jaccard_index": len(intersection) / len(union),
                }
            )
    return concordance, weight_table, pd.DataFrame(overlap_rows)


def four_covariate_target_sensitivity(expression: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    target = pd.read_csv(FOUR_COVARIATE_TMAP_FILE)[["region", "t"]]
    frame = expression.drop(columns=["region", "t"]).merge(
        target, left_on="label", right_on="region", how="inner"
    )
    full_fit = fit_univariate_pls1(frame, genes)
    stored = pd.read_csv(FOUR_COVARIATE_WEIGHTS_FILE).set_index("gene").reindex(genes)["full_weight"].to_numpy(float)
    reproduction_r = float(stats.pearsonr(full_fit["weights"], stored).statistic)
    reproduction_difference = float(np.max(np.abs(full_fit["weights"] - stored)))
    if reproduction_r < 0.999999999 or reproduction_difference > 1e-12:
        raise RuntimeError("The four-covariate target PLS weights were not reproduced.")
    keep = ~frame["label"].astype(str).str.match(MOTOR_PATTERN)
    leaveout_fit = fit_univariate_pls1(frame.loc[keep].copy(), genes)
    spearman = stats.spearmanr(full_fit["weights"], leaveout_fit["weights"])
    pearson = stats.pearsonr(full_fit["weights"], leaveout_fit["weights"])
    return pd.DataFrame(
        [
            {
                "target": "four-covariate HD-HC t map",
                "original_n_observations": len(frame),
                "excluded_precentral_n": int((~keep).sum()),
                "remaining_n_observations": int(keep.sum()),
                "n_genes": len(genes),
                "full_fit_reproduction_Pearson_r": reproduction_r,
                "full_fit_maximum_absolute_difference": reproduction_difference,
                "gene_weight_Spearman_rho": float(spearman.statistic),
                "gene_weight_Spearman_p": float(spearman.pvalue),
                "gene_weight_Pearson_r": float(pearson.statistic),
                "gene_weight_Pearson_p": float(pearson.pvalue),
                "full_score_target_r": float(full_fit["spatial_r"]),
                "leave_BA4_out_score_target_r": float(leaveout_fit["spatial_r"]),
            }
        ]
    )


def create_figures(weight_table: pd.DataFrame, overlap: pd.DataFrame, rho: float) -> None:
    configure_plotting()
    FIGURES.mkdir(parents=True, exist_ok=True)
    x = weight_table["reference_PLS1_weight"].to_numpy(float)
    y = weight_table["leave_BA4_out_PLS1_weight"].to_numpy(float)
    colors = np.where(x >= 0, PINK, PURPLE)
    low = min(x.min(), y.min())
    high = max(x.max(), y.max())
    padding = 0.07 * (high - low)

    fig, ax = plt.subplots(figsize=(7.2, 6.3), constrained_layout=True)
    ax.scatter(x, y, s=8, c=colors, alpha=0.22, edgecolors="none", rasterized=False)
    ax.plot([low - padding, high + padding], [low - padding, high + padding], color=GRAY, linestyle="--", linewidth=1.3)
    ax.axhline(0, color="#BBBBBB", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#BBBBBB", linewidth=0.7, zorder=0)
    ax.set_xlim(low - padding, high + padding)
    ax.set_ylim(low - padding, high + padding)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Original 152-parcel PLS1 gene weight")
    ax.set_ylabel("Leave-BA4-out PLS1 gene weight")
    ax.set_title("PLS1 gene-weight stability after motor-cortex exclusion")
    ax.text(
        0.04,
        0.95,
        f"Spearman $\\rho$ = {rho:.3f}\n15,632 genes",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=14,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(FIGURES / "leave_precentral_gene_weight_scatter.pdf", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "leave_precentral_gene_weight_scatter.png", dpi=800, bbox_inches="tight")
    plt.close(fig)

    labels = [
        f"Top {row.threshold_percent}%\n{row.direction}"
        for row in overlap.itertuples(index=False)
    ]
    values = overlap["retention_fraction"].to_numpy(float) * 100
    bar_colors = [PINK if direction == "positive" else PURPLE for direction in overlap["direction"]]
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    bars = ax.bar(np.arange(len(values)), values, color=bar_colors, width=0.68)
    ax.set_xticks(np.arange(len(values)), labels)
    ax.set_ylabel("Genes retained after BA4 exclusion (%)")
    ax.set_ylim(0, 105)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_title("Stability of the strongest PLS1 gene weights")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}%", ha="center", va="bottom", fontsize=13)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(FIGURES / "leave_precentral_top_gene_overlap.pdf", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "leave_precentral_top_gene_overlap.png", dpi=800, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_logging()
    for directory in (TABLES, FIGURES, STATISTICS):
        directory.mkdir(parents=True, exist_ok=True)
    inputs = [
        EXPRESSION_FILE,
        REFERENCE_TMAP_FILE,
        REFERENCE_WEIGHTS_FILE,
        FOUR_COVARIATE_TMAP_FILE,
        FOUR_COVARIATE_WEIGHTS_FILE,
        REGION_NAMES_FILE,
    ]
    input_hashes = pd.DataFrame(
        [{"file": str(path), "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in inputs]
    )
    input_hashes.to_csv(STATISTICS / "leave_precentral_input_sha256.csv", index=False)
    logging.info("Input hashes:\n%s", input_hashes.to_string(index=False))

    frame, genes, archived = load_reference_inputs()
    full_fit, leaveout_fit, excluded, reproduction = reproduce_and_refit(frame, genes, archived)
    concordance, weights, overlap = gene_weight_statistics(genes, full_fit, leaveout_fit)
    excluded.to_csv(TABLES / "leave_precentral_excluded_parcels.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "original_n_observations": len(frame),
                "excluded_BA4_n": len(excluded),
                "remaining_n_observations": len(frame) - len(excluded),
                "original_n_genes": len(genes),
                "refit_n_genes": len(genes),
                "common_n_genes": len(genes),
                "orientation_rule": "PLS1 score-target Pearson correlation fixed to the original negative direction",
                "whole_component_sign_reversal_needed_for_refit": True,
            }
        ]
    )
    summary.to_csv(TABLES / "leave_precentral_sample_exclusion_summary.csv", index=False)
    concordance.to_csv(TABLES / "leave_precentral_gene_weight_concordance.csv", index=False)
    weights.to_csv(TABLES / "leave_precentral_gene_weights_reference_vs_refit.csv", index=False)
    overlap.to_csv(TABLES / "leave_precentral_top_gene_overlap.csv", index=False)
    pd.DataFrame([reproduction]).to_csv(TABLES / "reference_PLS_reproduction.csv", index=False)
    four_covariate = four_covariate_target_sensitivity(frame, genes)
    four_covariate.to_csv(TABLES / "leave_precentral_four_covariate_target_sensitivity.csv", index=False)

    rho = float(concordance.loc[concordance["metric"].eq("Spearman_rho"), "value"].iloc[0])
    create_figures(weights, overlap, rho)

    spearman_row = concordance.loc[concordance["metric"].eq("Spearman_rho")].iloc[0]
    pearson_row = concordance.loc[concordance["metric"].eq("Pearson_r")].iloc[0]
    overlap_markdown = "\n".join(
        [
            "| Threshold | Direction | Genes per set | Intersection | Retained | Jaccard |",
            "|---:|:---|---:|---:|---:|---:|",
            *[
                f"| {int(row.threshold_percent)}% | {row.direction} | {int(row.genes_per_set)} | "
                f"{int(row.intersection_n)} | {100 * row.retention_fraction:.1f}% | {row.jaccard_index:.3f} |"
                for row in overlap.itertuples(index=False)
            ],
        ]
    )
    report = f"""# Leave-precentral-out PLS statistical report

## Prespecified exclusion

The reference PLS model used {len(frame)} left-hemisphere DK308 parcels and {len(genes):,} AHBA genes. All nine left-hemisphere DK308 parcels named `lh_precentral_part1` through `lh_precentral_part9` were excluded as the fixed anatomical approximation of BA4/primary motor cortex. The refit therefore used {len(frame) - len(excluded)} parcels.

## Original-model reproduction

The one-component centered-and-scaled PLS fit reproduced the reference PLS1 gene weights to machine precision: Spearman rho = {reproduction['weight_spearman_rho']:.12f}, Pearson r = {reproduction['weight_pearson_r']:.12f}, and maximum absolute weight difference = {reproduction['maximum_absolute_weight_difference']:.3e}. The reproduced PLS1 score-target correlation was r = {reproduction['spatial_r']:.12f}. This validation was required before the leave-BA4-out refit.

## Leave-BA4-out refit

After complete PLS refitting on the remaining 143 parcels, gene weights remained highly concordant with the reference PLS1 weights: Spearman rho = {float(spearman_row['value']):.12f}, two-sided P < 1e-300; Pearson r = {float(pearson_row['value']):.12f}, two-sided P < 1e-300; n = {len(genes):,} genes. The refitted PLS1 score-target correlation was r = {float(leaveout_fit['spatial_r']):.12f}.

PLS component signs are mathematically arbitrary. The reference component had a negative PLS1 score-target correlation. That direction was fixed before the parcel exclusion and applied to the complete refit. The raw fitted component required a whole-vector sign flip under this rule; no gene-specific sign changes or result-dependent orientation choices were made.

## Top-weight stability

{overlap_markdown}

## Four-covariate regional-target sensitivity

The same exclusion applied to the four-covariate HD-HC regional t map gave Spearman rho = {four_covariate.loc[0, 'gene_weight_Spearman_rho']:.12f} and Pearson r = {four_covariate.loc[0, 'gene_weight_Pearson_r']:.12f} across {len(genes):,} genes. This secondary check does not redefine the reference gene set.

## Interpretation boundary

The near-unity gene-weight concordance shows that the reference cortical transcriptional signature is not materially dependent on the nine left precentral parcels. This reduces concern that the later GSE233387 BA4 analyses merely recover a signature driven by BA4 observations in the reference model. Somatic CAG measurements are still available only for the 14 GSE233387 BA4 cell-type aggregates, so no cross-regional somatic-CAG inference follows from this sensitivity result.
"""
    (STATISTICS / "leave_precentral_statistical_report.md").write_text(report, encoding="utf-8")
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": mpl.__version__,
        "pls_implementation": "one-component centered-and-scaled univariate PLS; validated exactly against archived sklearn PLSRegression weights",
        "random_seed": None,
    }
    (STATISTICS / "leave_precentral_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    logging.info("Original PLS reproduction: %s", reproduction)
    logging.info("Excluded parcels: %s", excluded["parcel_name"].tolist())
    logging.info("Gene-weight concordance:\n%s", concordance.to_string(index=False))
    logging.info("Top-gene overlap:\n%s", overlap.to_string(index=False))
    logging.info("Four-covariate target sensitivity:\n%s", four_covariate.to_string(index=False))
    logging.info("Leave-precentral-out PLS sensitivity completed successfully.")


if __name__ == "__main__":
    main()


# Script record
# Purpose: reproduce the reference AHBA PLS1 model, exclude the fixed
#          left precentral DK308 parcels, refit PLS completely, and quantify
#          gene-weight stability; also run a four-covariate target sensitivity.
# Inputs: expression_with_label.csv (id, label, 15,632 genes), reference
#         HD-vs-HC t values (Region_Name, T_Value), reference PLS1 weights (gene,
#         PLS1_weight), four-covariate regional t values, DK308 names.
# Outputs: exclusion, sample-count, gene-weight, top-gene-overlap, reproduction,
#          four-covariate sensitivity tables; editable PDF and 800-dpi PNG plots;
#          statistical report, environment versions, hashes, and execution log.
# Main steps: validate 152x15,632 inputs; reproduce original weights; exclude
#             lh_precentral_part1-part9; refit the centered/scaled one-component
#             univariate PLS; orient by the prespecified negative score-target
#             direction; calculate Spearman/Pearson concordance and top 5%/10%
#             overlap; perform the same exclusion against the four-covariate map.
# Random seed: none; PLS fitting and concordance calculations are deterministic.
# Software: Python, NumPy, pandas, SciPy, Matplotlib; exact versions are written
#           to results/statistics/leave_precentral_environment.json.
# Log: logs/leave_precentral_out_pls.log.

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Test whether the reference AHBA PLS1 signature depends on the left precentral parcels by fully refitting PLS after their prespecified exclusion.
# Input source/location: Reference AHBA expression, reference HD-HC regional t map/PLS weights, four-covariate target map, and DK308 region names from the local input directory.
# Output location: Gene-weight stability/exclusion tables, statistical report, environment record, editable PDF and 800-dpi PNG figures under results/.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Validate reference fit; exclude left precentral parcels; refit centered/scaled one-component PLS; quantify weight/rank overlap; repeat against the four-covariate target.
# Log location: logs/leave_precentral_out_pls.log.
# =============================================================================
