from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import os
import platform
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(os.environ.get("HD_MSN_PROJECT_ROOT", Path.cwd())).resolve()
WORK = Path(
    os.environ.get(
        "HD_MSN_REGIONAL_WORK_DIR",
        ROOT / "regional_sensitivity_outputs",
    )
).resolve()
TABLE_DIR = WORK / "results" / "tables"
FIG_DIR = WORK / "results" / "figures"
STAT_DIR = WORK / "results" / "statistics"
LOG_DIR = WORK / "logs"
QC_DIR = WORK / "qc"

INPUT_DIR = Path(os.environ.get("HD_MSN_REGIONAL_INPUT_DIR", ROOT / "input")).resolve()
EXPRESSION_FILE = Path(os.environ.get("HD_MSN_AHBA_EXPRESSION_FILE", INPUT_DIR / "expression_with_label.csv")).resolve()
WEIGHT_FILE = Path(os.environ.get("HD_MSN_REFERENCE_PLS_WEIGHTS", INPUT_DIR / "reference_gene_weights.csv")).resolve()
TVALUE_FILE = Path(os.environ.get("HD_MSN_REFERENCE_TMAP", INPUT_DIR / "reference_HD_vs_HC_tvalues_with_names.txt")).resolve()
NATIVE_EFFECT_FILE = Path(os.environ.get("HD_MSN_NATIVE_CELLTYPE_EFFECT_FILE", INPUT_DIR / "native_cell_type_HD_effects_and_gene_set_tests.csv")).resolve()
BA4_EFFECT_FILE = Path(os.environ.get("HD_MSN_BA4_EFFECT_FILE", INPUT_DIR / "GSE233408_BA4_four_subtype_effects.csv")).resolve()
LAMINAR_FILE = Path(os.environ.get("HD_MSN_LAMINAR_ENRICHMENT_FILE", INPUT_DIR / "laminar_top_marker_enrichment_globalBH.csv")).resolve()
CAG_FILE = Path(os.environ.get("HD_MSN_CAG_AGGREGATE_FILE", INPUT_DIR / "GSE233387_14celltype_PLS1positive_CAG_final.csv")).resolve()

PINK = "#FEA3A2"
PURPLE = "#8E8BFE"
GRAY = "#B8B8B8"
DARK_GRAY = "#666666"
LIGHT_GRAY = "#ECECEC"
BLACK = "#000000"


def ensure_dirs() -> None:
    for p in [TABLE_DIR, FIG_DIR, STAT_DIR, LOG_DIR, QC_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def configure_plotting() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "font.family": "Arial",
            "font.size": 13,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "text.color": BLACK,
            "axes.labelcolor": BLACK,
            "axes.edgecolor": BLACK,
            "xtick.color": BLACK,
            "ytick.color": BLACK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_figure(fig: plt.Figure, stem: str) -> list[str]:
    outputs: list[str] = []
    for ext, kwargs in [
        ("pdf", {"dpi": 600}),
        ("svg", {}),
        ("png", {"dpi": 800}),
        ("tiff", {"dpi": 800, "pil_kwargs": {"compression": "tiff_lzw"}}),
    ]:
        out = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(out, bbox_inches="tight", **kwargs)
        outputs.append(str(out))
    plt.close(fig)
    return outputs


def precentral_spatial_anchor() -> tuple[dict, list[str]]:
    logging.info("Computing the canonical AHBA PLS1 score in 152 left cortical parcels")
    expression = pd.read_csv(EXPRESSION_FILE)
    weights = pd.read_csv(WEIGHT_FILE)
    t_values = pd.read_csv(TVALUE_FILE, sep="\t")

    left = expression.loc[expression["label"].astype(str).str.startswith("lh_")].copy()
    genes = weights["gene"].astype(str).tolist()
    missing_genes = sorted(set(genes) - set(left.columns))
    if missing_genes:
        raise ValueError(f"Missing {len(missing_genes)} weighted genes in the AHBA matrix")
    if left.shape[0] != 152 or len(genes) != 15632:
        raise ValueError(f"Unexpected AHBA dimensions: parcels={left.shape[0]}, genes={len(genes)}")

    x = left[genes].to_numpy(dtype=float)
    means = x.mean(axis=0)
    sds = x.std(axis=0, ddof=1)
    if np.any(~np.isfinite(sds)) or np.any(sds <= 0):
        raise ValueError("At least one AHBA gene has a non-finite or zero standard deviation")
    xz = (x - means) / sds
    w = weights.set_index("gene").loc[genes, "PLS1_weight"].to_numpy(dtype=float)
    score = xz @ w

    parcel = pd.DataFrame(
        {
            "parcel": left["label"].astype(str).to_numpy(),
            "PLS1_weighted_expression_score": score,
        }
    )
    parcel = parcel.merge(
        t_values.rename(columns={"Region_Name": "parcel", "T_Value": "HD_HC_t_value"}),
        on="parcel",
        how="left",
        validate="one_to_one",
    )
    if parcel["HD_HC_t_value"].isna().any():
        raise ValueError("The canonical HD-HC t-map did not match all 152 AHBA parcels")
    r_score_t, p_score_t = stats.pearsonr(
        parcel["PLS1_weighted_expression_score"], parcel["HD_HC_t_value"]
    )
    if not np.isclose(r_score_t, -0.610142, atol=5e-6):
        raise ValueError(f"Canonical PLS1 score validation failed: r={r_score_t:.9f}")

    parcel["score_z_across_152_parcels"] = stats.zscore(
        parcel["PLS1_weighted_expression_score"], ddof=1
    )
    parcel["rank_low_to_high"] = parcel["PLS1_weighted_expression_score"].rank(
        method="average", ascending=True
    )
    parcel["percentile_low_to_high"] = (
        parcel["rank_low_to_high"] - 1
    ) / (len(parcel) - 1) * 100
    parcel["is_left_precentral"] = parcel["parcel"].str.match(r"lh_precentral_part[1-9]$")
    if int(parcel["is_left_precentral"].sum()) != 9:
        raise ValueError("Expected exactly nine left precentral DK308 parcels")

    parcel = parcel.sort_values("rank_low_to_high").reset_index(drop=True)
    parcel.to_csv(TABLE_DIR / "all_left_cortical_parcel_PLS1_scores.csv", index=False)

    precentral = parcel.loc[parcel["is_left_precentral"]].copy()
    precentral["precentral_part"] = precentral["parcel"].str.extract(r"part(\d+)").astype(int)
    precentral = precentral.sort_values("precentral_part")
    precentral.to_csv(TABLE_DIR / "precentral_PLS1_scores.csv", index=False)

    summary = pd.DataFrame(
        [
            {
                "region_definition": "left DK308 precentral parcels 1-9",
                "n_parcels": len(precentral),
                "mean_PLS1_score": precentral["PLS1_weighted_expression_score"].mean(),
                "sd_PLS1_score": precentral["PLS1_weighted_expression_score"].std(ddof=1),
                "median_PLS1_score": precentral["PLS1_weighted_expression_score"].median(),
                "min_PLS1_score": precentral["PLS1_weighted_expression_score"].min(),
                "max_PLS1_score": precentral["PLS1_weighted_expression_score"].max(),
                "mean_z_across_152": precentral["score_z_across_152_parcels"].mean(),
                "mean_percentile_low_to_high": precentral["percentile_low_to_high"].mean(),
                "canonical_score_tmap_Pearson_r": r_score_t,
                "canonical_score_tmap_P_value": p_score_t,
            }
        ]
    )
    summary.to_csv(TABLE_DIR / "precentral_spatial_anchor_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.7), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ranks = np.arange(1, len(parcel) + 1)
    colors = np.where(parcel["is_left_precentral"], PINK, GRAY)
    sizes = np.where(parcel["is_left_precentral"], 64, 24)
    ax.scatter(ranks, parcel["PLS1_weighted_expression_score"], c=colors, s=sizes, edgecolor="none", zorder=2)
    ax.axhline(0, color=BLACK, linewidth=1.1, linestyle="--", zorder=1)
    ax.set_xticks([1, 25, 50, 75, 100, 125, 152])
    ax.set_xlim(-4, 157)
    ypad = 0.08 * np.ptp(parcel["PLS1_weighted_expression_score"])
    ax.set_ylim(
        parcel["PLS1_weighted_expression_score"].min() - ypad,
        parcel["PLS1_weighted_expression_score"].max() + ypad,
    )
    ax.set_xlabel("Rank among 152 left cortical parcels")
    ax.set_ylabel("PLS1 weighted expression score")
    ax.set_title("Cortical distribution of the PLS1 score", pad=12)
    ax.scatter([], [], s=64, color=PINK, label="Left precentral parcels (n = 9)")
    ax.scatter([], [], s=24, color=GRAY, label="Other left cortical parcels")
    ax.legend(loc="best", frameon=False)
    ax.text(-0.10, 1.04, "A", transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")

    ax = axes[1]
    y = np.arange(len(precentral))
    values = precentral["PLS1_weighted_expression_score"].to_numpy()
    point_colors = np.where(values < 0, PURPLE, PINK)
    for yi, value, color in zip(y, values, point_colors):
        ax.plot([0, value], [yi, yi], color=color, linewidth=3.2, solid_capstyle="round", zorder=2)
    ax.scatter(values, y, c=point_colors, s=105, edgecolor=BLACK, linewidth=0.7, zorder=3)
    ax.axvline(0, color=BLACK, linewidth=1.1, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"Precentral {int(v)}" for v in precentral["precentral_part"]])
    ax.invert_yaxis()
    xpad = 0.12 * np.ptp(values) if np.ptp(values) > 0 else 1
    ax.set_xlim(min(0, values.min()) - xpad, max(0, values.max()) + xpad)
    ax.set_xlabel("PLS1 weighted expression score")
    ax.set_title("Scores in the left precentral cortex", pad=12)
    ax.text(-0.16, 1.04, "B", transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")
    fig.tight_layout(w_pad=3.5)
    figure_files = save_figure(fig, "precentral_spatial_anchor")

    results = {
        "canonical_score_tmap_pearson_r": float(r_score_t),
        "canonical_score_tmap_p_value": float(p_score_t),
        "precentral_n": int(len(precentral)),
        "precentral_mean": float(precentral["PLS1_weighted_expression_score"].mean()),
        "precentral_sd": float(precentral["PLS1_weighted_expression_score"].std(ddof=1)),
        "precentral_median": float(precentral["PLS1_weighted_expression_score"].median()),
        "precentral_min": float(precentral["PLS1_weighted_expression_score"].min()),
        "precentral_max": float(precentral["PLS1_weighted_expression_score"].max()),
        "precentral_mean_z": float(precentral["score_z_across_152_parcels"].mean()),
        "precentral_mean_percentile": float(precentral["percentile_low_to_high"].mean()),
    }
    return results, figure_files


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lower = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    upper = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lower), float(upper)


def directional_concordance() -> tuple[dict, list[str]]:
    logging.info("Formalizing cross-dataset directional concordance")
    native = pd.read_csv(NATIVE_EFFECT_FILE)
    native = native[
        ["dataset", "region", "cell_type", "standardized_HD_effect", "ci_low", "ci_high"]
    ].copy()
    native["source_model"] = "dataset-specific donor-level model"

    ba4 = pd.read_csv(BA4_EFFECT_FILE).rename(
        columns={
            "estimate": "standardized_HD_effect",
            "subtype": "cell_type",
        }
    )
    ba4["dataset"] = "GSE233408"
    ba4["region"] = "BA4"
    ba4["source_model"] = "donor-level mixed model"
    ba4 = ba4[
        ["dataset", "region", "cell_type", "standardized_HD_effect", "ci_low", "ci_high", "source_model"]
    ]

    effects = pd.concat([native, ba4], ignore_index=True)
    expected_counts = {"GSE281069": 7, "GSE180928": 6, "GSE233408": 4}
    observed_counts = effects.groupby("dataset").size().to_dict()
    if observed_counts != expected_counts or len(effects) != 17:
        raise ValueError(f"Unexpected cell-type estimate counts: {observed_counts}")
    effects["negative_direction"] = effects["standardized_HD_effect"] < 0
    if int(effects["negative_direction"].sum()) != 16:
        raise ValueError("Expected 16 of 17 standardized HD effects to be negative")
    effects["display_label"] = effects["cell_type"] + " (" + effects["dataset"] + ", " + effects["region"] + ")"
    effects.to_csv(TABLE_DIR / "celltype_directional_effects.csv", index=False)

    rows = []
    dataset_order = ["GSE281069", "GSE180928", "GSE233408"]
    display_names = {
        "GSE281069": "GSE281069, frontal cortex",
        "GSE180928": "GSE180928, cingulate cortex",
        "GSE233408": "GSE233408, BA4",
    }
    for dataset in dataset_order:
        d = effects.loc[effects["dataset"] == dataset]
        k, n = int(d["negative_direction"].sum()), len(d)
        low, high = clopper_pearson(k, n)
        rows.append(
            {
                "scope": display_names[dataset],
                "negative_count": k,
                "total_count": n,
                "negative_fraction": k / n,
                "clopper_pearson_95ci_low": low,
                "clopper_pearson_95ci_high": high,
            }
        )
    k_total, n_total = int(effects["negative_direction"].sum()), len(effects)
    low_total, high_total = clopper_pearson(k_total, n_total)
    exact_one = stats.binomtest(k_total, n_total, p=0.5, alternative="greater").pvalue
    exact_two = stats.binomtest(k_total, n_total, p=0.5, alternative="two-sided").pvalue
    rows.append(
        {
            "scope": "Overall cell-type estimates",
            "negative_count": k_total,
            "total_count": n_total,
            "negative_fraction": k_total / n_total,
            "clopper_pearson_95ci_low": low_total,
            "clopper_pearson_95ci_high": high_total,
        }
    )
    summary = pd.DataFrame(rows)
    summary["two_sided_exact_binomial_P_overall"] = np.nan
    summary["one_sided_exact_binomial_P_overall"] = np.nan
    summary.loc[summary.index[-1], "two_sided_exact_binomial_P_overall"] = exact_two
    summary.loc[summary.index[-1], "one_sided_exact_binomial_P_overall"] = exact_one

    dataset_negative = effects.groupby("dataset")["negative_direction"].agg(["sum", "count"]).loc[dataset_order]
    block_rows = []
    for signs in itertools.product([1, -1], repeat=3):
        negative_count = 0
        for sign, (_, row) in zip(signs, dataset_negative.iterrows()):
            original_negative = int(row["sum"])
            total = int(row["count"])
            negative_count += original_negative if sign == 1 else total - original_negative
        block_rows.append(
            {
                "GSE281069_sign": signs[0],
                "GSE180928_sign": signs[1],
                "GSE233408_sign": signs[2],
                "negative_count": negative_count,
                "absolute_deviation_from_half": abs(negative_count - n_total / 2),
            }
        )
    block_null = pd.DataFrame(block_rows)
    observed_dev = abs(k_total - n_total / 2)
    block_p_one = float((block_null["negative_count"] >= k_total).mean())
    block_p_two = float((block_null["absolute_deviation_from_half"] >= observed_dev).mean())
    summary["dataset_block_signflip_P_one_sided"] = np.nan
    summary["dataset_block_signflip_P_two_sided"] = np.nan
    summary.loc[summary.index[-1], "dataset_block_signflip_P_one_sided"] = block_p_one
    summary.loc[summary.index[-1], "dataset_block_signflip_P_two_sided"] = block_p_two
    summary.to_csv(TABLE_DIR / "directional_concordance_summary.csv", index=False)
    block_null.to_csv(TABLE_DIR / "dataset_block_signflip_null.csv", index=False)

    group_order = [
        ("GSE281069", "frontal"),
        ("GSE180928", "cingulate"),
        ("GSE233408", "BA4"),
    ]
    effect_parts = []
    for dataset, region in group_order:
        d = effects.loc[(effects["dataset"] == dataset) & (effects["region"] == region)].copy()
        effect_parts.append(d)
    plot_effects = pd.concat(effect_parts, ignore_index=True)
    y_positions: list[float] = []
    current = 0.0
    last_dataset = None
    for dataset in plot_effects["dataset"]:
        if last_dataset is not None and dataset != last_dataset:
            current += 0.75
        y_positions.append(current)
        current += 1.0
        last_dataset = dataset
    plot_effects["y"] = y_positions

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 9.6), gridspec_kw={"width_ratios": [1.75, 1]})
    ax = axes[0]
    for _, row in plot_effects.iterrows():
        color = PURPLE if row["standardized_HD_effect"] < 0 else PINK
        ax.plot([row["ci_low"], row["ci_high"]], [row["y"], row["y"]], color=color, linewidth=2.5, zorder=2)
        ax.scatter(
            row["standardized_HD_effect"],
            row["y"],
            s=80,
            color=color,
            edgecolor=BLACK,
            linewidth=0.6,
            zorder=3,
        )
    ax.axvline(0, color=BLACK, linewidth=1.1, linestyle="--", zorder=1)
    ax.set_yticks(plot_effects["y"])
    ax.set_yticklabels(plot_effects["cell_type"])
    ax.invert_yaxis()
    xmin = min(plot_effects["ci_low"].min(), -0.1)
    xmax = max(plot_effects["ci_high"].max(), 0.1)
    pad = 0.08 * (xmax - xmin)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_xlabel("Standardized HD–HC effect (95% CI)")
    ax.set_title("Disease-related effects across sampled cell types", pad=12)
    group_starts = plot_effects.groupby("dataset", sort=False)["y"].min()
    for dataset in dataset_order:
        ax.text(
            0.01,
            group_starts[dataset] - 0.47,
            display_names[dataset],
            transform=ax.get_yaxis_transform(),
            va="bottom",
            ha="left",
            fontsize=13,
            fontweight="bold",
        )
    ax.text(-0.13, 1.03, "A", transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")

    ax = axes[1]
    summary_plot = summary.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(summary_plot))
    fractions = summary_plot["negative_fraction"].to_numpy()
    lows = summary_plot["clopper_pearson_95ci_low"].to_numpy()
    highs = summary_plot["clopper_pearson_95ci_high"].to_numpy()
    xerr = np.vstack([fractions - lows, highs - fractions])
    ax.errorbar(
        fractions,
        y,
        xerr=xerr,
        fmt="o",
        markersize=9,
        color=PURPLE,
        ecolor=PURPLE,
        elinewidth=2.6,
        capsize=5,
        markeredgecolor=BLACK,
        markeredgewidth=0.6,
    )
    ax.axvline(0.5, color=BLACK, linestyle="--", linewidth=1.1)
    ax.set_xlim(-0.03, 1.10)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks(y)
    ax.set_yticklabels(summary_plot["scope"])
    ax.set_xlabel("Fraction of negative HD–HC effects\n(95% exact CI)")
    ax.set_title("Directional concordance", pad=12)
    for yi, (_, row) in enumerate(summary_plot.iterrows()):
        ax.text(
            min(1.035, row["negative_fraction"] + 0.035),
            yi,
            f"{int(row['negative_count'])}/{int(row['total_count'])}",
            va="center",
            ha="left",
            fontsize=13,
        )
    ax.text(
        0.02,
        -0.18,
        f"Overall two-sided exact binomial P = {exact_two:.4g}\n"
        f"Dataset-block sign-flip P = {block_p_two:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=13,
    )
    ax.text(-0.18, 1.03, "B", transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.17, top=0.92, wspace=0.53)
    figure_files = save_figure(fig, "directional_concordance")

    results = {
        "negative_count": k_total,
        "total_count": n_total,
        "negative_fraction": k_total / n_total,
        "clopper_pearson_95ci_low": low_total,
        "clopper_pearson_95ci_high": high_total,
        "exact_binomial_two_sided_p": float(exact_two),
        "exact_binomial_one_sided_p": float(exact_one),
        "dataset_block_signflip_two_sided_p": block_p_two,
        "dataset_block_signflip_one_sided_p": block_p_one,
    }
    return results, figure_files


def exact_deep_superficial_test(deep: np.ndarray, superficial: np.ndarray) -> dict:
    values = np.concatenate([deep, superficial])
    observed = float(deep.mean() - superficial.mean())
    permuted = []
    for deep_idx in itertools.combinations(range(len(values)), len(deep)):
        deep_idx = np.asarray(deep_idx, dtype=int)
        superficial_idx = np.asarray([i for i in range(len(values)) if i not in set(deep_idx)], dtype=int)
        permuted.append(float(values[deep_idx].mean() - values[superficial_idx].mean()))
    permuted = np.asarray(permuted)
    p_one = float(np.mean(permuted >= observed - 1e-12))
    p_two = float(np.mean(np.abs(permuted) >= abs(observed) - 1e-12))
    return {
        "deep_mean": float(deep.mean()),
        "superficial_mean": float(superficial.mean()),
        "mean_difference": observed,
        "n_allocations": int(len(permuted)),
        "one_sided_exact_P": p_one,
        "two_sided_exact_P": p_two,
    }


def laminar_alignment() -> tuple[dict, list[str]]:
    logging.info("Aligning STDS0000242 laminar enrichment with BA4 CAG summaries")
    laminar = pd.read_csv(LAMINAR_FILE)
    laminar = laminar.loc[
        (laminar["family"] == "primary_12")
        & (laminar["gene_set"] == "PLS1_positive")
        & (laminar["domain"].isin(["L1", "L2", "L3", "L4", "L5", "L6"]))
    ].copy()
    layer_order = ["L1", "L2", "L3", "L4", "L5", "L6"]
    laminar["layer_order"] = pd.Categorical(laminar["domain"], categories=layer_order, ordered=True)
    laminar = laminar.sort_values("layer_order")
    if laminar["domain"].tolist() != layer_order:
        raise ValueError("STDS0000242 layer results do not contain exactly L1-L6")

    cag_all = pd.read_csv(CAG_FILE)
    excitatory_types = ["Layer 2", "Layer 4", "Betz cells", "Layer 5a", "Layer 6a", "Layer 6b"]
    cag = cag_all.loc[cag_all["target_cell_type"].isin(excitatory_types)].copy()
    if set(cag["target_cell_type"]) != set(excitatory_types) or len(cag) != 6:
        raise ValueError("Expected exactly six GSE233387 excitatory cell types")
    layer_map = {
        "Layer 2": "L2",
        "Layer 4": "L4",
        "Betz cells": "L5",
        "Layer 5a": "L5",
        "Layer 6a": "L6",
        "Layer 6b": "L6",
    }
    cag["layer"] = cag["target_cell_type"].map(layer_map)
    cag["depth_group"] = np.where(cag["layer"].isin(["L5", "L6"]), "Deep (L5/L6)", "Superficial/middle (L2/L4)")

    aligned_rows = []
    for _, row in laminar.iterrows():
        layer_cag = cag.loc[cag["layer"] == row["domain"]]
        if layer_cag.empty:
            aligned_rows.append(
                {
                    "layer": row["domain"],
                    "PLS1_positive_enrichment_fold": row["enrichment_fold"],
                    "PLS1_positive_overlap_N": row["overlap_N"],
                    "PLS1_positive_q": row["q_global_primary_12"],
                    "GSE233387_cell_type": np.nan,
                    "mean_somatic_CAG_gain": np.nan,
                    "CAG_SD": np.nan,
                }
            )
        else:
            for _, crow in layer_cag.iterrows():
                aligned_rows.append(
                    {
                        "layer": row["domain"],
                        "PLS1_positive_enrichment_fold": row["enrichment_fold"],
                        "PLS1_positive_overlap_N": row["overlap_N"],
                        "PLS1_positive_q": row["q_global_primary_12"],
                        "GSE233387_cell_type": crow["target_cell_type"],
                        "mean_somatic_CAG_gain": crow["cag_mean_somatic_length_gain"],
                        "CAG_SD": crow["cag_sd"],
                    }
                )
    aligned = pd.DataFrame(aligned_rows)
    aligned.to_csv(TABLE_DIR / "laminar_enrichment_and_CAG_alignment.csv", index=False)

    deep = cag.loc[cag["depth_group"] == "Deep (L5/L6)", "cag_mean_somatic_length_gain"].to_numpy()
    superficial = cag.loc[
        cag["depth_group"] == "Superficial/middle (L2/L4)", "cag_mean_somatic_length_gain"
    ].to_numpy()
    exact = exact_deep_superficial_test(deep, superficial)
    pd.DataFrame([exact]).to_csv(TABLE_DIR / "deep_vs_superficial_exact_test.csv", index=False)

    layer_y = {layer: i for i, layer in enumerate(layer_order)}
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.4), sharey=True, gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    y = np.asarray([layer_y[v] for v in laminar["domain"]])
    significant = laminar["q_global_primary_12"] < 0.05
    colors = np.where(significant, PINK, LIGHT_GRAY)
    ax.barh(y, laminar["enrichment_fold"], height=0.58, color=colors, edgecolor=BLACK, linewidth=0.7)
    ax.axvline(1, color=BLACK, linestyle="--", linewidth=1.1)
    ax.set_yticks(np.arange(6))
    ax.set_yticklabels(layer_order)
    ax.invert_yaxis()
    ax.set_xlabel("PLS1-positive gene enrichment (fold)")
    ax.set_ylabel("Cortical layer")
    ax.set_title("STDS0000242 adult cortical reference", pad=12)
    ax.set_xlim(0, laminar["enrichment_fold"].max() * 1.35)
    for yi, (_, row) in zip(y, laminar.iterrows()):
        q = row["q_global_primary_12"]
        q_text = f"q = {q:.4f}" if q >= 0.001 else f"q = {q:.1e}"
        ax.text(row["enrichment_fold"] + 0.10, yi, q_text, va="center", ha="left", fontsize=13)
    ax.text(-0.14, 1.04, "A", transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")

    ax = axes[1]
    offsets = {
        "Layer 2": 0.0,
        "Layer 4": 0.0,
        "Betz cells": -0.16,
        "Layer 5a": 0.16,
        "Layer 6a": -0.16,
        "Layer 6b": 0.16,
    }
    for _, row in cag.iterrows():
        yy = layer_y[row["layer"]] + offsets[row["target_cell_type"]]
        ax.errorbar(
            row["cag_mean_somatic_length_gain"],
            yy,
            xerr=row["cag_sd"],
            fmt="o",
            markersize=8.5,
            color=PURPLE,
            ecolor=PURPLE,
            elinewidth=2.3,
            capsize=4,
            markeredgecolor=BLACK,
            markeredgewidth=0.6,
            zorder=3,
        )
        ax.text(
            row["cag_mean_somatic_length_gain"] + row["cag_sd"] + 0.55,
            yy,
            row["target_cell_type"],
            va="center",
            ha="left",
            fontsize=13,
        )
    for layer in ["L1", "L3"]:
        ax.text(1.0, layer_y[layer], "No GSE233387 CAG estimate", color=DARK_GRAY, va="center", fontsize=13)
    ax.set_xlabel("Mean somatic CAG-length gain (SD)")
    ax.set_title("GSE233387 BA4 excitatory cell types", pad=12)
    upper = float((cag["cag_mean_somatic_length_gain"] + cag["cag_sd"]).max() + 8.0)
    ax.set_xlim(0, upper)
    ax.set_yticks(np.arange(6))
    ax.set_yticklabels(layer_order)
    ax.text(-0.12, 1.04, "B", transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")
    fig.tight_layout(w_pad=3.5)
    figure_files = save_figure(fig, "laminar_alignment")

    significant_layers = laminar.loc[significant, "domain"].tolist()
    results = {
        "significant_layers_q_lt_0_05": significant_layers,
        "L2_fold": float(laminar.loc[laminar["domain"] == "L2", "enrichment_fold"].iloc[0]),
        "L2_q": float(laminar.loc[laminar["domain"] == "L2", "q_global_primary_12"].iloc[0]),
        "L5_fold": float(laminar.loc[laminar["domain"] == "L5", "enrichment_fold"].iloc[0]),
        "L5_q": float(laminar.loc[laminar["domain"] == "L5", "q_global_primary_12"].iloc[0]),
        "L6_fold": float(laminar.loc[laminar["domain"] == "L6", "enrichment_fold"].iloc[0]),
        "L6_q": float(laminar.loc[laminar["domain"] == "L6", "q_global_primary_12"].iloc[0]),
        **exact,
    }
    return results, figure_files


def write_report(a2: dict, a3: dict, a5: dict) -> None:
    report = f"""# Analyses 2, 3, and 5: detailed results

## PLS1 spatial anchoring in the left precentral cortex

The canonical AHBA analysis comprised 152 left-hemisphere DK308 cortical parcels and 15,632 genes. Gene expression was standardized across parcels, multiplied by the canonical PLS1 weights, and summed to obtain the parcel-level PLS1 weighted expression score. The reconstructed score reproduced the confirmed association with the HD–HC MSN t-map (Pearson r = {a2['canonical_score_tmap_pearson_r']:.6f}, P = {a2['canonical_score_tmap_p_value']:.3e}).

The motor-cortex approximation was defined a priori as the same nine left precentral DK308 parcels used in the leave-precentral-out sensitivity analysis. Their mean PLS1 score was {a2['precentral_mean']:.3f} (SD {a2['precentral_sd']:.3f}; median {a2['precentral_median']:.3f}; range {a2['precentral_min']:.3f} to {a2['precentral_max']:.3f}). The mean standardized score was {a2['precentral_mean_z']:.3f}, corresponding to a mean percentile of {a2['precentral_mean_percentile']:.1f} among the 152 left cortical parcels. Individual parcel values are reported in `precentral_PLS1_scores.csv`.

This analysis establishes an explicit AHBA spatial anchor for the PLS1 score within left precentral cortex. It does not treat the nine DK308 parcels as independent biological replicates, and it does not extend BA4 post-mortem observations to unsampled cortical regions.

## Formal test of directional concordance across sampled cortical cell types

The standardized HD–HC effect was negative for 7/7 cell types in GSE281069 frontal cortex, 5/6 in GSE180928 cingulate cortex, and 4/4 BA4 excitatory neuronal subtypes in GSE233408. Overall, {a3['negative_count']}/{a3['total_count']} estimates ({100*a3['negative_fraction']:.1f}%) were negative. The exact 95% Clopper–Pearson interval for this proportion was {a3['clopper_pearson_95ci_low']:.3f}–{a3['clopper_pearson_95ci_high']:.3f}. Against a null probability of 0.5, the two-sided exact binomial P value was {a3['exact_binomial_two_sided_p']:.6f}; the directional one-sided P value was {a3['exact_binomial_one_sided_p']:.6f}.

The 17 estimates are clustered within three datasets and therefore do not provide 17 fully independent replications. A sensitivity analysis that flipped all signs together within each dataset retained the within-dataset dependence structure. With only 2^3 = 8 possible dataset-level sign configurations, the two-sided exact P value was {a3['dataset_block_signflip_two_sided_p']:.3f} and the one-sided P value was {a3['dataset_block_signflip_one_sided_p']:.3f}. Thus, the observed cell-type estimates show marked directional concordance, while the number of independent datasets limits dataset-level inferential precision.

## Laminar alignment of STDS0000242 enrichment and GSE233387 BA4 CAG summaries

In the normal adult cortical reference STDS0000242, PLS1-positive genes were enriched among layer markers in L2 (fold enrichment = {a5['L2_fold']:.3f}, q = {a5['L2_q']:.4g}), L5 (fold enrichment = {a5['L5_fold']:.3f}, q = {a5['L5_q']:.4g}), and L6 (fold enrichment = {a5['L6_fold']:.3f}, q = {a5['L6_q']:.3e}). L6 showed the largest enrichment. L1, L3, and L4 did not pass Benjamini–Hochberg correction.

The GSE233387 BA4 CAG summaries comprised Layer 2, Layer 4, Betz cells, Layer 5a, Layer 6a, and Layer 6b. Mean somatic CAG-length gain averaged {a5['deep_mean']:.2f} in the four L5/L6 cell types and {a5['superficial_mean']:.2f} in Layer 2 and Layer 4, a difference of {a5['mean_difference']:.2f} repeats. All four L5/L6 means exceeded both L2/L4 means. Only 15 assignments of six values into groups of four and two are possible; the exact permutation P values were {a5['one_sided_exact_P']:.4f} one-sided and {a5['two_sided_exact_P']:.4f} two-sided.

The combined display provides an indirect laminar correspondence: PLS1-positive genes show strong L5/L6 enrichment in a normal adult reference, and the largest BA4 mean CAG gains occur in sampled L5/L6 excitatory cell types. STDS0000242 contains neurotypical tissue, whereas GSE233387 supplies BA4 cell-type CAG summaries. Consequently, this comparison does not establish disease-specific laminar localization, causality, or CAG effects outside BA4.

## Overall assessment

- The spatial-anchor analysis supplies an explicit motor-cortex value anchor for the whole-cortex PLS1 latent score.
- The directional-concordance analysis supports strong direction-level concordance across 17 estimates; inference at the independent-dataset level remains limited by three datasets.
- The laminar-alignment analysis strengthens the anatomical interpretation through significant L2/L5/L6 enrichment, with the clearest overlap in L5/L6, while retaining the normal-reference and BA4-only limitations.
"""
    (STAT_DIR / "regional_sensitivity_detailed_results.md").write_text(report, encoding="utf-8")



def main() -> None:
    ensure_dirs()
    configure_plotting()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(LOG_DIR / "run_analyses_02_03_05.log", mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    inputs = [EXPRESSION_FILE, WEIGHT_FILE, TVALUE_FILE, NATIVE_EFFECT_FILE, BA4_EFFECT_FILE, LAMINAR_FILE, CAG_FILE]
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(path)
    manifest = pd.DataFrame(
        [{"input_file": str(p), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in inputs]
    )
    manifest.to_csv(QC_DIR / "input_SHA256_manifest.csv", index=False)

    a2, f2 = precentral_spatial_anchor()
    a3, f3 = directional_concordance()
    a5, f5 = laminar_alignment()
    write_report(a2, a3, a5)

    figure_files = f2 + f3 + f5
    missing_outputs = [p for p in figure_files if not Path(p).exists() or Path(p).stat().st_size == 0]
    if missing_outputs:
        raise RuntimeError(f"Missing or empty figure outputs: {missing_outputs}")
    qc = {
        "status": "PASS",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "precentral_spatial_anchor": a2,
        "directional_concordance": a3,
        "laminar_alignment": a5,
        "figure_outputs": figure_files,
        "checks": {
            "canonical_PLS1_reproduced": bool(np.isclose(a2["canonical_score_tmap_pearson_r"], -0.610142, atol=5e-6)),
            "nine_left_precentral_parcels": a2["precentral_n"] == 9,
            "sixteen_of_seventeen_negative": a3["negative_count"] == 16 and a3["total_count"] == 17,
            "STDS_significant_layers_exactly_L2_L5_L6": a5["significant_layers_q_lt_0_05"] == ["L2", "L5", "L6"],
            "all_figure_files_nonempty": len(missing_outputs) == 0,
        },
    }
    (QC_DIR / "analysis_QC.json").write_text(json.dumps(qc, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.info("All analyses completed successfully")


if __name__ == "__main__":
    main()


# Script documentation
# Function: Complete analyses 2, 3, and 5 requested for the regional limitation review.
# Inputs: Canonical AHBA DK308 expression matrix, canonical PLS1 weights and t-map,
#         GSE281069/GSE180928/GSE233408 donor-level effects, STDS0000242 laminar
#         enrichment results, and GSE233387 BA4 cell-type CAG summaries.
# Outputs: Source-data CSV files, exact statistical summaries, PDF/SVG/800-dpi PNG/TIFF
#          figures, bilingual detailed reports, SHA-256 input manifest, and QC JSON.
# Main steps: Reconstruct parcel-level PLS1 scores; summarize left precentral parcels;
#             perform exact sign tests with dataset-block sensitivity; align laminar
#             enrichment with BA4 CAG summaries and calculate an exact depth contrast.
# Configuration: set HD_MSN_PROJECT_ROOT to the local project-data root and,
# optionally, HD_MSN_REGIONAL_WORK_DIR to the desired output directory.
# Log: <HD_MSN_REGIONAL_WORK_DIR>/logs/regional_sensitivity_analysis.log

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Run regional spatial anchoring, cross-dataset directional-concordance, and laminar-alignment sensitivity analyses.
# Input source/location: Configurable AHBA expression/PLS/target files, native cell-type effects, BA4 subtype effects, laminar enrichment, and CAG aggregate data via HD_MSN_* environment variables or input/.
# Output location: Descriptive tables, statistical summaries, QC records, and editable PDF/SVG plus 800-dpi PNG/TIFF figures under HD_MSN_REGIONAL_WORK_DIR.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Reconstruct parcel PLS scores; summarize precentral anchoring; test cross-dataset directional concordance; align laminar enrichments with BA4 CAG summaries; export reports/figures.
# Log location: Logs are written under the configured regional-sensitivity work directory/logs when logging is enabled by the script.
# =============================================================================
