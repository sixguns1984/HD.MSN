#!/usr/bin/env python3
"""Generate the BA4-restricted CAG/transcriptional-alignment figure and supplementary figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats


PINK = "#FEA3A2"
PURPLE = "#8E8BFE"
TEAL = "#52B6A8"
GOLD = "#E9B44C"
BLUE = "#5D8FD6"
GREEN = "#79B96E"
GRAY = "#9AA0A6"
DARK = "#202124"
LIGHT = "#EEF0F4"

CELL_COLORS = {
    "Layer 2": "#FDB6B3", "Layer 4": "#F58E8B", "Betz cells": "#D86778",
    "Layer 5a": "#A765A6", "Layer 6a": PURPLE, "Layer 6b": "#6969C7",
    "VIP": "#3FB8AF", "RELN": "#65C8A3", "LAMP5": "#4E9E8F", "PVALB": "#277F7B",
    "Astrocytes": "#E7B74C", "Microglia": "#D98E3D",
    "Oligodendrocytes": "#5D8FD6", "OPCs": "#86B7E6",
}

ORDER14 = ["Layer 2", "Layer 4", "Betz cells", "Layer 5a", "Layer 6a", "Layer 6b",
           "VIP", "RELN", "LAMP5", "PVALB", "Astrocytes", "Microglia",
           "Oligodendrocytes", "OPCs"]
ORDER4 = ["Betz cells", "Layer 5a", "Layer 6a", "Layer 6b"]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    return p.parse_args()


def setup() -> None:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 14,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "axes.edgecolor": DARK,
        "axes.linewidth": 0.8,
        "xtick.color": "black",
        "ytick.color": "black",
        "text.color": "black",
        "axes.labelcolor": "black",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })


def clean(ax: plt.Axes, grid: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color="#E5E7EB", linewidth=0.7, zorder=0)


def label(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.14, 1.09, letter, transform=ax.transAxes, fontsize=18,
            fontweight="bold", va="top", ha="left")


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def regress(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    slope, intercept = np.polyfit(x, y, 1)
    xx = np.linspace(np.min(x), np.max(x), 100)
    ax.plot(xx, intercept + slope * xx, color=color, linewidth=2.2, zorder=2)


def jitter(values: np.ndarray, seed: int, width: float = 0.09) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-width, width, len(values))


def main_figure(root: Path, out: Path) -> None:
    src = root / "figure_source_data"
    cag = pd.read_csv(src / "Pressl_BA4_CAG_by_cell_population.csv").set_index("target_cell_type").loc[ORDER14].reset_index()
    b = pd.read_csv(src / "GSE233387_BA4_excitatory_PLS1positive_vs_CAG.csv").set_index("target_cell_type").loc[ORDER14[:6]].reset_index()
    c = pd.read_csv(src / "GSE233408_BA4_deep_neuron_alignment.csv").set_index("target_cell_type").loc[ORDER4].reset_index()
    d = pd.read_csv(src / "GSE233408_BA4_donor_signed_PLS1_scores.csv")

    fig = plt.figure(figsize=(17.2, 13.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], width_ratios=[1.05, 0.95],
                          hspace=0.46, wspace=0.34)
    axa = fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])

    # A: full BA4 CAG landscape with multiple biologically meaningful colors.
    x = np.arange(len(cag))
    for i, row in cag.iterrows():
        color = CELL_COLORS[row.target_cell_type]
        axa.vlines(i, 0, row.cag_mean_somatic_length_gain, color=color, linewidth=2.5, zorder=2)
        axa.errorbar(i, row.cag_mean_somatic_length_gain, yerr=row.cag_sd, fmt="o",
                     color=color, mec="white", mew=0.9, ms=8, capsize=3, linewidth=1.3, zorder=3)
    axa.set_xticks(x, [x.replace("Oligodendrocytes", "Oligos") for x in ORDER14],
                   rotation=52, ha="right")
    axa.set_ylabel("Mean somatic CAG-length gain")
    axa.set_ylim(0, 28)
    axa.set_yticks(np.arange(0, 29, 4))
    axa.set_title("Motor-cortex CAG expansion across 14 cell types", loc="left", pad=13)
    clean(axa)
    label(axa, "A")
    handles = [Line2D([0], [0], marker="o", linestyle="", color=c, label=l, markersize=7)
               for l, c in [("Excitatory", "#D86778"), ("Inhibitory", TEAL), ("Glial", GOLD)]]
    axa.legend(handles=handles, frameon=False, ncol=3, loc="upper left",
               bbox_to_anchor=(0, 1.01), handletextpad=0.5, columnspacing=1.5)

    # B: primary six-excitatory association.
    xb = b.cag_mean_somatic_length_gain.to_numpy()
    yb = b.value.to_numpy()
    regress(axb, xb, yb, PURPLE)
    for _, row in b.iterrows():
        axb.scatter(row.cag_mean_somatic_length_gain, row.value, s=82,
                    color=CELL_COLORS[row.target_cell_type], edgecolor="white", linewidth=1, zorder=3)
        offsets = {"Layer 2": (8, 8), "Layer 4": (8, -20), "Betz cells": (-74, -18),
                   "Layer 5a": (-68, 10), "Layer 6a": (9, -20), "Layer 6b": (9, 10)}
        dx, dy = offsets[row.target_cell_type]
        axb.annotate(row.target_cell_type, (row.cag_mean_somatic_length_gain, row.value),
                     xytext=(dx, dy), textcoords="offset points", fontsize=13)
    axb.set_xlabel("Mean somatic CAG-length gain")
    axb.set_ylabel("PLS1-positive expression score")
    axb.set_xlim(7.5, 23.3)
    axb.set_title("GSE233387: six BA4 excitatory neuronal subtypes", loc="left", pad=13)
    axb.text(0.03, 0.04, r"Spearman $\rho$ = -0.886" + "\n" +
             "exact P = 0.033; Q = 0.300", transform=axb.transAxes, va="bottom",
             fontsize=13, bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D1D5DB"))
    clean(axb)
    label(axb, "B")

    # C: direct four-population alignment: baseline expression plus disease direction.
    pos = np.arange(4)
    axc.axvline(0, color="#B8BCC4", linewidth=1)
    cag_offsets = {
        "Betz cells": (0.014, -0.16, "left"),
        "Layer 5a": (0.014, -0.18, "left"),
        "Layer 6a": (-0.014, -0.16, "right"),
        "Layer 6b": (0.014, -0.18, "left"),
    }
    for i, row in c.iterrows():
        axc.plot([0, row.pls1_positive_expression_score], [i, i], color=CELL_COLORS[row.target_cell_type], linewidth=3)
        axc.scatter(row.pls1_positive_expression_score, i, s=70 + 3.2 * row.cag_mean_somatic_length_gain,
                    color=CELL_COLORS[row.target_cell_type], edgecolor="white", linewidth=1, zorder=3)
        dx, dy, ha = cag_offsets[row.target_cell_type]
        axc.text(row.pls1_positive_expression_score + dx, i + dy,
                 f"CAG {row.cag_mean_somatic_length_gain:.1f}", ha=ha, fontsize=13)
    axc.set_yticks(pos, ORDER4)
    axc.invert_yaxis()
    axc.set_xlabel("PLS1-positive expression score")
    axc.set_title("GSE233408: four directly matched BA4 neuronal subtypes", loc="left", pad=13)
    axc.set_xlim(-0.115, 0.145)
    axc.text(0.00, -0.28, r"Baseline score vs CAG: $\rho$ = 1.00, exact P = 0.083" + "\n" +
             r"Disease-direction z vs CAG: $\rho$ = 1.00, exact P = 0.083",
             transform=axc.transAxes, va="top", fontsize=13, clip_on=False,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D1D5DB"))
    clean(axc, "x")
    label(axc, "C")

    # D: donor-level signed score; points are actual donors, not summaries.
    xloc = {name: i for i, name in enumerate(ORDER4)}
    for i, ct in enumerate(ORDER4):
        for j, cond in enumerate(["CTRL", "HD"]):
            vals = d[(d.target_cell_type == ct) & (d.condition == cond)].signed_pls1_score.to_numpy()
            center = i + (-0.17 if cond == "CTRL" else 0.17)
            color = PURPLE if cond == "CTRL" else PINK
            xx = center + jitter(vals, 1700 + i * 10 + j)
            axd.scatter(xx, vals, s=26, color=color, alpha=0.82, edgecolor="white", linewidth=0.45, zorder=3)
            mean = np.mean(vals)
            sem = stats.sem(vals) if len(vals) > 1 else 0
            axd.errorbar(center, mean, yerr=sem, color="black", fmt="_", markersize=15,
                         capsize=3, linewidth=1.2, zorder=4)
        q = float(c.loc[c.target_cell_type == ct, "global_q_all_eligible_region_celltype_sets"].iloc[0])
        ytop = max(d[d.target_cell_type == ct].signed_pls1_score.max(), 0.12) + 0.025
        axd.text(i, ytop, f"Q = {q:.3g}", ha="center", va="bottom", fontsize=13)
    axd.axhline(0, color="#B8BCC4", linewidth=0.9)
    axd.set_xticks(np.arange(4), ORDER4, rotation=20, ha="right")
    axd.set_ylabel("Signed PLS1 expression score")
    axd.set_ylim(-0.145, 0.235)
    axd.set_title("GSE233408: donor-level BA4 disease differences", loc="left", pad=13)
    axd.legend(handles=[Line2D([0], [0], marker="o", linestyle="", color=PURPLE, label="Control"),
                        Line2D([0], [0], marker="o", linestyle="", color=PINK, label="HD")],
               frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    clean(axd)
    label(axd, "D")

    save(fig, out / "main" / "BA4_CAG_transcriptional_alignment")


def coverage_figure(root: Path, out: Path) -> None:
    src = root / "figure_source_data"
    meta = pd.read_csv(src / "GSE233387_BA4_donor_cell_counts.csv")
    mapping = pd.read_csv(src / "GSE233387_Pressl_population_mapping.csv")
    donors = meta.donor.drop_duplicates().tolist()
    conditions = meta.drop_duplicates("donor").set_index("donor").loc[donors].condition
    mat = meta.pivot(index="donor", columns="target_cell_type", values="n_cells").reindex(index=donors, columns=ORDER14).fillna(0)
    passes = meta.pivot(index="donor", columns="target_cell_type", values="passes_min_cells").reindex(index=donors, columns=ORDER14).fillna(False)

    fig, (a, b) = plt.subplots(1, 2, figsize=(17.5, 7.3),
                              gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.34})
    for iy, donor in enumerate(donors):
        for ix, ct in enumerate(ORDER14):
            n = mat.loc[donor, ct]
            if n > 0:
                color = PURPLE if conditions.loc[donor] == "CTRL" else PINK
                a.scatter(ix, iy, s=15 + 30 * np.log10(n + 1), color=color,
                          alpha=1.0 if passes.loc[donor, ct] else 0.25,
                          edgecolor="white", linewidth=0.4)
    a.set_xticks(range(14), [x.replace("Oligodendrocytes", "Oligos") for x in ORDER14], rotation=50, ha="right")
    a.set_yticks(range(len(donors)), donors)
    a.invert_yaxis()
    a.set_title("GSE233387 BA4 nuclei retained by donor and mapped cell type", loc="left", pad=16)
    a.legend(handles=[Line2D([0], [0], marker="o", linestyle="", color=PURPLE, label="Control"),
                      Line2D([0], [0], marker="o", linestyle="", color=PINK, label="HD")],
             frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    clean(a, None); label(a, "A")

    map_count = mapping.groupby(["target_cell_type", "pressl_sfANS_population"]).size().reset_index(name="n_subtypes")
    rows = ORDER14
    sfans = map_count.pressl_sfANS_population.drop_duplicates().tolist()
    for _, row in map_count.iterrows():
        iy = rows.index(row.target_cell_type); ix = sfans.index(row.pressl_sfANS_population)
        b.scatter(ix, iy, s=65 + 35 * row.n_subtypes, color=CELL_COLORS[row.target_cell_type],
                  edgecolor="white", linewidth=0.6)
    b.set_xticks(range(len(sfans)), sfans, rotation=50, ha="right")
    b.set_yticks(range(len(rows)), rows)
    b.invert_yaxis()
    b.set_title("Exclusive mapping to 14 Pressl BA4 CAG-defined cell types", loc="left", pad=16)
    clean(b, None); label(b, "B")
    save(fig, out / "supplementary" / "BA4_mapping_and_coverage")


def score_tests_figure(root: Path, out: Path) -> None:
    src = root / "figure_source_data"
    t = pd.read_csv(src / "nine_score_CAG_permutation_tests.csv")
    null = np.load(root / "analysis_results" / "score_cag_null_distributions.npz")
    fig = plt.figure(figsize=(17.2, 11.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.15], hspace=0.46, wspace=0.42)
    a = fig.add_subplot(gs[:, 0]); b = fig.add_subplot(gs[0, 1]); c = fig.add_subplot(gs[1, 1])

    score_short = {"Signed PLS1 expression score": "Signed",
                   "PLS1-positive expression score": "PLS1-positive",
                   "PLS1-negative expression score": "PLS1-negative"}
    fam_short = {"Six cortical excitatory populations": "Excitatory neuronal subtypes (6)",
                 "Ten neuronal populations": "Neuronal cell types (10)",
                 "All 14 populations, broad-class restricted permutation": "All mapped cell types (14)"}
    t["label"] = t.analysis.map(fam_short) + "  |  " + t.score.map(score_short)
    y = np.arange(len(t))
    colors = [PINK if "positive" in s else PURPLE if "negative" in s else GRAY for s in t.score]
    a.axvline(0, color="#B8BCC4", linewidth=1)
    for i, row in t.iterrows():
        a.plot([0, row.spearman_rho], [i, i], color=colors[i], linewidth=2.5)
        a.scatter(row.spearman_rho, i, s=70, color=colors[i], edgecolor="white", linewidth=0.8, zorder=3)
        a.text(1.58, i,
               f"P={row.permutation_p_two_sided:.3f}  Q={row.q_across_nine_score_tests:.3f}",
               ha="right", va="center", fontsize=13,
               bbox=dict(boxstyle="square,pad=0.08", fc="white", ec="none", alpha=0.92))
    a.set_yticks(y, t.label); a.invert_yaxis(); a.set_xlim(-1.08, 1.68)
    a.set_xlabel("Spearman ρ")
    a.set_title("CAG association across prespecified score tests", loc="left")
    clean(a, "x"); label(a, "A")

    key = "Six_cortical_excitatory_populations_|_PLS1-positive_expression_score"
    vals = null[key]
    b.hist(vals, bins=np.linspace(-1, 1, 32), color="#D9D7FA", edgecolor="white")
    b.axvline(-0.885714, color=PURPLE, linewidth=2.5)
    b.axvline(0.885714, color=PURPLE, linewidth=1.1, linestyle="--")
    b.set_xlabel("Permuted Spearman ρ"); b.set_ylabel("Permutation count")
    b.set_title("Exact null: six excitatory neuronal subtypes (720 permutations)", loc="left")
    clean(b); label(b, "B")

    key2 = "Ten_neuronal_populations_|_PLS1-negative_expression_score"
    v2 = null[key2]
    c.hist(v2[::500], bins=np.linspace(-1, 1, 32), color="#D9D7FA", edgecolor="white")
    c.axvline(-0.575758, color=PURPLE, linewidth=2.5)
    c.set_xlabel("Permuted Spearman ρ"); c.set_ylabel("Subsampled permutation count")
    c.set_title("Monte Carlo null: ten neuronal cell types", loc="left")
    clean(c); label(c, "C")
    save(fig, out / "supplementary" / "score_tests_and_permutation_nulls")


def sensitivity_figure(root: Path, out: Path) -> None:
    s = pd.read_csv(root / "figure_source_data" / "score_CAG_sensitivity.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(16.2, 6.6),
                              gridspec_kw={"width_ratios": [1.4, 0.8], "wspace": 0.40})
    core = s[(s.family.isin(["excitatory6", "neuronal10", "all14"]))
             & s.score.eq("PLS1-positive expression score")].copy()
    fam_order = ["excitatory6", "neuronal10", "all14"]
    sens_order = ["CTRL_zscore", "CTRL_rank", "HD_zscore", "HD_rank", "ALL_zscore", "ALL_rank"]
    for j, fam in enumerate(fam_order):
        for i, ss in enumerate(sens_order):
            row = core[(core.family == fam) & (core.sensitivity == ss)].iloc[0]
            xpos = i + (j - 1) * 0.2
            a.scatter(xpos, row.spearman_rho, s=58, color=[PINK, PURPLE, TEAL][j],
                      edgecolor="white", linewidth=0.7, zorder=3)
    a.axhline(0, color="#B8BCC4", linewidth=1)
    a.set_xticks(range(6), ["CTRL\nz", "CTRL\nrank", "HD\nz", "HD\nrank", "All\nz", "All\nrank"])
    a.set_ylabel("Spearman ρ with CAG gain")
    a.set_ylim(-1.05, 1.05)
    a.set_title("Expression-summary and cell-type-set sensitivity", loc="left")
    a.legend(handles=[Line2D([0], [0], marker="o", linestyle="", color=PINK, label="Excitatory neuronal subtypes (6)"),
                      Line2D([0], [0], marker="o", linestyle="", color=PURPLE, label="Neuronal cell types (10)"),
                      Line2D([0], [0], marker="o", linestyle="", color=TEAL, label="All mapped cell types (14)")],
             frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.20))
    clean(a); label(a, "A")

    loo = s[s.family.eq("excitatory6_LOO") &
            s.score.eq("PLS1-positive expression score")].copy()
    names = loo.sensitivity.str.replace("omit_", "", regex=False).tolist()
    b.barh(np.arange(len(loo)), loo.spearman_rho, color=[CELL_COLORS[n] for n in names])
    b.axvline(-0.885714, color="black", linestyle="--", linewidth=1.1,
              label="Full six-subtype estimate")
    b.set_yticks(range(len(loo)), [f"Omit {n}" for n in names]); b.invert_yaxis()
    b.set_xlabel("Spearman ρ")
    b.set_title("Leave-one-neuronal-subtype-out: PLS1-positive score", loc="left")
    b.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.16))
    clean(b, "x"); label(b, "B")
    save(fig, out / "supplementary" / "score_CAG_sensitivity")


def gene_figure(root: Path, out: Path) -> None:
    g = pd.read_csv(root / "figure_source_data" / "gene_level_CAG_permutation_results.csv.gz")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(18.6, 7.4),
                                 gridspec_kw={"width_ratios": [1.25, 1.25, 1.0], "wspace": 0.42})
    for ax, fam, title in [(a, "neuronal10", "Ten neuronal cell types"),
                           (b, "all14_broad_class_adjusted", "All 14 neuronal and glial cell types\n(class-restricted permutations)")]:
        d = g[g.family.eq(fam)].copy()
        d["mlogq"] = -np.log10(d.global_bh_q.clip(lower=1e-300))
        sig = d.global_bh_q < 0.05
        colors = np.where(d.gene_set.eq("PLS1+"), PINK, PURPLE)
        ax.scatter(d.spearman_rho, d.mlogq, color=colors, alpha=np.where(sig, 0.95, 0.42),
                   s=np.where(sig, 30, 17), edgecolor="none")
        ax.axhline(-math.log10(0.05), color="black", linestyle="--", linewidth=1)
        ax.axvline(0, color="#B8BCC4", linewidth=0.8)
        if fam == "neuronal10":
            label_genes = ["ASS1", "NT5M", "CCDC136", "PTCHD1", "ADTRP", "RASD2"]
        else:
            label_genes = ["NCOA3", "B3GALT6", "NPHS1", "PTCHD1", "LYRM9"]
        top = d[d.gene.isin(label_genes)]
        if fam == "neuronal10":
            positions = {
                "ASS1": (-0.93, 1.69), "NT5M": (-0.93, 1.60),
                "CCDC136": (-0.66, 1.69), "RASD2": (-0.66, 1.55),
                "PTCHD1": (0.76, 1.69), "ADTRP": (0.76, 1.60),
            }
        else:
            positions = {
                "NCOA3": (-0.76, 0.055), "B3GALT6": (-0.57, 0.09),
                "NPHS1": (0.54, 0.13), "PTCHD1": (0.74, 0.055),
                "LYRM9": (0.84, 0.09),
            }
        for _, r in top.iterrows():
            tx, ty = positions[r.gene]
            ax.annotate(r.gene, (r.spearman_rho, r.mlogq), xytext=(tx, ty),
                        textcoords="data", fontsize=13,
                        arrowprops=dict(arrowstyle="-", color="#777777", lw=0.55))
        ax.set_xlabel("Gene-level Spearman ρ")
        ax.set_ylabel("−log10(global BH Q)")
        ax.set_title(title, loc="left")
        clean(ax); label(ax, "A" if fam == "neuronal10" else "B")
    hits = g[(g.family == "neuronal10") & (g.global_bh_q < 0.05)].sort_values("spearman_rho")
    c.barh(np.arange(len(hits)), hits.spearman_rho,
           color=[PINK if x == "PLS1+" else PURPLE for x in hits.gene_set])
    c.set_yticks(range(len(hits)), hits.gene)
    c.axvline(0, color="#B8BCC4", linewidth=0.8)
    c.set_xlabel("Spearman ρ")
    c.set_title("17 neuron-only candidates\n(global BH Q < 0.05)", loc="left")
    clean(c, "x"); label(c, "C")
    c.legend(handles=[Line2D([0], [0], color=PINK, lw=5, label="PLS1-positive"),
                      Line2D([0], [0], color=PURPLE, lw=5, label="PLS1-negative")],
             frameon=False, loc="upper left", bbox_to_anchor=(0, -0.14))
    save(fig, out / "supplementary" / "gene_level_CAG_sensitivity")


def exact4_figure(root: Path, out: Path) -> None:
    src = root / "figure_source_data"
    c = pd.read_csv(src / "GSE233408_BA4_deep_neuron_alignment.csv").set_index("target_cell_type").loc[ORDER4].reset_index()
    t = pd.read_csv(src / "GSE233408_BA4_exact_permutation_tests.csv")
    n = pd.read_csv(src / "GSE233408_exact_null_distributions.csv")
    fig, axs = plt.subplots(2, 3, figsize=(18.2, 12.0),
                           gridspec_kw={"hspace": 0.43, "wspace": 0.38})
    measures = [
        ("Signed PLS1 expression score", "signed_pls1_expression_score", "Signed expression score"),
        ("PLS1-positive expression score", "pls1_positive_expression_score", "PLS1-positive score"),
        ("PLS1-negative expression score", "pls1_negative_expression_score", "PLS1-negative score"),
        ("HD-control signed-score difference", "hd_minus_control_signed_score", "HD − control signed score"),
        ("PLS1-positive directional camera z", "directional_z", "Directional gene-set z"),
    ]
    stat_positions = {
        0: (0.97, 0.04, "right", "bottom"),
        1: (0.03, 0.97, "left", "top"),
        2: (0.97, 0.97, "right", "top"),
        3: (0.97, 0.97, "right", "top"),
        4: (0.03, 0.97, "left", "top"),
    }
    for i, (measure, col, title) in enumerate(measures):
        ax = axs.flat[i]
        x = c.cag_mean_somatic_length_gain.to_numpy(); y = c[col].to_numpy()
        regress(ax, x, y, PINK if "positive" in measure else PURPLE)
        for _, r in c.iterrows():
            ax.scatter(r.cag_mean_somatic_length_gain, r[col], s=75,
                       color=CELL_COLORS[r.target_cell_type], edgecolor="white", linewidth=0.8, zorder=3)
            ax.annotate(r.target_cell_type, (r.cag_mean_somatic_length_gain, r[col]),
                        xytext=(7, 6), textcoords="offset points", fontsize=13)
        tr = t[t.measure.eq(measure)].iloc[0]
        tx, ty, ha, va = stat_positions[i]
        ax.text(tx, ty, f"ρ={tr.spearman_rho:.2f}; exact P={tr.complete_permutation_p_two_sided:.3f}",
                transform=ax.transAxes, fontsize=13, ha=ha, va=va,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#D1D5DB", alpha=0.9))
        ax.set_xlabel("Mean somatic CAG-length gain"); ax.set_ylabel(title)
        ax.set_title(title, loc="left")
        clean(ax); label(ax, chr(65 + i))
    f = axs.flat[5]
    null_measure = "PLS1-positive_directional_camera_z"
    vals = n[n.measure.eq(null_measure)].null_spearman_rho
    counts = vals.value_counts().sort_index()
    f.bar(counts.index, counts.values, width=0.13, color="#D9D7FA", edgecolor="white")
    f.axvline(1.0, color=PINK, linewidth=2.5)
    f.set_xlabel("Exact-null Spearman ρ"); f.set_ylabel("Permutation count")
    f.set_title("Four-neuronal-subtype exact null (24 permutations)", loc="left")
    clean(f); label(f, "F")
    save(fig, out / "supplementary" / "GSE233408_four_population_exact_tests")


def no_enrichment_figure(root: Path, out: Path) -> None:
    meta = json.loads((root / "analysis_results" / "neuronal10_17gene_gprofiler_metadata.json").read_text())
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.axis("off")
    ax.text(0.5, 0.69, "No robust pathway-level enrichment", ha="center", va="center",
            fontsize=17, fontweight="bold")
    ax.text(0.5, 0.48,
            "The 17 neuron-only candidate genes were tested with g:Profiler\n"
            "against the 485 tested PLS1-positive/negative genes.\n"
            "No GO, Reactome, or WikiPathways term met FDR < 0.05.",
            ha="center", va="center", fontsize=13, linespacing=1.5)
    ax.text(0.5, 0.18, "This result does not support a pathway-enrichment panel in the main figure.",
            ha="center", va="center", fontsize=13, color="#555555")
    save(fig, out / "supplementary" / "pathway_enrichment_negative_result")


def main() -> None:
    setup()
    root = Path(args().root).resolve()
    out = root / "figures"
    main_figure(root, out)
    coverage_figure(root, out)
    score_tests_figure(root, out)
    sensitivity_figure(root, out)
    gene_figure(root, out)
    exact4_figure(root, out)
    print(json.dumps({"output": str(out), "rendered_files": len(list(out.rglob('*.*')))}, indent=2))


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Generate the composite BA4 transcriptional/CAG alignment visualization from prepared source-data tables.
# Input source/location: Prepared BA4 CAG figure source-data tables from the configured BA4 results directory.
# Output location: Editable/vector and high-resolution raster BA4 CAG visualization files in the configured figure directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Load source-data panels; validate required fields; construct panel layouts; apply publication plotting settings; save vector and raster outputs.
# Log location: No dedicated log file; plotting errors are raised to the caller.
# =============================================================================
