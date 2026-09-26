#!/usr/bin/env python3
"""Plot the analysis-derived GSE233387 standardized HD-control differences."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "04_figures"
FIGURES.mkdir(parents=True, exist_ok=True)

effects = pd.read_csv(RESULTS / "GSE233387_HD_control_effects.csv")
omnibus = pd.read_csv(RESULTS / "GSE233387_omnibus_interaction.csv").iloc[0]
cell_order = [
    "Layer 2", "Layer 4", "Betz cells", "Layer 5a", "Layer 6a", "Layer 6b",
    "VIP", "RELN", "LAMP5", "PVALB", "Astrocytes", "Microglia",
    "Oligodendrocytes", "OPCs",
]
effects = effects.set_index("cell_type").loc[cell_order].reset_index()
effects["cell_class"] = ["Excitatory neurons"] * 6 + ["Inhibitory neurons"] * 4 + ["Glial"] * 4

colors = {"Excitatory neurons": "#ff9896", "Inhibitory neurons": "#8E8BFE", "Glial": "#c5b0d5"}
markers = {"Excitatory neurons": "o", "Inhibitory neurons": "s", "Glial": "^"}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 14,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

fig, ax = plt.subplots(figsize=(9.2, 8.6))
ci_min = float(effects.ci_low.min())
ci_max = float(effects.ci_high.max())
span = ci_max - ci_min
q_x = ci_max + 0.11 * span
x_min = ci_min - 0.04 * span
x_max = ci_max + 0.34 * span

for i, row in effects.iterrows():
    cls = row.cell_class
    ax.plot([row.ci_low, row.ci_high], [i, i], color=colors[cls], lw=2.4, alpha=0.9, solid_capstyle="round")
    ax.scatter(row.standardized_HD_control_difference, i, s=92, marker=markers[cls],
               facecolor=colors[cls], edgecolor="black", linewidth=1.0, zorder=3)
    q = float(row.camera_BH_q_value)
    label = f" = {q:.3f}" if q >= 0.001 else f" = {q:.1e}"
    ax.text(q_x, i, "q", va="center", ha="left", fontsize=13, color="black",
            fontfamily="Arial", fontstyle="italic")
    ax.annotate(label, xy=(q_x, i), xytext=(8, 0), textcoords="offset points",
                va="center", ha="left", fontsize=13, color="black",
                fontfamily="Arial")

ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
ax.set_yticks(np.arange(len(effects)), effects.cell_type.tolist())
ax.invert_yaxis()
ax.set_xlim(x_min, x_max)
ax.set_xlabel("Standardized HD–control difference")
title = HPacker(children=[
    TextArea("GSE233387 BA4 cell types   HD × cell type: ",
             textprops={"fontsize": 16, "fontfamily": "Arial", "color": "black"}),
    TextArea("P", textprops={"fontsize": 16, "fontfamily": "Arial",
                              "fontstyle": "italic", "color": "black"}),
    TextArea(f" = {float(omnibus.p_value):.3f}",
             textprops={"fontsize": 16, "fontfamily": "Arial", "color": "black"}),
], align="center", pad=0, sep=0)
title_box = AnchoredOffsetbox(loc="lower center", child=title, pad=0, frameon=False,
                              bbox_to_anchor=(0.5, 1.025), bbox_transform=ax.transAxes,
                              borderpad=0)
ax.add_artist(title_box)
ax.grid(axis="x", color="#E5E5E5", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for name in ("left", "bottom"):
    ax.spines[name].set_linewidth(1.1)
ax.tick_params(length=5, width=1.0, color="black")

handles = [
    plt.Line2D([], [], marker=markers[cls], linestyle="None", markersize=9,
               markerfacecolor=colors[cls], markeredgecolor="black", label=cls)
    for cls in ("Excitatory neurons", "Inhibitory neurons", "Glial")
]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, 1.105), ncol=3,
          frameon=False, handlelength=0.8, columnspacing=1.3, handletextpad=0.45)
fig.text(0.015, 0.985, "B", ha="left", va="top", fontsize=20, fontweight="bold")
fig.subplots_adjust(left=0.26, right=0.95, bottom=0.11, top=0.79)

base = FIGURES / "GSE233387_standardized_HD_control_difference"
fig.savefig(base.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
fig.savefig(base.with_suffix(".svg"), dpi=600, bbox_inches="tight")
fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight")
fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)

# Function: plot the 14-cell-type standardized HD-control differences and CAMERA q values.
# Inputs: analysis-derived effect table and donor-fixed-effect omnibus interaction result.
# Outputs: editable PDF/SVG and 600-dpi PNG/TIFF publication-quality candidates.
# Main steps: order cell types, draw effects with 95% CIs, annotate q values and omnibus P.
# Logs: no separate log is produced; plotting errors are returned by the calling process.

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Plot GSE233387 HD-control cell-type effect estimates with confidence intervals and statistical annotations.
# Input source/location: Cell-type effect/model result tables produced by the accompanying R analysis script.
# Output location: Editable/vector and high-resolution raster effect plots plus any plot-source tables in the local figure directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Load model results; order cell types; draw effect estimates and 95% confidence intervals; annotate q values/omnibus statistics; export figures.
# Log location: No dedicated log file; failures are raised to the caller.
# =============================================================================
