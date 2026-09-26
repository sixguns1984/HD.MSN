from __future__ import annotations

from pathlib import Path
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from private_clinical_schema import indicator_column, indicator_id, load_private_clinical_schema


ROOT = Path(os.environ.get("HD_MSN_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA = Path(os.environ.get("HD_MSN_CLINICAL_INPUT", ROOT / "data" / "clinical_associations"))
OUT = Path(os.environ.get("HD_MSN_CLINICAL_FIGURE_DIR", ROOT / "results" / "clinical_associations" / "figures")).resolve()
STEM = OUT / "clinical_associations"

PURPLE = "#8E8BFE"
SALMON = "#FEA3A2"
INK = "#25262B"
MUTED = "#666A73"
GRID = "#DADCE3"


def configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 14,
            "axes.titlesize": 13,
            "axes.labelsize": 13,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ms = pd.read_csv(DATA / "HD_regional_ms.csv", index_col=0)
    clinical_path = Path(os.environ.get("HD_MSN_HD_CLINICAL_FILE", DATA / "hd_clinical.xlsx"))
    clinical = pd.read_excel(clinical_path)
    clinical.columns = [str(c).strip() for c in clinical.columns]
    id_column = os.environ.get("HD_MSN_HD_CLINICAL_ID_COLUMN", "NUM")
    if id_column not in clinical.columns:
        raise KeyError("Configured de-identified clinical subject-ID column is absent.")
    clinical = clinical.set_index(id_column)
    stats = pd.read_csv(DATA / "all_four_covariate_clinical_associations.csv")
    return ms, clinical, stats


def load_panel_specs() -> list[dict]:
    """Load public-safe panel structure and private clinical bindings from the external schema."""
    schema = load_private_clinical_schema()
    panels = schema.get("figure_panels")
    if not isinstance(panels, list) or not panels:
        raise ValueError("Private schema requires a non-empty figure_panels list.")
    return panels


def data_for(
    ms: pd.DataFrame,
    clinical: pd.DataFrame,
    region: str,
    source_column: str,
) -> pd.DataFrame:
    common = ms.columns.intersection(clinical.index)
    x = pd.to_numeric(ms.loc[region, common], errors="coerce")
    y = pd.to_numeric(clinical.loc[common, source_column], errors="coerce")
    return pd.DataFrame({"ms": x, "outcome": y}).dropna()


def plot_panel(
    ax: plt.Axes,
    spec: dict,
    ms: pd.DataFrame,
    clinical: pd.DataFrame,
    stats: pd.DataFrame,
    schema: dict,
) -> dict:
    index = int(spec["indicator_index"])
    public_id = indicator_id(index)
    source_column = indicator_column(index, schema)
    region = str(spec["region"])
    region_label = str(spec["region_label"])
    ylabel = str(spec.get("display_label", public_id))
    letter = str(spec["panel"])
    direction = str(spec.get("direction", ""))
    ylim = tuple(float(x) for x in spec["ylim"])
    major = float(spec["major_tick"])
    ceiling = spec.get("ceiling")

    if source_column not in clinical.columns:
        raise KeyError(f"Private source column for {public_id} is absent from the clinical table.")

    frame = data_for(ms, clinical, region, source_column)
    x = frame["ms"].to_numpy(float)
    y = frame["outcome"].to_numpy(float)
    q_row = stats[
        (stats["Region"] == region)
        & (stats["Clinical_Indicator_ID"] == public_id)
    ]
    if q_row.empty:
        raise KeyError(f"Missing four-covariate association result for {region} / {public_id}.")

    adjusted_r = float(q_row["R"].iloc[0])
    adjusted_p = float(q_row["P"].iloc[0])
    q = float(q_row["P_corrected"].iloc[0])
    color = SALMON if adjusted_r > 0 else PURPLE

    ax.scatter(x, y, s=27, color=color, alpha=0.74, edgecolor="white", linewidth=0.45, zorder=3)
    coef = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 200)
    ax.plot(xx, np.polyval(coef, xx), color=INK, lw=1.65, zorder=4)

    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(MultipleLocator(major))
    pad = max((x.max() - x.min()) * 0.05, 0.002)
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_xlabel(f"Regional MS: {region_label}")
    ax.set_ylabel(ylabel)
    ax.text(-0.16, 1.07, letter, transform=ax.transAxes, fontsize=15, fontweight="bold", color=INK, va="top")

    if ceiling is not None:
        ceiling = float(ceiling)
        ax.text(0.02, 1.025, direction, transform=ax.transAxes, va="bottom", ha="left", fontsize=13, color=INK, fontweight="bold", clip_on=False)
        ax.text(0.98, 1.025, f"Protocol ceiling: {ceiling:g}", transform=ax.transAxes, va="bottom", ha="right", fontsize=13, color=INK, clip_on=False)
        stat_y, stat_va = 0.06, "bottom"
        ax.axhline(ceiling, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2, clip_on=False)
    else:
        ax.text(0.02, 0.97, direction, transform=ax.transAxes, va="top", ha="left", fontsize=13, color=INK, fontweight="bold")
        stat_y, stat_va = 0.97, "top"

    ax.text(
        0.98,
        stat_y,
        f"partial r = {adjusted_r:.3f}\nq = {q:.3f}",
        transform=ax.transAxes,
        va=stat_va,
        ha="right",
        fontsize=13,
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.5},
    )

    ax.grid(axis="both", color=GRID, lw=0.55, alpha=0.72)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    return {
        "panel": letter,
        "region": region,
        "clinical_indicator_id": public_id,
        "n": len(frame),
        "partial_r_four_covariates": adjusted_r,
        "p_four_covariates": adjusted_p,
        "q_four_covariates": q,
        "covariates": "age;sex;education_years;eTIV",
        "raw_scatter_for_display_only": True,
        "y_min": y.min(),
        "y_max": y.max(),
    }


def main() -> None:
    configure()
    schema = load_private_clinical_schema()
    panels = load_panel_specs()
    ms, clinical, stats = load_data()

    n_panels = len(panels)
    n_cols = int(schema.get("figure_columns", 2))
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9.3, 3.7 * n_rows))
    axes_array = np.atleast_1d(axes).ravel()
    fig.subplots_adjust(left=0.105, right=0.975, top=0.93, bottom=0.075, wspace=0.31, hspace=0.33)

    records = []
    for ax, spec in zip(axes_array, panels):
        records.append(plot_panel(ax, spec, ms, clinical, stats, schema))
    for ax in axes_array[len(panels):]:
        ax.set_visible(False)

    fig.suptitle(
        "Regional morphometric similarity and clinical measures: four-covariate inference",
        x=0.105,
        y=0.975,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(STEM.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(STEM.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    pd.DataFrame(records).to_csv(OUT / "clinical_association_statistics.csv", index=False)
    plt.close(fig)
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Build clinical-association visualizations from four-covariate MSN inference while keeping private clinical field names outside the public source tree.
# Input source/location: Clinical-association tables, private clinical metadata, and the external private clinical schema from configurable paths.
# Output location: Clinical-association figure files and source-data tables under HD_MSN_CLINICAL_FIGURE_DIR or results/clinical_associations/figures.
# Input/output notes: The public script uses anonymous clinical IDs. Private source-column names and display labels are resolved only at runtime from an external schema.
# Main steps: Load private panel bindings; read four-covariate association estimates; display descriptive scatter points; annotate adjusted partial-r/FDR statistics; export vector and high-resolution raster products.
# Log location: No dedicated log file; errors are raised to the caller.
# =============================================================================
