#!/usr/bin/env python3
"""Cell-type-aggregate bootstrap confidence interval and power analysis."""

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
from scipy.stats import norm, rankdata, spearmanr, t


SEED = 20260911
BOOTSTRAP_B = 20_000
PERMUTATIONS = 1_000_000
COARSE_SIMULATIONS = 20_000
FINE_SIMULATIONS = 50_000
OBSERVED_EFFECT_SIMULATIONS = 100_000
N_UNITS = 14
ALPHA = 0.05

SCRIPT = Path(__file__).resolve()
ROOT = SCRIPT.parents[1]
INPUT = ROOT / "input" / "GSE233387_14celltype_PLS1positive_CAG_final.csv"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
STATISTICS = ROOT / "results" / "statistics"
LOGS = ROOT / "logs"
LOG_FILE = LOGS / "cag_bootstrap_power.log"

X_COLUMN = "pls1_positive_expression_score"
Y_COLUMN = "cag_mean_somatic_length_gain"
CELL_COLUMN = "target_cell_type"
CANONICAL_CELL_ORDER = [
    "Layer 2", "Layer 4", "Betz cells", "Layer 5a", "Layer 6a", "Layer 6b",
    "VIP", "RELN", "LAMP5", "PVALB", "Astrocytes", "Microglia", "Oligodendrocytes", "OPCs",
]


def configure_logging() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"), logging.StreamHandler()],
    )
    logging.getLogger("fontTools").setLevel(logging.WARNING)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def monte_carlo_permutation_p(
    x: np.ndarray, y: np.ndarray, observed: float
) -> tuple[float, int, float, float]:
    """Reproduce the primary permutation test and estimate its alpha-level critical value."""
    rng = np.random.default_rng(20260827)
    x_rank = rankdata(x)
    y_rank = rankdata(y)
    x_centered = x_rank - x_rank.mean()
    denominator = math.sqrt(float(np.sum(x_centered**2) * np.sum((y_rank - y_rank.mean()) ** 2)))
    extreme = 0
    generated = 0
    chunk = 20_000
    absolute_null_values: list[np.ndarray] = []
    while generated < PERMUTATIONS:
        size = min(chunk, PERMUTATIONS - generated)
        keys = rng.random((size, len(y_rank)))
        order = np.argsort(keys, axis=1)
        permuted_y = y_rank[order] - y_rank.mean()
        rho = (permuted_y @ x_centered) / denominator
        absolute_null_values.append(np.abs(rho))
        extreme += int(np.count_nonzero(np.abs(rho) >= abs(observed) - 1e-15))
        generated += size
    absolute_null = np.concatenate(absolute_null_values)
    unique_values, counts = np.unique(absolute_null, return_counts=True)
    tail_counts = np.cumsum(counts[::-1])[::-1]
    tail_probabilities = (tail_counts + 1) / (PERMUTATIONS + 1)
    eligible = np.flatnonzero(tail_probabilities < ALPHA)
    if len(eligible) == 0:
        raise RuntimeError("A permutation-calibrated critical value could not be determined.")
    critical_index = int(eligible[0])
    critical_value = float(unique_values[critical_index])
    achieved_alpha = float(tail_probabilities[critical_index])
    return (extreme + 1) / (PERMUTATIONS + 1), extreme, critical_value, achieved_alpha


def bootstrap_correlations(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(SEED)
    values: list[float] = []
    invalid = 0
    for _ in range(BOOTSTRAP_B):
        indices = rng.integers(0, len(x), size=len(x))
        rho = spearmanr(x[indices], y[indices]).statistic
        if np.isfinite(rho):
            values.append(float(rho))
        else:
            invalid += 1
    return np.asarray(values, dtype=float), invalid


def bca_interval(
    observed: float, bootstrap_values: np.ndarray, x: np.ndarray, y: np.ndarray, alpha: float = 0.05
) -> tuple[float, float, float, float]:
    """Bias-corrected and accelerated interval with leave-one-unit-out acceleration."""
    proportion_less = (np.count_nonzero(bootstrap_values < observed) + 0.5) / (len(bootstrap_values) + 1.0)
    z0 = float(norm.ppf(np.clip(proportion_less, 1e-12, 1 - 1e-12)))
    jackknife = []
    for omitted in range(len(x)):
        keep = np.arange(len(x)) != omitted
        jackknife.append(float(spearmanr(x[keep], y[keep]).statistic))
    jackknife_values = np.asarray(jackknife)
    jack_mean = float(jackknife_values.mean())
    numerator = float(np.sum((jack_mean - jackknife_values) ** 3))
    denominator = 6.0 * float(np.sum((jack_mean - jackknife_values) ** 2)) ** 1.5
    acceleration = numerator / denominator if denominator > 0 else 0.0
    z_low, z_high = norm.ppf([alpha / 2, 1 - alpha / 2])

    def adjusted_probability(z_alpha: float) -> float:
        return float(norm.cdf(z0 + (z0 + z_alpha) / (1 - acceleration * (z0 + z_alpha))))

    q_low = float(np.clip(adjusted_probability(float(z_low)), 0, 1))
    q_high = float(np.clip(adjusted_probability(float(z_high)), 0, 1))
    low, high = np.quantile(bootstrap_values, [q_low, q_high])
    return float(low), float(high), z0, acceleration


def latent_pearson_from_spearman(rho_s: float) -> float:
    """Inverse Gaussian-copula relationship: Pearson r = 2 sin(pi*rho_s/6)."""
    return float(2 * np.sin(np.pi * rho_s / 6.0))


def rowwise_spearman_no_ties(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorized Spearman correlations for continuous simulated observations."""
    n = x.shape[1]
    ranks_x = np.argsort(np.argsort(x, axis=1), axis=1).astype(float)
    ranks_y = np.argsort(np.argsort(y, axis=1), axis=1).astype(float)
    midpoint = (n - 1) / 2.0
    denominator = n * (n**2 - 1) / 12.0
    return np.sum((ranks_x - midpoint) * (ranks_y - midpoint), axis=1) / denominator


def simulate_power(
    rho_true: float,
    simulations: int,
    seed_sequence: np.random.SeedSequence,
    permutation_critical_rho: float,
) -> dict[str, float | int]:
    rng = np.random.default_rng(seed_sequence)
    latent = latent_pearson_from_spearman(rho_true)
    x = rng.standard_normal((simulations, N_UNITS))
    noise = rng.standard_normal((simulations, N_UNITS))
    y = latent * x + math.sqrt(max(0.0, 1.0 - latent**2)) * noise
    sample_rho = rowwise_spearman_no_ties(x, y)
    statistic = sample_rho * np.sqrt((N_UNITS - 2) / np.maximum(1e-15, 1.0 - sample_rho**2))
    p_values = 2.0 * t.sf(np.abs(statistic), df=N_UNITS - 2)
    permutation_significant = int(
        np.count_nonzero(np.abs(sample_rho) >= permutation_critical_rho - 1e-15)
    )
    asymptotic_significant = int(np.count_nonzero(p_values < ALPHA))
    return {
        "rho_true": rho_true,
        "latent_pearson_r": latent,
        "n": N_UNITS,
        "alpha": ALPHA,
        "simulations": simulations,
        "permutation_critical_sample_rho": permutation_critical_rho,
        "significant_n": permutation_significant,
        "empirical_power": permutation_significant / simulations,
        "asymptotic_significant_n": asymptotic_significant,
        "asymptotic_empirical_power": asymptotic_significant / simulations,
    }


def create_figures(
    bootstrap_values: np.ndarray,
    observed: float,
    percentile_ci: tuple[float, float],
    power_table: pd.DataFrame,
    thresholds: dict[float, float],
    observed_power: float,
) -> None:
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
        }
    )
    FIGURES.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    bins = np.linspace(-1.0, 1.0, 51)
    ax.hist(bootstrap_values, bins=bins, color="#8E8BFE", edgecolor="white", linewidth=0.6)
    ax.axvline(observed, color="#FEA3A2", linewidth=2.5, label=f"Observed $\\rho$ = {observed:.3f}")
    ax.axvline(percentile_ci[0], color="black", linestyle="--", linewidth=1.6)
    ax.axvline(percentile_ci[1], color="black", linestyle="--", linewidth=1.6)
    ax.plot([], [], color="black", linestyle="--", linewidth=1.6, label="Percentile 95% CI")
    ax.set_xlim(-1.02, 1.02)
    ax.set_xticks(np.arange(-1.0, 1.01, 0.25))
    ax.set_xlabel("Bootstrap Spearman $\\rho$")
    ax.set_ylabel("Frequency")
    ax.set_title("Cell-type-aggregate bootstrap distribution")
    ax.legend(frameon=False, loc="upper left")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(FIGURES / "cag_bootstrap_distribution.pdf", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "cag_bootstrap_distribution.png", dpi=800, bbox_inches="tight")
    plt.close(fig)

    plotted = power_table.sort_values("rho_true").drop_duplicates("rho_true", keep="last")
    fig, ax = plt.subplots(figsize=(8.4, 5.6), constrained_layout=True)
    ax.plot(plotted["rho_true"], plotted["empirical_power"], color="#8E8BFE", linewidth=2.5)
    ax.scatter(plotted["rho_true"], plotted["empirical_power"], color="#8E8BFE", s=22, zorder=3)
    for target, linestyle in ((0.50, ":"), (0.80, "--"), (0.90, "-.")):
        ax.axhline(target, color="#666666", linestyle=linestyle, linewidth=1.2)
        threshold = thresholds[target]
        ax.axvline(threshold, color="#666666", linestyle=linestyle, linewidth=1.2)
        ax.text(threshold + 0.008, target + 0.018, f"{int(target*100)}%: $\\rho$ ≈ {threshold:.2f}", fontsize=13)
    ax.scatter([observed], [observed_power], color="#FEA3A2", edgecolor="black", linewidth=0.6, s=85, zorder=5)
    ax.annotate(
        f"Observed magnitude reference\n$\\rho$ = {observed:.3f}; power ≈ {observed_power:.2f}",
        xy=(observed, observed_power),
        xytext=(0.35, 0.34),
        arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.0},
        fontsize=13,
    )
    ax.set_xlim(0, 0.92)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(np.arange(0, 0.91, 0.1))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel("True Spearman correlation magnitude")
    ax.set_ylabel("Empirical power")
    ax.set_title("Power of the permutation-calibrated Spearman test ($n$ = 14)")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.savefig(FIGURES / "cag_power_curve.pdf", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "cag_power_curve.png", dpi=800, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_logging()
    for directory in (TABLES, FIGURES, STATISTICS):
        directory.mkdir(parents=True, exist_ok=True)
    logging.info("Input: %s", INPUT)
    logging.info("Input SHA-256: %s", sha256(INPUT))
    frame = pd.read_csv(INPUT)
    required = {CELL_COLUMN, X_COLUMN, Y_COLUMN}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(frame.columns))}")
    if len(frame) != N_UNITS or frame[CELL_COLUMN].nunique() != N_UNITS:
        raise ValueError("The final input must contain exactly 14 unique cell-type aggregate rows.")
    if frame[[X_COLUMN, Y_COLUMN]].isna().any().any():
        raise ValueError("The two analysis variables contain missing values.")
    if set(frame[CELL_COLUMN]) != set(CANONICAL_CELL_ORDER):
        raise ValueError("The 14 cell-type labels do not match the prespecified GSE233387 mapping.")
    frame = frame.set_index(CELL_COLUMN).loc[CANONICAL_CELL_ORDER].reset_index()
    x = frame[X_COLUMN].to_numpy(dtype=float)
    y = frame[Y_COLUMN].to_numpy(dtype=float)

    observed_result = spearmanr(x, y)
    observed = float(observed_result.statistic)
    asymptotic_p = float(observed_result.pvalue)
    permutation_p, extreme, permutation_critical_rho, achieved_null_alpha = monte_carlo_permutation_p(
        x, y, observed
    )
    logging.info(
        "Observed n=%d, Spearman rho=%.12f, asymptotic P=%.12g, permutation P=%.12g (%d extremes)",
        len(frame), observed, asymptotic_p, permutation_p, extreme,
    )
    if not (abs(observed - 0.6351648351648351) < 1e-12 and abs(permutation_p - 0.017323982676017324) < 1e-15):
        raise RuntimeError("The prespecified cell-type aggregate result was not reproduced; stopping before bootstrap and power analysis.")

    bootstrap_values, invalid = bootstrap_correlations(x, y)
    percentile_low, percentile_high = np.quantile(bootstrap_values, [0.025, 0.975])
    bca_low, bca_high, z0, acceleration = bca_interval(observed, bootstrap_values, x, y)
    bootstrap_summary = pd.DataFrame(
        [
            {
                "n_units": N_UNITS,
                "observed_rho": observed,
                "asymptotic_two_sided_p": asymptotic_p,
                "permutation_p": permutation_p,
                "permutation_extreme_n": extreme,
                "permutation_repetitions": PERMUTATIONS,
                "permutation_critical_sample_rho": permutation_critical_rho,
                "permutation_critical_achieved_null_alpha": achieved_null_alpha,
                "bootstrap_B": BOOTSTRAP_B,
                "valid_bootstrap_B": len(bootstrap_values),
                "invalid_bootstrap_B": invalid,
                "invalid_bootstrap_proportion": invalid / BOOTSTRAP_B,
                "bootstrap_median_rho": float(np.median(bootstrap_values)),
                "bootstrap_mean_rho": float(np.mean(bootstrap_values)),
                "bootstrap_ci95_low": float(percentile_low),
                "bootstrap_ci95_high": float(percentile_high),
                "bca_ci95_low": bca_low,
                "bca_ci95_high": bca_high,
                "bca_bias_correction_z0": z0,
                "bca_acceleration": acceleration,
                "seed": SEED,
            }
        ]
    )
    bootstrap_summary.to_csv(TABLES / "cag_bootstrap_summary.csv", index=False)
    pd.DataFrame({"bootstrap_replicate": np.arange(1, len(bootstrap_values) + 1), "spearman_rho": bootstrap_values}).to_csv(
        TABLES / "cag_bootstrap_replicates.csv", index=False
    )

    coarse_rhos = np.round(np.arange(0.00, 0.901, 0.05), 3)
    fine_rhos = np.round(np.arange(0.30, 0.901, 0.01), 3)
    all_specs = [(float(rho), COARSE_SIMULATIONS, "coarse") for rho in coarse_rhos]
    all_specs += [(float(rho), FINE_SIMULATIONS, "fine") for rho in fine_rhos]
    all_specs += [(observed, OBSERVED_EFFECT_SIMULATIONS, "observed_magnitude_reference")]
    seeds = np.random.SeedSequence(SEED).spawn(len(all_specs))
    rows = []
    for index, ((rho, simulations, grid), sequence) in enumerate(zip(all_specs, seeds), start=1):
        result = simulate_power(rho, simulations, sequence, permutation_critical_rho)
        result["grid"] = grid
        rows.append(result)
        if index % 20 == 0 or index == len(all_specs):
            logging.info("Power simulations completed: %d/%d", index, len(all_specs))
    power_table = pd.DataFrame(rows)
    power_table.to_csv(TABLES / "cag_power_curve.csv", index=False)

    fine = power_table[power_table["grid"] == "fine"].sort_values("rho_true")
    thresholds: dict[float, float] = {}
    for target in (0.50, 0.80, 0.90):
        crossing = fine[fine["empirical_power"] >= target]
        thresholds[target] = float(crossing.iloc[0]["rho_true"]) if len(crossing) else float("nan")
    observed_power = float(
        power_table.loc[power_table["grid"] == "observed_magnitude_reference", "empirical_power"].iloc[0]
    )
    t_critical = float(t.ppf(1 - ALPHA / 2, df=N_UNITS - 2))
    sample_rho_critical = float(math.sqrt(t_critical**2 / (t_critical**2 + N_UNITS - 2)))
    mde = pd.DataFrame(
        [
            {
                "n": N_UNITS,
                "alpha": ALPHA,
                "test": "two-sided Spearman correlation; permutation-calibrated significance rule",
                "sample_rho_significance_threshold": permutation_critical_rho,
                "permutation_critical_achieved_null_alpha": achieved_null_alpha,
                "asymptotic_sample_rho_significance_threshold": sample_rho_critical,
                "power_50_rho": thresholds[0.50],
                "power_80_rho": thresholds[0.80],
                "power_90_rho": thresholds[0.90],
                "mde_grid_resolution": 0.01,
                "fine_grid_simulations_per_rho": FINE_SIMULATIONS,
                "observed_rho": observed,
                "power_at_observed_rho_as_true_magnitude": observed_power,
                "observed_effect_simulations": OBSERVED_EFFECT_SIMULATIONS,
                "seed": SEED,
            }
        ]
    )
    mde.to_csv(TABLES / "cag_mde_summary.csv", index=False)
    create_figures(
        bootstrap_values,
        observed,
        (float(percentile_low), float(percentile_high)),
        power_table,
        thresholds,
        observed_power,
    )

    report = f"""# CAG bootstrap and power statistical report

## Input and reproducibility

- Input: `{INPUT}`
- SHA-256: `{sha256(INPUT)}`
- Statistical unit: 14 GSE233387 BA4 cell-type aggregates.
- Expression variable: `{X_COLUMN}`.
- Somatic-CAG variable: `{Y_COLUMN}`.
- Observed Spearman correlation: rho = {observed:.6f}.
- One-million-label-permutation two-sided P = {permutation_p:.6f} ({extreme:,} extreme permutations; seed 20260827).

## Bootstrap uncertainty

- Nonparametric resampling unit: cell-type aggregate row.
- Bootstrap repetitions: {BOOTSTRAP_B:,}; valid: {len(bootstrap_values):,}; invalid: {invalid:,}.
- Percentile bootstrap 95% CI: [{percentile_low:.3f}, {percentile_high:.3f}].
- BCa bootstrap 95% CI: [{bca_low:.3f}, {bca_high:.3f}].
- Bootstrap median rho = {np.median(bootstrap_values):.3f}; mean rho = {np.mean(bootstrap_values):.3f}.

## Simulation-based power

- Simulation model: Gaussian copula calibrated to each target Spearman correlation with latent Pearson r = 2 sin(pi*rho/6).
- Sample size per simulated dataset: n = {N_UNITS}.
- Test: two-sided Spearman correlation at alpha = {ALPHA}; significance evaluated with the sample-correlation threshold estimated from 1,000,000 null label permutations.
- Permutation-calibrated sample-correlation threshold: |rho| >= {permutation_critical_rho:.3f}; achieved null tail probability = {achieved_null_alpha:.4f}.
- The standard t-approximation threshold was |rho| >= {sample_rho_critical:.3f} and is retained in the source table as a method sensitivity value.
- Minimum true Spearman magnitude for at least 50% power: {thresholds[0.50]:.2f}.
- Minimum true Spearman magnitude for at least 80% power: {thresholds[0.80]:.2f}.
- Minimum true Spearman magnitude for at least 90% power: {thresholds[0.90]:.2f}.
- If the true population Spearman correlation equalled the observed sample magnitude ({observed:.3f}), estimated power would be {observed_power:.3f}. This is a magnitude reference and does not treat the observed sample estimate as the known population parameter.

## Interpretation

The positive association is reproduced and its permutation P value remains below 0.05. The bootstrap interval quantifies substantial uncertainty arising from 14 ecological units. The power analysis shows that this design is mainly capable of detecting large monotonic associations. The result should be retained as a BA4 cell-type-level ecological association, accompanied by its confidence interval and sample-size limitation; it does not establish an individual-cell, donor-level, gene-specific, spatial, or causal effect.

## Software

- Python {platform.python_version()}
- NumPy {np.__version__}
- pandas {pd.__version__}
- SciPy {scipy.__version__}
- Matplotlib {mpl.__version__}
"""
    (STATISTICS / "cag_bootstrap_power_statistical_report.md").write_text(report, encoding="utf-8")
    environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": mpl.__version__,
        "platform": platform.platform(),
        "input_sha256": sha256(INPUT),
        "seed": SEED,
    }
    (STATISTICS / "cag_bootstrap_power_environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    logging.info("Bootstrap percentile 95%% CI: [%.6f, %.6f]", percentile_low, percentile_high)
    logging.info("Power thresholds: %s", thresholds)
    logging.info("Power at observed magnitude %.6f: %.6f", observed, observed_power)
    logging.info("CAG bootstrap and power analysis completed successfully.")


if __name__ == "__main__":
    main()


# Script function: reproduce the 14-cell-type Spearman result, estimate cell-type-row bootstrap CIs, and simulate power/MDE.
# Input source: input/GSE233387_14celltype_PLS1positive_CAG_final.csv.
# Output location: results/tables, results/figures, results/statistics under the configured regional-sensitivity work directory.
# Input fields: target_cell_type, pls1_positive_expression_score, cag_mean_somatic_length_gain.
# Output fields: observed/permutation statistics, bootstrap summaries and replicates, power curve, MDE thresholds, figures, report.
# Main steps: strict 14-row validation; 1,000,000 label permutations; 20,000 row bootstraps; Gaussian-copula power simulations calibrated to the permutation-test threshold.
# Random seeds: 20260911 for bootstrap/power; 20260827 for reproduction of the reference permutation analysis.
# Software versions: written at runtime to results/statistics/cag_bootstrap_power_environment.json and the statistical report.
# Log location: logs/cag_bootstrap_power.log.

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Quantify uncertainty and detectable-effect characteristics for the 14-unit BA4 cell-type aggregate CAG/transcriptional association.
# Input source/location: input/GSE233387_14celltype_PLS1positive_CAG_final.csv.
# Output location: Bootstrap/permutation/power tables, statistical report, environment record, and editable PDF plus 800-dpi PNG figures under results/.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Validate 14 units; reproduce the Spearman/permutation result; bootstrap the correlation; calculate BCa/percentile intervals; simulate power and minimum detectable effects.
# Log location: logs/cag_bootstrap_power.log.
# =============================================================================
