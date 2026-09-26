from __future__ import annotations

import itertools
import json
import math
import warnings
from pathlib import Path
import os

import numpy as np
import pandas as pd
from patsy import build_design_matrices
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable


ROOT = Path(os.environ.get("HD_MSN_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
PREVIOUS = Path(os.environ.get("HD_MSN_CELLCLASS_RESULTS", ROOT / "results" / "hierarchical_cellclass_analysis")).resolve()
CAG_ROOT = Path(os.environ.get("HD_MSN_BA4_CAG_ROOT", ROOT / "results" / "ba4_cag_reanalysis")).resolve()
OUT = Path(os.environ.get("HD_MSN_NATIVE_CELLTYPE_OUT", ROOT / "results" / "native_celltype_validation")).resolve()
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260827
BOOTSTRAPS = 5000

NATIVE_TYPES = {
    "GSE281069": [
        "Ex_neuron", "SST/NPY+ IN", "VIP+ IN", "Astrocyte",
        "Microglia", "OLG", "OPC",
    ],
    "GSE180928": [
        "Ex_neuron", "In_neuron", "Astrocyte", "NFO", "OLG", "OPC",
    ],
}


def bh(values: pd.Series) -> np.ndarray:
    x = values.to_numpy(float)
    out = np.full(len(x), np.nan)
    valid = np.isfinite(x)
    xv = x[valid]
    order = np.argsort(xv)
    ranked = xv[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(ranked) / np.arange(1, len(ranked) + 1))[::-1]
    )[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[valid] = restored
    return out


def ols_native_effect(group: pd.DataFrame, dataset: str) -> tuple[dict, object]:
    current = group.copy()
    current["HD"] = current.condition.eq("HD").astype(int)
    if dataset == "GSE281069":
        current["age_z"] = (current.age - current.age.mean()) / current.age.std(ddof=1)
        fit = smf.ols("PLS1_positive_score_z ~ age_z + C(sex) + HD", current).fit()
    else:
        fit = smf.ols("PLS1_positive_score_z ~ HD", current).fit()
    rank = int(np.linalg.matrix_rank(fit.model.exog))
    if rank != fit.model.exog.shape[1]:
        raise RuntimeError(f"Rank-deficient native-cell-type model: {dataset}/{group.cell_type.iloc[0]}")
    if fit.df_resid < 3:
        raise RuntimeError(f"Residual df < 3: {dataset}/{group.cell_type.iloc[0]}")
    estimate = float(fit.params["HD"])
    se = float(fit.bse["HD"])
    critical = float(stats.t.ppf(0.975, fit.df_resid))
    return {
        "dataset": dataset,
        "region": group.region.iloc[0],
        "cell_type": group.cell_type.iloc[0],
        "n_control": int((group.condition == "Control").sum()),
        "n_hd": int((group.condition == "HD").sum()),
        "standardized_HD_effect": estimate,
        "standard_error": se,
        "ci_low": estimate - critical * se,
        "ci_high": estimate + critical * se,
        "score_p_value": float(fit.pvalues["HD"]),
        "design_rank": rank,
        "residual_df": int(fit.df_resid),
    }, fit


def bootstrap_native(group: pd.DataFrame, dataset: str) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    current = group.copy().reset_index(drop=True)
    current["HD"] = current.condition.eq("HD").astype(int)
    if dataset == "GSE281069":
        current["age_z"] = (current.age - current.age.mean()) / current.age.std(ddof=1)
        current["sex_M"] = current.sex.eq("M").astype(int)
        columns = ["intercept", "age_z", "sex_M", "HD"]
    else:
        columns = ["intercept", "HD"]
    current["intercept"] = 1.0
    y = current.PLS1_positive_score_z.to_numpy(float)
    X = current[columns].to_numpy(float)
    hd_col = columns.index("HD")
    strata = [g.index.to_numpy() for _, g in current.groupby("condition", observed=True)]
    estimates: list[float] = []
    attempts = 0
    while len(estimates) < BOOTSTRAPS and attempts < BOOTSTRAPS * 20:
        attempts += 1
        chosen = np.concatenate([rng.choice(idx, size=len(idx), replace=True) for idx in strata])
        draw_x = X[chosen]
        if np.linalg.matrix_rank(draw_x) != draw_x.shape[1]:
            continue
        beta = np.linalg.lstsq(draw_x, y[chosen], rcond=None)[0]
        estimates.append(float(beta[hd_col]))
    if len(estimates) != BOOTSTRAPS:
        raise RuntimeError(f"Bootstrap full-rank failure: {dataset}/{group.cell_type.iloc[0]}")
    arr = np.asarray(estimates)
    summary = {
        "dataset": dataset,
        "cell_type": group.cell_type.iloc[0],
        "bootstrap_replicates": BOOTSTRAPS,
        "attempted_draws": attempts,
        "bootstrap_mean": float(arr.mean()),
        "bootstrap_ci_low": float(np.quantile(arr, 0.025)),
        "bootstrap_ci_high": float(np.quantile(arr, 0.975)),
    }
    full_beta = np.linalg.lstsq(X, y, rcond=None)[0][hd_col]
    lodo = []
    for donor in current.donor_id.unique():
        keep = current.donor_id.ne(donor).to_numpy()
        rank_ok = np.linalg.matrix_rank(X[keep]) == X.shape[1]
        effect = float(np.linalg.lstsq(X[keep], y[keep], rcond=None)[0][hd_col]) if rank_ok else np.nan
        lodo.append({
            "dataset": dataset,
            "cell_type": group.cell_type.iloc[0],
            "omitted_donor": donor,
            "effect": effect,
            "full_rank": bool(rank_ok),
            "direction_matches_full": bool(np.sign(effect) == np.sign(full_beta)) if np.isfinite(effect) else False,
        })
    return summary, pd.DataFrame(lodo)


def estimate_from_contrast(result, contrast: np.ndarray) -> dict:
    params = np.asarray(result.fe_params if hasattr(result, "fe_params") else result.params, float)
    covariance = np.asarray(result.cov_params(), float)[: len(params), : len(params)]
    estimate = float(contrast @ params)
    variance = float(contrast @ covariance @ contrast)
    se = math.sqrt(max(variance, 0.0))
    z = estimate / se if se > 0 else np.nan
    return {
        "estimate": estimate,
        "standard_error": se,
        "ci_low": estimate - 1.959963984540054 * se,
        "ci_high": estimate + 1.959963984540054 * se,
        "p_value": float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else np.nan,
    }


def design_contrast(result, high: pd.DataFrame, low: pd.DataFrame) -> np.ndarray:
    info = result.model.data.design_info
    x_high = np.asarray(build_design_matrices([info], high)[0], float)
    x_low = np.asarray(build_design_matrices([info], low)[0], float)
    return (x_high - x_low).ravel()


def joint_native_model(data: pd.DataFrame, dataset: str, categories: list[str]) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    current = data[(data.dataset == dataset) & data.cell_type.isin(categories)].copy()
    current["cell_type"] = pd.Categorical(current.cell_type, categories=categories, ordered=True)
    current["HD"] = current.condition.eq("HD").astype(int)
    current["donor_cluster"] = dataset + "::" + current.donor_id.astype(str)
    reference = categories[0]
    if dataset == "GSE281069":
        current["age_z"] = (current.age - current.age.mean()) / current.age.std(ddof=1)
        full_formula = f'PLS1_positive_score_z ~ age_z + C(sex) + HD * C(cell_type, Treatment(reference="{reference}"))'
        reduced_formula = f'PLS1_positive_score_z ~ age_z + C(sex) + HD + C(cell_type, Treatment(reference="{reference}"))'
        other = {"age_z": 0.0, "sex": current.sex.mode().iloc[0]}
    else:
        full_formula = f'PLS1_positive_score_z ~ HD * C(cell_type, Treatment(reference="{reference}"))'
        reduced_formula = f'PLS1_positive_score_z ~ HD + C(cell_type, Treatment(reference="{reference}"))'
        other = {}

    mixed_full = mixed_reduced = None
    messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for optimizer in ("lbfgs", "powell", "bfgs", "cg"):
            try:
                candidate_full = smf.mixedlm(full_formula, current, groups=current.donor_cluster).fit(
                    reml=False, method=optimizer, maxiter=4000, disp=False
                )
                candidate_reduced = smf.mixedlm(reduced_formula, current, groups=current.donor_cluster).fit(
                    reml=False, method=optimizer, maxiter=4000, disp=False
                )
                if candidate_full.converged and candidate_reduced.converged:
                    mixed_full, mixed_reduced = candidate_full, candidate_reduced
                    messages.append(f"optimizer={optimizer}")
                    break
                messages.append(f"optimizer={optimizer}: not converged")
            except Exception as exc:
                messages.append(f"optimizer={optimizer}: {type(exc).__name__}: {exc}")
        messages.extend(str(w.message) for w in caught)

    if mixed_full is None or mixed_reduced is None:
        raise RuntimeError(f"Mixed model not estimable for {dataset}: {' | '.join(messages)}")
    omnibus_df = len(mixed_full.fe_params) - len(mixed_reduced.fe_params)
    lrt = max(0.0, 2 * (mixed_full.llf - mixed_reduced.llf))
    mixed_p = float(stats.chi2.sf(lrt, omnibus_df))

    gee = smf.gee(
        full_formula, groups="donor_cluster", data=current,
        family=sm.families.Gaussian(), cov_struct=Exchangeable(),
    ).fit(maxiter=500)
    interaction_indices = [i for i, name in enumerate(gee.params.index) if "HD:C(cell_type" in name]
    restriction = np.zeros((len(interaction_indices), len(gee.params)))
    for row, col in enumerate(interaction_indices):
        restriction[row, col] = 1.0
    gee_test = gee.wald_test(restriction, scalar=True)

    diagnostics = {
        "dataset": dataset,
        "n_observations": int(len(current)),
        "n_donors": int(current.donor_id.nunique()),
        "n_cell_types": len(categories),
        "mixed_model_converged": bool(mixed_full.converged and mixed_reduced.converged),
        "mixed_model_optimizer_and_warnings": " | ".join(dict.fromkeys(messages)),
        "mixed_model_random_intercept_variance": float(np.asarray(mixed_full.cov_re)[0, 0]),
        "mixed_model_design_rank": int(np.linalg.matrix_rank(mixed_full.model.exog)),
        "mixed_model_residual_df": int(len(current) - len(mixed_full.fe_params)),
        "HD_by_cell_type_df": int(omnibus_df),
        "HD_by_cell_type_LRT": lrt,
        "HD_by_cell_type_mixed_model_p": mixed_p,
        "GEE_converged": bool(getattr(gee, "converged", True)),
        "GEE_design_rank": int(np.linalg.matrix_rank(gee.model.exog)),
        "HD_by_cell_type_GEE_Wald": float(np.asarray(gee_test.statistic).squeeze()),
        "HD_by_cell_type_GEE_p": float(np.asarray(gee_test.pvalue).squeeze()),
    }

    mixed_effects = []
    gee_effects = []
    mixed_contrasts: dict[str, np.ndarray] = {}
    for category in categories:
        high = {"HD": 1, "cell_type": category, **other}
        low = {"HD": 0, "cell_type": category, **other}
        mixed_contrast = design_contrast(mixed_full, pd.DataFrame([high]), pd.DataFrame([low]))
        mixed_contrasts[category] = mixed_contrast
        mixed_row = estimate_from_contrast(mixed_full, mixed_contrast)
        mixed_row.update({"dataset": dataset, "cell_type": category})
        mixed_effects.append(mixed_row)
        gee_row = estimate_from_contrast(
            gee, design_contrast(gee, pd.DataFrame([high]), pd.DataFrame([low]))
        )
        gee_row.update({"dataset": dataset, "cell_type": category})
        gee_effects.append(gee_row)
    pairwise_rows = []
    for first, second in itertools.combinations(categories, 2):
        row = estimate_from_contrast(mixed_full, mixed_contrasts[first] - mixed_contrasts[second])
        row.update({"dataset": dataset, "first_cell_type": first, "second_cell_type": second})
        pairwise_rows.append(row)
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise["BH_q_within_dataset"] = bh(pairwise.p_value)
    lodo_rows = []
    for omitted in current.donor_id.unique():
        subset = current[current.donor_id.ne(omitted)].copy()
        fit_full = fit_reduced = None
        optimizer_used = None
        for optimizer in ("lbfgs", "powell", "bfgs", "cg"):
            try:
                candidate_full = smf.mixedlm(full_formula, subset, groups=subset.donor_cluster).fit(
                    reml=False, method=optimizer, maxiter=4000, disp=False
                )
                candidate_reduced = smf.mixedlm(reduced_formula, subset, groups=subset.donor_cluster).fit(
                    reml=False, method=optimizer, maxiter=4000, disp=False
                )
                if candidate_full.converged and candidate_reduced.converged:
                    fit_full, fit_reduced = candidate_full, candidate_reduced
                    optimizer_used = optimizer
                    break
            except Exception:
                continue
        if fit_full is None or fit_reduced is None:
            lodo_rows.append({
                "dataset": dataset, "omitted_donor": omitted, "converged": False,
                "optimizer": "", "omnibus_df": np.nan, "LRT": np.nan, "omnibus_p_value": np.nan,
            })
            continue
        df = len(fit_full.fe_params) - len(fit_reduced.fe_params)
        statistic = max(0.0, 2 * (fit_full.llf - fit_reduced.llf))
        lodo_rows.append({
            "dataset": dataset, "omitted_donor": omitted, "converged": True,
            "optimizer": optimizer_used, "omnibus_df": df, "LRT": statistic,
            "omnibus_p_value": float(stats.chi2.sf(statistic, df)),
        })
    return diagnostics, pd.DataFrame(mixed_effects), pd.DataFrame(gee_effects), pairwise, pd.DataFrame(lodo_rows)


def residualized_rank_association(x: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
    xr = stats.rankdata(x, method="average")
    yr = stats.rankdata(y, method="average")
    design = pd.get_dummies(pd.Series(classes), drop_first=True, dtype=float)
    design.insert(0, "intercept", 1.0)
    matrix = design.to_numpy(float)
    x_resid = xr - matrix @ np.linalg.lstsq(matrix, xr, rcond=None)[0]
    y_resid = yr - matrix @ np.linalg.lstsq(matrix, yr, rcond=None)[0]
    return float(np.corrcoef(x_resid, y_resid)[0, 1])


def exact_restricted_p(x: np.ndarray, y: np.ndarray, classes: np.ndarray) -> tuple[float, float, int]:
    observed = residualized_rank_association(x, y, classes)
    xr = stats.rankdata(x, method="average")
    yr = stats.rankdata(y, method="average")
    design = pd.get_dummies(pd.Series(classes), drop_first=True, dtype=float)
    design.insert(0, "intercept", 1.0)
    matrix = design.to_numpy(float)
    x_resid = xr - matrix @ np.linalg.lstsq(matrix, xr, rcond=None)[0]
    y_resid = yr - matrix @ np.linalg.lstsq(matrix, yr, rcond=None)[0]
    denominator = math.sqrt(float(x_resid @ x_resid) * float(y_resid @ y_resid))
    blocks = [np.flatnonzero(classes == label) for label in pd.unique(classes)]
    permutations = [np.array(list(itertools.permutations(block)), dtype=int) for block in blocks]
    total = math.prod(len(p) for p in permutations)
    extreme = 0
    for first in permutations[0]:
        for second in permutations[1]:
            idx = np.tile(np.arange(len(y)), (len(permutations[2]), 1))
            idx[:, blocks[0]] = first
            idx[:, blocks[1]] = second
            idx[:, blocks[2]] = permutations[2]
            correlations = (y_resid[idx] @ x_resid) / denominator
            extreme += int(np.sum(np.abs(correlations) >= abs(observed) - 1e-12))
    return observed, extreme / total, total


def run_cag() -> tuple[pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(CAG_ROOT / "analysis_results" / "family_specific_celltype_scores.csv")
    scores = scores[
        scores.analysis.eq("All 14 populations, broad-class restricted permutation")
        & scores.donor_group.eq("CTRL")
        & scores.expression_standardization.eq("zscore")
    ].copy()
    wide = scores.pivot(index="target_cell_type", columns="score", values="value").reset_index()
    cag = pd.read_csv(CAG_ROOT / "source_data" / "Pressl_Table1_BA4_CAG.csv")
    data = cag.merge(wide, on="target_cell_type", validate="one_to_one")
    if data.broad_class.value_counts().to_dict() != {"excitatory": 6, "inhibitory": 4, "glial": 4}:
        raise RuntimeError("Unexpected GSE233387 14-cell-type broad-class composition")
    rows = []
    for score in [
        "PLS1-positive expression score",
        "Signed PLS1 expression score",
        "PLS1-negative expression score",
    ]:
        rho, p_value, n_perm = exact_restricted_p(
            data[score].to_numpy(float),
            data.cag_mean_somatic_length_gain.to_numpy(float),
            data.broad_class.to_numpy(str),
        )
        lodo = []
        for omitted in data.target_cell_type:
            subset = data[data.target_cell_type.ne(omitted)]
            lodo.append(residualized_rank_association(
                subset[score].to_numpy(float),
                subset.cag_mean_somatic_length_gain.to_numpy(float),
                subset.broad_class.to_numpy(str),
            ))
        rows.append({
            "score": score,
            "n_cell_types": len(data),
            "class_adjusted_spearman_rho": rho,
            "exact_two_sided_p": p_value,
            "complete_restricted_permutations": n_perm,
            "lodo_min_rho": float(np.min(lodo)),
            "lodo_max_rho": float(np.max(lodo)),
            "lodo_same_direction_fraction": float(np.mean(np.sign(lodo) == np.sign(rho))),
        })
    result = pd.DataFrame(rows)
    result["BH_q_across_three_expression_scores"] = bh(result.exact_two_sided_p)
    return result, data


def main() -> None:
    scores = pd.read_csv(PREVIOUS / "results" / "zenodo_specific_cell_type_scores.csv")
    coverage_rows = []
    effect_rows = []
    bootstrap_rows = []
    lodo_rows = []
    omnibus_rows = []
    mixed_rows = []
    gee_rows = []
    pairwise_rows = []
    interaction_lodo_rows = []

    for dataset, categories in NATIVE_TYPES.items():
        current = scores[(scores.dataset == dataset) & scores.cell_type.isin(categories)].copy()
        observed = current.cell_type.drop_duplicates().tolist()
        if set(observed) != set(categories):
            raise RuntimeError(f"Native label mismatch for {dataset}: {observed}")
        for cell_type in categories:
            group = current[current.cell_type == cell_type].copy()
            coverage_rows.append({
                "dataset": dataset,
                "region": group.region.iloc[0],
                "cell_type": cell_type,
                "n_control": int((group.condition == "Control").sum()),
                "n_hd": int((group.condition == "HD").sum()),
                "control_nuclei": int(group.loc[group.condition == "Control", "n_nuclei"].sum()),
                "hd_nuclei": int(group.loc[group.condition == "HD", "n_nuclei"].sum()),
            })
            row, _ = ols_native_effect(group, dataset)
            effect_rows.append(row)
            boot, lodo = bootstrap_native(group, dataset)
            bootstrap_rows.append(boot)
            lodo_rows.append(lodo)
        diagnostics, mixed_effects, gee_effects, pairwise, interaction_lodo = joint_native_model(current, dataset, categories)
        omnibus_rows.append(diagnostics)
        mixed_rows.append(mixed_effects)
        gee_rows.append(gee_effects)
        pairwise_rows.append(pairwise)
        interaction_lodo_rows.append(interaction_lodo)

    coverage = pd.DataFrame(coverage_rows)
    effects = pd.DataFrame(effect_rows)
    effects["score_BH_q_within_dataset"] = effects.groupby("dataset", observed=True).score_p_value.transform(
        lambda x: bh(x)
    )

    camera = pd.read_csv(
        Path(os.environ.get("HD_MSN_CAMERA_RESULTS", ROOT / "results" / "sampled_cortex_validation" / "camera_directional_results_globalBH.csv")).resolve()
    )
    camera = camera[camera.gene_set.eq("PLS1_positive")].copy()
    camera["dataset"] = np.where(camera.region.eq("frontal"), "GSE281069", "GSE180928")
    camera = camera[camera.apply(lambda r: r.cell_type in NATIVE_TYPES[r.dataset], axis=1)]
    camera["camera_BH_q_within_dataset"] = camera.groupby("dataset", observed=True).PValue.transform(lambda x: bh(x))
    camera = camera[[
        "dataset", "region", "cell_type", "NGenes", "Direction", "PValue",
        "camera_BH_q_within_dataset",
    ]]
    effects = effects.merge(camera, on=["dataset", "region", "cell_type"], validate="one_to_one")

    bootstrap = pd.DataFrame(bootstrap_rows)
    lodo = pd.concat(lodo_rows, ignore_index=True)
    lodo_summary = lodo.groupby(["dataset", "cell_type"], observed=True).agg(
        lodo_runs=("omitted_donor", "size"),
        full_rank_runs=("full_rank", "sum"),
        direction_agreement=("direction_matches_full", "mean"),
    ).reset_index()
    lodo = lodo.merge(lodo_summary, on=["dataset", "cell_type"], validate="many_to_one")

    cag_results, cag_data = run_cag()

    coverage.to_csv(OUT / "native_cell_type_coverage.csv", index=False)
    effects.to_csv(OUT / "native_cell_type_HD_effects_and_gene_set_tests.csv", index=False)
    pd.DataFrame(omnibus_rows).to_csv(OUT / "native_cell_type_omnibus_interactions.csv", index=False)
    pd.concat(mixed_rows, ignore_index=True).to_csv(OUT / "native_cell_type_joint_mixed_model_effects.csv", index=False)
    pd.concat(gee_rows, ignore_index=True).to_csv(OUT / "native_cell_type_joint_GEE_effects.csv", index=False)
    pd.concat(pairwise_rows, ignore_index=True).to_csv(OUT / "native_cell_type_pairwise_HD_effect_differences.csv", index=False)
    pd.concat(interaction_lodo_rows, ignore_index=True).to_csv(OUT / "native_cell_type_interaction_leave_one_donor_out.csv", index=False)
    bootstrap.to_csv(OUT / "native_cell_type_donor_bootstrap_5000.csv", index=False)
    lodo.to_csv(OUT / "native_cell_type_leave_one_donor_out.csv", index=False)
    cag_results.to_csv(OUT / "GSE233387_14celltype_class_adjusted_CAG_association.csv", index=False)
    cag_data.to_csv(OUT / "GSE233387_14celltype_CAG_analysis_input.csv", index=False)

    ba4_effects = pd.read_csv(PREVIOUS / "results" / "GSE233408_BA4_subtype_effects.csv")
    ba4_camera = pd.read_csv(PREVIOUS / "results" / "GSE233408_BA4_competitive_gene_set_results.csv")
    ba4_effects.to_csv(OUT / "GSE233408_BA4_four_subtype_effects.csv", index=False)
    ba4_camera.to_csv(OUT / "GSE233408_BA4_four_subtype_gene_set_tests.csv", index=False)

    qc = {
        "seed": SEED,
        "bootstrap_replicates_per_cell_type": BOOTSTRAPS,
        "native_labels": NATIVE_TYPES,
        "n_native_cell_type_models": int(len(effects)),
        "all_native_models_full_rank": bool((effects.design_rank < effects.n_control + effects.n_hd).all()),
        "all_native_models_residual_df_at_least_3": bool((effects.residual_df >= 3).all()),
        "all_bootstraps_complete": bool((bootstrap.bootstrap_replicates == BOOTSTRAPS).all()),
        "all_joint_models_converged": bool(pd.DataFrame(omnibus_rows).mixed_model_converged.all()),
        "cag_complete_restricted_permutations": int(cag_results.complete_restricted_permutations.iloc[0]),
        "cag_class_sizes": cag_data.broad_class.value_counts().to_dict(),
    }
    (OUT / "analysis_QC.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Validate the transcriptional signature in native sampled-cortex cell types and summarize HD effects across external single-cell datasets.
# Input source/location: External cell-type effect tables, reference gene sets/PLS weights, CAG summaries, and CAMERA results provided through environment variables or project-relative inputs.
# Output location: Native-cell-type validation tables, statistics, and figures under HD_MSN_NATIVE_CELLTYPE_OUT or results/native_celltype_validation.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Load external effects; harmonize cell-type labels; score reference gene sets; test direction/enrichment; integrate BA4/CAG summaries; export validation products.
# Log location: No dedicated log file; progress is written to standard output.
# =============================================================================
