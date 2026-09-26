#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
import os

import numpy as np
import pandas as pd
from scipy import stats


BASE = Path(os.environ.get("HD_MSN_DATA_ROOT", Path(__file__).resolve().parents[1] / "data"))
WORK = Path(os.environ.get("HD_MSN_SIGNED_EDGE_WORK", Path(__file__).resolve().parents[1] / "results" / "signed_edge_sensitivity"))
INPUT = WORK / "input"
OUTPUT = WORK / "results_locked"
OUTPUT.mkdir(parents=True, exist_ok=True)

MSN_COVARIATES = ["age", "sex", "education_years", "eTIV"]


def qvalues(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.full(values.size, np.nan)
    valid = np.isfinite(values)
    if not valid.any():
        return result
    p = values[valid]
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    restored = np.empty_like(adjusted)
    restored[order] = np.minimum(adjusted, 1.0)
    result[valid] = restored
    return result


def robust_group_model(
    y: np.ndarray, group: np.ndarray, age: np.ndarray, sex: np.ndarray,
    education_years: np.ndarray, etiv: np.ndarray,
) -> tuple[float, float, float]:
    """HC3-robust HD-HC model with the fixed four MSN nuisance covariates."""
    age_z = (age - np.mean(age)) / np.std(age, ddof=1)
    education_z = (education_years - np.mean(education_years)) / np.std(education_years, ddof=1)
    etiv_z = (etiv - np.mean(etiv)) / np.std(etiv, ddof=1)
    design = np.column_stack([np.ones(len(y)), group, age_z, sex, education_z, etiv_z])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    xtx_inv = np.linalg.pinv(design.T @ design)
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    corrected = residual / np.maximum(1.0 - leverage, 1e-8)
    meat = design.T @ ((corrected[:, None] ** 2) * design)
    covariance = xtx_inv @ meat @ xtx_inv
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    t_value = float(beta[1] / standard_error)
    degrees_freedom = max(len(y) - design.shape[1], 1)
    p_value = float(2.0 * stats.t.sf(abs(t_value), degrees_freedom))
    return float(beta[1]), t_value, p_value


def matrix_components(matrix: np.ndarray) -> dict[str, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float).copy()
    if matrix.shape != (308, 308):
        raise ValueError(f"Expected 308x308 matrix, received {matrix.shape}")
    np.fill_diagonal(matrix, 0.0)
    denominator = matrix.shape[0] - 1
    positive = np.maximum(matrix, 0.0).sum(axis=1) / denominator
    negative = np.minimum(matrix, 0.0).sum(axis=1) / denominator
    signed = matrix.sum(axis=1) / denominator
    positive_proportion = (matrix > 0).sum(axis=1) / denominator
    return {
        "signed_mean": signed,
        "positive_contribution": positive,
        "negative_contribution": negative,
        "absolute_negative_contribution": -negative,
        "positive_edge_proportion": positive_proportion,
    }


def load_expected(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    region_column = next(column for column in frame.columns if column.lower() == "region_index")
    frame = frame.set_index(region_column)
    frame.index = [f"Region_{int(value)}" for value in frame.index]
    return frame.astype(float)


def select_matrix(subject: str, expected: np.ndarray, candidates: list[Path]) -> tuple[Path, dict[str, np.ndarray], float]:
    best = None
    for candidate in candidates:
        try:
            matrix = pd.read_csv(candidate, index_col=0).to_numpy(float)
            components = matrix_components(matrix)
            error = float(np.max(np.abs(components["signed_mean"] - expected)))
        except Exception:
            continue
        if best is None or error < best[2]:
            best = (candidate, components, error)
    if best is None or best[2] > 1e-9:
        raise RuntimeError(f"No matrix reproduced the regional MSN vector for {subject}; best={best}")
    return best


def normalize_hd(value: str) -> str:
    match = re.match(r"(HD_sub\d+)", str(value))
    return match.group(1) if match else str(value)


def normalize_hc(value: str) -> str:
    match = re.match(r"(?:HC_)?sub(\d+)", str(value), flags=re.IGNORECASE)
    return f"HC_sub{int(match.group(1)):03d}" if match else str(value)


def canonicalize_sex(series: pd.Series) -> pd.Series:
    """Standardize numeric sex coding to 1/0 for all MSN models."""
    values = pd.to_numeric(series, errors="coerce")
    observed = set(values.dropna().unique().tolist())
    if observed and observed.issubset({1, 2}):
        return values.map({1: 1.0, 2: 0.0})
    if observed.issubset({0, 1}):
        return values.astype(float)
    raise ValueError("Sex must be encoded as 0/1 or 1/2 for MSN modeling.")


def load_group(group_name: str, expected: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    records: dict[str, dict[str, np.ndarray]] = {}
    audit_rows = []
    if group_name == "HD":
        search_roots = [BASE / "PreHD", BASE / "mHD", BASE / "HD308"]
        filename = lambda subject: f"{subject}_similarity_matrix.csv"
    else:
        search_roots = [BASE / "HC308"]
        filename = lambda subject: f"{re.sub(r'^HC_', '', subject, flags=re.IGNORECASE)}_similarity_matrix.csv"

    for subject in expected.columns:
        candidates = []
        target = filename(subject)
        for root in search_roots:
            candidates.extend(root.glob(f"**/{target}"))
        chosen, components, error = select_matrix(subject, expected[subject].to_numpy(float), candidates)
        records[subject] = components
        audit_rows.append({"group": group_name, "subject": subject, "matrix": str(chosen), "maximum_absolute_reproduction_error": error})

    components_by_name = {}
    for component in next(iter(records.values())):
        components_by_name[component] = pd.DataFrame(
            {subject: values[component] for subject, values in records.items()},
            index=expected.index,
        ).T
    return components_by_name, pd.DataFrame(audit_rows)


def main() -> None:
    hd_expected = load_expected(INPUT / "HD_all_subjects_regional_msn.csv")
    hc_expected = load_expected(INPUT / "HC_all_subjects_regional_msn.csv")
    hd_components, hd_audit = load_group("HD", hd_expected)
    hc_components, hc_audit = load_group("HC", hc_expected)
    pd.concat([hd_audit, hc_audit], ignore_index=True).to_csv(OUTPUT / "matrix_source_audit.csv", index=False)

    hd_meta = pd.read_excel(INPUT / "HD_MSN_clinical_plus_education_years.xlsx")
    hd_meta.columns = [str(column).strip() for column in hd_meta.columns]
    hd_meta["subject_key"] = hd_meta["NUM"].astype(str).str.strip().map(normalize_hd)
    hd_meta["age"] = pd.to_numeric(hd_meta["Age"], errors="coerce")
    hd_meta["sex"] = canonicalize_sex(hd_meta["Gender"])
    normalized_hd = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in hd_meta.columns}
    hd_edu_col = next((normalized_hd[k] for k in ["educationyears", "educationyear"] if k in normalized_hd), None)
    if hd_edu_col is None:
        raise KeyError("HD education years are required for the signed-edge MSN model.")
    hd_meta["education_years"] = pd.to_numeric(hd_meta[hd_edu_col], errors="coerce")

    etiv_path = INPUT / "eTIV_HD_HC_reconall.csv"
    if not etiv_path.exists():
        raise FileNotFoundError("eTIV_HD_HC_reconall.csv is required for the signed-edge MSN model.")
    etiv = pd.read_csv(etiv_path)
    etiv["subject_key"] = etiv["source_id"].astype(str).map(lambda x: normalize_hd(x) if "HD" in x else normalize_hc(x))
    etiv = etiv.dropna(subset=["subject_key"]).drop_duplicates("subject_key").set_index("subject_key")
    hd_meta = hd_meta.drop_duplicates("subject_key").set_index("subject_key")
    hd_meta["eTIV"] = pd.to_numeric(etiv["eTIV_mm3"], errors="coerce").reindex(hd_meta.index)

    hc_meta = pd.read_csv(INPUT / "hc_metadata.csv")
    hc_meta["subject_key"] = hc_meta["subject"].map(normalize_hc)
    hc_meta["age"] = pd.to_numeric(hc_meta["age"], errors="coerce")
    hc_meta["sex"] = canonicalize_sex(hc_meta["sex"])
    hc_meta["eTIV"] = pd.to_numeric(hc_meta["eTIV"], errors="coerce")
    hc_edu_path = INPUT / "hc_education_years.xlsx"
    if not hc_edu_path.exists():
        raise FileNotFoundError("HC education-years data are required for the signed-edge MSN model.")
    hc_edu = pd.read_excel(hc_edu_path)
    norm_hc = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in hc_edu.columns}
    id_col = norm_hc.get("subid") or norm_hc.get("subject")
    if id_col is None:
        raise KeyError("Could not identify the HC subject-ID column in the education table.")
    edu_candidates = [c for c in hc_edu.columns if re.sub(r"[^a-z0-9]", "", str(c).lower()) in {"educationyear", "educationyears"}]
    if not edu_candidates:
        raise KeyError("Could not identify the HC education-years column.")
    hc_edu["subject_key"] = hc_edu[id_col].astype(str).map(normalize_hc)
    hc_edu["education_years"] = pd.to_numeric(hc_edu[edu_candidates[0]], errors="coerce")
    hc_edu = hc_edu.drop_duplicates("subject_key").set_index("subject_key")
    hc_meta = hc_meta.drop_duplicates("subject_key").set_index("subject_key")
    hc_meta["education_years"] = hc_edu["education_years"].reindex(hc_meta.index)

    hd_subjects = [subject for subject in hd_expected.columns if normalize_hd(subject) in hd_meta.index]
    hc_subjects = [subject for subject in hc_expected.columns if normalize_hc(subject) in hc_meta.index]
    hd_cov = np.array([[hd_meta.loc[normalize_hd(subject), c] for c in MSN_COVARIATES] for subject in hd_subjects], dtype=float)
    hc_cov = np.array([[hc_meta.loc[normalize_hc(subject), c] for c in MSN_COVARIATES] for subject in hc_subjects], dtype=float)
    valid_hd = np.isfinite(hd_cov).all(axis=1)
    valid_hc = np.isfinite(hc_cov).all(axis=1)
    hd_subjects = list(np.asarray(hd_subjects)[valid_hd])
    hc_subjects = list(np.asarray(hc_subjects)[valid_hc])
    hd_cov = hd_cov[valid_hd]
    hc_cov = hc_cov[valid_hc]

    age = np.concatenate([hd_cov[:, 0], hc_cov[:, 0]])
    sex = np.concatenate([hd_cov[:, 1], hc_cov[:, 1]])
    education_years = np.concatenate([hd_cov[:, 2], hc_cov[:, 2]])
    etiv_values = np.concatenate([hd_cov[:, 3], hc_cov[:, 3]])
    group = np.concatenate([np.ones(len(hd_subjects)), np.zeros(len(hc_subjects))])
    model_rows = []
    maps = {}
    for component in hd_components:
        combined = np.vstack([
            hd_components[component].loc[hd_subjects].to_numpy(float),
            hc_components[component].loc[hc_subjects].to_numpy(float),
        ])
        component_rows = []
        for index, region in enumerate(hd_expected.index):
            beta, t_value, p_value = robust_group_model(combined[:, index], group, age, sex, education_years, etiv_values)
            component_rows.append({
                "component": component,
                "region_index": index + 1,
                "region": region,
                "beta_HD_minus_HC": beta,
                "t": t_value,
                "p": p_value,
                "mean_HD": float(np.mean(combined[:len(hd_subjects), index])),
                "mean_HC": float(np.mean(combined[len(hd_subjects):, index])),
                "N_HD": len(hd_subjects),
                "N_HC": len(hc_subjects),
            })
        table = pd.DataFrame(component_rows)
        table["q"] = qvalues(table["p"].to_numpy(float))
        table["FDR_significant"] = table["q"] < 0.05
        maps[component] = table
        model_rows.append(table)
    regional_models = pd.concat(model_rows, ignore_index=True)
    regional_models.to_csv(OUTPUT / "signed_edge_component_regional_models.csv", index=False)

    signed = maps["signed_mean"]
    similarity_rows = []
    for component, table in maps.items():
        x = signed["t"].to_numpy(float)
        y = table["t"].to_numpy(float)
        sig_x = signed["FDR_significant"].to_numpy(bool)
        sig_y = table["FDR_significant"].to_numpy(bool)
        union = int(np.sum(sig_x | sig_y))
        similarity_rows.append({
            "component": component,
            "pearson_r_with_signed_tmap": float(stats.pearsonr(x, y).statistic),
            "spearman_rho_with_signed_tmap": float(stats.spearmanr(x, y).statistic),
            "sign_agreement_with_signed_tmap": float(np.mean(np.sign(x) == np.sign(y))),
            "FDR_regions": int(sig_y.sum()),
            "FDR_overlap_with_signed": int(np.sum(sig_x & sig_y)),
            "FDR_jaccard_with_signed": float(np.sum(sig_x & sig_y) / union) if union else np.nan,
        })
    similarity = pd.DataFrame(similarity_rows)
    similarity.to_csv(OUTPUT / "signed_edge_map_similarity.csv", index=False)

    identity_error = max(
        float(np.max(np.abs(hd_components["signed_mean"].to_numpy() - hd_components["positive_contribution"].to_numpy() - hd_components["negative_contribution"].to_numpy()))),
        float(np.max(np.abs(hc_components["signed_mean"].to_numpy() - hc_components["positive_contribution"].to_numpy() - hc_components["negative_contribution"].to_numpy()))),
    )
    summary = {
        "N_HD": len(hd_subjects),
        "N_HC": len(hc_subjects),
        "regions": 308,
        "model": "HC3-robust linear regression adjusted for age, sex, education years, and eTIV",
        "covariates": MSN_COVARIATES,
        "signed_decomposition_identity_maximum_error": identity_error,
        "signed_FDR_regions": int(maps["signed_mean"]["FDR_significant"].sum()),
        "positive_contribution_FDR_regions": int(maps["positive_contribution"]["FDR_significant"].sum()),
        "negative_contribution_FDR_regions": int(maps["negative_contribution"]["FDR_significant"].sum()),
        "absolute_negative_contribution_FDR_regions": int(maps["absolute_negative_contribution"]["FDR_significant"].sum()),
        "positive_edge_proportion_FDR_regions": int(maps["positive_edge_proportion"]["FDR_significant"].sum()),
    }
    with open(OUTPUT / "signed_edge_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(similarity.to_string(index=False))


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Test signed-edge MSN HD-HC effects with robust regression adjusted for age, sex, education_years, and eTIV.
# Input source/location: HD/HC signed-edge or regional MSN data plus de-identified age, sex, education-years, and eTIV metadata from the configured input directory.
# Output location: Signed-edge sensitivity effect tables and associated summaries in the configured results directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Load and align subjects; enforce complete four-covariate cases; standardize continuous covariates; fit HC3-robust group models; export effects and QC summaries.
# Log location: No dedicated log file; progress and failures are written to standard output.
# =============================================================================
