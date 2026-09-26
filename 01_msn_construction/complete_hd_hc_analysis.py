#!/usr/bin/env python3
"""
Complete HD-versus-HC regional morphometric-similarity workflow.

Subject-level MSN matrices are treated as already constructed network objects.
Demographic nuisance variables are not inserted into the within-subject MSN
construction formula. All cross-subject MSN inference in this script uses the
fixed nuisance-covariate set: age, sex, education_years, and eTIV.
"""

import glob
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from private_clinical_schema import indicator_columns, indicator_id, load_private_clinical_schema

warnings.filterwarnings("ignore")

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = os.environ.get(
    "HD_MSN_DATA_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"),
)

HD308_DIR = os.path.join(BASE_DIR, "HD308")
HC308_DIR = os.path.join(BASE_DIR, "HC308")
CLINICAL_FILE = os.environ.get(
    "HD_MSN_CLINICAL_FILE",
    os.path.join(BASE_DIR, "clinical", "clinical_data.xlsx"),
)
HC_METADATA_FILE = os.environ.get(
    "HD_MSN_HC_METADATA_FILE",
    os.path.join(BASE_DIR, "clinical", "hc_metadata.csv"),
)
HC_EDUCATION_FILE = os.environ.get(
    "HD_MSN_HC_EDUCATION_FILE",
    os.path.join(BASE_DIR, "clinical", "hc_education_years.xlsx"),
)
ETIV_FILE = os.environ.get(
    "HD_MSN_ETIV_FILE",
    os.path.join(BASE_DIR, "clinical", "eTIV_HD_HC_reconall.csv"),
)

OUTPUT_BASE = os.path.join(BASE_DIR, "clinical_correlation_results")
HD_MS_DIR = os.path.join(BASE_DIR, "HD_regional_ms")
HC_MS_DIR = os.path.join(BASE_DIR, "HC_regional_ms")
HD_T_DIR = os.path.join(BASE_DIR, "HD_HC_group_statistics")

MSN_COVARIATES = ["age", "sex", "education_years", "eTIV"]

# Clinical-variable names are loaded at runtime from a private external schema.



# ============================================================================
# Utility functions
# ============================================================================

def ensure_dir(path):
    """Create a directory when it does not already exist."""
    os.makedirs(path, exist_ok=True)


def normalize_name(value):
    """Normalize a column label for robust schema matching."""
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(frame, *candidates):
    """Return the first matching column after punctuation/case normalization."""
    normalized = {normalize_name(column): column for column in frame.columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    return None


def normalize_hd_id(value):
    """Normalize an HD participant identifier without exposing source-system IDs."""
    match = re.search(r"HD_sub\d+", str(value), flags=re.IGNORECASE)
    return match.group(0) if match else str(value).strip()


def normalize_hc_id(value):
    """Normalize an HC participant identifier to HC_subNNN when possible."""
    match = re.search(r"(?:HC_)?sub(\d+)", str(value), flags=re.IGNORECASE)
    if match:
        return f"HC_sub{int(match.group(1)):03d}"
    return str(value).strip()


def canonicalize_sex(series):
    """Convert numeric sex coding to a consistent binary representation."""
    values = pd.to_numeric(series, errors="coerce")
    observed = set(values.dropna().unique().tolist())
    if observed and observed.issubset({1, 2}):
        return values.map({1: 1.0, 2: 0.0})
    if observed.issubset({0, 1}):
        return values.astype(float)
    raise ValueError(
        "Sex must be encoded numerically as 0/1 or 1/2 before MSN modeling."
    )


def load_similarity_matrix(file_path):
    """Load a similarity matrix from CSV."""
    try:
        return pd.read_csv(file_path, index_col=0).values
    except Exception as exc:
        print(f"Failed to load {file_path}: {exc}")
        return None


def matrix_to_ms_vector(matrix):
    """Aggregate a square MSN matrix to regional row means excluding the diagonal."""
    matrix = np.asarray(matrix, dtype=float)
    n = matrix.shape[0]
    masked = matrix.copy()
    np.fill_diagonal(masked, np.nan)
    return np.nanmean(masked, axis=1)


def extract_subject_id(filename):
    """Extract a de-identified participant ID from an MSN filename."""
    hd_match = re.search(r"(HD_sub\d+)", filename, flags=re.IGNORECASE)
    if hd_match:
        return normalize_hd_id(hd_match.group(1))
    hc_match = re.search(r"(?:HC_)?sub(\d+)", filename, flags=re.IGNORECASE)
    if hc_match:
        return normalize_hc_id(hc_match.group(0))
    return None


def robust_ols(y, design):
    """Fit OLS with HC3 robust standard errors and return beta/t/p vectors."""
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    xtx_inv = np.linalg.pinv(design.T @ design)
    leverage = np.einsum("ij,jk,ik->i", design, xtx_inv, design)
    denominator = np.maximum(1.0 - leverage, 1e-8)
    meat = design.T @ (((residual / denominator) ** 2)[:, None] * design)
    covariance = xtx_inv @ meat @ xtx_inv
    standard_error = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    t_value = np.divide(
        beta,
        standard_error,
        out=np.full_like(beta, np.nan, dtype=float),
        where=standard_error > 0,
    )
    degrees_of_freedom = max(len(y) - design.shape[1], 1)
    p_value = 2.0 * stats.t.sf(np.abs(t_value), degrees_of_freedom)
    return beta, t_value, p_value


def standardized_design(frame, include_group=False):
    """Create an inferential design using the fixed four nuisance covariates."""
    local = frame.copy()
    for column in ["age", "education_years", "eTIV"]:
        sd = local[column].std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            raise RuntimeError(f"Cannot standardize required covariate {column}.")
        local[f"{column}_z"] = (local[column] - local[column].mean()) / sd

    columns = [np.ones(len(local))]
    names = ["intercept"]
    if include_group:
        columns.append(local["group"].to_numpy(float))
        names.append("group")
    columns.extend(
        [
            local["age_z"].to_numpy(float),
            local["sex"].to_numpy(float),
            local["education_years_z"].to_numpy(float),
            local["eTIV_z"].to_numpy(float),
        ]
    )
    names.extend(MSN_COVARIATES)
    return np.column_stack(columns), names


# ============================================================================
# Covariate loading
# ============================================================================

def load_clinical_data(clinical_file):
    """Load de-identified HD clinical data and create a normalized subject index."""
    frame = pd.read_excel(clinical_file, sheet_name=0, header=0)
    frame.columns = [str(column).strip() for column in frame.columns]
    id_column = find_column(frame, "NUM", "subject", "subject_id")
    if id_column is None:
        raise KeyError("HD clinical data require a de-identified subject-ID column.")
    frame["ID"] = frame[id_column].astype(str).map(normalize_hd_id)
    frame = frame.drop_duplicates("ID").set_index("ID", drop=False)
    print(f"Loaded HD clinical data: {frame.shape}")
    return frame


def load_etiv_lookup():
    """Load eTIV values for both groups from a de-identified lookup table."""
    if not os.path.exists(ETIV_FILE):
        return pd.Series(dtype=float)
    frame = pd.read_csv(ETIV_FILE)
    source_column = find_column(frame, "source_id", "subject", "subject_id")
    etiv_column = find_column(frame, "eTIV_mm3", "eTIV")
    if source_column is None or etiv_column is None:
        raise KeyError("The eTIV table must contain a subject ID and eTIV column.")

    def normalize_any(value):
        value = str(value)
        return normalize_hd_id(value) if "HD" in value.upper() else normalize_hc_id(value)

    frame["subject_key"] = frame[source_column].astype(str).map(normalize_any)
    frame["eTIV_value"] = pd.to_numeric(frame[etiv_column], errors="coerce")
    return frame.drop_duplicates("subject_key").set_index("subject_key")["eTIV_value"]


def load_hd_covariates(clinical_df):
    """Extract the fixed four MSN nuisance covariates for HD participants."""
    age_column = find_column(clinical_df, "Age", "age")
    sex_column = find_column(clinical_df, "Gender", "Sex", "sex")
    education_column = find_column(
        clinical_df, "education_years", "education years", "educationyear", "educationyears"
    )
    if age_column is None or sex_column is None or education_column is None:
        raise KeyError(
            "HD metadata must contain age, sex/gender, and numeric education years."
        )

    meta = pd.DataFrame(index=clinical_df.index)
    meta["age"] = pd.to_numeric(clinical_df[age_column], errors="coerce")
    meta["sex"] = canonicalize_sex(clinical_df[sex_column])
    meta["education_years"] = pd.to_numeric(
        clinical_df[education_column], errors="coerce"
    )

    etiv_column = find_column(clinical_df, "eTIV")
    if etiv_column is not None:
        meta["eTIV"] = pd.to_numeric(clinical_df[etiv_column], errors="coerce")
    else:
        meta["eTIV"] = np.nan
    etiv_lookup = load_etiv_lookup()
    if not etiv_lookup.empty:
        meta["eTIV"] = meta["eTIV"].fillna(etiv_lookup.reindex(meta.index))
    return meta


def load_hc_covariates():
    """Load the fixed four MSN nuisance covariates for HC participants."""
    if not os.path.exists(HC_METADATA_FILE):
        raise FileNotFoundError(
            "HC metadata are required for the four-covariate HD-HC MSN model."
        )
    frame = pd.read_csv(HC_METADATA_FILE)
    id_column = find_column(frame, "subject", "subject_id", "subid")
    age_column = find_column(frame, "age")
    sex_column = find_column(frame, "sex", "gender")
    if id_column is None or age_column is None or sex_column is None:
        raise KeyError("HC metadata must contain subject ID, age, and sex.")
    frame["subject_key"] = frame[id_column].astype(str).map(normalize_hc_id)
    frame = frame.drop_duplicates("subject_key").set_index("subject_key")

    meta = pd.DataFrame(index=frame.index)
    meta["age"] = pd.to_numeric(frame[age_column], errors="coerce")
    meta["sex"] = canonicalize_sex(frame[sex_column])

    education_column = find_column(frame, "education_years", "education years")
    if education_column is not None:
        meta["education_years"] = pd.to_numeric(frame[education_column], errors="coerce")
    else:
        meta["education_years"] = np.nan

    if os.path.exists(HC_EDUCATION_FILE):
        education = pd.read_excel(HC_EDUCATION_FILE)
        education_id = find_column(education, "subject", "subject_id", "subid")
        education_value = find_column(
            education, "education_years", "education years", "educationyear", "educationyears"
        )
        if education_id is None or education_value is None:
            raise KeyError(
                "HC education data must contain a subject ID and numeric education-years column."
            )
        education["subject_key"] = education[education_id].astype(str).map(normalize_hc_id)
        education["education_years"] = pd.to_numeric(
            education[education_value], errors="coerce"
        )
        education = education.drop_duplicates("subject_key").set_index("subject_key")
        meta["education_years"] = meta["education_years"].fillna(
            education["education_years"].reindex(meta.index)
        )

    etiv_column = find_column(frame, "eTIV")
    if etiv_column is not None:
        meta["eTIV"] = pd.to_numeric(frame[etiv_column], errors="coerce")
    else:
        meta["eTIV"] = np.nan
    etiv_lookup = load_etiv_lookup()
    if not etiv_lookup.empty:
        meta["eTIV"] = meta["eTIV"].fillna(etiv_lookup.reindex(meta.index))
    return meta


# ============================================================================
# Regional MSN extraction
# ============================================================================

def calculate_ms_vectors():
    """Compute regional MSN row means for every HD and HC participant."""
    print("\n" + "=" * 70)
    print("Compute regional MSN vectors")
    print("=" * 70)
    ensure_dir(HD_MS_DIR)
    ensure_dir(HC_MS_DIR)

    outputs = {}
    for label, source_dir, destination in [
        ("HD", HD308_DIR, os.path.join(HD_MS_DIR, "HD_regional_ms.csv")),
        ("HC", HC308_DIR, os.path.join(HC_MS_DIR, "HC_regional_ms.csv")),
    ]:
        records = {}
        files = sorted(glob.glob(os.path.join(source_dir, "*_similarity_matrix.csv")))
        print(f"Processing {label}: {len(files)} matrices")
        for file_path in tqdm(files, desc=label):
            subject_id = extract_subject_id(os.path.basename(file_path))
            matrix = load_similarity_matrix(file_path)
            if subject_id is None or matrix is None or matrix.shape != (308, 308):
                continue
            records[subject_id] = matrix_to_ms_vector(matrix)
        frame = pd.DataFrame(records)
        frame.index = [f"Region_{index + 1}" for index in range(308)]
        frame.to_csv(destination)
        outputs[label] = frame
        print(f"Saved {label} regional MSN: {destination}; shape={frame.shape}")
    return outputs["HD"], outputs["HC"]


# ============================================================================
# Four-covariate HD-HC inference
# ============================================================================

def calculate_group_effects(hd_ms_df, hc_ms_df, hd_meta, hc_meta):
    """Fit HD-HC regional models with age, sex, education years, and eTIV."""
    print("\n" + "=" * 70)
    print("Four-covariate HD-HC regional MSN analysis")
    print("=" * 70)
    ensure_dir(HD_T_DIR)

    hd = hd_ms_df.T.copy()
    hd.index = [normalize_hd_id(value) for value in hd.index]
    hc = hc_ms_df.T.copy()
    hc.index = [normalize_hc_id(value) for value in hc.index]
    hd = hd[~hd.index.duplicated(keep="first")]
    hc = hc[~hc.index.duplicated(keep="first")]

    combined_y = pd.concat([hd, hc], axis=0)
    meta = pd.concat(
        [
            hd_meta.reindex(hd.index).assign(group=1.0),
            hc_meta.reindex(hc.index).assign(group=0.0),
        ],
        axis=0,
    )
    rows = []
    for region in tqdm(combined_y.columns, desc="regions"):
        data = meta[["group", *MSN_COVARIATES]].join(combined_y[[region]]).dropna()
        if len(data) < 10 or data["group"].nunique() != 2:
            continue
        design, design_names = standardized_design(data, include_group=True)
        if np.linalg.matrix_rank(design) < design.shape[1]:
            continue
        beta, t_value, p_value = robust_ols(data[region].to_numpy(float), design)
        group_index = design_names.index("group")
        hd_values = data.loc[data["group"].eq(1), region].to_numpy(float)
        hc_values = data.loc[data["group"].eq(0), region].to_numpy(float)
        rows.append(
            {
                "Region": region,
                "Beta_HD_minus_HC": float(beta[group_index]),
                "T_stat": float(t_value[group_index]),
                "P_value": float(p_value[group_index]),
                "N_HD": len(hd_values),
                "N_HC": len(hc_values),
                "Mean_HD": float(np.mean(hd_values)),
                "Mean_HC": float(np.mean(hc_values)),
                "Covariates": ";".join(MSN_COVARIATES),
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No complete HD-HC regional models could be fitted.")
    _, results["P_corrected"] = fdrcorrection(results["P_value"], alpha=0.05)
    results["Significant"] = results["P_corrected"] < 0.05
    results = results.sort_values("T_stat", ascending=False)
    full_path = os.path.join(HD_T_DIR, "HD_vs_HC_four_covariate_results.csv")
    significant_path = os.path.join(HD_T_DIR, "HD_vs_HC_four_covariate_significant_regions.csv")
    results.to_csv(full_path, index=False)
    results.loc[results["Significant"]].to_csv(significant_path, index=False)
    print(f"Saved four-covariate group results: {full_path}")
    return results


# ============================================================================
# Four-covariate clinical associations
# ============================================================================

def partial_pearson(x, y, covariates):
    """Partial Pearson correlation after residualizing both variables on four covariates."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    design = np.column_stack([np.ones(len(x)), covariates])
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return np.nan, np.nan
    x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if np.std(x_residual, ddof=1) == 0 or np.std(y_residual, ddof=1) == 0:
        return np.nan, np.nan
    r_value = float(stats.pearsonr(x_residual, y_residual).statistic)
    df = len(x) - covariates.shape[1] - 2
    if df <= 0 or abs(r_value) >= 1:
        return r_value, np.nan
    t_value = r_value * np.sqrt(df / max(1.0 - r_value ** 2, 1e-15))
    p_value = float(2.0 * stats.t.sf(abs(t_value), df))
    return r_value, p_value


def clinical_correlation_analysis(hd_ms_df, clinical_df, hd_meta):
    """Analyze private regional MSN-clinical associations using four covariates."""
    print("\n" + "=" * 70)
    print("Four-covariate regional MSN-clinical association analysis")
    print("=" * 70)
    ensure_dir(OUTPUT_BASE)

    schema = load_private_clinical_schema()
    configured_indicators = indicator_columns(schema)
    missing = [column for column in configured_indicators if column not in clinical_df.columns]
    if missing:
        raise KeyError(
            f"The private clinical schema references {len(missing)} source columns that are absent from the input table."
        )
    indicator_pairs = [(indicator_id(i), column) for i, column in enumerate(configured_indicators)]
    print(f"Clinical measures analyzed: {len(indicator_pairs)} private variables")

    hd = hd_ms_df.T.copy()
    hd.index = [normalize_hd_id(value) for value in hd.index]
    hd = hd[~hd.index.duplicated(keep="first")]
    correlations = []
    for region in tqdm(hd.columns, desc="regions"):
        for public_id, source_column in indicator_pairs:
            indicator_values = pd.to_numeric(clinical_df[source_column], errors="coerce")
            data = hd[[region]].join(indicator_values.rename("clinical"), how="inner")
            data = data.join(hd_meta[MSN_COVARIATES], how="inner").dropna()
            if len(data) < len(MSN_COVARIATES) + 4:
                continue
            cov = data[MSN_COVARIATES].copy()
            for column in ["age", "education_years", "eTIV"]:
                sd = cov[column].std(ddof=1)
                if not np.isfinite(sd) or sd == 0:
                    cov = None
                    break
                cov[column] = (cov[column] - cov[column].mean()) / sd
            if cov is None:
                continue
            r_value, p_value = partial_pearson(
                data[region].to_numpy(float),
                data["clinical"].to_numpy(float),
                cov[MSN_COVARIATES].to_numpy(float),
            )
            if np.isfinite(r_value) and np.isfinite(p_value):
                correlations.append(
                    {
                        "Region": region,
                        "Clinical_Indicator_ID": public_id,
                        "R": r_value,
                        "P": p_value,
                        "N": len(data),
                        "Covariates": ";".join(MSN_COVARIATES),
                    }
                )

    if not correlations:
        print("No valid four-covariate clinical association models were available.")
        return None
    results = pd.DataFrame(correlations)
    _, results["P_corrected"] = fdrcorrection(results["P"], alpha=0.05)
    results["Significant"] = results["P_corrected"] < 0.05
    full_path = os.path.join(OUTPUT_BASE, "all_four_covariate_clinical_associations.csv")
    significant_path = os.path.join(
        OUTPUT_BASE, "significant_four_covariate_clinical_associations.csv"
    )
    results.to_csv(full_path, index=False)
    results.loc[results["Significant"]].to_csv(significant_path, index=False)
    print(f"Saved four-covariate clinical associations: {full_path}")
    return results


# ============================================================================
# Main entry point
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("Complete HD-versus-HC regional MSN analysis")
    print("=" * 70)

    clinical_df = load_clinical_data(CLINICAL_FILE)
    hd_meta = load_hd_covariates(clinical_df)
    hc_meta = load_hc_covariates()

    hd_ms_df, hc_ms_df = calculate_ms_vectors()
    calculate_group_effects(hd_ms_df, hc_ms_df, hd_meta, hc_meta)
    clinical_correlation_analysis(hd_ms_df, clinical_df, hd_meta)

    print("\nAnalysis completed.")
    print(f"MS vectors: {HD_MS_DIR}, {HC_MS_DIR}")
    print(f"Four-covariate HD-HC statistics: {HD_T_DIR}")
    print(f"Four-covariate clinical associations: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Derive regional MSN summaries and run HD-HC plus MSN-clinical inference with the fixed nuisance covariates age, sex, education_years, and eTIV.
# Input source/location: De-identified HD/HC subject-level MSN matrices, private clinical metadata, HC demographic/education metadata, eTIV metadata, and an external private clinical schema.
# Output location: Project-relative regional MSN, HD-HC four-covariate statistics, and four-covariate clinical-association directories.
# Input/output notes: Within-subject MSN matrices are not demographically residualized inside this script; the four nuisance covariates are applied consistently at cross-subject inferential stages. Clinical-variable names are never embedded in the public source code.
# Main steps: Validate metadata; normalize de-identified IDs and sex coding; extract regional MSN row means; fit HC3-robust HD-HC models; residualize MSN and clinical measures on the four covariates; apply FDR correction; export machine-readable results.
# Log location: No dedicated log file; progress and validation messages are written to standard output.
# =============================================================================
