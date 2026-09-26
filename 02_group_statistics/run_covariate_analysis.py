from __future__ import annotations

import re
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from private_clinical_schema import indicator_column, indicator_id, load_private_clinical_schema


ROOT = Path(os.environ.get("HD_MSN_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
BASE = Path(os.environ.get("HD_MSN_COVARIATE_WORK_DIR", ROOT / "results" / "covariate_analysis")).resolve()
INP = BASE / "input"
OUT = BASE / os.environ.get("COVARIATE_OUT_DIR", "results")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260807
ALPHA = 0.05
MSN_COVARIATES = ["age", "sex", "education_years", "eTIV"]


def robust_ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OLS coefficients, HC3 standard errors and two-sided t p-values."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    xtx_inv = np.linalg.pinv(X.T @ X)
    leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    denom = np.maximum(1.0 - leverage, 1e-8)
    meat = X.T @ ((resid / denom)[:, None] ** 2 * X)
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tval = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    df = max(X.shape[0] - X.shape[1], 1)
    pval = 2.0 * stats.t.sf(np.abs(tval), df=df)
    return beta, tval, pval


def normalize_id(x: object) -> str:
    m = re.match(r"(HD_sub\d+)", str(x))
    return m.group(1) if m else str(x)


def normalize_hc_id(x: object) -> str:
    m = re.match(r"(?:HC_)?sub(\d+)", str(x), flags=re.IGNORECASE)
    return f"HC_sub{int(m.group(1)):03d}" if m else str(x)


def canonicalize_sex(series: pd.Series) -> pd.Series:
    """Standardize numeric sex coding to 1/0 for consistent HD-HC modeling."""
    values = pd.to_numeric(series, errors="coerce")
    observed = set(values.dropna().unique().tolist())
    if observed and observed.issubset({1, 2}):
        return values.map({1: 1.0, 2: 0.0})
    if observed.issubset({0, 1}):
        return values.astype(float)
    raise ValueError("Sex must be encoded as 0/1 or 1/2 for MSN modeling.")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hd_raw = pd.read_csv(INP / "HD_individual_regional_ms" / "all_subjects_regional_msn.csv")
    hc_raw = pd.read_csv(INP / "HC_individual_regional_ms" / "all_subjects_regional_msn.csv")
    regions = [f"Region_{int(x) + 1}" for x in hd_raw["Region_Index"]]
    hd = hd_raw.drop(columns=["Region_Index"]).T
    hd.columns = regions
    hd.index = [normalize_id(x) for x in hd.index]
    hd = hd[~hd.index.duplicated(keep="first")].astype(float)
    hc = hc_raw.drop(columns=["Region_Index"]).T
    hc.columns = regions
    hc.index = [normalize_hc_id(x) for x in hc.index]
    hc = hc.astype(float)

    clinical_path = Path(os.environ.get("HD_MSN_HD_CLINICAL_FILE", INP / "hd_clinical_with_education_years.xlsx")).resolve()
    clinical = pd.read_excel(clinical_path)
    clinical.columns = [str(c).strip() for c in clinical.columns]
    clinical["subject"] = clinical["NUM"].astype(str).str.strip()
    clinical = clinical.set_index("subject")
    clinical["age"] = pd.to_numeric(clinical["Age"], errors="coerce")
    clinical["sex"] = canonicalize_sex(clinical["Gender"])
    stage_col = next(c for c in clinical.columns if c.lower().replace(" ", "") == "tfcstages")
    clinical["stage_raw"] = clinical[stage_col].astype(str).str.strip()
    clinical["stage_group"] = clinical["stage_raw"].map(stage_group)
    normalized_columns = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in clinical.columns}
    edu_col = next((normalized_columns[k] for k in ["educationyears", "educationyear"] if k in normalized_columns), None)
    if edu_col is None:
        raise KeyError("A numeric education-years column is required for the MSN covariate model.")
    clinical["education_years"] = pd.to_numeric(clinical[edu_col], errors="coerce")

    # Prefer eTIV parsed from the final FreeSurfer aseg.stats files.
    etiv_csv = INP / "eTIV_HD_HC_reconall.csv"
    if etiv_csv.exists():
        et = pd.read_csv(etiv_csv)
        et = et[et["group"].eq("HD")].copy()
        et["subject"] = et["source_id"].astype(str).str.extract(r"(HD_sub\d+)")[0]
        et = et.dropna(subset=["subject"]).drop_duplicates("subject").set_index("subject")
        clinical["eTIV"] = pd.to_numeric(et["eTIV_mm3"], errors="coerce").reindex(clinical.index)
    else:
        # Legacy fallback: discover the HD_sub ID column in the older workbook.
        etiv_source = pd.read_excel(INP / "hd_clinical_data.xlsx")
        id_candidates = [(col, int(etiv_source[col].astype(str).str.startswith("HD_sub").sum())) for col in etiv_source.columns]
        id_candidates = [x for x in id_candidates if x[1] >= 50]
        if id_candidates:
            id_col = max(id_candidates, key=lambda x: x[1])[0]
            et = etiv_source[[id_col, "eTIV"]].copy()
            et["subject"] = et[id_col].astype(str).str.extract(r"(HD_sub\d+)")[0]
            et = et.dropna(subset=["subject"]).drop_duplicates("subject").set_index("subject")
            clinical["eTIV"] = pd.to_numeric(et["eTIV"], errors="coerce").reindex(clinical.index)
        else:
            clinical["eTIV"] = np.nan

    # Keep only the clinical rows represented in the individual MSN matrix.
    meta = clinical.reindex(hd.index).copy()
    meta.index.name = "subject"
    return hd, hc, meta, clinical


def stage_group(value: object) -> str | float:
    s = str(value).strip().lower()
    if s in {"pre-hd", "prehd", "pre hd"}:
        return "preHD"
    if s == "1":
        return "stage1"
    if s in {"2", "3", "4"}:
        return "stage2plus"
    return np.nan



def qvalues(p: np.ndarray) -> np.ndarray:
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        pv = np.asarray(p[ok], dtype=float)
        order = np.argsort(pv)
        ranked = pv[order]
        qrank = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
        qrank = np.minimum.accumulate(qrank[::-1])[::-1]
        q = np.empty_like(qrank)
        q[order] = np.minimum(qrank, 1.0)
        out[ok] = q
    return out


def load_hc_covariates(hc_index: pd.Index) -> pd.DataFrame:
    """Load HC age, sex, education years, and eTIV using the fixed MSN schema."""
    hc_path = INP / "hc_metadata.csv"
    if not hc_path.exists():
        raise FileNotFoundError("HC metadata are required for the four-covariate MSN model.")
    hc_meta = pd.read_csv(hc_path)
    hc_meta["subject"] = hc_meta["subject"].astype(str).map(normalize_hc_id)
    hc_meta = hc_meta.drop_duplicates("subject").set_index("subject")
    hc_meta["age"] = pd.to_numeric(hc_meta["age"], errors="coerce")
    hc_meta["sex"] = canonicalize_sex(hc_meta["sex"])
    hc_meta["eTIV"] = pd.to_numeric(hc_meta["eTIV"], errors="coerce")

    hc_edu_path = INP / "hc_education_years.xlsx"
    if not hc_edu_path.exists():
        raise FileNotFoundError("HC education-years data are required for the four-covariate MSN model.")
    he = pd.read_excel(hc_edu_path)
    normalized = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in he.columns}
    subject_col = normalized.get("subid") or normalized.get("subject")
    if subject_col is None:
        raise KeyError("Could not identify the HC subject-ID column in the education-years table.")
    edu_candidates = [
        c for c in he.columns
        if c != subject_col and re.sub(r"[^a-z0-9]", "", str(c).lower()) in {"educationyear", "educationyears"}
    ]
    if not edu_candidates:
        raise KeyError("Could not identify a numeric HC education-years column.")
    he["subject"] = he[subject_col].astype(str).map(normalize_hc_id)
    he["education_years"] = pd.to_numeric(he[edu_candidates[0]], errors="coerce")
    dup_keep = os.environ.get("HC_EDU_DUP_POLICY", "first")
    if dup_keep == "exclude":
        he = he[~he["subject"].duplicated(keep=False)]
    else:
        he = he.drop_duplicates("subject", keep="last" if dup_keep == "last" else "first")
    he = he.set_index("subject")
    hc_meta["education_years"] = he["education_years"].reindex(hc_meta.index)
    return hc_meta.reindex(hc_index)[MSN_COVARIATES]


def fit_hd_hc_covariate_map(
    hd_subset: pd.DataFrame, hc: pd.DataFrame, hd_meta: pd.DataFrame, hc_meta: pd.DataFrame, model_name: str
) -> pd.DataFrame:
    """Fit an HD-HC regional map with exactly the four required nuisance covariates."""
    y = pd.concat([hd_subset, hc], axis=0)
    meta = pd.concat([hd_meta.reindex(hd_subset.index)[MSN_COVARIATES], hc_meta.reindex(hc.index)[MSN_COVARIATES]], axis=0)
    group = pd.Series([1.0] * len(hd_subset) + [0.0] * len(hc), index=y.index)
    table, _, status = fit_group_roi_models(y, meta, group, model_name, MSN_COVARIATES)
    if status != "ok":
        raise RuntimeError(f"Four-covariate HD-HC model failed: {status}")
    table["covariates"] = ";".join(MSN_COVARIATES)
    return table


def bootstrap_hd_hc_adjusted_ci(
    hd_subset: pd.DataFrame, hc: pd.DataFrame, hd_meta: pd.DataFrame, hc_meta: pd.DataFrame,
    label: str, rng: np.random.Generator, n_iter: int = 1000,
) -> dict[str, float]:
    """Bootstrap the four-covariate-adjusted HD-HC group coefficient for every ROI."""
    hd_dat = hd_meta.reindex(hd_subset.index)[MSN_COVARIATES].join(hd_subset).dropna()
    hc_dat = hc_meta.reindex(hc.index)[MSN_COVARIATES].join(hc).dropna()
    roi_cols = list(hd_subset.columns)
    boot = np.empty((n_iter, len(roi_cols)), dtype=float)

    def group_beta(hd_frame: pd.DataFrame, hc_frame: pd.DataFrame) -> np.ndarray:
        dat = pd.concat([hd_frame, hc_frame], axis=0).copy()
        group = np.r_[np.ones(len(hd_frame)), np.zeros(len(hc_frame))]
        columns = [np.ones(len(dat)), group]
        for covariate in ["age", "education_years", "eTIV"]:
            values = dat[covariate].to_numpy(float)
            sd = values.std(ddof=1)
            columns.append((values - values.mean()) / sd)
        columns.insert(3, dat["sex"].to_numpy(float))
        X = np.column_stack(columns)
        Y = dat[roi_cols].to_numpy(float)
        return np.linalg.lstsq(X, Y, rcond=None)[0][1]

    observed = group_beta(hd_dat, hc_dat)
    for iteration in range(n_iter):
        hd_sample = hd_dat.iloc[rng.integers(0, len(hd_dat), size=len(hd_dat))]
        hc_sample = hc_dat.iloc[rng.integers(0, len(hc_dat), size=len(hc_dat))]
        boot[iteration] = group_beta(hd_sample, hc_sample)
    lo = np.quantile(boot, 0.025, axis=0)
    hi = np.quantile(boot, 0.975, axis=0)
    sign_prob = np.mean(np.sign(boot) == np.sign(observed)[None, :], axis=0)
    pd.DataFrame({
        "ROI": roi_cols, "adjusted_group_beta": observed, "bootstrap_ci_low": lo,
        "bootstrap_ci_high": hi, "direction_stability": sign_prob,
        "ci_excludes_zero": (lo > 0) | (hi < 0), "covariates": ";".join(MSN_COVARIATES),
    }).to_csv(OUT / f"bootstrap_ci_{label}_vs_HC_four_covariates.csv", index=False)
    return {
        "bootstrap_iterations": n_iter,
        "bootstrap_ci_excludes_zero": int(((lo > 0) | (hi < 0)).sum()),
        "median_ci_width": float(np.median(hi - lo)),
        "median_direction_stability": float(np.median(sign_prob)),
        "q025_direction_stability": float(np.quantile(sign_prob, .025)),
        "q975_direction_stability": float(np.quantile(sign_prob, .975)),
        "covariates": ";".join(MSN_COVARIATES),
    }


def map_similarity(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, float]:
    aa = a.set_index("ROI").reindex(b["ROI"]).reset_index()
    x = aa["t"].to_numpy(float)
    y = b["t"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    sig_a = a.set_index("ROI").reindex(b["ROI"])["significant"].fillna(False).to_numpy(bool)[ok]
    sig_b = b.set_index("ROI").reindex(b["ROI"])["significant"].fillna(False).to_numpy(bool)[ok]
    inter = int(np.sum(sig_a & sig_b)); union = int(np.sum(sig_a | sig_b))
    return {
        "pearson_r_tmap": float(stats.pearsonr(x, y).statistic) if len(x) > 2 else np.nan,
        "spearman_r_tmap": float(stats.spearmanr(x, y).statistic) if len(x) > 2 else np.nan,
        "sign_agreement": float(np.mean(np.sign(x) == np.sign(y))) if len(x) else np.nan,
        "fdr_sig_A": int(sig_a.sum()), "fdr_sig_B": int(sig_b.sum()),
        "fdr_overlap": inter, "fdr_union": union,
        "fdr_jaccard": float(inter / union) if union else np.nan,
    }


def run_ablation(hd: pd.DataFrame, hc: pd.DataFrame, meta: pd.DataFrame, hc_meta: pd.DataFrame) -> None:
    """Run stage-removal and stage-contrast sensitivities with the same four MSN covariates."""
    valid = meta["stage_group"].notna()
    stage_ids = meta.index[valid]
    stage = meta.loc[stage_ids, "stage_group"]
    groups = {
        "all_stages": stage.index,
        "remove_preHD": stage.index[stage.isin(["stage1", "stage2plus"])],
        "stage2plus_only": stage.index[stage.eq("stage2plus")],
    }
    maps = {}
    for name, ids in groups.items():
        maps[name] = fit_hd_hc_covariate_map(
            hd.loc[list(ids)], hc, meta.loc[list(ids)], hc_meta, f"{name}_four_covariates"
        )
        maps[name].to_csv(OUT / f"ablation_{name}_vs_HC_four_covariates.csv", index=False)

    rows = []
    base = maps["all_stages"]
    boot_rows = []
    boot_rng = np.random.default_rng(SEED + 1)
    for name, tab in maps.items():
        rows.append({"analysis": name, "N_HD": len(groups[name]), "N_HC": len(hc), **map_similarity(base, tab), "covariates": ";".join(MSN_COVARIATES)})
        boot_rows.append({
            "analysis": name, "N_HD": len(groups[name]), "N_HC": len(hc),
            **bootstrap_hd_hc_adjusted_ci(hd.loc[list(groups[name])], hc, meta.loc[list(groups[name])], hc_meta, name, boot_rng),
        })

    pair_specs = [
        ("preHD_vs_stage1", "preHD", "stage1"),
        ("preHD_vs_stage2plus", "preHD", "stage2plus"),
        ("stage1_vs_stage2plus", "stage1", "stage2plus"),
    ]
    pair_maps = {}
    for name, a, b in pair_specs:
        ia = stage.index[stage.eq(a)]
        ib = stage.index[stage.eq(b)]
        ids = ia.append(ib)
        group = pd.Series([1.0] * len(ia) + [0.0] * len(ib), index=ids)
        table, _, status = fit_group_roi_models(
            hd.loc[ids], meta.loc[ids], group, f"{name}_four_covariates", MSN_COVARIATES
        )
        if status != "ok":
            raise RuntimeError(f"Stage contrast {name} failed: {status}")
        table["covariates"] = ";".join(MSN_COVARIATES)
        pair_maps[name] = table
        table.to_csv(OUT / f"stage_contrast_{name}_four_covariates.csv", index=False)
        rows.append({"analysis": name, "N_A": len(ia), "N_B": len(ib), **map_similarity(table, table), "covariates": ";".join(MSN_COVARIATES)})
    pd.DataFrame(rows).to_csv(OUT / "ablation_and_stage_map_stability_four_covariates.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(OUT / "ablation_bootstrap_ci_summary_four_covariates.csv", index=False)

    # Balanced resampling for the Stage 1 versus Stage 2+ comparison.
    rng = np.random.default_rng(SEED)
    ia = stage.index[stage.eq("stage1")].to_numpy()
    ib = stage.index[stage.eq("stage2plus")].to_numpy()
    full = pair_maps["stage1_vs_stage2plus"]
    n_bal = min(len(ia), len(ib))
    n_iter = 1000
    beta_maps = np.empty((n_iter, hd.shape[1]), dtype=float)
    sig_maps = np.zeros((n_iter, hd.shape[1]), dtype=bool)
    roi_cols = list(hd.columns)
    for iteration in range(n_iter):
        a_ids = rng.choice(ia, size=n_bal, replace=False)
        b_ids = rng.choice(ib, size=n_bal, replace=False)
        ids = pd.Index(np.r_[a_ids, b_ids])
        group = pd.Series(np.r_[np.ones(n_bal), np.zeros(n_bal)], index=ids)
        dat = meta.loc[ids, MSN_COVARIATES].join(hd.loc[ids]).dropna()
        group = group.reindex(dat.index)
        X = pd.DataFrame({"intercept": 1.0, "group": group}, index=dat.index)
        for covariate in ["age", "education_years", "eTIV"]:
            values = dat[covariate].astype(float)
            X[covariate] = (values - values.mean()) / values.std(ddof=1)
        X["sex"] = dat["sex"].astype(float)
        X = X[["intercept", "group", "age", "sex", "education_years", "eTIV"]]
        Xm = X.to_numpy(float)
        Y = dat[roi_cols].to_numpy(float)
        inv = np.linalg.pinv(Xm.T @ Xm)
        betas = inv @ Xm.T @ Y
        residuals = Y - Xm @ betas
        df = max(len(dat) - Xm.shape[1], 1)
        sigma2 = np.sum(residuals ** 2, axis=0) / df
        se = np.sqrt(np.maximum(sigma2 * inv[1, 1], 0.0))
        tvals = np.divide(betas[1], se, out=np.zeros_like(se), where=se > 0)
        pvals = 2 * stats.t.sf(np.abs(tvals), df)
        beta_maps[iteration] = betas[1]
        sig_maps[iteration] = qvalues(pvals) < ALPHA

    full_beta = full["beta"].to_numpy(float)
    corrs = np.array([stats.pearsonr(row, full_beta).statistic for row in beta_maps])
    signs = np.mean(np.sign(beta_maps) == np.sign(full_beta)[None, :], axis=1)
    fdr_freq = sig_maps.mean(axis=0)
    pd.DataFrame([{
        "contrast": "stage1_vs_stage2plus", "iterations": n_iter, "balanced_n_each": n_bal,
        "beta_map_r_median": np.nanmedian(corrs), "beta_map_r_q025": np.nanquantile(corrs, .025), "beta_map_r_q975": np.nanquantile(corrs, .975),
        "sign_agreement_median": np.nanmedian(signs), "sign_agreement_q025": np.nanquantile(signs, .025), "sign_agreement_q975": np.nanquantile(signs, .975),
        "covariates": ";".join(MSN_COVARIATES),
    }]).to_csv(OUT / "stage1_stage2plus_balanced_resampling_four_covariates_summary.csv", index=False)
    freq = full[["ROI", "beta", "q", "significant"]].copy()
    freq["fdr_selection_frequency"] = fdr_freq
    freq["covariates"] = ";".join(MSN_COVARIATES)
    freq.to_csv(OUT / "stage1_stage2plus_balanced_resampling_four_covariates_region_frequency.csv", index=False)


def design_matrix(meta: pd.DataFrame, covars: list[str], group: pd.Series | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Build a design matrix from explicitly named numeric covariates."""
    dat = meta.copy()
    X = pd.DataFrame(index=dat.index)
    cols: list[str] = []
    if group is not None:
        X["group"] = pd.to_numeric(group, errors="coerce")
        cols.append("group")
    for covariate in covars:
        if covariate not in {"age", "sex", "education_years", "eTIV"}:
            raise ValueError(f"Unsupported MSN nuisance covariate: {covariate}")
        X[covariate] = pd.to_numeric(dat[covariate], errors="coerce")
        cols.append(covariate)
    X.insert(0, "intercept", 1.0)
    return X, cols

def fit_roi_models(y: pd.DataFrame, meta: pd.DataFrame, predictor: pd.Series, model_name: str, covars: list[str], group: pd.Series | None = None) -> tuple[pd.DataFrame, int, str]:
    X0, predictor_cols = design_matrix(meta, covars, group=group)
    X0["predictor"] = pd.to_numeric(predictor, errors="coerce")
    dat = pd.concat([y, X0], axis=1)
    needed = list(y.columns) + list(X0.columns)
    dat = dat.dropna(subset=needed)
    if dat.empty:
        return pd.DataFrame(), 0, "no_complete_cases"
    X = dat[X0.columns].astype(float)
    if np.linalg.matrix_rank(X.to_numpy()) < X.shape[1] or len(dat) - X.shape[1] < 5:
        return pd.DataFrame(), len(dat), "rank_or_df_guard"
    rows = []
    for roi in y.columns:
        beta, tval, pval = robust_ols(dat[roi].astype(float).to_numpy(), X.to_numpy())
        j = list(X.columns).index("predictor")
        rows.append((roi, float(beta[j]), float(tval[j]), float(pval[j])))
    out = pd.DataFrame(rows, columns=["ROI", "beta", "t", "p"])
    out["q"] = qvalues(out["p"].to_numpy(float)); out["significant"] = out["q"] < ALPHA
    out["model"] = model_name; out["N"] = len(dat)
    return out, len(dat), "ok"


def fit_group_roi_models(y: pd.DataFrame, meta: pd.DataFrame, group: pd.Series,
                         model_name: str, covars: list[str]) -> tuple[pd.DataFrame, int, str]:
    """Fit a group/stage contrast for every ROI using HC3 robust standard errors."""
    X, _ = design_matrix(meta, covars, group=None)
    X.insert(1, "group", pd.to_numeric(group, errors="coerce"))
    dat = pd.concat([y, X], axis=1).dropna(subset=list(y.columns) + list(X.columns))
    if dat.empty:
        return pd.DataFrame(), 0, "no_complete_cases"
    Xdat = dat[X.columns].astype(float)
    if np.linalg.matrix_rank(Xdat.to_numpy()) < Xdat.shape[1] or len(dat) - Xdat.shape[1] < 5:
        return pd.DataFrame(), len(dat), "rank_or_df_guard"
    rows = []
    j = list(X.columns).index("group")
    for roi in y.columns:
        beta, tval, pval = robust_ols(dat[roi].astype(float).to_numpy(), Xdat.to_numpy())
        rows.append((roi, float(beta[j]), float(tval[j]), float(pval[j])))
    out = pd.DataFrame(rows, columns=["ROI", "beta", "t", "p"])
    out["q"] = qvalues(out["p"].to_numpy(float)); out["significant"] = out["q"] < ALPHA
    out["model"] = model_name; out["N"] = len(dat)
    return out, len(dat), "ok"


def compare_tables(base: pd.DataFrame, sens: pd.DataFrame, prefix: str) -> dict[str, float]:
    a = base.set_index("ROI").reindex(sens["ROI"]); b = sens.set_index("ROI").reindex(sens["ROI"])
    x = a["beta"].to_numpy(float); y = b["beta"].to_numpy(float); ok = np.isfinite(x) & np.isfinite(y)
    sa = a["significant"].fillna(False).to_numpy(bool); sb = b["significant"].fillna(False).to_numpy(bool)
    inter = int(np.sum(sa & sb)); union = int(np.sum(sa | sb))
    return {f"{prefix}_beta_pearson": float(stats.pearsonr(x[ok], y[ok]).statistic), f"{prefix}_beta_spearman": float(stats.spearmanr(x[ok], y[ok]).statistic),
            f"{prefix}_sign_agreement": float(np.mean(np.sign(x[ok]) == np.sign(y[ok]))), f"{prefix}_fdr_sig_base": int(sa.sum()), f"{prefix}_fdr_sig_sens": int(sb.sum()),
            f"{prefix}_fdr_jaccard": float(inter / union) if union else np.nan}


def run_clinical_models(hd: pd.DataFrame, meta: pd.DataFrame) -> None:
    """Fit selected private clinical-association models with four MSN covariates."""
    schema = load_private_clinical_schema()
    indices = schema.get("focused_outcome_indices", [])
    if not isinstance(indices, list) or not indices:
        raise ValueError("Private schema requires a non-empty focused_outcome_indices list for this analysis.")

    summaries = []
    all_tables = []
    model_name = "age_sex_education_years_eTIV"
    for raw_index in indices:
        index = int(raw_index)
        out_col = indicator_column(index, schema)
        public_id = indicator_id(index)
        if out_col not in meta.columns:
            raise KeyError(f"Private clinical source column for {public_id} is absent from the metadata table.")
        predictor = pd.to_numeric(meta[out_col], errors="coerce")
        tab, n, status = fit_roi_models(hd, meta, predictor, model_name, MSN_COVARIATES)
        summaries.append({
            "outcome_id": public_id, "model": model_name, "N": n, "status": status,
            "outcome_missing": int(predictor.isna().sum()),
            "covariates": ";".join(MSN_COVARIATES),
        })
        if status == "ok":
            current = tab.assign(outcome_id=public_id)
            current.to_csv(OUT / f"clinical_model_{public_id}_four_covariates.csv", index=False)
            all_tables.append(current)
    pd.DataFrame(summaries).to_csv(OUT / "clinical_covariate_model_summary.csv", index=False)
    if all_tables:
        pd.concat(all_tables, ignore_index=True).to_csv(OUT / "clinical_covariate_models_all.csv", index=False)


def run_stage_covariate_models(hd: pd.DataFrame, meta: pd.DataFrame) -> None:
    """Fit stage contrasts with age, sex, education years, and eTIV as nuisance covariates."""
    stage = meta["stage_group"]
    pairs = [("stage1", "stage2plus"), ("preHD", "stage2plus"), ("preHD", "stage1")]
    rows = []
    model_name = "age_sex_education_years_eTIV"
    for a, b in pairs:
        ids = stage.index[stage.isin([a, b])]
        local_meta = meta.loc[ids].copy()
        group = stage.loc[ids].eq(a).astype(float)
        tab, n, status = fit_group_roi_models(
            hd.loc[ids], local_meta, group=group, model_name=model_name, covars=MSN_COVARIATES
        )
        if status == "ok":
            tab.to_csv(OUT / f"stage_contrast_{a}_vs_{b}_four_covariates.csv", index=False)
        rows.append({
            "contrast": f"{a}_vs_{b}", "model": model_name, "N": n,
            "N_A": int((group == 1).sum()), "N_B": int((group == 0).sum()),
            "status": status, "covariates": ";".join(MSN_COVARIATES),
        })
    pd.DataFrame(rows).to_csv(OUT / "stage_contrast_covariate_audit.csv", index=False)

def run_hd_hc_four_covariates(hd: pd.DataFrame, hc: pd.DataFrame, meta: pd.DataFrame, hc_meta: pd.DataFrame) -> None:
    """Fit the primary HD-HC MSN model adjusted for the fixed four covariates."""
    tab = fit_hd_hc_covariate_map(hd, hc, meta, hc_meta, "age_sex_education_years_eTIV")
    tab.to_csv(OUT / "HD_vs_HC_four_covariates.csv", index=False)
    pd.DataFrame([{
        "model": "age_sex_education_years_eTIV",
        "N_HD": len(hd), "N_HC": len(hc),
        "FDR_significant": int(tab["significant"].sum()),
        "covariates": ";".join(MSN_COVARIATES),
    }]).to_csv(OUT / "HD_vs_HC_covariate_summary.csv", index=False)


def main() -> None:
    hd, hc, meta, clinical = load_inputs()
    hc_meta = load_hc_covariates(hc.index)
    meta.to_csv(OUT / "matched_HD_metadata_with_stage_education_years_eTIV.csv")
    run_ablation(hd, hc, meta, hc_meta)
    run_clinical_models(hd, meta)
    run_hd_hc_four_covariates(hd, hc, meta, hc_meta)
    run_stage_covariate_models(hd, meta)
    print("completed", OUT)


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Run regional MSN group, clinical, and disease-stage models using the fixed nuisance covariates age, sex, education_years, and eTIV.
# Input source/location: HD/HC regional MSN matrices plus de-identified clinical, education-years, and eTIV metadata from the configured input directory.
# Output location: Covariate-adjusted regional effect tables, model audits, resampling summaries, and metadata QC tables under results/covariate_analysis or HD_MSN_COVARIATE_WORK_DIR.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Harmonize metadata; require numeric four-covariate data; fit robust regional models; run clinical/stage contrasts and resampling sensitivities; export model diagnostics.
# Log location: No dedicated log file; exceptions and progress are written to standard output.
# =============================================================================
