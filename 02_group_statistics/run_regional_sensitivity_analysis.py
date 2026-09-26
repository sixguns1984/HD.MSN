#!/usr/bin/env python3
"""Run regional MSN robustness analyses with portable, project-neutral paths.

Every numerical result is written to a machine-readable table before figures are
generated. All adjusted MSN models use age, sex, education years, and eTIV.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.cross_decomposition import PLSRegression

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from private_clinical_schema import indicator_column, load_private_clinical_schema


PURPLE = "#8E8BFE"
PINK = "#FEA3A2"
INK = "#111111"
GRAY = "#767676"
LIGHT = "#E7E7EC"
SEED = 20260814
MSN_COVARIATES = ["age", "sex", "education_years", "eTIV"]


def qvalues(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order]
    qrank = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    qrank = np.minimum.accumulate(qrank[::-1])[::-1]
    restored = np.empty_like(qrank)
    restored[order] = np.minimum(qrank, 1.0)
    out[ok] = restored
    return out


def robust_ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    xtx_inv = np.linalg.pinv(X.T @ X)
    leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    denom = np.maximum(1.0 - leverage, 1e-8)
    meat = X.T @ (((resid / denom) ** 2)[:, None] * X)
    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    tval = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    df = max(X.shape[0] - X.shape[1], 1)
    pval = 2 * stats.t.sf(np.abs(tval), df)
    return beta, cov, tval, pval


def normalize_hd_id(value: object) -> str:
    m = re.search(r"HD_sub\d+", str(value))
    return m.group(0) if m else str(value).strip()


def canonicalize_sex(series: pd.Series) -> pd.Series:
    """Standardize numeric sex coding to 1/0 for all MSN models."""
    values = pd.to_numeric(series, errors="coerce")
    observed = set(values.dropna().unique().tolist())
    if observed and observed.issubset({1, 2}):
        return values.map({1: 1.0, 2: 0.0})
    if observed.issubset({0, 1}):
        return values.astype(float)
    raise ValueError("Sex must be encoded as 0/1 or 1/2 for MSN modeling.")


def parent_region(name: str) -> str:
    return re.sub(r"_part\d+$", "", str(name))


def stage_group(value: object) -> str | float:
    s = str(value).strip().lower().replace(" ", "")
    if s in {"pre-hd", "prehd"}:
        return "preHD"
    if s in {"1", "1.0"}:
        return "stage1"
    if s in {"2", "2.0", "3", "3.0", "4", "4.0"}:
        return "stage2plus"
    return np.nan

def find_column(frame: pd.DataFrame, *normalized_names: str) -> str | None:
    normalized = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in frame.columns}
    return next((normalized[name] for name in normalized_names if name in normalized), None)


def attach_four_covariates(
    meta: pd.DataFrame, study_indexed: pd.DataFrame, clinical: pd.DataFrame, msn_root: Path
) -> pd.DataFrame:
    """Attach age, sex, education years, and eTIV to every participant in the MSN model."""
    out = meta.copy()

    study_edu = find_column(study_indexed, "educationyears", "educationyear")
    study_etiv = find_column(study_indexed, "etiv")
    if study_edu is not None:
        out["education_years"] = pd.to_numeric(study_indexed[study_edu], errors="coerce").reindex(out.index)
    else:
        out["education_years"] = np.nan
    if study_etiv is not None:
        out["eTIV"] = pd.to_numeric(study_indexed[study_etiv], errors="coerce").reindex(out.index)
    else:
        out["eTIV"] = np.nan

    hd_ids = out.index[out["diagnosis"].eq(1)]
    clinical_edu = find_column(clinical, "educationyears", "educationyear")
    if clinical_edu is not None:
        out.loc[hd_ids, "education_years"] = out.loc[hd_ids, "education_years"].fillna(
            pd.to_numeric(clinical[clinical_edu], errors="coerce").reindex(hd_ids)
        )
    clinical_etiv = find_column(clinical, "etiv")
    if clinical_etiv is not None:
        out.loc[hd_ids, "eTIV"] = out.loc[hd_ids, "eTIV"].fillna(
            pd.to_numeric(clinical[clinical_etiv], errors="coerce").reindex(hd_ids)
        )

    etiv_path = msn_root / "clinical" / "eTIV_HD_HC_reconall.csv"
    if etiv_path.exists():
        et = pd.read_csv(etiv_path)
        if {"source_id", "eTIV_mm3"}.issubset(et.columns):
            et["subject"] = et["source_id"].astype(str).map(
                lambda x: normalize_hd_id(x) if "HD" in x else (x if x.startswith("HC_") else "HC_" + re.sub(r"^sub", "", x))
            )
            et = et.drop_duplicates("subject").set_index("subject")
            out["eTIV"] = out["eTIV"].fillna(pd.to_numeric(et["eTIV_mm3"], errors="coerce").reindex(out.index))

    hc_ids = out.index[out["diagnosis"].eq(0)]
    hc_meta_path = msn_root / "clinical" / "hc_metadata.csv"
    if hc_meta_path.exists():
        hc = pd.read_csv(hc_meta_path)
        hc["subject_key"] = hc["subject"].astype(str).map(lambda x: x if x.startswith("HC_") else "HC_" + re.sub(r"^sub", "", x))
        hc = hc.drop_duplicates("subject_key").set_index("subject_key")
        if "eTIV" in hc.columns:
            out.loc[hc_ids, "eTIV"] = out.loc[hc_ids, "eTIV"].fillna(pd.to_numeric(hc["eTIV"], errors="coerce").reindex(hc_ids))

    hc_edu_path = msn_root / "clinical" / "hc_education_years.xlsx"
    if hc_edu_path.exists():
        he = pd.read_excel(hc_edu_path)
        id_col = find_column(he, "subid", "subject")
        edu_col = find_column(he, "educationyears", "educationyear")
        if id_col is not None and edu_col is not None:
            he["subject_key"] = he[id_col].astype(str).map(lambda x: x if x.startswith("HC_") else "HC_" + re.sub(r"^sub", "", x))
            he = he.drop_duplicates("subject_key").set_index("subject_key")
            out.loc[hc_ids, "education_years"] = out.loc[hc_ids, "education_years"].fillna(
                pd.to_numeric(he[edu_col], errors="coerce").reindex(hc_ids)
            )

    missing = out[MSN_COVARIATES].isna().sum()
    if int(missing.sum()) > 0:
        raise RuntimeError(f"Missing required MSN covariates after metadata merge: {missing.to_dict()}")
    return out


def fit_group_map(y: pd.DataFrame, meta: pd.DataFrame, include: pd.Index) -> pd.DataFrame:
    """Fit HD-HC regional MSN models using the fixed four nuisance covariates."""
    ids = include.intersection(y.index).intersection(meta.index)
    required = ["diagnosis", *MSN_COVARIATES]
    dat = meta.loc[ids, required].copy().join(y.loc[ids]).dropna()
    for column in ["age", "education_years", "eTIV"]:
        sd = dat[column].std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            raise RuntimeError(f"Cannot standardize required covariate {column}.")
        dat[f"{column}_z"] = (dat[column] - dat[column].mean()) / sd
    X = np.column_stack([
        np.ones(len(dat)),
        dat["diagnosis"].to_numpy(float),
        dat["age_z"].to_numpy(float),
        dat["sex"].to_numpy(float),
        dat["education_years_z"].to_numpy(float),
        dat["eTIV_z"].to_numpy(float),
    ])
    rows = []
    for roi in y.columns:
        beta, cov, tval, pval = robust_ols(dat[roi].to_numpy(float), X)
        rows.append((roi, beta[1], tval[1], pval[1]))
    out = pd.DataFrame(rows, columns=["region", "beta_HD_minus_HC", "t", "p"])
    out["q"] = qvalues(out["p"])
    out["significant"] = out["q"] < 0.05
    out["N"] = len(dat)
    out["N_HD"] = int(dat["diagnosis"].sum())
    out["N_HC"] = int((1 - dat["diagnosis"]).sum())
    out["covariates"] = ";".join(MSN_COVARIATES)
    return out

def map_comparison(base: pd.DataFrame, sensitivity: pd.DataFrame, value: str = "t") -> dict:
    a = base.set_index("region").reindex(sensitivity["region"])
    b = sensitivity.set_index("region")
    x, y = a[value].to_numpy(float), b[value].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    sa = a["significant"].fillna(False).to_numpy(bool)
    sb = b["significant"].fillna(False).to_numpy(bool)
    inter, union = int((sa & sb).sum()), int((sa | sb).sum())
    return {
        "pearson_r": float(stats.pearsonr(x[ok], y[ok]).statistic),
        "spearman_rho": float(stats.spearmanr(x[ok], y[ok]).statistic),
        "sign_agreement": float(np.mean(np.sign(x[ok]) == np.sign(y[ok]))),
        "median_absolute_change": float(np.median(np.abs(x[ok] - y[ok]))),
        "maximum_absolute_change": float(np.max(np.abs(x[ok] - y[ok]))),
        "fdr_regions_base": int(sa.sum()),
        "fdr_regions_sensitivity": int(sb.sum()),
        "fdr_overlap": inter,
        "fdr_jaccard": float(inter / union) if union else np.nan,
    }


def align_pls(scores: np.ndarray, weights: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(weights) & np.isfinite(reference)
    if stats.spearmanr(weights[ok], reference[ok]).statistic < 0:
        return -scores, -weights
    return scores, weights


def fit_pls(expression: pd.DataFrame, tmap: pd.DataFrame, reference: pd.Series) -> dict:
    merged = expression.merge(tmap[["region", "t"]], left_on="label", right_on="region", how="inner")
    genes = [c for c in expression.columns if c not in {"id", "label"}]
    X = merged[genes].to_numpy(float)
    y = merged["t"].to_numpy(float)
    complete = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[complete], y[complete]
    regions = merged.loc[complete, "region"].tolist()
    model = PLSRegression(n_components=1)
    model.fit(X, y)
    scores = model.x_scores_[:, 0]
    weights = model.x_weights_[:, 0]
    ref = reference.reindex(genes).to_numpy(float)
    scores, weights = align_pls(scores, weights, ref)
    return {
        "genes": np.asarray(genes), "regions": regions, "scores": scores,
        "weights": weights, "spatial_r": float(stats.pearsonr(scores, y).statistic),
        "n_regions": len(y), "n_genes": len(genes),
    }


def pls_comparison(a: dict, b: dict, positive_n: int, negative_n: int) -> dict:
    wa, wb = a["weights"], b["weights"]
    genes = a["genes"]
    pos_a = set(genes[np.argsort(wa)[-positive_n:]])
    pos_b = set(genes[np.argsort(wb)[-positive_n:]])
    neg_a = set(genes[np.argsort(wa)[:negative_n]])
    neg_b = set(genes[np.argsort(wb)[:negative_n]])
    return {
        "weight_pearson_r": float(stats.pearsonr(wa, wb).statistic),
        "weight_spearman_rho": float(stats.spearmanr(wa, wb).statistic),
        "positive_tail_n": positive_n,
        "positive_tail_jaccard": len(pos_a & pos_b) / len(pos_a | pos_b),
        "negative_tail_n": negative_n,
        "negative_tail_jaccard": len(neg_a & neg_b) / len(neg_a | neg_b),
        "spatial_r_base": a["spatial_r"],
        "spatial_r_sensitivity": b["spatial_r"],
        "n_regions_base": a["n_regions"],
        "n_regions_sensitivity": b["n_regions"],
    }


def design_vif(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in frame.columns:
        y = frame[col].to_numpy(float)
        others = [c for c in frame.columns if c != col]
        X = np.column_stack([np.ones(len(frame)), frame[others].to_numpy(float)])
        resid = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        tss = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum(resid ** 2) / tss if tss > 0 else 1.0
        rows.append((col, 1 / max(1 - r2, 1e-12)))
    return pd.DataFrame(rows, columns=["predictor", "VIF"])


def fit_stage_model(y: pd.DataFrame, meta: pd.DataFrame, covariates: list[str], model_name: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    needed = ["stage_group"] + covariates
    dat = meta[needed].join(y).dropna()
    stage1 = dat["stage_group"].eq("stage1").astype(float)
    stage2 = dat["stage_group"].eq("stage2plus").astype(float)
    Xdf = pd.DataFrame({"stage1_vs_preHD": stage1, "stage2plus_vs_preHD": stage2}, index=dat.index)
    for c in covariates:
        Xdf[c] = pd.to_numeric(dat[c], errors="coerce")
    X = np.column_stack([np.ones(len(dat)), Xdf.to_numpy(float)])
    names = ["intercept"] + Xdf.columns.tolist()
    rows, omnibus = [], []
    reduced_cols = [i for i, name in enumerate(names) if name not in {"stage1_vs_preHD", "stage2plus_vs_preHD"}]
    for roi in y.columns:
        yy = dat[roi].to_numpy(float)
        beta, cov, tval, pval = robust_ols(yy, X)
        for j, contrast in [(1, "stage1_vs_preHD"), (2, "stage2plus_vs_preHD")]:
            rows.append((roi, contrast, beta[j], tval[j], pval[j]))
        b = beta[[1, 2]]
        vc = cov[np.ix_([1, 2], [1, 2])]
        wald = float(b @ np.linalg.pinv(vc) @ b)
        p_joint = float(stats.chi2.sf(wald, 2))
        resid_full = yy - X @ beta
        Xr = X[:, reduced_cols]
        resid_red = yy - Xr @ np.linalg.lstsq(Xr, yy, rcond=None)[0]
        partial_r2 = max(0.0, 1 - np.sum(resid_full ** 2) / np.sum(resid_red ** 2))
        omnibus.append((roi, wald, p_joint, partial_r2))
    coef = pd.DataFrame(rows, columns=["region", "contrast", "beta", "t", "p"])
    coef["q"] = coef.groupby("contrast")["p"].transform(lambda s: qvalues(s.to_numpy()))
    coef["significant"] = coef["q"] < 0.05
    coef["model"] = model_name
    coef["N"] = len(dat)
    omni = pd.DataFrame(omnibus, columns=["region", "wald_chi2_df2", "p", "partial_R2_stage"])
    omni["q"] = qvalues(omni["p"])
    omni["significant"] = omni["q"] < 0.05
    omni["model"] = model_name
    omni["N"] = len(dat)
    vif = design_vif(Xdf).assign(model=model_name, N=len(dat))
    return coef, omni, vif


def compare_stage_coefficients(base: pd.DataFrame, adjusted: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for contrast in ["stage1_vs_preHD", "stage2plus_vs_preHD"]:
        a = base[base["contrast"].eq(contrast)].set_index("region")
        b = adjusted[adjusted["contrast"].eq(contrast)].set_index("region")
        x, y = a["beta"].to_numpy(float), b["beta"].to_numpy(float)
        sa, sb = a["significant"].to_numpy(bool), b["significant"].to_numpy(bool)
        union = int((sa | sb).sum())
        rows.append({
            "comparison": label, "contrast": contrast,
            "pearson_r": stats.pearsonr(x, y).statistic,
            "spearman_rho": stats.spearmanr(x, y).statistic,
            "sign_agreement": np.mean(np.sign(x) == np.sign(y)),
            "median_absolute_beta_change": np.median(np.abs(x - y)),
            "fdr_regions_base": int(sa.sum()), "fdr_regions_adjusted": int(sb.sum()),
            "fdr_jaccard": int((sa & sb).sum()) / union if union else np.nan,
        })
    return pd.DataFrame(rows)


def partial_spearman(x: np.ndarray, y: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    data = np.column_stack([x, y, cov])
    ranks = np.column_stack([stats.rankdata(data[:, j]) for j in range(data.shape[1])])
    C = np.column_stack([np.ones(len(data)), ranks[:, 2:]])
    rx = ranks[:, 0] - C @ np.linalg.lstsq(C, ranks[:, 0], rcond=None)[0]
    ry = ranks[:, 1] - C @ np.linalg.lstsq(C, ranks[:, 1], rcond=None)[0]
    return stats.pearsonr(rx, ry)


def tobit_nll(theta: np.ndarray, y: np.ndarray, X: np.ndarray, censored: np.ndarray, ceiling: float) -> float:
    beta, log_sigma = theta[:-1], theta[-1]
    sigma = np.exp(log_sigma)
    mu = X @ beta
    z = (y - mu) / sigma
    ll = np.empty(len(y))
    ll[~censored] = stats.norm.logpdf(z[~censored]) - log_sigma
    zc = (ceiling - mu[censored]) / sigma
    ll[censored] = stats.norm.logsf(zc)
    if not np.isfinite(ll).all():
        return 1e100
    return float(-ll.sum())


def numerical_hessian(fun, theta: np.ndarray) -> np.ndarray:
    n = len(theta)
    h = 1e-4 * np.maximum(np.abs(theta), 1.0)
    H = np.zeros((n, n))
    f0 = fun(theta)
    for i in range(n):
        ei = np.zeros(n); ei[i] = h[i]
        H[i, i] = (fun(theta + ei) - 2 * f0 + fun(theta - ei)) / (h[i] ** 2)
        for j in range(i):
            ej = np.zeros(n); ej[j] = h[j]
            H[i, j] = H[j, i] = (
                fun(theta + ei + ej) - fun(theta + ei - ej)
                - fun(theta - ei + ej) + fun(theta - ei - ej)
            ) / (4 * h[i] * h[j])
    return H


def fit_tobit(y: np.ndarray, X: np.ndarray, ceiling: float) -> dict:
    censored = y >= ceiling - 1e-10
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    sigma0 = max(np.std(y - X @ ols, ddof=X.shape[1]), 1.0)
    theta0 = np.r_[ols, np.log(sigma0)]
    fun = lambda th: tobit_nll(th, y, X, censored, ceiling)
    fit = optimize.minimize(fun, theta0, method="L-BFGS-B", options={"maxiter": 1000, "ftol": 1e-11})
    H = numerical_hessian(fun, fit.x)
    cov = np.linalg.pinv(H)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    z = fit.x / se
    p = 2 * stats.norm.sf(np.abs(z))
    return {
        "theta": fit.x, "se": se, "z": z, "p": p,
        "success": bool(fit.success), "nll": float(fit.fun),
        "censored_n": int(censored.sum()), "hessian_condition": float(np.linalg.cond(H)),
    }


def timed_outcome_analysis(
    hd_ms_path: Path, clinical_path: Path, region_names: list[str], out: Path, hd_meta: pd.DataFrame,
    source_column: str, ceiling: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Analyze a private timed clinical outcome with the fixed four MSN nuisance covariates."""
    ms = pd.read_csv(hd_ms_path, index_col=0)
    ms.index = region_names
    clin = pd.read_excel(clinical_path)
    clin["subject"] = clin["NUM"].astype(str).str.strip()
    clin = clin.set_index("subject")
    ids = ms.columns.intersection(clin.index).intersection(hd_meta.index)
    raw = clin.loc[ids, source_column]
    ynum = pd.to_numeric(raw, errors="coerce")
    base = hd_meta.loc[ids, MSN_COVARIATES].copy()
    base["timed_outcome"] = ynum
    base = base.dropna(subset=["timed_outcome", *MSN_COVARIATES])

    y = base["timed_outcome"].to_numpy(float)
    cov_raw = base[MSN_COVARIATES].to_numpy(float)
    cov_design = np.column_stack([
        (base["age"] - base["age"].mean()) / base["age"].std(ddof=1),
        base["sex"],
        (base["education_years"] - base["education_years"].mean()) / base["education_years"].std(ddof=1),
        (base["eTIV"] - base["eTIV"].mean()) / base["eTIV"].std(ddof=1),
    ]).astype(float)
    rows = []
    for roi in region_names:
        x = ms.loc[roi, base.index].to_numpy(float)
        xz = (x - x.mean()) / x.std(ddof=1)
        rho, p_rho = partial_spearman(x, y, cov_raw)
        X = np.column_stack([np.ones(len(y)), xz, cov_design])
        beta, cov, tval, pval = robust_ols(y, X)
        tob = fit_tobit(y, X, ceiling)
        ysd = y.std(ddof=1)
        rows.append({
            "region": roi, "N": len(y), "ceiling_n": int((y >= ceiling).sum()),
            "partial_spearman_rho": rho, "partial_spearman_p": p_rho,
            "ols_beta_seconds_per_SD_MS": beta[1], "ols_se": math.sqrt(max(cov[1, 1], 0)), "ols_p": pval[1],
            "ols_standardized_beta": beta[1] / ysd,
            "tobit_beta_seconds_per_SD_MS": tob["theta"][1], "tobit_se": tob["se"][1], "tobit_p": tob["p"][1],
            "tobit_standardized_beta": tob["theta"][1] / ysd,
            "tobit_success": tob["success"], "tobit_hessian_condition": tob["hessian_condition"],
            "covariates": ";".join(MSN_COVARIATES),
        })
    tab = pd.DataFrame(rows)
    for pcol, qcol in [("partial_spearman_p", "partial_spearman_q"), ("ols_p", "ols_q"), ("tobit_p", "tobit_q")]:
        tab[qcol] = qvalues(tab[pcol])

    inclusive = hd_meta.loc[ids, MSN_COVARIATES].copy()
    inclusive["timed_outcome"] = ynum
    inclusive["raw"] = raw
    noncompletion = inclusive["raw"].notna() & inclusive["timed_outcome"].isna()
    inclusive.loc[noncompletion, "timed_outcome"] = ceiling
    inclusive = inclusive.dropna(subset=["timed_outcome", *MSN_COVARIATES])
    yi = inclusive["timed_outcome"].to_numpy(float)
    cov_i = np.column_stack([
        (inclusive["age"] - inclusive["age"].mean()) / inclusive["age"].std(ddof=1),
        inclusive["sex"],
        (inclusive["education_years"] - inclusive["education_years"].mean()) / inclusive["education_years"].std(ddof=1),
        (inclusive["eTIV"] - inclusive["eTIV"].mean()) / inclusive["eTIV"].std(ddof=1),
    ]).astype(float)
    inc_rows = []
    for roi in region_names:
        x = ms.loc[roi, inclusive.index].to_numpy(float)
        xz = (x - x.mean()) / x.std(ddof=1)
        X = np.column_stack([np.ones(len(yi)), xz, cov_i])
        tob = fit_tobit(yi, X, ceiling)
        inc_rows.append({
            "region": roi, "N": len(yi), "ceiling_or_noncompletion_n": int((yi >= ceiling).sum()),
            "tobit_beta_seconds_per_SD_MS": tob["theta"][1], "tobit_se": tob["se"][1],
            "tobit_p": tob["p"][1], "tobit_standardized_beta": tob["theta"][1] / yi.std(ddof=1),
            "tobit_success": tob["success"], "covariates": ";".join(MSN_COVARIATES),
        })
    inc = pd.DataFrame(inc_rows)
    inc["tobit_q"] = qvalues(inc["tobit_p"])

    rng = np.random.default_rng(SEED)
    n_sim = 2000
    xs = rng.normal(size=n_sim)
    age_s = rng.normal(size=n_sim)
    sex_s = rng.integers(0, 2, n_sim)
    edu_s = rng.normal(size=n_sim)
    etiv_s = rng.normal(size=n_sim)
    Xs = np.column_stack([np.ones(n_sim), xs, age_s, sex_s, edu_s, etiv_s])
    true_effect = 0.075 * ceiling
    true_beta = np.array([0.75 * ceiling, true_effect, 0.033 * ceiling, 0.021 * ceiling, 0.017 * ceiling, 0.013 * ceiling])
    latent = Xs @ true_beta + rng.normal(scale=0.146 * ceiling, size=n_sim)
    observed = np.minimum(latent, ceiling)
    sim = fit_tobit(observed, Xs, ceiling)
    validation = {
        "N": n_sim, "true_beta_MS": true_effect, "estimated_beta_MS": float(sim["theta"][1]),
        "relative_error": float(abs(sim["theta"][1] - true_effect) / true_effect),
        "censored_fraction": float(np.mean(observed >= ceiling)), "optimizer_success": sim["success"],
        "covariates": MSN_COVARIATES,
    }
    tab.to_csv(out / "timed_outcome_rank_tobit_results.csv", index=False)
    inc.to_csv(out / "timed_outcome_inclusive_tobit_results.csv", index=False)
    (out / "timed_outcome_tobit_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    return tab, inc, validation

def configure_plotting() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 14, "axes.labelsize": 13, "axes.titlesize": 13,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "text.color": INK,
        "axes.labelcolor": INK, "axes.edgecolor": INK, "xtick.color": INK,
        "ytick.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def add_identity(ax, x: np.ndarray, y: np.ndarray) -> None:
    lo, hi = min(np.nanmin(x), np.nanmin(y)), max(np.nanmax(x), np.nanmax(y))
    ax.plot([lo, hi], [lo, hi], color=GRAY, lw=1.1, ls=(0, (4, 3)), zorder=1)


def make_imaging_figure(out: Path, full_map: pd.DataFrame, low_map: pd.DataFrame,
                        full_pls: dict, low_pls: dict, distributions: pd.DataFrame,
                        stage_coefs: dict[str, pd.DataFrame], dk68: pd.DataFrame,
                        dk_summary: dict, low_summary: dict, pls_low_summary: dict) -> None:
    configure_plotting()
    fig, axes = plt.subplots(3, 2, figsize=(14.2, 16.2))
    fig.subplots_adjust(left=.09, right=.98, top=.965, bottom=.07, hspace=.40, wspace=.30)

    ax = axes[0, 0]
    x, y = full_map["t"].to_numpy(), low_map["t"].to_numpy()
    ax.scatter(x, y, c=np.where(y >= 0, PINK, PURPLE), s=30, alpha=.78, edgecolor="white", linewidth=.35)
    add_identity(ax, x, y)
    ax.set(xlabel="All-carrier HD–HC t statistic", ylabel="After private genetic-range exclusion")
    ax.text(.03, .96, f"r = {low_summary['pearson_r']:.3f}\nρ = {low_summary['spearman_rho']:.3f}\nsign agreement = {100*low_summary['sign_agreement']:.1f}%", transform=ax.transAxes, va="top")

    ax = axes[0, 1]
    wa, wb = full_pls["weights"], low_pls["weights"]
    ax.scatter(wa, wb, c=np.where(wb >= 0, PINK, PURPLE), s=7, alpha=.22, edgecolor="none", rasterized=True)
    add_identity(ax, wa, wb)
    ax.set(xlabel="All-carrier PLS1 gene weight", ylabel="Private genetic range excluded")
    ax.text(.03, .96, f"weight ρ = {pls_low_summary['weight_spearman_rho']:.3f}\nPLS1+ Jaccard = {pls_low_summary['positive_tail_jaccard']:.3f}\nPLS1− Jaccard = {pls_low_summary['negative_tail_jaccard']:.3f}", transform=ax.transAxes, va="top")

    order = ["preHD", "stage1", "stage2plus"]
    labels = ["Premanifest", "Stage 1", "Stage 2+"]
    for ax, value, title in [(axes[1, 0], "genetic_measure", "Private genetic measure"), (axes[1, 1], "burden_measure", "Private burden measure")]:
        vals = [distributions.loc[distributions.stage_group.eq(g), value].dropna().to_numpy() for g in order]
        vp = ax.violinplot(vals, positions=np.arange(3), showextrema=False, widths=.82)
        for body in vp["bodies"]:
            body.set_facecolor(PINK); body.set_edgecolor(INK); body.set_alpha(.42)
        rng = np.random.default_rng(SEED)
        for i, v in enumerate(vals):
            ax.scatter(i + rng.uniform(-.11, .11, len(v)), v, s=24, color=PURPLE, alpha=.66, edgecolor="white", linewidth=.3)
            ax.boxplot(v, positions=[i], widths=.22, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor="white", edgecolor=INK), medianprops=dict(color=INK, lw=1.7),
                       whiskerprops=dict(color=INK), capprops=dict(color=INK))
        ax.set_xticks(range(3), labels)
        ax.set_ylabel(title)

    ax = axes[2, 0]
    base_genetic = stage_coefs["base_genetic_same_cases"]
    adj_genetic = stage_coefs["genetic_adjusted"]
    base_burden = stage_coefs["base_burden_same_cases"]
    adj_burden = stage_coefs["burden_adjusted"]
    specs = [(base_genetic, adj_genetic, PINK, "Add private genetic measure"), (base_burden, adj_burden, PURPLE, "Add private burden measure")]
    for base, adj, color, label in specs:
        for contrast, marker in [("stage1_vs_preHD", "o"), ("stage2plus_vs_preHD", "^")]:
            x = base.loc[base.contrast.eq(contrast), "beta"].to_numpy()
            y = adj.loc[adj.contrast.eq(contrast), "beta"].to_numpy()
            ax.scatter(x, y, s=20, marker=marker, color=color, alpha=.42, edgecolor="none", label=f"{label}; {contrast.replace('_vs_', ' vs ')}")
    allx = np.r_[base_genetic.beta, base_burden.beta]; ally = np.r_[adj_genetic.beta, adj_burden.beta]
    add_identity(ax, allx, ally)
    ax.set(xlabel="Base stage coefficient", ylabel="Genetic-burden-adjusted coefficient")
    ax.legend(frameon=False, fontsize=13, loc="best")

    ax = axes[2, 1]
    x, y = dk68["fine_parent_mean_t"].to_numpy(), dk68["DK68_t"].to_numpy()
    ax.scatter(x, y, c=np.where(y >= 0, PINK, PURPLE), s=45, alpha=.82, edgecolor="white", linewidth=.5)
    add_identity(ax, x, y)
    ax.set(xlabel="Mean DK308 t within DK68 parent", ylabel="DK68 participant-level aggregate t")
    ax.text(.03, .96, f"r = {dk_summary['t_pearson_r']:.3f}\nρ = {dk_summary['t_spearman_rho']:.3f}\nsign agreement = {100*dk_summary['sign_agreement']:.1f}%\nPLS gene-weight ρ = {dk_summary['PLS_weight_spearman_rho']:.3f}", transform=ax.transAxes, va="top")

    for letter, ax in zip("ABCDEF", axes.flat):
        ax.text(-.13, 1.06, letter, transform=ax.transAxes, fontsize=17, fontweight="bold", color=INK, va="top")
        ax.grid(color=LIGHT, lw=.65, alpha=.75); ax.spines[["top", "right"]].set_visible(False)
    stem = out / "genetic_stage_and_DK68_sensitivities"
    fig.savefig(stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def make_timed_outcome_figure(out: Path, tab: pd.DataFrame, inclusive: pd.DataFrame) -> None:
    configure_plotting()
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.4))
    fig.subplots_adjust(left=.07, right=.985, top=.92, bottom=.20, wspace=.46)
    ax = axes[0]
    x, y = tab["ols_standardized_beta"].to_numpy(), tab["partial_spearman_rho"].to_numpy()
    ax.scatter(x, y, c=np.where(y >= 0, PINK, PURPLE), s=31, alpha=.72, edgecolor="white", linewidth=.35)
    add_identity(ax, x, y)
    ax.set(xlabel="Four-covariate-adjusted linear coefficient (standardized)", ylabel="Partial Spearman ρ")
    ax.text(.03, .96, f"regional concordance ρ = {stats.spearmanr(x, y).statistic:.3f}", transform=ax.transAxes, va="top")

    ax = axes[1]
    x, y = tab["ols_standardized_beta"].to_numpy(), tab["tobit_standardized_beta"].to_numpy()
    ax.scatter(x, y, c=np.where(y >= 0, PINK, PURPLE), s=31, alpha=.72, edgecolor="white", linewidth=.35)
    add_identity(ax, x, y)
    ax.set(xlabel="Linear coefficient (standardized)", ylabel="Upper-censored Tobit coefficient (standardized)")
    ax.text(.03, .96, f"regional concordance ρ = {stats.spearmanr(x, y).statistic:.3f}", transform=ax.transAxes, va="top")

    ax = axes[2]
    # The primary clinical visualization displays Region_232 and Region_198; the table is in one-based
    # DK308 order, hence Python positions 231 and 197.
    fine_selected = [tab.iloc[231], tab.iloc[197]]
    metrics = [("Linear", "ols_standardized_beta", PINK), ("Partial Spearman", "partial_spearman_rho", PURPLE), ("Tobit", "tobit_standardized_beta", "#D96B8A")]
    ypos = np.arange(2)
    offsets = [-.18, 0, .18]
    for (label, col, color), off in zip(metrics, offsets):
        vals = np.array([r[col] for r in fine_selected])
        ax.hlines(ypos + off, 0, vals, color=color, lw=2.2)
        ax.scatter(vals, ypos + off, color=color, s=62, label=label, zorder=3, edgecolor="white", linewidth=.5)
    ax.axvline(0, color=GRAY, lw=1)
    ax.set_yticks(ypos, ["Postcentral gyrus", "Lateral orbitofrontal\ncortex"])
    ax.set_xlabel("Association with higher timed-outcome value")
    ax.legend(frameon=False, fontsize=13, loc="center", bbox_to_anchor=(.58, .50))

    for letter, ax in zip("ABC", axes):
        ax.text(-.14, 1.07, letter, transform=ax.transAxes, fontsize=17, fontweight="bold", va="top")
        ax.grid(color=LIGHT, lw=.65, alpha=.75); ax.spines[["top", "right"]].set_visible(False)
    stem = out / "timed_outcome_rank_and_censoring_sensitivities"
    fig.savefig(stem.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root, args.out
    out.mkdir(parents=True, exist_ok=True)

    msn = Path(os.environ.get("HD_MSN_MSN_ROOT", root / "msn")).resolve()
    full_path = Path(os.environ.get("HD_MSN_PARTICIPANT_MSN_FILE", msn / "group_data" / "full_compiled_data.csv")).resolve()
    clinical_path = Path(os.environ.get("HD_MSN_CLINICAL_FILE", msn / "clinical" / "clinical_data.xlsx")).resolve()
    names_path = Path(os.environ.get("HD_MSN_REGION_NAMES_FILE", root / "parcellation" / "308_regions_names.txt")).resolve()
    expression_path = Path(os.environ.get("HD_MSN_AHBA_EXPRESSION_FILE", root / "ahba" / "expression_with_label.csv")).resolve()
    weights_path = Path(os.environ.get("HD_MSN_REFERENCE_PLS_WEIGHTS", root / "pls" / "reference_pls1_gene_weights.csv")).resolve()
    hd_ms_path = Path(os.environ.get("HD_MSN_HD_REGIONAL_MS_FILE", msn / "regional_ms" / "HD_regional_ms.csv")).resolve()

    region_names = [line.strip() for line in names_path.read_text().splitlines() if line.strip()]
    if len(region_names) != 308:
        raise RuntimeError(f"Expected 308 names, found {len(region_names)}")

    full = pd.read_csv(full_path)
    roi_cols = [f"MSN_Region_{i}" for i in range(1, 309)]
    study = full[full["Group"].isin(["HD", "HC"])].copy()
    study["subject"] = np.where(study["Group"].eq("HD"), study["patient_id"].map(normalize_hd_id), "HC_" + study["sub_id"].astype(str))
    study["diagnosis"] = study["Group"].eq("HD").astype(float)
    study["age"] = pd.to_numeric(study["age"], errors="coerce")
    study["sex"] = canonicalize_sex(study["sex"])
    study_indexed = study.set_index("subject")
    y308 = study_indexed[roi_cols].astype(float)
    y308.columns = region_names
    meta = study_indexed[["diagnosis", "age", "sex", "Group"]].copy()

    clinical = pd.read_excel(clinical_path)
    clinical.columns = [str(c).strip() for c in clinical.columns]
    clinical["subject"] = clinical["NUM"].astype(str).str.strip()
    clinical = clinical.set_index("subject")

    private_schema = load_private_clinical_schema()
    roles = private_schema.get("regional_sensitivity_roles", {})
    required_role_keys = {
        "genetic_measure_index", "burden_measure_index", "timed_outcome_index",
        "stage_column", "genetic_exclusion_min", "genetic_exclusion_max", "timed_ceiling",
    }
    missing_roles = required_role_keys.difference(roles)
    if missing_roles:
        raise ValueError(f"Private schema is missing regional sensitivity role keys: {sorted(missing_roles)}")

    genetic_source_column = indicator_column(int(roles["genetic_measure_index"]), private_schema)
    burden_source_column = indicator_column(int(roles["burden_measure_index"]), private_schema)
    timed_source_column = indicator_column(int(roles["timed_outcome_index"]), private_schema)
    stage_col = str(roles["stage_column"])
    genetic_exclusion_min = float(roles["genetic_exclusion_min"])
    genetic_exclusion_max = float(roles["genetic_exclusion_max"])
    timed_ceiling = float(roles["timed_ceiling"])

    for source_column in [genetic_source_column, burden_source_column, timed_source_column, stage_col]:
        if source_column not in clinical.columns:
            raise KeyError("A private regional-sensitivity source column is absent from the clinical table.")

    meta = attach_four_covariates(meta, study_indexed, clinical, msn)
    hd_ids = meta.index[meta["diagnosis"].eq(1)]
    hd_meta = meta.loc[hd_ids].join(clinical, how="left", rsuffix="_clinical")
    hd_meta["genetic_measure"] = pd.to_numeric(hd_meta[genetic_source_column], errors="coerce")
    hd_meta["burden_measure"] = pd.to_numeric(hd_meta[burden_source_column], errors="coerce")
    hd_meta["stage_group"] = hd_meta[stage_col].map(stage_group)

    # Full and private genetic-range-excluded HD-HC group models.
    all_ids = meta.index
    low_ids = hd_meta.index[hd_meta["genetic_measure"].between(genetic_exclusion_min, genetic_exclusion_max, inclusive="both")]
    retained = all_ids.difference(low_ids)
    full_map = fit_group_map(y308, meta, all_ids)
    excluded_map = fit_group_map(y308, meta, retained)
    full_map.to_csv(out / "HD_HC_full_four_covariates.csv", index=False)
    excluded_map.to_csv(out / "HD_HC_genetic_range_excluded_four_covariates.csv", index=False)
    low_summary = map_comparison(full_map, excluded_map)
    low_summary.update({"excluded_n": len(low_ids), "excluded_ids": sorted(low_ids.tolist()), "genetic_measure_available_n": int(hd_meta["genetic_measure"].notna().sum())})

    expression = pd.read_csv(expression_path)
    reference_table = pd.read_csv(weights_path)
    reference = reference_table.set_index("gene")["PLS1_weight"]
    pos_n = int(reference_table["gene_set"].eq("PLS1_positive").sum()) or 350
    neg_n = int(reference_table["gene_set"].eq("PLS1_negative").sum()) or 154
    # The manuscript PLS is left-hemisphere only because right-hemisphere AHBA
    # tissue coverage is sparse. Preserve that exact 152-parcel analysis space.
    expression_fine_left = expression[expression["label"].astype(str).str.startswith("lh_")].copy()
    full_pls = fit_pls(expression_fine_left, full_map, reference)
    excluded_pls = fit_pls(expression_fine_left, excluded_map, reference)
    pls_low_summary = pls_comparison(full_pls, excluded_pls, pos_n, neg_n)
    pd.DataFrame({"gene": full_pls["genes"], "weight_all": full_pls["weights"], "weight_genetic_range_excluded": excluded_pls["weights"]}).to_csv(out / "genetic_range_exclusion_PLS_gene_weight_stability.csv", index=False)
    (out / "genetic_range_exclusion_summary.json").write_text(json.dumps({"regional_map": low_summary, "PLS": pls_low_summary}, indent=2) + "\n")

    # Disease-stage distribution and private-measure-adjusted carrier models.
    stage_data = hd_meta[["stage_group", "genetic_measure", "burden_measure", *MSN_COVARIATES]].copy()
    stage_data.to_csv(out / "stage_private_measures_analysis_table.csv")
    desc = []
    for group in ["preHD", "stage1", "stage2plus"]:
        for value in ["genetic_measure", "burden_measure"]:
            s = stage_data.loc[stage_data.stage_group.eq(group), value].dropna()
            desc.append({"stage_group": group, "measure": value, "N": len(s), "mean": s.mean(), "SD": s.std(ddof=1), "median": s.median(), "Q1": s.quantile(.25), "Q3": s.quantile(.75)})
    desc = pd.DataFrame(desc)
    desc.to_csv(out / "private_measures_by_stage_descriptive.csv", index=False)
    dist_tests = []
    for value in ["genetic_measure", "burden_measure"]:
        groups = [stage_data.loc[stage_data.stage_group.eq(g), value].dropna().to_numpy() for g in ["preHD", "stage1", "stage2plus"]]
        kw = stats.kruskal(*groups)
        dist_tests.append({"measure": value, "test": "Kruskal-Wallis", "statistic": kw.statistic, "p": kw.pvalue})
        for i, a in enumerate(["preHD", "stage1", "stage2plus"]):
            for b in ["preHD", "stage1", "stage2plus"][i+1:]:
                xa = stage_data.loc[stage_data.stage_group.eq(a), value].dropna()
                xb = stage_data.loc[stage_data.stage_group.eq(b), value].dropna()
                test = stats.mannwhitneyu(xa, xb, alternative="two-sided")
                dist_tests.append({"measure": value, "test": f"Mann-Whitney {a} vs {b}", "statistic": test.statistic, "p": test.pvalue})
    dist_tests = pd.DataFrame(dist_tests)
    dist_tests["q_within_measure"] = dist_tests.groupby("measure")["p"].transform(lambda s: qvalues(s.to_numpy()))
    dist_tests.to_csv(out / "private_measures_distribution_tests.csv", index=False)

    y_hd = y308.loc[hd_ids]
    genetic_cases = stage_data.dropna(subset=["stage_group", *MSN_COVARIATES, "genetic_measure"]).index
    burden_cases = stage_data.dropna(subset=["stage_group", *MSN_COVARIATES, "burden_measure"]).index
    stage_models = {}
    for label, cases, covars in [
        ("base_genetic_same_cases", genetic_cases, MSN_COVARIATES),
        ("genetic_adjusted", genetic_cases, [*MSN_COVARIATES, "genetic_measure"]),
        ("base_burden_same_cases", burden_cases, MSN_COVARIATES),
        ("burden_adjusted", burden_cases, [*MSN_COVARIATES, "burden_measure"]),
    ]:
        coef, omni, vif = fit_stage_model(y_hd.loc[cases], stage_data.loc[cases], covars, label)
        coef.to_csv(out / f"stage_coefficients_{label}.csv", index=False)
        omni.to_csv(out / f"stage_omnibus_{label}.csv", index=False)
        vif.to_csv(out / f"stage_VIF_{label}.csv", index=False)
        stage_models[label] = coef
    stage_compare = pd.concat([
        compare_stage_coefficients(stage_models["base_genetic_same_cases"], stage_models["genetic_adjusted"], "add_genetic_measure"),
        compare_stage_coefficients(stage_models["base_burden_same_cases"], stage_models["burden_adjusted"], "add_burden_measure"),
    ], ignore_index=True)
    stage_compare.to_csv(out / "stage_effect_stability_summary.csv", index=False)

    # Participant-level aggregation from DK308 parcels to 68 parent regions.
    parent_lookup = pd.Series({r: parent_region(r) for r in region_names})
    parent_names = list(dict.fromkeys(parent_lookup.tolist()))
    y68 = pd.DataFrame(index=y308.index)
    for parent in parent_names:
        children = parent_lookup.index[parent_lookup.eq(parent)].tolist()
        y68[parent] = y308[children].mean(axis=1)
    dk68_map = fit_group_map(y68, meta, all_ids)
    fine_parent = full_map.assign(parent=full_map["region"].map(parent_region)).groupby("parent", sort=False).agg(
        fine_parent_mean_t=("t", "mean"), fine_parent_mean_beta=("beta_HD_minus_HC", "mean"),
        fine_any_FDR=("significant", "any"), child_n=("region", "size"),
    ).reset_index()
    dk68 = dk68_map.rename(columns={"region": "parent", "t": "DK68_t", "beta_HD_minus_HC": "DK68_beta", "q": "DK68_q", "significant": "DK68_FDR"}).merge(fine_parent, on="parent")
    dk68.to_csv(out / "DK68_vs_DK308_parent_comparison.csv", index=False)
    x, y = dk68["fine_parent_mean_t"].to_numpy(), dk68["DK68_t"].to_numpy()
    union = (dk68["fine_any_FDR"] | dk68["DK68_FDR"]).sum()
    dk_summary = {
        "N_DK308": 308, "N_DK68": 68,
        "t_pearson_r": float(stats.pearsonr(x, y).statistic),
        "t_spearman_rho": float(stats.spearmanr(x, y).statistic),
        "sign_agreement": float(np.mean(np.sign(x) == np.sign(y))),
        "DK68_FDR_regions": int(dk68["DK68_FDR"].sum()),
        "DK68_parents_with_any_DK308_FDR_child": int(dk68["fine_any_FDR"].sum()),
        "FDR_parent_jaccard": float((dk68["fine_any_FDR"] & dk68["DK68_FDR"]).sum() / union) if union else np.nan,
    }
    expr_left = expression[expression["label"].isin([r for r in region_names if r.startswith("lh_")])].copy()
    expr_left["parent"] = expr_left["label"].map(parent_region)
    genes = [c for c in expression.columns if c not in {"id", "label"}]
    expr68 = expr_left.groupby("parent", sort=False)[genes].mean().reset_index().rename(columns={"parent": "label"})
    expr68.insert(0, "id", np.arange(len(expr68)))
    dk68_pls_map = dk68_map[dk68_map.region.str.startswith("lh_")]
    dk68_pls = fit_pls(expr68, dk68_pls_map, reference)
    dk_pls_compare = pls_comparison(full_pls, dk68_pls, pos_n, neg_n)
    dk_summary.update({
        "PLS_weight_pearson_r": dk_pls_compare["weight_pearson_r"],
        "PLS_weight_spearman_rho": dk_pls_compare["weight_spearman_rho"],
        "PLS_positive_tail_jaccard": dk_pls_compare["positive_tail_jaccard"],
        "PLS_negative_tail_jaccard": dk_pls_compare["negative_tail_jaccard"],
        "PLS_spatial_r_DK308": full_pls["spatial_r"], "PLS_spatial_r_DK68": dk68_pls["spatial_r"],
        "PLS_regions_DK308": full_pls["n_regions"], "PLS_regions_DK68": dk68_pls["n_regions"],
    })
    pd.DataFrame({"gene": full_pls["genes"], "DK308_weight": full_pls["weights"], "DK68_weight": dk68_pls["weights"]}).to_csv(out / "DK68_DK308_PLS_gene_weight_stability.csv", index=False)
    (out / "parcellation_scale_summary.json").write_text(json.dumps(dk_summary, indent=2) + "\n")

    # Private timed-outcome rank/censoring sensitivity analysis.
    timed_tab, timed_inclusive, timed_validation = timed_outcome_analysis(
        hd_ms_path, clinical_path, region_names, out, hd_meta, timed_source_column, timed_ceiling
    )
    timed_summary = {
        "numeric_records_N": int(timed_tab.N.iloc[0]), "numeric_ceiling_N": int(timed_tab.ceiling_n.iloc[0]),
        "all_tobit_optimizers_successful": bool(timed_tab.tobit_success.all()),
        "OLS_vs_partial_Spearman_regional_rho": float(stats.spearmanr(timed_tab.ols_standardized_beta, timed_tab.partial_spearman_rho).statistic),
        "OLS_vs_Tobit_regional_rho": float(stats.spearmanr(timed_tab.ols_standardized_beta, timed_tab.tobit_standardized_beta).statistic),
        "Tobit_numeric_vs_inclusive_regional_rho": float(stats.spearmanr(timed_tab.tobit_standardized_beta, timed_inclusive.tobit_standardized_beta).statistic),
        "partial_Spearman_FDR_regions": int((timed_tab.partial_spearman_q < .05).sum()),
        "Tobit_FDR_regions": int((timed_tab.tobit_q < .05).sum()),
        "inclusive_Tobit_FDR_regions": int((timed_inclusive.tobit_q < .05).sum()),
        "implementation_validation": timed_validation,
    }
    (out / "timed_outcome_rank_censoring_summary.json").write_text(json.dumps(timed_summary, indent=2) + "\n")

    make_imaging_figure(out, full_map, excluded_map, full_pls, excluded_pls,
                        stage_data, stage_models, dk68, dk_summary, low_summary, pls_low_summary)
    make_timed_outcome_figure(out, timed_tab, timed_inclusive)

    input_manifest = pd.DataFrame([
        {"input": str(full_path), "role": "participant-level DK308 MSN and HD/HC covariates"},
        {"input": str(clinical_path), "role": "private sensitivity phenotypes and stage metadata"},
        {"input": str(names_path), "role": "DK308-to-DK68 parent labels"},
        {"input": str(expression_path), "role": "AHBA DK308 expression matrix"},
        {"input": str(weights_path), "role": "PLS1 reference gene ranking"},
        {"input": str(hd_ms_path), "role": "participant-level regional MS for clinical visualization"},
    ])
    input_manifest.to_csv(out / "input_provenance.csv", index=False)
    final = {"genetic_range_exclusion": {"regional_map": low_summary, "PLS": pls_low_summary}, "stage_robustness": stage_compare.to_dict("records"), "parcellation_scale": dk_summary, "timed_outcome_rank_censoring": timed_summary}
    (out / "regional_sensitivity_summary.json").write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Run prespecified regional MSN sensitivity analyses, including four-covariate HD-HC models, disease-stage contrasts, and clinical robustness checks.
# Input source/location: Regional MSN matrices and de-identified demographic/clinical covariates from the configured project input tree.
# Output location: Sensitivity tables and figures in the configured results directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Attach age, sex, education_years, and eTIV; fit robust regional models; evaluate stage/clinical sensitivities; calculate multiple-comparison statistics; save results and figures.
# Log location: No dedicated log file unless configured externally; progress is written to standard output.
# =============================================================================
