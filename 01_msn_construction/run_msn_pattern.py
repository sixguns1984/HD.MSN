#!/usr/bin/env python3

import os
import sys
import re
import math
import argparse
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 14,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

def save_plot_with_pdf(png_path: str) -> None:
    """Save the active Matplotlib figure as 600-dpi PNG and matching 600-dpi PDF."""
    base, _ = os.path.splitext(png_path)
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    plt.savefig(base + '.pdf', dpi=600, bbox_inches='tight')


try:
    import pyreadr  # for .rds support if needed
    HAS_PYREADR = True
except Exception:
    HAS_PYREADR = False

VALID_EXT = ('.csv', '.tsv', '.txt', '.rds', '.Rds', '.RDS')
MSN_COVARIATES = ['age', 'sex', 'education_years', 'eTIV']


def parse_args():
    p = argparse.ArgumentParser(description='MSN pattern analysis (Python)')
    p.add_argument('hc_dir', type=str, help='HC directory containing one table per subject or a group matrix')
    p.add_argument('pt_dir', type=str, help='Patient directory containing one table per subject or a group matrix')
    p.add_argument('roi_names_file', type=str, help='Path to the ROI-name file, one ROI per line, expected length 308')
    p.add_argument('out_dir', type=str, help='Output directory')
    p.add_argument('--covariates_file', type=str, required=True, help='CSV/XLSX table with subject IDs and age, sex, education_years, and eTIV')
    p.add_argument('--z_thr', type=float, default=3.0, help='Outlier threshold |Z|')
    p.add_argument('--seed', type=int, default=1234, help='Random seed')
    return p.parse_args()


def _normalize_col(value: object) -> str:
    return re.sub(r'[^a-z0-9]', '', str(value).lower())


def canonicalize_sex(series: pd.Series) -> pd.Series:
    """Standardize numeric sex coding to 1/0 for all MSN models."""
    values = pd.to_numeric(series, errors='coerce')
    observed = set(values.dropna().unique().tolist())
    if observed and observed.issubset({1, 2}):
        return values.map({1: 1.0, 2: 0.0})
    if observed.issubset({0, 1}):
        return values.astype(float)
    raise ValueError('Sex must be encoded as 0/1 or 1/2 for MSN modeling.')


def load_covariates(path: str) -> pd.DataFrame:
    """Load a de-identified subject-by-covariate table with the fixed four MSN covariates."""
    ext = os.path.splitext(path)[1].lower()
    frame = pd.read_excel(path) if ext in {'.xlsx', '.xls'} else pd.read_csv(path)
    normalized = {_normalize_col(c): c for c in frame.columns}
    id_col = next((normalized[k] for k in ['subject', 'subjectid', 'participantid', 'patientid', 'id'] if k in normalized), None)
    if id_col is None:
        raise KeyError('Covariate table requires a de-identified subject-ID column.')
    mapping = {}
    for target, candidates in {
        'age': ['age'],
        'sex': ['sex', 'gender'],
        'education_years': ['educationyears', 'educationyear'],
        'eTIV': ['etiv'],
    }.items():
        source = next((normalized[k] for k in candidates if k in normalized), None)
        if source is None:
            raise KeyError(f'Missing required MSN covariate: {target}')
        mapping[target] = source
    out = pd.DataFrame(index=frame[id_col].astype(str).str.strip())
    out.index.name = 'subject'
    out['age'] = pd.to_numeric(frame[mapping['age']], errors='coerce').to_numpy()
    out['sex'] = canonicalize_sex(frame[mapping['sex']]).to_numpy()
    out['education_years'] = pd.to_numeric(frame[mapping['education_years']], errors='coerce').to_numpy()
    out['eTIV'] = pd.to_numeric(frame[mapping['eTIV']], errors='coerce').to_numpy()
    return out[~out.index.duplicated(keep='first')]


def covariate_adjust_msn(hc_mat: pd.DataFrame, pt_mat: pd.DataFrame, covariates: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Residualize regional MSN values using HC normative regressions with the fixed four covariates."""
    missing_hc = hc_mat.index.difference(covariates.index)
    missing_pt = pt_mat.index.difference(covariates.index)
    if len(missing_hc) or len(missing_pt):
        raise KeyError(f'Missing covariates for {len(missing_hc)} HC and {len(missing_pt)} patient participants.')
    hc_cov = covariates.reindex(hc_mat.index)[MSN_COVARIATES].copy()
    pt_cov = covariates.reindex(pt_mat.index)[MSN_COVARIATES].copy()
    if hc_cov.isna().any().any() or pt_cov.isna().any().any():
        raise RuntimeError('Complete age, sex, education_years, and eTIV are required for all MSN pattern participants.')

    hc_design_parts = [np.ones(len(hc_cov))]
    pt_design_parts = [np.ones(len(pt_cov))]
    for column in ['age', 'education_years', 'eTIV']:
        mean = hc_cov[column].mean()
        sd = hc_cov[column].std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            raise RuntimeError(f'Cannot standardize required covariate {column}.')
        hc_design_parts.append(((hc_cov[column] - mean) / sd).to_numpy(float))
        pt_design_parts.append(((pt_cov[column] - mean) / sd).to_numpy(float))
    # Insert sex between age and education_years to preserve the canonical covariate order.
    hc_design_parts.insert(2, hc_cov['sex'].to_numpy(float))
    pt_design_parts.insert(2, pt_cov['sex'].to_numpy(float))
    X_hc = np.column_stack(hc_design_parts)
    X_pt = np.column_stack(pt_design_parts)
    if np.linalg.matrix_rank(X_hc) < X_hc.shape[1]:
        raise RuntimeError('HC covariate design is rank deficient.')

    Y_hc = hc_mat.to_numpy(float)
    Y_pt = pt_mat.to_numpy(float)
    beta = np.linalg.lstsq(X_hc, Y_hc, rcond=None)[0]
    hc_residual = Y_hc - X_hc @ beta
    pt_residual = Y_pt - X_pt @ beta
    return (
        pd.DataFrame(hc_residual, index=hc_mat.index, columns=hc_mat.columns),
        pd.DataFrame(pt_residual, index=pt_mat.index, columns=pt_mat.columns),
    )


def read_roi_names(roi_file: str) -> List[str]:
    if not os.path.exists(roi_file):
        raise FileNotFoundError(f'ROI-name file does not exist: {roi_file}')
    df = pd.read_csv(roi_file, header=None, sep='\t|,', engine='python')
    names = df.iloc[:, 0].astype(str).str.strip().tolist()
    names = [n for n in names if n != '' and n.lower() != 'nan']
    if len(names) < 50:
        raise ValueError(f'Too few ROI names({len(names)}).')
    return names


def list_data_files(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        raise NotADirectoryError(f'Directory does not exist: {directory}')
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    files = [f for f in files if os.path.isfile(f) and f.endswith(VALID_EXT)]
    files.sort()
    if len(files) == 0:
        raise FileNotFoundError(f'No readable data files found in directory: {directory}')
    return files


def read_table_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        return pd.read_csv(path)
    if ext == '.tsv':
        return pd.read_csv(path, sep='\t')
    if ext == '.txt':
        # try tsv then csv
        try:
            return pd.read_csv(path, sep='\t')
        except Exception:
            return pd.read_csv(path)
    if ext in ('.rds', '.rds', '.rds'):
        if not HAS_PYREADR:
            raise RuntimeError(f'pyreadr is required to read RDS files: {path}')
        obj = pyreadr.read_r(path)
        # take the first object
        if len(obj.keys()) == 0:
            raise ValueError(f'RDS file is empty: {path}')
        key = list(obj.keys())[0]
        dt = obj[key]
        if isinstance(dt, pd.DataFrame):
            return dt
        else:
            raise ValueError(f'RDS file does not contain a DataFrame: {path}')
    raise ValueError(f'Unsupported file type: {path}')


def coerce_subject_vector(df: pd.DataFrame, subject_id: Optional[str] = None) -> pd.Series:
    df = df.loc[:, [c for c in df.columns if not (df[c].isna().all())]]

    # Case A: looks like an (n x (n+1)) similarity matrix with first column as row index
    try:
        if df.shape[1] >= 3:
            first = df.columns[0]
            rest = df.columns[1:]
            if all([col.isdigit() for col in map(str, rest)]) and df.shape[0] + 1 == df.shape[1]:
                mat = df.drop(columns=[first])
                n = mat.shape[0]
                if mat.shape[1] == n:
                    mat = mat.apply(pd.to_numeric, errors='coerce')
                    # row mean excluding diagonal
                    diag = np.diag(mat.values)
                    row_sum = mat.sum(axis=1).values
                    with np.errstate(invalid='ignore'):
                        vec = (row_sum - diag) / np.maximum(1, (n - 1))
                    v = pd.Series(vec)
                    # use numeric string indices 0..n-1 so we can later remap by position
                    v.index = [str(i) for i in range(n)]
                    return v
    except Exception:
        pass

    roi_col_candidates = ['roi','ROI','region','Region','label','Label','name','Name']
    has_roi_col = any([c in df.columns for c in roi_col_candidates])

    if has_roi_col:
        roi_col = [c for c in roi_col_candidates if c in df.columns][0]
        vals = df.drop(columns=[roi_col])
        num_cols = [c for c in vals.columns if pd.api.types.is_numeric_dtype(vals[c])]
        if len(num_cols) == 0:
            raise ValueError('No numeric column could be identified as the MSN value')
        v = vals[num_cols[0]]
        v.index = df[roi_col].astype(str).values
        return pd.to_numeric(v, errors='coerce')

    # if first column non-numeric and can be ROI names
    first = df.columns[0]
    if not pd.api.types.is_numeric_dtype(df[first]):
        num_cols = [c for c in df.columns[1:] if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) >= 1:
            v = df[num_cols[0]]
            v.index = df[first].astype(str).values
            return pd.to_numeric(v, errors='coerce')

    # otherwise take first numeric column as values, and if exists any non-numeric col, use it as names
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) >= 1:
        v = df[num_cols[0]]
        chr_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
        if len(chr_cols) >= 1:
            v.index = df[chr_cols[0]].astype(str).values
        return pd.to_numeric(v, errors='coerce')

    # fallback: flatten
    v = pd.Series(pd.to_numeric(df.stack(), errors='coerce').values)
    return v


def read_group(directory: str) -> Dict[str, pd.Series]:
    files = list_data_files(directory)
    # single matrix file case
    if len(files) == 1:
        df = read_table_any(files[0])
        # if first column is non-numeric, treat as rownames
        if df.shape[1] >= 2:
            first = df.columns[0]
            if not pd.api.types.is_numeric_dtype(df[first]):
                rownames = df[first].astype(str).values
                df = df.drop(columns=[first])
                df.index = rownames
        # majority numeric?
        num_ratio = np.mean([pd.api.types.is_numeric_dtype(df[c]) for c in df.columns])
        if num_ratio > 0.5:
            mat = df.copy()
            mat = mat.apply(pd.to_numeric, errors='coerce')
            # decide which axis is ROI (prefer 308 rows)
            if mat.shape[0] == 308:
                roi_names = list(mat.index)
                out = {}
                for j, col in enumerate(mat.columns):
                    v = pd.to_numeric(mat.iloc[:, j], errors='coerce')
                    v.index = roi_names
                    out[str(col)] = v
                return out
            elif mat.shape[1] == 308:
                roi_names = list(mat.columns)
                out = {}
                for i in range(mat.shape[0]):
                    v = pd.to_numeric(mat.iloc[i, :], errors='coerce')
                    v.index = roi_names
                    sid = str(mat.index[i]) if mat.index is not None else f'subj_{i+1}'
                    out[sid] = v
                return out
            else:
                raise ValueError('Could not determine which matrix dimension represents 308 ROIs; split the data by subject or provide ROI names.')
        raise ValueError('The single-file format could not be identified automatically; provide explicit matrix, column, or row labels.')

    # multi-file: one subject per file
    out = {}
    for f in files:
        sid = os.path.splitext(os.path.basename(f))[0]
        df = read_table_any(f)
        v = coerce_subject_vector(df, sid)
        out[sid] = v
    return out


def align_subjects(sub_map: Dict[str, pd.Series], roi_order: Optional[List[str]] = None) -> pd.DataFrame:
    # Determine names availability
    all_names = [list(v.index) if v.index is not None else [] for v in sub_map.values()]
    has_names = all(len(n) > 0 for n in all_names)

    if roi_order is not None and has_names:
        # Strategy 1: direct name overlap with provided order
        present_sets = [set(v.index) for v in sub_map.values()]
        common = set.intersection(*present_sets)
        roi_list = [r for r in roi_order if r in common]
        if len(roi_list) >= 50:
            rows = []
            index = []
            for sid, vec in sub_map.items():
                vv = pd.to_numeric(vec.reindex(roi_list), errors='coerce')
                rows.append(vv.values)
                index.append(sid)
            mat = pd.DataFrame(rows, index=index, columns=roi_list)
            mat = mat.loc[:, ~(mat.isna().all(axis=0))]
            return mat
        # Strategy 2: if indices look like numeric strings 0..n-1, align by position to provided ROI order
        lengths = {sid: len(v) for sid, v in sub_map.items()}
        if len(set(lengths.values())) == 1:
            L = list(lengths.values())[0]
            # check names are numeric strings increasing from 0..L-1 across subjects
            def is_numeric_0_to_Lm1(idx):
                try:
                    nums = [int(str(x)) for x in idx]
                    return sorted(nums) == list(range(L))
                except Exception:
                    return False
            if all(is_numeric_0_to_Lm1(v.index) for v in sub_map.values()):
                roi_list = roi_order[:L]
                rows = []
                index = []
                for sid, vec in sub_map.items():
                    vv = pd.to_numeric(pd.Series(vec).reset_index(drop=True), errors='coerce')
                    rows.append(vv.values)
                    index.append(sid)
                mat = pd.DataFrame(rows, index=index, columns=roi_list)
                mat = mat.loc[:, ~(mat.isna().all(axis=0))]
                return mat
        # if both strategies fail, error
        raise ValueError('The supplied ROI names have insufficient overlap with the data.')

    if has_names:
        # intersect names
        present_sets = [set(v.index) for v in sub_map.values()]
        roi_list = sorted(list(set.intersection(*present_sets)))
        if len(roi_list) == 0:
            raise ValueError('ROI naming is inconsistent across files and cannot be aligned.')
        rows = []
        index = []
        for sid, vec in sub_map.items():
            vv = pd.to_numeric(vec.reindex(roi_list), errors='coerce')
            rows.append(vv.values)
            index.append(sid)
        mat = pd.DataFrame(rows, index=index, columns=roi_list)
        mat = mat.loc[:, ~(mat.isna().all(axis=0))]
        return mat

    # fallback by position: use shortest length as template
    lengths = {sid: len(v) for sid, v in sub_map.items()}
    template_sid = min(lengths, key=lengths.get)
    L = lengths[template_sid]
    roi_list = [f'ROI_{i+1}' for i in range(L)]
    rows = []
    index = []
    for sid, vec in sub_map.items():
        vv = pd.to_numeric(pd.Series(vec).iloc[:L], errors='coerce')
        rows.append(vv.values)
        index.append(sid)
    mat = pd.DataFrame(rows, index=index, columns=roi_list)
    mat = mat.loc[:, ~(mat.isna().all(axis=0))]
    return mat


def infer_hemisphere(roi_names: List[str]) -> List[Optional[str]]:
    hemi = []
    for r in roi_names:
        low = r.lower()
        if re.search(r'(^l[^a-z]?|_l$|_lh$|^lh_|left|_left|/left)', low):
            hemi.append('L')
        elif re.search(r'(^r[^a-z]?|_r$|_rh$|^rh_|right|_right|/right)', low):
            hemi.append('R')
        else:
            hemi.append(None)
    return hemi


def save_tsne_plot(df_embed: pd.DataFrame, out_png: str, title: str):
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=df_embed, x='TSNE1', y='TSNE2', hue='Group', palette={'HC':'#1f78b4','PT':'#e31a1c'}, s=20, alpha=0.85, linewidth=0)
    plt.title(title)
    plt.tight_layout()
    save_plot_with_pdf(out_png)
    plt.close()


def save_tsne_patients_plot(df_embed: pd.DataFrame, out_png: str, title: str):
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=df_embed, x='TSNE1', y='TSNE2', hue='Cluster', palette='Set1', s=25, alpha=0.9, linewidth=0)
    plt.title(title)
    plt.tight_layout()
    save_plot_with_pdf(out_png)
    plt.close()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    hc_dir = os.path.abspath(args.hc_dir)
    pt_dir = os.path.abspath(args.pt_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    roi_order = read_roi_names(os.path.abspath(args.roi_names_file))

    print(f'Reading HC data: {hc_dir}')
    hc_map = read_group(hc_dir)
    print(f'Reading patient data: {pt_dir}')
    pt_map = read_group(pt_dir)

    # Align by provided ROI order, then enforce the fixed four-covariate policy.
    hc_mat = align_subjects(hc_map, roi_order=roi_order)
    pt_mat = align_subjects(pt_map, roi_order=roi_order)

    common_roi = [r for r in roi_order if r in hc_mat.columns and r in pt_mat.columns]
    if len(common_roi) < 50:
        raise ValueError('Too few shared ROIs between HC and patients (<50); check ROI naming and atlas consistency.')
    hc_mat = hc_mat.loc[:, common_roi]
    pt_mat = pt_mat.loc[:, common_roi]

    covariates = load_covariates(os.path.abspath(args.covariates_file))
    hc_adjusted, pt_adjusted = covariate_adjust_msn(hc_mat, pt_mat, covariates)

    # Normative Z scores are calculated from four-covariate-adjusted HC residuals.
    hc_mean = hc_adjusted.mean(axis=0)
    hc_sd = hc_adjusted.std(axis=0, ddof=1).replace(0, np.nan)
    pt_z = (pt_adjusted - hc_mean) / hc_sd

    z_out = pt_z.copy()
    z_out.insert(0, 'PatientID', z_out.index)
    z_out.to_csv(os.path.join(out_dir, 'patient_four_covariate_Z_scores.csv'), index=False)

    hc_stats = pd.DataFrame({'ROI': hc_mean.index, 'HC_adjusted_mean': hc_mean.values, 'HC_adjusted_sd': hc_sd.values})
    hc_stats['covariates'] = ';'.join(MSN_COVARIATES)
    hc_stats.to_csv(os.path.join(out_dir, 'HC_four_covariate_residual_mean_sd.csv'), index=False)

    # one-sample t-test per ROI against 0
    rows = []
    for r in common_roi:
        x = pt_z[r].replace([np.inf, -np.inf], np.nan).dropna().values
        if x.size < 3:
            rows.append((r, x.size, np.nan, np.nan, np.nan))
        else:
            tt = stats.ttest_1samp(x, popmean=0.0, nan_policy='omit')
            rows.append((r, x.size, float(np.nanmean(x)), float(tt.statistic), float(tt.pvalue)))
    roi_stats = pd.DataFrame(rows, columns=['ROI','n','meanZ','t','p'])
    mask = roi_stats['p'].notna()
    pvals = roi_stats.loc[mask, 'p'].values
    fdr = np.full_like(roi_stats['p'].values, np.nan, dtype=float)
    if pvals.size > 0:
        _, p_fdr, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
        fdr[mask.values] = p_fdr
    roi_stats['p_fdr'] = fdr
    roi_stats = roi_stats.sort_values('p_fdr', na_position='last')
    roi_stats['covariates'] = ';'.join(MSN_COVARIATES)
    roi_stats.to_csv(os.path.join(out_dir, 'one_sample_four_covariate_Z_FDR.csv'), index=False)

    # bar plot top meanZ
    top_n = min(25, len(common_roi))
    top_df = roi_stats.sort_values('meanZ', ascending=False).head(top_n).copy()
    top_df['sig'] = top_df['p_fdr'] < 0.05
    plt.figure(figsize=(8, 6))
    sns.barplot(data=top_df, y='ROI', x='meanZ', hue='sig', palette={False:'grey', True:'tomato'})
    plt.title(f'Mean four-covariate-adjusted patient MSN Z score (top {top_n} ROIs)')
    plt.ylabel('ROI')
    plt.xlabel('Mean Z')
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    save_plot_with_pdf(os.path.join(out_dir, 'roi_top_meanZ_bar.png'))
    plt.close()

    # t-SNE on four-covariate-adjusted MSN residuals.
    all_mat = pd.concat([hc_adjusted, pt_adjusted], axis=0)
    all_labels = np.array(['HC'] * hc_mat.shape[0] + ['PT'] * pt_mat.shape[0])
    perp = max(5, min(30, (all_mat.shape[0] - 1) // 3))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=args.seed, init='pca', learning_rate='auto')
    Y = tsne.fit_transform(all_mat.fillna(all_mat.mean()).values)
    embed = pd.DataFrame(Y, columns=['TSNE1','TSNE2'], index=all_mat.index)
    embed['Group'] = all_labels
    embed['ID'] = embed.index

    save_tsne_plot(embed.copy(), os.path.join(out_dir, 'tsne_all_participants.png'), f't-SNE on four-covariate-adjusted MSN (perplexity={perp})')

    # Patients clustering on t-SNE
    pt_embed = embed[embed['Group'] == 'PT'][['TSNE1','TSNE2']].copy()
    pt_ids = pt_embed.index.tolist()

    def choose_k(X: np.ndarray, k_range=range(2, 9)) -> Tuple[int, Optional[float]]:
        best_k, best_sil = 2, -np.inf
        for k in k_range:
            km = KMeans(n_clusters=k, n_init=20, random_state=args.seed)
            labels = km.fit_predict(X)
            if len(set(labels)) < 2:
                continue
            sil = silhouette_score(X, labels)
            if sil > best_sil:
                best_sil, best_k = sil, k
        if not np.isfinite(best_sil):
            return 5, None
        return best_k, float(best_sil)

    if pt_embed.shape[0] >= 5:
        k_sel, sil = choose_k(pt_embed.values, range(2, 9))
        km = KMeans(n_clusters=k_sel, n_init=50, random_state=args.seed)
        clus = km.fit_predict(pt_embed.values)
        clus_series = pd.Series(clus + 1, index=pt_ids)  # 1-based
    else:
        k_sel, sil = 1, None
        clus_series = pd.Series(np.ones(pt_embed.shape[0], dtype=int), index=pt_ids)

    embed['PT_cluster'] = np.nan
    embed.loc[pt_ids, 'PT_cluster'] = clus_series.values

    pt_embed_plot = embed.loc[pt_ids, ['TSNE1','TSNE2','PT_cluster']].copy()
    pt_embed_plot['Cluster'] = pt_embed_plot['PT_cluster'].astype(int).astype(str)
    save_tsne_patients_plot(pt_embed_plot, os.path.join(out_dir, 'tsne_patients_clusters.png'), f'Patient t-SNE clustering (k={k_sel})')

    pd.DataFrame({'PatientID': pt_ids, 'Cluster': clus_series.loc[pt_ids].astype(int).values}).to_csv(
        os.path.join(out_dir, 'patient_clusters.csv'), index=False
    )

    # burden and lateralization
    hemi = infer_hemisphere(common_roi)
    has_hemi = any([h in ('L','R') for h in hemi])

    abs_z = pt_z.abs()
    burden_total = (abs_z > args.z_thr).sum(axis=1).astype(int)

    if has_hemi:
        L_idx = [i for i, h in enumerate(hemi) if h == 'L']
        R_idx = [i for i, h in enumerate(hemi) if h == 'R']
        pt_L = (abs_z.iloc[:, L_idx] > args.z_thr).sum(axis=1).astype(int) if len(L_idx) > 0 else pd.Series(0, index=abs_z.index)
        pt_R = (abs_z.iloc[:, R_idx] > args.z_thr).sum(axis=1).astype(int) if len(R_idx) > 0 else pd.Series(0, index=abs_z.index)
        df_burden = pd.DataFrame({'PatientID': abs_z.index, 'Burden_L': pt_L.values, 'Burden_R': pt_R.values})
        df_burden['Burden_total'] = df_burden['Burden_L'] + df_burden['Burden_R']
        def lateral(row):
            if row['Burden_total'] == 0:
                return 'None'
            if row['Burden_L'] > 0 and row['Burden_R'] > 0:
                return 'Bilateral'
            if row['Burden_L'] > 0 and row['Burden_R'] == 0:
                return 'Left'
            if row['Burden_L'] == 0 and row['Burden_R'] > 0:
                return 'Right'
            return 'Unknown'
        df_burden['Lateralization'] = df_burden.apply(lateral, axis=1)
    else:
        df_burden = pd.DataFrame({'PatientID': abs_z.index, 'Burden_total': burden_total.values, 'Lateralization': 'Unknown'})

    df_burden.to_csv(os.path.join(out_dir, 'patient_burden_lateralization.csv'), index=False)

    # pie chart if we have hemi
    if 'Burden_L' in df_burden.columns:
        dist_tab = df_burden[df_burden['Burden_total'] > 0]['Lateralization'].value_counts(normalize=True)
        if dist_tab.sum() > 0:
            plt.figure(figsize=(6,5))
            plt.pie(dist_tab.values, labels=dist_tab.index, autopct='%1.1f%%', colors=sns.color_palette('Set2'))
            plt.title('MSN alteration burden lateralization (|Z|>3)')
            plt.tight_layout()
            save_plot_with_pdf(os.path.join(out_dir, 'burden_lateralization_pie.png'))
            plt.close()

    # heatmap of patient Z
    # order by cluster if available
    order = pt_ids
    if len(clus_series) == len(order):
        order = list(pd.Series(clus_series.values, index=clus_series.index).sort_values().index)
    z_for_heat = pt_z.loc[order, :]
    z_for_heat_plot = z_for_heat.clip(lower=-5, upper=5)

    plt.figure(figsize=(10, 8))
    sns.heatmap(z_for_heat_plot, cmap='vlag', center=0, cbar_kws={'label': 'Z'}, xticklabels=False, yticklabels=False)
    plt.tight_layout()
    save_plot_with_pdf(os.path.join(out_dir, 'patient_Z_heatmap.png'))
    plt.close()

    # summary
    with open(os.path.join(out_dir, 'summary.txt'), 'w', encoding='utf-8') as f:
        f.write('Dataset size\n')
        f.write(f'Number of HC participants: {hc_mat.shape[0]}\n')
        f.write(f'Number of patients: {pt_mat.shape[0]}\n')
        f.write(f'Number of shared ROIs: {len(common_roi)}\n\n')
        sig_num = int((roi_stats['p_fdr'] < 0.05).sum()) if roi_stats['p_fdr'].notna().any() else 0
        f.write(f'Number of significant ROIs in one-sample tests of four-covariate-adjusted Z scores (FDR < 0.05): {sig_num}\n')
        if sig_num > 0:
            top_row = roi_stats.sort_values(['meanZ'], ascending=False).iloc[0]
            f.write('Example top ROIs ranked by descending mean Z:\n')
            f.write(str(top_row.to_dict()) + '\n')
        if 'Burden_L' in df_burden.columns:
            dist = df_burden[df_burden['Burden_total'] > 0]['Lateralization'].value_counts()
            f.write('\nLaterality distribution among patients with nonzero load:\n')
            f.write(str(dist.to_dict()) + '\n')
        if pt_embed.shape[0] > 0:
            counts = pd.Series(clus_series.values).value_counts().sort_index()
            f.write(f"\nPatient clusters (k={k_sel}) counts by cluster:\n")
            f.write(str(counts.to_dict()) + '\n')

    print(f'Analysis completed. Results written to: {out_dir}')


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Quantify subject-level MSN pattern abnormalities after HC-normative adjustment for age, sex, education_years, and eTIV.
# Input source/location: Regional MSN values, ROI labels, and a de-identified subject-level covariate table supplied through command-line arguments.
# Output location: Four-covariate-adjusted Z-score tables, regional rankings, burden/laterality summaries, optional clustering results, text summaries, and visualization files in the selected output directory.
# Input/output notes: Complete values for all four MSN nuisance covariates are mandatory; outputs contain derived analysis products only.
# Main steps: Load and align MSN values; fit HC normative regressions with age, sex, education_years, and eTIV; residualize HC/patient regional MSN; derive adjusted Z scores; run regional tests; calculate burden/laterality metrics; optionally cluster adjusted patient patterns; save outputs.
# Log location: No dedicated log file; command-line status messages are written to standard output.
# =============================================================================
