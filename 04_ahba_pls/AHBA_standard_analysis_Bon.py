import abagen
import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils import resample
from scipy.stats import zscore, pearsonr, norm
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import nibabel as nb

# Helper function: save figures in PNG, PDF, and EPS formats
def save_figure_both_formats(filepath_base, dpi_png=600, dpi_pdf=600):
    """
    Save the current figure in PNG, PDF, and EPS formats
    Args:
        filepath_base: file path without an extension
        dpi_png: DPI for the PNG version
        dpi_pdf: DPI for the PDF version, typically higher for publication
    """
    png_path = f'{filepath_base}.png'
    pdf_path = f'{filepath_base}.pdf'
    eps_path = f'{filepath_base}.eps'
    plt.savefig(png_path, dpi=dpi_png, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=dpi_pdf, bbox_inches='tight')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    return png_path, pdf_path, eps_path

def save_figure_untitled_pdf(filepath_base, dpi_pdf=600):
    """
    Save the current figure as title-free and label-free PDF and EPS versions
    Remove the x-axis label, y-axis label, and figure title
    Args:
        filepath_base: file path without an extension
        dpi_pdf: DPI for the PDF version
    """
    # Get all axes in the current figure
    fig = plt.gcf()
    axes = fig.get_axes()
    
    # Store all original labels
    original_labels = []
    for ax in axes:
        original_labels.append({
            'xlabel': ax.get_xlabel(),
            'ylabel': ax.get_ylabel(),
            'title': ax.get_title()
        })
    
    # Remove all labels
    for ax in axes:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
    
    # Save PDF and EPS versions
    pdf_path = f'{filepath_base}.pdf'
    eps_path = f'{filepath_base}.eps'
    plt.savefig(pdf_path, dpi=dpi_pdf, bbox_inches='tight')
    plt.savefig(eps_path, format='eps', bbox_inches='tight')
    
    # Restore the original labels
    for ax, labels in zip(axes, original_labels):
        ax.set_xlabel(labels['xlabel'])
        ax.set_ylabel(labels['ylabel'])
        ax.set_title(labels['title'])
    
    return pdf_path, eps_path

# Set fonts and plotting style using the same configuration as the reference MSN plotting script
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.labelweight'] = 'normal'
plt.rcParams['axes.titleweight'] = 'normal'
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
# Use standard line widths consistent with the cell-type panel plots
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['lines.linewidth'] = 1.0

# Path configuration
PROJECT_ROOT = Path(os.environ.get('HD_MSN_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
DATA_ROOT = Path(os.environ.get('HD_MSN_DATA_ROOT', PROJECT_ROOT / 'data'))
RESULTS_ROOT = Path(os.environ.get('HD_MSN_RESULTS_ROOT', PROJECT_ROOT / 'results'))
atlas_img = str(Path(os.environ.get('HD_MSN_ATLAS_IMAGE', DATA_ROOT / 'parcellation' / '500.aparc.nii')))
atlas_info = str(Path(os.environ.get('HD_MSN_ATLAS_INFO', DATA_ROOT / 'ahba' / 'DK_all_atlas_info.csv')))
ahba_dir = str(Path(os.environ.get('HD_MSN_AHBA_RAW', DATA_ROOT / 'ahba' / 'raw')))
tmap_dir = str(Path(os.environ.get('HD_MSN_TMAP_DIR', DATA_ROOT / 'msn' / 'regional_statistics')))
results_dir = str(Path(os.environ.get('HD_MSN_AHBA_RESULTS', RESULTS_ROOT / 'ahba_pls')))

# FreeSurfer surface and annotation paths used to generate visualization files
SURFACE_DIR = str(Path(os.environ.get("HD_MSN_SURFACE_DIR", DATA_ROOT / "freesurfer" / "fsaverage" / "surf")))
PARCELLATION_DIR = str(Path(os.environ.get("HD_MSN_PARCELLATION_DIR", DATA_ROOT / "parcellation")))
ANNOT_NAME = "500.aparc.annot"

tmap_files = [
    'HD_vs_HC_tvalues_with_names.txt',
    'mHD_vs_HC_tvalues_with_names.txt',
    'preHD_vs_HC_tvalues_with_names.txt',
    'mHD_vs_preHD_tvalues_with_names.txt'
]

# Create the results directory
os.makedirs(results_dir, exist_ok=True)

def save_region_values_to_surface_txt(region_vals_dict, base_name, out_dir):
    """
    Map regional values to FreeSurfer vertices and save visualization files
    Equivalent in purpose to the save_surface_vertex_files helper used by the cell-type panel workflow
    
    Args:
        region_vals_dict: {region_name: value} dictionary
        base_name: base name for output files
        out_dir: output directory
    """
    os.makedirs(out_dir, exist_ok=True)
    hemisphere_surf_maps = {}
    
    for short_hemi in ['lh', 'rh']:
        annot_path = os.path.join(PARCELLATION_DIR, f'{short_hemi}.{ANNOT_NAME}')
        
        if not os.path.exists(annot_path):
            print(f"Warning: annotation file not found {annot_path}")
            continue
        
        # Read the annotation
        roi_map, _, names_bytes = nb.freesurfer.read_annot(annot_path)
        annot_region_names = [n.decode('utf-8') for n in names_bytes]
        
        surf_map = np.zeros(len(roi_map), dtype=float)
        unique_indices = np.unique(roi_map)
        matched = 0
        
        for index in unique_indices:
            if index == 0:
                continue
            if index < len(annot_region_names):
                region_name_from_annot = annot_region_names[index]
                # Try multiple naming conventions
                full_name_formats = [
                    f"{short_hemi}_{region_name_from_annot}",
                    region_name_from_annot
                ]
                
                value = None
                for fmt_name in full_name_formats:
                    if fmt_name in region_vals_dict:
                        value = region_vals_dict[fmt_name]
                        break
                
                if value is not None:
                    surf_map[roi_map == index] = value
                    matched += 1
        
        # Save the MGH file regardless of whether matching regions were found
        mgh_output_path = os.path.join(out_dir, f'{base_name}_{short_hemi}.mgh')
        mgh_data = surf_map.reshape(-1, 1, 1).astype(np.float32)
        mgh_image = nb.MGHImage(mgh_data, affine=None)
        nb.save(mgh_image, mgh_output_path)
        hemisphere_surf_maps[short_hemi] = surf_map
        
        if matched > 0:
            print(f"  Save {short_hemi} hemisphere: {matched} regions matched -> {mgh_output_path}")
        else:
            print(f"  Save {short_hemi} hemisphere: 0 regions matched (all zeros)-> {mgh_output_path}")
    
    # Save a combined text file containing both hemispheres; a missing hemisphere is filled with zeros
    if 'lh' in hemisphere_surf_maps:
        lh_map = hemisphere_surf_maps['lh']
    else:
        # If the left hemisphere cannot be read, create an all-zero array
        lh_map = np.zeros(160842, dtype=float)  # standard FreeSurfer fsaverage left-hemisphere vertex count
        print(f"  Left hemisphere could not be read; using an all-zero array")
    
    if 'rh' in hemisphere_surf_maps:
        rh_map = hemisphere_surf_maps['rh']
    else:
        # If the right hemisphere cannot be read, create an all-zero array
        rh_map = np.zeros(160842, dtype=float)  # standard FreeSurfer fsaverage right-hemisphere vertex count
        print(f"  Right hemisphere could not be read; using an all-zero array")
    
    # Combine data from both hemispheres
    combined = np.concatenate((lh_map, rh_map))
    txt_output_path = os.path.join(out_dir, f'{base_name}_both_hemispheres.txt')
    np.savetxt(txt_output_path, combined, fmt='%.10f')
    print(f"  Save combined text file (left hemisphere {len(lh_map)} vertices + right hemisphere {len(rh_map)} vertices): {txt_output_path}")


def perform_permutation_test(expression_mat, t_values, n_permutations=1000):
    """
    📊 Permutation test: randomly shuffle t values to assess statistical significance of the PLS model
    """
    print(f"🔄 Running permutation test with {n_permutations} permutations...")
    
    # Compute the observed correlation coefficient
    pls_observed = PLSRegression(n_components=1)
    pls_observed.fit(expression_mat, t_values)
    pls1_scores_observed = pls_observed.x_scores_[:, 0]
    r_observed, _ = pearsonr(pls1_scores_observed, t_values)
    
    print(f"📈 Observed PLS1-t-value correlation: r = {r_observed:.4f}")
    
    # Generate the null distribution
    null_r_values = []
    for i in range(n_permutations):
        if i % 100 == 0:
            print(f"  Permutation {i+1}/{n_permutations}...")
        
        # Randomly shuffle t values
        t_values_shuffled = t_values.copy()
        np.random.shuffle(t_values_shuffled)
        
        # Refit the PLS model
        pls_null = PLSRegression(n_components=1)
        pls_null.fit(expression_mat, t_values_shuffled)
        pls1_scores_null = pls_null.x_scores_[:, 0]
        
        # Compute the null correlation coefficient
        r_null, _ = pearsonr(pls1_scores_null, t_values_shuffled)
        null_r_values.append(r_null)
    
    null_r_values = np.array(null_r_values)
    
    # Compute the two-sided p value
    observed_abs = np.abs(r_observed)
    null_abs = np.abs(null_r_values)
    p_value_perm = (np.sum(null_abs >= observed_abs) + 1) / (len(null_r_values) + 1)
    
    print(f"🎯 Permutation-test result: p_perm = {p_value_perm:.6f}")
    
    return p_value_perm, null_r_values, r_observed

def bootstrap_pls_weights(expression_mat, t_values, n_bootstrap=1000):
    """
    🔧 Bootstrap analysis: estimate standard errors of PLS1 weights
    """
    print(f"🔄 Running bootstrap analysis with {n_bootstrap} resamples...")
    
    n_genes = expression_mat.shape[1]
    boot_weights = np.zeros((n_bootstrap, n_genes))
    
    for i in range(n_bootstrap):
        if i % 100 == 0:
            print(f"  Bootstrap {i+1}/{n_bootstrap}...")
        
        # Sample with replacement
        idx = resample(np.arange(expression_mat.shape[0]), replace=True, random_state=i)
        
        # Fit PLS to the bootstrap sample
        pls_boot = PLSRegression(n_components=1)
        pls_boot.fit(expression_mat.iloc[idx], t_values[idx])
        boot_weights[i, :] = pls_boot.x_weights_[:, 0]
    
    # Compute bootstrap standard errors
    pls1w_bse = boot_weights.std(axis=0)
    
    print(f"✅ Bootstrap completed; estimated standard errors for {len(pls1w_bse)} genes")
    
    return pls1w_bse, boot_weights

def calculate_corrected_weights(pls1_weights, pls1w_bse):
    """
    📐 Compute corrected PLS1 weights using the published formula:
    PLS1W_corr = PLS1W / PLS1W_BSE (this is the Z statistic itself)
    """
    print("📐 Computing corrected weights using the published formula...")
    print("🔬 Formula: PLS1W_corr = PLS1W / PLS1W_BSE (Z statistic)")
    
    # Avoid division-by-zero errors
    pls1w_bse_safe = np.where(pls1w_bse == 0, np.finfo(float).eps, pls1w_bse)
    
    # Compute PLS1W / PLS1W_BSE, which is the Z statistic
    pls1w_corr_z = pls1_weights / pls1w_bse_safe
    
    print(f"✅ Corrected-weight calculation completed")
    print(f"📊 Z statisticrange: [{pls1w_corr_z.min():.3f}, {pls1w_corr_z.max():.3f}]")
    
    return pls1w_corr_z

def identify_significant_genes_Bonferroni(genes, pls1w_corr_z, bonferroni_alpha=0.05):
    """
    🎯 Identify significant genes using Bonferroni correction, a stringent multiple-testing procedure
    """
    print(f"🎯 Identifying significant genes using Bonferroni correction...")
    print(f"📏 Bonferroni-adjusted alpha level: α = {bonferroni_alpha}")
    
    # Compute the raw two-sided p value for each gene
    raw_p_values = 2 * (1 - norm.cdf(np.abs(pls1w_corr_z)))
    
    # Bonferroni correction
    n_tests = len(raw_p_values)
    bonferroni_corrected_p = raw_p_values * n_tests
    # Ensure adjusted p values do not exceed 1.0
    bonferroni_corrected_p = np.minimum(bonferroni_corrected_p, 1.0)
    
    print(f"📊 Bonferroni correction completed:")
    print(f"   Number of tests: {n_tests}")
    print(f"   Raw p-value range: [{raw_p_values.min():.6f}, {raw_p_values.max():.6f}]")
    print(f"   Adjusted p-value range: [{bonferroni_corrected_p.min():.6f}, {bonferroni_corrected_p.max():.6f}]")
    
    # Determine significance using the Bonferroni-adjusted threshold
    significant_mask = bonferroni_corrected_p < bonferroni_alpha
    
    # Select Bonferroni-significant genes
    significant_genes = genes[significant_mask]
    significant_z = pls1w_corr_z[significant_mask]
    significant_p_raw = raw_p_values[significant_mask]
    significant_p_bonferroni = bonferroni_corrected_p[significant_mask]
    
    print(f"📊 Number of nominally significant genes (p < 0.05): {np.sum(raw_p_values < 0.05)}")
    print(f"📊 Number of Bonferroni-significant genes: {len(significant_genes)} (Bonferroni < {bonferroni_alpha})")
    
    if len(significant_genes) > 0:
        print(f"📊 Z statisticrange: [{significant_z.min():.3f}, {significant_z.max():.3f}]")
        
        # PLS+ gene set (positive association, Z > 0)
        pos_mask = significant_z > 0
        pls1_pos_genes = significant_genes[pos_mask]
        
        # PLS- gene set (negative association, Z < 0)  
        neg_mask = significant_z < 0
        pls1_neg_genes = significant_genes[neg_mask]
        
        print(f"🔺 PLS+ gene-set size: {len(pls1_pos_genes)} (Z > 0, Bonferroni < {bonferroni_alpha})")
        print(f"🔻 PLS- gene-set size: {len(pls1_neg_genes)} (Z < 0, Bonferroni < {bonferroni_alpha})")
        
        # Create a detailed results DataFrame
        significant_results = pd.DataFrame({
            'gene': significant_genes,
            'Z_score': significant_z,
            'p_raw': significant_p_raw,
            'p_Bonferroni': significant_p_bonferroni,
            'direction': ['PLS+' if z > 0 else 'PLS-' for z in significant_z]
        })
        
        # Sort by absolute Z value
        significant_results = significant_results.reindex(
            significant_results['Z_score'].abs().sort_values(ascending=False).index
        )
        
        return pls1_pos_genes, pls1_neg_genes, significant_results, raw_p_values, bonferroni_corrected_p
    else:
        print("📊 Z statisticrange: no Bonferroni-significant genes")
        empty_df = pd.DataFrame(columns=['gene', 'Z_score', 'p_raw', 'p_Bonferroni', 'direction'])
        return np.array([]), np.array([]), empty_df, raw_p_values, bonferroni_corrected_p

def check_existing_files(file_prefix):
    """Check for existing files to avoid redundant computation"""
    expression_file = f'{results_dir}/expression_left_{file_prefix}.csv'
    if os.path.exists(expression_file):
        print(f"✅ Found an existing expression-matrix file: {expression_file}")
        return True
    return False

def load_existing_expression_data(file_prefix):
    """Load existing expression data"""
    expression_file = f'{results_dir}/expression_left_{file_prefix}.csv'
    return pd.read_csv(expression_file, index_col=0)

def create_pls1_weighted_gene_rank_table(genes, pls1w_corr_z, top_n=10, bottom_n=10):
    """
    📊 Create a PLS1 weighted-gene ranking table showing genes with the largest and smallest Z scores
    Top: genes with the largest positive Z values; bottom: genes with the smallest negative Z values
    """
    print(f"📊 Creating the PLS1 weighted-gene ranking table...")
    print(f"   Showing the top {top_n} genes with the largest Z values and the bottom {bottom_n} genes with the smallest Z values")
    
    # Create a DataFrame containing gene names and Z scores
    gene_rank_df = pd.DataFrame({
        'GENE': genes,
        'Z-Score': pls1w_corr_z
    })
    
    # Sort directly by Z score in descending order
    gene_rank_df = gene_rank_df.sort_values('Z-Score', ascending=False)
    
    # Get genes with the highest and lowest Z scores
    top_genes = gene_rank_df.head(top_n)
    bottom_genes = gene_rank_df.tail(bottom_n)
    
    # Create the complete ranking table with highest Z values first, lowest Z values last, and an ellipsis between them
    rank_table = pd.concat([
        top_genes,
        pd.DataFrame({
            'GENE': ['......'],
            'Z-Score': ['......']
        }),
        bottom_genes
    ], ignore_index=True)
    
    print(f"✅ PLS1 weighted-gene ranking table completed")
    print(f"   Gene with highest Z value: {top_genes.iloc[0]['GENE']} (Z = {top_genes.iloc[0]['Z-Score']:.2f})")
    print(f"   Gene with lowest Z value: {bottom_genes.iloc[-1]['GENE']} (Z = {bottom_genes.iloc[-1]['Z-Score']:.2f})")
    
    return rank_table, top_genes, bottom_genes

# 1. Check and extract the expression matrix
print("🔍 Checking gene-expression data...")
expression_base_file = f'{results_dir}/expression.csv'
expression_with_label_file = f'{results_dir}/expression_with_label.csv'

if os.path.exists(expression_with_label_file):
    print(f"✅ Found existing gene-expression data: {expression_with_label_file}")
    print("📖 Loading existing data...")
    expression = pd.read_csv(expression_with_label_file)
else:
    print("📥 Extracting gene-expression data using the complete seven-step workflow...")
    print("🔬 Applying seven key AHBA preprocessing steps:")
    print("   1️⃣ Match microarray probes to gene annotations")
    print("   2️⃣ Filter probes failing the 50% intensity-based background criterion")
    print("   3️⃣ Select probes with the most consistent regional expression pattern")
    print("   4️⃣ Spatial matching to the D-K atlas using a 2-mm tolerance")
    print("   5️⃣ Robust sigmoid normalization")
    print("   6️⃣ Gene selection within the differential-stability framework")
    print("   7️⃣ Compute cross-donor averages")
    
    expression = abagen.get_expression_data(
        atlas_img,
        atlas_info=atlas_info,
        data_dir=ahba_dir,
        # Step 1: probe-to-gene matching (handled automatically)
        # Step 2: filter probes failing the 50% background-intensity criterion
        ibf_threshold=0.5,  # 50% intensity-based filtering threshold
        # Step 3: select the most consistent probe
        probe_selection='diff_stability',  # Select the probe with the highest differential stability
        # Step 4: spatial matching using a 2-mm Euclidean-distance tolerance
        missing='centroids',  # Use centroids for spatial matching
        tolerance=2,  # 2-mm distance tolerance
        # Step 5: robust sigmoid normalization
        norm_structures=True,  # Normalize across brain regions
        norm_matched=True,     # Normalize matched samples
        # Steps 6 and 7: cross-donor processing
        donor_probes='aggregate',  # Aggregate donor data
        lr_mirror='bidirectional', # Bidirectional left-right hemisphere mirroring
        # Additional quality control
        corrected_mni=True,    # Use corrected MNI coordinates
        reannotated=True,      # Use reannotated probes
        return_counts=False,   # Do not return sample counts
        return_donors=False    # Do not return donor information
    )
    expression.to_csv(expression_base_file, header=True, index=True)
    
    # Add the label column
    atlas_info_df = pd.read_csv(atlas_info)
    expression.index.name = 'id'
    expression['label'] = expression.index.map(dict(zip(atlas_info_df['id'], atlas_info_df['label'])))
    expression = expression.reset_index()
    expression.to_csv(expression_with_label_file, index=False)

# 2. Iterate over each t map
for tmap_file in tmap_files:
    print(f"\n{'='*80}")
    print(f"📋 Processing contrast: {tmap_file}")
    print(f"{'='*80}")
    
    file_prefix = tmap_file.replace('.txt', '')
    
    # Create an untitled directory for label-free PDF versions
    untitled_dir = os.path.join(results_dir, 'untitled')
    os.makedirs(untitled_dir, exist_ok=True)
    
    # Load and process t-map data
    tmap_path = os.path.join(tmap_dir, tmap_file)
    tmap = pd.read_csv(tmap_path, sep='\t')
    print(f"📋 Original t map contains {len(tmap)} regions")
    
    # Extract left-hemisphere data
    tmap_left = tmap[tmap['Region_Name'].str.startswith('lh_')]
    print(f"📊 Left hemisphere contains {len(tmap_left)} regions")
    
    # Match gene-expression data
    expression_left = expression[expression['label'].isin(tmap_left['Region_Name'])].copy()
    merged = pd.merge(expression_left, tmap_left, left_on='label', right_on='Region_Name')
    
    # Prepare analysis data
    expr_cols = [col for col in expression.columns if col not in ['id', 'label']]
    expression_mat = merged[expr_cols]  # Keep as a DataFrame
    t_values_left = merged['T_Value'].values
    region_names_left = merged['Region_Name'].tolist()
    genes = expression_mat.columns.values  # Get the array of gene names
    
    print(f"✅ Analysis dimensions: {expression_mat.shape[0]} left-hemisphere regions x {expression_mat.shape[1]} genes")
    
    # Check whether the expression matrix needs to be saved
    if not check_existing_files(file_prefix):
        expression_mat.to_csv(f'{results_dir}/expression_left_{file_prefix}.csv')
    
    print("✅ abagen already performed internal robust-sigmoid normalization; no additional normalization is required")
    print(f"📊 Final gene-expression data dimensions: {expression_mat.shape}")
    print(f"📊 T-value dimensions: {t_values_left.shape}")

    # 3. Step 1: PLS analysis
    print(f"\n⚗️ Step 1: PLS regression analysis...")
    print(f"📊 Data matrix: {expression_mat.shape[0]} regions x {expression_mat.shape[1]} genes")
    print(f"📊 Response variable: {len(t_values_left)} t values")
    
    pls = PLSRegression(n_components=1)
    pls.fit(expression_mat.values, t_values_left)  # Use abagen-preprocessed data
    
    # Extract PLS1 scores and weights
    pls1_scores = pls.x_scores_[:, 0]
    pls1_weights = pls.x_weights_[:, 0]
    
    # Compute explained variance
    r_observed, _ = pearsonr(pls1_scores, t_values_left)
    r_squared = r_observed ** 2
    
    print(f"📈 PLS1 results:")
    print(f"   PLS1-t-value correlation: r = {r_observed:.4f}")
    print(f"   Explained variance: R² = {r_squared:.2%}")
    
    # 4. Step 2: permutation test (1,000 permutations)
    print(f"\n🎲 Step 2: permutation test (1,000 permutations)...")
    p_value_perm, null_r_values, _ = perform_permutation_test(expression_mat.values, t_values_left, n_permutations=1000)
    
    # 5. Step 3: bootstrap analysis (1,000 resamples)
    print(f"\n🔧 Step 3: bootstrap analysis (1,000 resamples)...")
    pls1w_bse, boot_weights = bootstrap_pls_weights(expression_mat, t_values_left, n_bootstrap=1000)
    
    # 6. Step 4: compute corrected weights using the published formula
    print(f"\n📐 Step 4: Compute corrected weights using the published formula...")
    pls1w_corr_z = calculate_corrected_weights(pls1_weights, pls1w_bse)
    
    # 7. Step 5: identify significant genes using Bonferroni correction
    print(f"\n🎯 Step 5: Identify significant genes using Bonferroni correction...")
    pls1_pos_genes, pls1_neg_genes, significant_results, raw_p_values, bonferroni_p_values = identify_significant_genes_Bonferroni(genes, pls1w_corr_z, bonferroni_alpha=0.05)
    
    # 8. Assess overall model significance
    print(f"\n📊 Overall analysis results:")
    print(f"   Permutation-test p value: p_perm = {p_value_perm:.6f}")
    print(f"   Model significant: {'yes' if p_value_perm <= 0.05 else 'no'} (p ≤ 0.05)")
    
    if p_value_perm > 0.05:
        print(f"⚠️  Warning: the PLS model is not significant (p_perm = {p_value_perm:.6f} > 0.05)")
        print(f"⚠️  Gene-weight results may not have statistical support.")
    
    # 9. Save results
    print(f"\n💾 Saving analysis results...")
    
    # Save PLS1 scores
    pls1_scores_df = pd.DataFrame({
        'Region': region_names_left,
        'T_Value': t_values_left,
        'PLS1_Score': pls1_scores
    })
    pls1_scores_df.to_csv(f'{results_dir}/PLS1_scores_{file_prefix}.csv', index=False)
    
    # Save gene ranking including Bonferroni-adjusted p values
    gene_rank = pd.DataFrame({
        'gene': genes,
        'PLS1_weight': pls1_weights,
        'PLS1W_BSE': pls1w_bse,
        'PLS1W_corr_Z': pls1w_corr_z,
        'p_raw': raw_p_values,
        'p_Bonferroni': bonferroni_p_values,
        'significant_Bonferroni': bonferroni_p_values < 0.05
    })
    gene_rank['abs_Z'] = np.abs(gene_rank['PLS1W_corr_Z'])
    gene_rank = gene_rank.sort_values('abs_Z', ascending=False)  # Sort by absolute Z value
    gene_rank = gene_rank.drop('abs_Z', axis=1)  # Remove the temporary column
    gene_rank['permutation_p_value'] = p_value_perm
    gene_rank['model_significant'] = p_value_perm <= 0.05
    gene_rank['method'] = 'standard_permutation_bootstrap_Bonferroni'
    
    gene_rank.to_csv(f'{results_dir}/PLS1_gene_rank_{file_prefix}.csv', index=False)
    
    # Save detailed information for Bonferroni-significant genes
    if len(significant_results) > 0:
        significant_results.to_csv(f'{results_dir}/significant_genes_Bonferroni_{file_prefix}.csv', index=False)
    
    # Create the PLS1 weighted-gene ranking table
    print(f"\n📊 Creating the PLS1 weighted-gene ranking table...")
    rank_table, top_genes, bottom_genes = create_pls1_weighted_gene_rank_table(
        genes, pls1w_corr_z, top_n=10, bottom_n=10
    )
    
    # Save the ranking table
    rank_table.to_csv(f'{results_dir}/PLS1_weighted_gene_rank_{file_prefix}.csv', index=False)
    top_genes.to_csv(f'{results_dir}/PLS1_top_genes_{file_prefix}.csv', index=False)
    bottom_genes.to_csv(f'{results_dir}/PLS1_bottom_genes_{file_prefix}.csv', index=False)
    
    print(f"💾 Save PLS1 weighted-gene ranking table:")
    print(f"   Complete ranking table: PLS1_weighted_gene_rank_{file_prefix}.csv")
    print(f"   Top 10 genes: PLS1_top_genes_{file_prefix}.csv")
    print(f"   Bottom 10 genes: PLS1_bottom_genes_{file_prefix}.csv")
    
    # Save detailed tables for significant gene sets
    if len(pls1_pos_genes) > 0:
        # Create the detailed PLS+ gene table
        pos_gene_details = []
        for gene in pls1_pos_genes:
            gene_idx = np.where(genes == gene)[0][0]
            pos_gene_details.append({
                'gene': gene,
                'Z_score': pls1w_corr_z[gene_idx],
                'p_raw': raw_p_values[gene_idx],
                'p_Bonferroni': bonferroni_p_values[gene_idx],
                'PLS1_weight': pls1_weights[gene_idx],
                'PLS1W_BSE': pls1w_bse[gene_idx]
            })
        
        pls_pos_df = pd.DataFrame(pos_gene_details)
        pls_pos_df = pls_pos_df.sort_values('Z_score', ascending=False)  # Sort by descending Z value
        pls_pos_df.to_csv(f'{results_dir}/PLS1_pos_genes_detailed_Bonferroni_{file_prefix}.csv', index=False)
        
        # Simple gene-name list
        pd.DataFrame({'gene': pls1_pos_genes}).to_csv(f'{results_dir}/PLS1_pos_genes_Bonferroni_{file_prefix}.csv', index=False)
        
        print(f"💾 Save PLS+ gene set: {len(pls1_pos_genes)} genes")
        print(f"   Detailed table: PLS1_pos_genes_detailed_Bonferroni_{file_prefix}.csv")
        print(f"   Gene list: PLS1_pos_genes_Bonferroni_{file_prefix}.csv")
    else:
        # If there are no PLS+ genes, create an empty file
        pd.DataFrame({'gene': []}).to_csv(f'{results_dir}/PLS1_pos_genes_Bonferroni_{file_prefix}.csv', index=False)
        pd.DataFrame(columns=['gene', 'Z_score', 'p_raw', 'p_Bonferroni', 'PLS1_weight', 'PLS1W_BSE']).to_csv(f'{results_dir}/PLS1_pos_genes_detailed_Bonferroni_{file_prefix}.csv', index=False)
    
    if len(pls1_neg_genes) > 0:
        # Create the detailed PLS- gene table
        neg_gene_details = []
        for gene in pls1_neg_genes:
            gene_idx = np.where(genes == gene)[0][0]
            neg_gene_details.append({
                'gene': gene,
                'Z_score': pls1w_corr_z[gene_idx],
                'p_raw': raw_p_values[gene_idx],
                'p_Bonferroni': bonferroni_p_values[gene_idx],
                'PLS1_weight': pls1_weights[gene_idx],
                'PLS1W_BSE': pls1w_bse[gene_idx]
            })
        
        pls_neg_df = pd.DataFrame(neg_gene_details)
        pls_neg_df = pls_neg_df.sort_values('Z_score', ascending=True)  # Sort by ascending Z value, with the most negative values first
        pls_neg_df.to_csv(f'{results_dir}/PLS1_neg_genes_detailed_Bonferroni_{file_prefix}.csv', index=False)
        
        # Simple gene-name list
        pd.DataFrame({'gene': pls1_neg_genes}).to_csv(f'{results_dir}/PLS1_neg_genes_Bonferroni_{file_prefix}.csv', index=False)
        
        print(f"💾 Save PLS- gene set: {len(pls1_neg_genes)} genes")
        print(f"   Detailed table: PLS1_neg_genes_detailed_Bonferroni_{file_prefix}.csv")
        print(f"   Gene list: PLS1_neg_genes_Bonferroni_{file_prefix}.csv")
    else:
        # If there are no PLS- genes, create an empty file
        pd.DataFrame({'gene': []}).to_csv(f'{results_dir}/PLS1_neg_genes_Bonferroni_{file_prefix}.csv', index=False)
        pd.DataFrame(columns=['gene', 'Z_score', 'p_raw', 'p_Bonferroni', 'PLS1_weight', 'PLS1W_BSE']).to_csv(f'{results_dir}/PLS1_neg_genes_detailed_Bonferroni_{file_prefix}.csv', index=False)
    
    # Save permutation-test results
    perm_results = pd.DataFrame({
        'null_r_values': null_r_values,
        'observed_r': r_observed,
        'p_perm': p_value_perm
    })
    perm_results.to_csv(f'{results_dir}/permutation_test_results_{file_prefix}.csv', index=False)
    
    # ==================== Additional outputs ====================
    
    # 1. Save PLS1-score visualization files for FreeSurfer
    print(f"\n💾 Generating FreeSurfer visualization files for PLS1 scores...")
    pls1_viz_dir = os.path.join(results_dir, 'PLS1_score_visualization')
    pls1_region_dict = dict(zip(region_names_left, pls1_scores))
    save_region_values_to_surface_txt(pls1_region_dict, f'PLS1_score_{file_prefix}', pls1_viz_dir)
    
    # 2. Identify the genes with the highest and lowest Z values
    print(f"\n🔍 Identifying genes with the highest and lowest Z values...")
    max_z_idx = np.argmax(pls1w_corr_z)
    min_z_idx = np.argmin(pls1w_corr_z)
    
    max_z_gene = genes[max_z_idx]
    min_z_gene = genes[min_z_idx]
    max_z_value = pls1w_corr_z[max_z_idx]
    min_z_value = pls1w_corr_z[min_z_idx]
    
    print(f"  Gene with highest Z value: {max_z_gene} (Z = {max_z_value:.4f})")
    print(f"  Gene with lowest Z value: {min_z_gene} (Z = {min_z_value:.4f})")
    
    # 3. Generate expression values for these two genes and convert them to Z scores
    max_z_gene_expr = expression_mat[max_z_gene].values
    min_z_gene_expr = expression_mat[min_z_gene].values
    
    # Convert expression values to standardized Z scores
    max_z_gene_expr_zscore = zscore(max_z_gene_expr)
    min_z_gene_expr_zscore = zscore(min_z_gene_expr)
    
    # 4. Save visualization files for the expression of these two genes using Z scores
    print(f"\n💾 Generating FreeSurfer visualization files for extreme-Z genes...")
    extreme_genes_viz_dir = os.path.join(results_dir, 'extreme_genes_visualization')
    
    max_z_gene_dict = dict(zip(region_names_left, max_z_gene_expr_zscore))
    save_region_values_to_surface_txt(max_z_gene_dict, f'{max_z_gene}_expr_zscore_{file_prefix}', extreme_genes_viz_dir)
    
    min_z_gene_dict = dict(zip(region_names_left, min_z_gene_expr_zscore))
    save_region_values_to_surface_txt(min_z_gene_dict, f'{min_z_gene}_expr_zscore_{file_prefix}', extreme_genes_viz_dir)
    
    # 5. Generate the gene-expression-by-region matrix after transposition: rows are genes and columns are regions
    print(f"\n💾 Saving the gene-expression-by-region matrix...")
    gene_by_region_matrix = expression_mat.T  # Transpose so rows are genes and columns are regions
    gene_by_region_matrix.index.name = 'gene'
    gene_by_region_matrix.columns = region_names_left
    gene_by_region_output = os.path.join(results_dir, f'gene_by_region_expression_{file_prefix}.csv')
    gene_by_region_matrix.to_csv(gene_by_region_output)
    print(f"  Saved to: {gene_by_region_output}")
    print(f"  Matrix dimensions: {gene_by_region_matrix.shape[0]} genes x {gene_by_region_matrix.shape[1]} regions")
    
    # 6. Generate scatter plots of the highest- and lowest-Z genes against the t map using Z-scored expression
    print(f"\n📊 Generating scatter plots for extreme-Z genes using Z-scored expression...")
    
    # Compute correlations using Z-scored expression
    r_max_z, p_max_z = pearsonr(max_z_gene_expr_zscore, t_values_left)
    r_min_z, p_min_z = pearsonr(min_z_gene_expr_zscore, t_values_left)
    
    # Create the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Highest-Z gene panel using Z-scored expression
    blueviolet_color = '#6a5acd'
    sns.regplot(x=max_z_gene_expr_zscore, y=t_values_left, ax=ax1,
                scatter_kws={'alpha': 0.6, 's': 30, 'color': blueviolet_color, 'edgecolor': 'none'},
                line_kws={'color': blueviolet_color, 'linewidth': 1.5})
    
    ax1.set_xlabel(f'{max_z_gene} Expression (Z-score)', fontsize=14, fontweight='normal')
    ax1.set_ylabel('T-Map', fontsize=14, fontweight='normal')
    ax1.set_title(f'Highest Z-score Gene: {max_z_gene}', fontsize=16, fontweight='normal')
    
    # Use standard axis line widths consistent with the cell-type panels
    for spine in ax1.spines.values():
        spine.set_linewidth(0.8)
    ax1.tick_params(width=0.8, labelsize=12)
    
    # Add statistical information and left-align legend text
    p_text_max = f'p < 0.0001' if p_max_z < 0.0001 else f'p = {p_max_z:.4f}'
    textstr_max = f'r = {r_max_z:.3f}\n{p_text_max}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1)
    # Keep the legend in the upper left with left-aligned text
    legend_loc_max = (0.05, 0.95)
    legend_ha_max = 'left'
    ax1.text(legend_loc_max[0], legend_loc_max[1], textstr_max, transform=ax1.transAxes, fontsize=13,
            verticalalignment='top', horizontalalignment=legend_ha_max, bbox=props, fontweight='normal')
    
    # Lowest-Z gene panel using Z-scored expression
    limegreen_color = '#32cd32'
    sns.regplot(x=min_z_gene_expr_zscore, y=t_values_left, ax=ax2,
                scatter_kws={'alpha': 0.6, 's': 30, 'color': limegreen_color, 'edgecolor': 'none'},
                line_kws={'color': limegreen_color, 'linewidth': 1.5})
    
    ax2.set_xlabel(f'{min_z_gene} Expression (Z-score)', fontsize=14, fontweight='normal')
    ax2.set_ylabel('T-Map', fontsize=14, fontweight='normal')
    ax2.set_title(f'Lowest Z-score Gene: {min_z_gene}', fontsize=16, fontweight='normal')
    
    # Use standard axis line widths consistent with the cell-type panels
    for spine in ax2.spines.values():
        spine.set_linewidth(0.8)
    ax2.tick_params(width=0.8, labelsize=12)
    
    # Add statistical information and left-align legend text
    p_text_min = f'p < 0.0001' if p_min_z < 0.0001 else f'p = {p_min_z:.4f}'
    textstr_min = f'r = {r_min_z:.3f}\n{p_text_min}'
    # Keep the legend in the upper left with left-aligned text
    legend_loc_min = (0.05, 0.95)
    legend_ha_min = 'left'
    ax2.text(legend_loc_min[0], legend_loc_min[1], textstr_min, transform=ax2.transAxes, fontsize=13,
            verticalalignment='top', horizontalalignment=legend_ha_min, bbox=props, fontweight='normal')
    
    plt.tight_layout()
    extreme_genes_plot_base = os.path.join(results_dir, f'extreme_Z_genes_vs_TMap_{file_prefix}')
    png_path, pdf_path, eps_path = save_figure_both_formats(extreme_genes_plot_base, dpi_png=300, dpi_pdf=600)
    
    # Generate label-free PDF and EPS versions
    untitled_pdf_path = os.path.join(untitled_dir, f'extreme_Z_genes_vs_TMap_{file_prefix}.pdf')
    untitled_pdf, untitled_eps = save_figure_untitled_pdf(untitled_pdf_path.replace('.pdf', ''), dpi_pdf=600)
    
    plt.close()
    print(f"  Saved to: {png_path}, {pdf_path} and {eps_path}")
    print(f"  Label-free versions: {untitled_pdf} and {untitled_eps}")
    
    # Save information for the extreme-Z genes
    extreme_genes_info = pd.DataFrame({
        'gene': [max_z_gene, min_z_gene],
        'Z_score': [max_z_value, min_z_value],
        'correlation_with_Tmap': [r_max_z, r_min_z],
        'p_value': [p_max_z, p_min_z],
        'type': ['Highest_Z', 'Lowest_Z']
    })
    extreme_genes_info.to_csv(f'{results_dir}/extreme_Z_genes_info_{file_prefix}.csv', index=False)
    print(f"  Saved extreme-Z gene information to: extreme_Z_genes_info_{file_prefix}.csv")
    
    # ==================== Continue with the main visualization outputs ====================
    
    # 10. Visualization
    print("📊 Generating visualization figures...")
    
    # Correlation between PLS1 scores and the t map
    plt.figure(figsize=(10, 6))
    
    # Set the blue-purple color
    blueviolet_color = '#6a5acd'
    
    sns.regplot(x=pls1_scores, y=t_values_left, 
                scatter_kws={'alpha': 0.6, 's': 30, 'color': blueviolet_color}, 
                line_kws={'color': blueviolet_color, 'linewidth': 1.5})
    
    plt.xlabel('PLS1 Score', fontsize=14, fontweight='normal')
    plt.ylabel('T-Map', fontsize=14, fontweight='normal')
    plt.title('Correlation between PLS1 Score and T-Map', fontsize=16, fontweight='normal')
    
    # Use standard axis line widths consistent with the cell-type panels
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    ax.tick_params(width=0.8, labelsize=12)
    
    # Create a text box for statistical information
    p_text = f'p < 0.0001' if p_value_perm < 0.0001 else f'p = {p_value_perm:.4f}'
    textstr = f'r = {r_observed:.3f}\n{p_text}'
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1)
    # Keep the legend in the upper left with left-aligned text
    legend_loc = (0.05, 0.95)
    legend_ha = 'left'
    ax.text(legend_loc[0], legend_loc[1], textstr, transform=ax.transAxes, fontsize=13,
            verticalalignment='top', horizontalalignment=legend_ha, bbox=props, fontweight='normal')
    
    plt.tight_layout()
    pls1_vs_tmap_base = f'{results_dir}/PLS1_vs_TMap_{file_prefix}'
    png_path, pdf_path, eps_path = save_figure_both_formats(pls1_vs_tmap_base, dpi_png=300, dpi_pdf=600)
    
    # Generate label-free PDF and EPS versions
    untitled_pdf_path = os.path.join(untitled_dir, f'PLS1_vs_TMap_{file_prefix}.pdf')
    untitled_pdf, untitled_eps = save_figure_untitled_pdf(untitled_pdf_path.replace('.pdf', ''), dpi_pdf=600)
    
    plt.close()
    print(f"  Saved to: {png_path}, {pdf_path} and {eps_path}")
    print(f"  Label-free versions: {untitled_pdf} and {untitled_eps}")
    
    # If Bonferroni-significant genes are available, generate PLS+ and PLS- gene-set correlations with the t map
    if len(significant_results) > 0:
        print(f"📊 Generating PLS+ and PLS- gene-set correlation plots...")
        
        # Compute mean expression for the PLS+ and PLS- gene sets
        if len(pls1_pos_genes) > 0:
            pos_gene_indices = [i for i, gene in enumerate(genes) if gene in pls1_pos_genes]
            pls_pos_expression = expression_mat.iloc[:, pos_gene_indices].mean(axis=1).values
        else:
            pls_pos_expression = np.zeros(len(t_values_left))
            
        if len(pls1_neg_genes) > 0:
            neg_gene_indices = [i for i, gene in enumerate(genes) if gene in pls1_neg_genes]
            pls_neg_expression = expression_mat.iloc[:, neg_gene_indices].mean(axis=1).values
        else:
            pls_neg_expression = np.zeros(len(t_values_left))
        
        # Compute correlations
        if len(pls1_pos_genes) > 0:
            r_pos, p_pos = pearsonr(pls_pos_expression, t_values_left)
        else:
            r_pos, p_pos = 0, 1
            
        if len(pls1_neg_genes) > 0:
            r_neg, p_neg = pearsonr(pls_neg_expression, t_values_left)
        else:
            r_neg, p_neg = 0, 1
        
        # Create the figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Set colors
        pls_pos_color = '#6a5acd'  # blue-purple (PLS+)
        pls_neg_color = '#32cd32'  # bright green (PLS-)
        
        # PLS+ gene-set panel
        if len(pls1_pos_genes) > 0:
            sns.regplot(x=pls_pos_expression, y=t_values_left, ax=ax1,
                       scatter_kws={'alpha': 0.6, 's': 30, 'color': pls_pos_color, 'edgecolor': 'none'}, 
                       line_kws={'color': pls_pos_color, 'linewidth': 1.5})
            ax1.set_xlabel(f'PLS+ Gene Set Expression\n({len(pls1_pos_genes)} genes)', fontsize=14, fontweight='normal')
            ax1.set_ylabel('T-Map', fontsize=14, fontweight='normal')
            ax1.set_title('Correlation between PLS+ Gene Set and T-Map', fontsize=14, fontweight='normal')
            
            # Add statistical information to the text box and left-align legend text
            p_text_pos = f'p < 0.0001' if p_pos < 0.0001 else f'p = {p_pos:.4f}'
            textstr_pos = f'r = {r_pos:.3f}\n{p_text_pos}'
            props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1)
            legend_loc_pos = (0.05, 0.95)
            legend_ha_pos = 'left'
            ax1.text(legend_loc_pos[0], legend_loc_pos[1], textstr_pos, transform=ax1.transAxes, fontsize=13,
                    verticalalignment='top', horizontalalignment=legend_ha_pos, bbox=props, fontweight='normal')
            
        else:
            ax1.text(0.5, 0.5, 'No PLS+ genes found', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=14, fontweight='normal')
            ax1.set_title('PLS+ Gene Set (No significant genes)', fontsize=14, fontweight='normal')
        
        # Use standard axis line widths consistent with the cell-type panels
        for spine in ax1.spines.values():
            spine.set_linewidth(0.8)
        ax1.tick_params(width=0.8, labelsize=12)
        
        # PLS- gene-set panel  
        if len(pls1_neg_genes) > 0:
            sns.regplot(x=pls_neg_expression, y=t_values_left, ax=ax2,
                       scatter_kws={'alpha': 0.6, 's': 30, 'color': pls_neg_color, 'edgecolor': 'none'}, 
                       line_kws={'color': pls_neg_color, 'linewidth': 1.5})
            ax2.set_xlabel(f'PLS- Gene Set Expression\n({len(pls1_neg_genes)} genes)', fontsize=14, fontweight='normal')
            ax2.set_ylabel('T-Map', fontsize=14, fontweight='normal')
            ax2.set_title('Correlation between PLS- Gene Set and T-Map', fontsize=14, fontweight='normal')
            
            # Add statistical information to the text box and left-align legend text
            p_text_neg = f'p < 0.0001' if p_neg < 0.0001 else f'p = {p_neg:.4f}'
            textstr_neg = f'r = {r_neg:.3f}\n{p_text_neg}'
            props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black', linewidth=1)
            legend_loc_neg = (0.05, 0.95)
            legend_ha_neg = 'left'
            ax2.text(legend_loc_neg[0], legend_loc_neg[1], textstr_neg, transform=ax2.transAxes, fontsize=13,
                    verticalalignment='top', horizontalalignment=legend_ha_neg, bbox=props, fontweight='normal')
            
        else:
            ax2.text(0.5, 0.5, 'No PLS- genes found', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14, fontweight='normal')
            ax2.set_title('PLS- Gene Set (No significant genes)', fontsize=14, fontweight='normal')
            
        # Use standard axis line widths consistent with the cell-type panels
        for spine in ax2.spines.values():
            spine.set_linewidth(0.8)
        ax2.tick_params(width=0.8, labelsize=12)
        
        plt.tight_layout()
        pls_geneset_base = f'{results_dir}/PLS_geneset_correlations_{file_prefix}'
        png_path, pdf_path, eps_path = save_figure_both_formats(pls_geneset_base, dpi_png=300, dpi_pdf=600)
        
        # Generate label-free PDF and EPS versions
        untitled_pdf_path = os.path.join(untitled_dir, f'PLS_geneset_correlations_{file_prefix}.pdf')
        untitled_pdf, untitled_eps = save_figure_untitled_pdf(untitled_pdf_path.replace('.pdf', ''), dpi_pdf=600)
        
        plt.close()
        print(f"  Saved to: {png_path}, {pdf_path} and {eps_path}")
        print(f"  Label-free versions: {untitled_pdf} and {untitled_eps}")
    
    # Gene-weight distribution and Bonferroni-correction comparison
    plt.figure(figsize=(12, 6))
    
    # Panel 1: raw weights versus corrected weights
    ax1 = plt.subplot(1, 2, 1)
    plt.scatter(pls1_weights, pls1w_corr_z, alpha=0.6, s=20, color='#6a5acd')
    plt.xlabel('PLS1 Weight', fontsize=14, fontweight='normal')
    plt.ylabel('PLS1W_corr_Z', fontsize=14, fontweight='normal')
    plt.title('Original vs Corrected Weights', fontsize=14, fontweight='normal')
    plt.axhline(y=1.96, color='orange', linestyle='--', alpha=0.7, linewidth=1.0, label='p=0.05 (uncorrected)')
    plt.axhline(y=-1.96, color='orange', linestyle='--', alpha=0.7, linewidth=1.0)
    if len(significant_results) > 0:
        min_sig_z = np.abs(significant_results['Z_score']).min()
        plt.axhline(y=min_sig_z, color='red', linestyle='-', alpha=0.7, linewidth=1.0, label=f'Bonferroni threshold (|Z|>{min_sig_z:.2f})')
        plt.axhline(y=-min_sig_z, color='red', linestyle='-', alpha=0.7, linewidth=1.0)
    plt.legend(fontsize=13, frameon=True, fancybox=False, shadow=False, edgecolor='black')
    
    # Use standard axis line widths consistent with the cell-type panels
    for spine in ax1.spines.values():
        spine.set_linewidth(0.8)
    ax1.tick_params(width=0.8, labelsize=12)
    
    # Panel 2: comparison of Bonferroni-significant gene counts
    ax2 = plt.subplot(1, 2, 2)
    raw_sig = np.sum(raw_p_values < 0.05)
    bonferroni_sig = len(significant_results)
    categories = ['Raw Method\n(p<0.05)', 'Bonferroni Corrected\n(α=0.05)', 'PLS+\n(Bonferroni)', 'PLS-\n(Bonferroni)']
    counts = [raw_sig, bonferroni_sig, len(pls1_pos_genes), len(pls1_neg_genes)]
    colors = ['orange', 'red', '#6a5acd', '#32cd32']  # Use the consistent blue-purple and bright-green colors
    bars = plt.bar(categories, counts, color=colors, alpha=0.7, linewidth=0.8, edgecolor='black')
    plt.ylabel('Number of genes', fontsize=14, fontweight='normal')
    plt.title('Bonferroni vs Raw Significance', fontsize=14, fontweight='normal')
    plt.xticks(rotation=45, fontsize=13, fontweight='normal')
    
    # Add value labels to the bar chart
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='normal', fontsize=13)
    
    # Use standard axis line widths consistent with the cell-type panels
    for spine in ax2.spines.values():
        spine.set_linewidth(0.8)
    ax2.tick_params(width=0.8, labelsize=12)
    
    plt.tight_layout()
    analysis_summary_base = f'{results_dir}/analysis_summary_Bonferroni_{file_prefix}'
    png_path, pdf_path, eps_path = save_figure_both_formats(analysis_summary_base, dpi_png=300, dpi_pdf=600)
    
    # Generate label-free PDF and EPS versions
    untitled_pdf_path = os.path.join(untitled_dir, f'analysis_summary_Bonferroni_{file_prefix}.pdf')
    untitled_pdf, untitled_eps = save_figure_untitled_pdf(untitled_pdf_path.replace('.pdf', ''), dpi_pdf=600)
    
    plt.close()
    print(f"  Saved to: {png_path}, {pdf_path} and {eps_path}")
    print(f"  Label-free versions: {untitled_pdf} and {untitled_eps}")
    
    # Generate a visualization of the PLS1 weighted-gene ranking table
    print(f"📊 Generating PLS1 weighted-gene ranking-table visualization...")
    
    # Create a table-style figure
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    table_data.append(['GENE', 'Z-Score'])  # header row
    
    # Add the top 10 genes
    for _, row in top_genes.iterrows():
        table_data.append([row['GENE'], f"{row['Z-Score']:.2f}"])
    
    # Add the ellipsis row
    table_data.append(['......', '......'])
    
    # Add the bottom 10 genes
    for _, row in bottom_genes.iterrows():
        table_data.append([row['GENE'], f"{row['Z-Score']:.2f}"])
    
    # Create the table
    table = ax.table(cellText=table_data, 
                    colLabels=None,  # The header row was added manually
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.6, 0.4])
    
    # Set table styling
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)  # Adjust row height
    
    # Set header styling
    for i in range(2):
        table[(0, i)].set_facecolor('#4A90E2')  # blue header
        table[(0, i)].set_text_props(weight='normal', color='white')
    
    # Set data-row styling
    for i in range(1, len(table_data)):
        for j in range(2):
            if i == len(top_genes) + 1:  # ellipsis row
                table[(i, j)].set_facecolor('#F0F0F0')  # gray
                table[(i, j)].set_text_props(style='italic')
            else:
                table[(i, j)].set_facecolor('white')
                table[(i, j)].set_text_props(weight='normal')
    
    # Add table borders
    for i in range(len(table_data)):
        for j in range(2):
            table[(i, j)].set_edgecolor('black')
            table[(i, j)].set_linewidth(1)
    
    # Add the title
    plt.title('PLS1 Weighted Gene Rank', fontsize=16, fontweight='normal', pad=20)
    
    plt.tight_layout()
    pls1_rank_table_base = f'{results_dir}/PLS1_weighted_gene_rank_table_{file_prefix}'
    png_path, pdf_path, eps_path = save_figure_both_formats(pls1_rank_table_base, dpi_png=300, dpi_pdf=600)
    
    # Generate label-free PDF and EPS versions with the title removed
    title_obj = plt.title('')
    untitled_pdf_path = os.path.join(untitled_dir, f'PLS1_weighted_gene_rank_table_{file_prefix}.pdf')
    untitled_pdf, untitled_eps = save_figure_untitled_pdf(untitled_pdf_path.replace('.pdf', ''), dpi_pdf=600)
    
    plt.close()
    
    print(f"💾 Saved PLS1 weighted-gene ranking-table visualization: {png_path}, {pdf_path} and {eps_path}")
    print(f"  Label-free versions: {untitled_pdf} and {untitled_eps}")

print("\n" + "="*80)
print("🎉 Bonferroni-corrected PLS Analysis Completed!")
print("📊 Results saved to", results_dir)
print("🔬 Analysis Pipeline:")
print("   1️⃣ PLS Regression Analysis")
print("   2️⃣ Permutation Test (1000 iterations)")
print("   3️⃣ Bootstrap Analysis (1000 iterations)")
print("   4️⃣ Calculate corrected weights using article formula: PLS1W_corr = PLS1W / PLS1W_BSE")
print("   5️⃣ Bonferroni correction for significant gene identification (most conservative method)")
print("   6️⃣ Generate PLS1 Weighted Gene Rank tables (top/bottom genes with Z-scores)")
print("   7️⃣ Create visual table representations")
print("📈 This is the most stringent multiple comparison correction method!")
print("📊 New Features Added:")
print("   • PLS1_weighted_gene_rank_{prefix}.csv - Complete gene ranking table")
print("   • PLS1_top_genes_{prefix}.csv - Top 10 genes by Z-score")
print("   • PLS1_bottom_genes_{prefix}.csv - Bottom 10 genes by Z-score")
print("   • PLS1_weighted_gene_rank_table_{prefix}.png - Visual table representation")
print("="*80)

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Run the primary AHBA cortical transcriptional PLS analysis and associated gene-weight/enrichment summaries.
# Input source/location: AHBA expression matrix, atlas mapping, HD-HC regional MSN target map, and optional gene-annotation resources from configurable project paths.
# Output location: PLS scores/weights, permutation/bootstrap statistics, enrichment tables, and visualization products in the configured AHBA results directories.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Validate and align parcels/genes; standardize data; fit PLS; orient components consistently; run null/uncertainty analyses; generate gene-level summaries and figures.
# Log location: No dedicated log file unless redirected externally; progress is written to standard output.
# =============================================================================
