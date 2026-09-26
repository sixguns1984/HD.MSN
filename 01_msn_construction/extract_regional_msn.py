#!/usr/bin/env python3
"""
Extract regional MSN values from similarity matrices
Regional MSN is the mean correlation between a target ROI and all other ROIs, excluding the diagonal
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import re

def extract_regional_msn(matrix_file):
    """Extract regional MSN values from a similarity matrix"""
    df = pd.read_csv(matrix_file, index_col=0)
    n = df.shape[0]
    
    # Compute regional MSN for each ROI as the row mean excluding the diagonal.
    regional_msn = []
    for i in range(n):
        row = df.iloc[i, :].values
        mask = np.arange(n) != i
        regional_msn.append(np.mean(row[mask]))
    
    return np.array(regional_msn)

def extract_subject_id(filename):
    """Extract the subject ID from the filename and standardize it to three zero-padded digits"""
    # Match HD_subXXX
    match = re.search(r'HD_sub(\d+)', filename)
    if match:
        num = int(match.group(1))
        return f"HD_sub{num:03d}"
    
    # Match HC_subXXX or subXXX
    match = re.search(r'(?:HC_)?sub(\d+)', filename)
    if match:
        num = int(match.group(1))
        return f"HC_sub{num:03d}"
    
    return None

def load_actual_roi_names():
    """Load region labels from feature files"""
    data_root = os.environ.get("HD_MSN_DATA_ROOT", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
    feature_dir = os.path.join(data_root, "featureHD")
    
    # Locate the first feature file
    feature_files = sorted([f for f in os.listdir(feature_dir) if f.endswith('_features.csv')])
    if len(feature_files) == 0:
        print("Warning: No feature files found, using generic ROI names")
        return None
    
    # Read the first feature file to obtain region names
    feature_file = os.path.join(feature_dir, feature_files[0])
    df = pd.read_csv(feature_file)
    
    if 'Region' not in df.columns:
        print("Warning: 'Region' column not found in feature file")
        return None
    
    roi_names = df['Region'].tolist()
    print(f"Loaded {len(roi_names)} actual ROI names from {feature_files[0]}")
    print(f"  Examples: {roi_names[:3]} ... {roi_names[-3:]}")
    
    return roi_names

def main():
    # Input directories
    data_root = os.environ.get("HD_MSN_DATA_ROOT", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"))
    hd_dir = os.path.join(data_root, "HD308")
    hc_dir = os.path.join(data_root, "HC308")
    
    # Output directory
    out_dir = os.environ.get("HD_MSN_PATTERN_INPUT", os.path.join(data_root, "Pattern_input"))
    os.makedirs(out_dir, exist_ok=True)
    
    # ROI names: load anatomical region labels from the feature files
    print("Loading ROI names...")
    roi_names = load_actual_roi_names()
    
    # If label loading fails, inspect matrix dimensions and use generic region names
    if roi_names is None:
        test_file = os.path.join(hd_dir, os.listdir(hd_dir)[0]) if os.path.exists(hd_dir) and os.listdir(hd_dir) else None
        if test_file and test_file.endswith('.csv'):
            test_mat = pd.read_csv(test_file, index_col=0)
            n_rois = test_mat.shape[0]
            print(f"Detected {n_rois} ROIs from matrix files")
        else:
            n_rois = 308
            print(f"Using default 308 ROIs")
        
        roi_names = [f"ROI_{i+1}" for i in range(n_rois)]
        print(f"Warning: Using generic ROI names: ROI_1 to ROI_{n_rois}")
    
    print(f"Total ROIs: {len(roi_names)}")
    
    # Process HD data
    print(f"\nProcessing HD data from {hd_dir}")
    hd_files = sorted([f for f in os.listdir(hd_dir) if f.endswith('_similarity_matrix.csv')])
    print(f"Found {len(hd_files)} HD similarity matrices")
    
    hd_data = []
    hd_subjects = []
    for fname in hd_files:
        subject_id = extract_subject_id(fname)
        if subject_id is None:
            print(f"  Warning: Could not extract subject ID from {fname}")
            continue
        
        fpath = os.path.join(hd_dir, fname)
        try:
            regional_msn = extract_regional_msn(fpath)
            if len(regional_msn) != len(roi_names):
                print(f"  Warning: {fname} has {len(regional_msn)} ROIs, expected {len(roi_names)}, skipping")
                continue
            hd_data.append(regional_msn)
            hd_subjects.append(subject_id)
        except Exception as e:
            print(f"  Error processing {fname}: {e}")
    
    # Save HD regional MSN
    if len(hd_data) > 0:
        hd_df = pd.DataFrame(hd_data, columns=roi_names)
        hd_df.insert(0, 'subject', hd_subjects)
        hd_out = os.path.join(out_dir, 'HD_regional_msn.csv')
        hd_df.to_csv(hd_out, index=False)
        print(f"Saved HD regional MSN: {hd_out} ({len(hd_df)} subjects x {len(roi_names)} ROIs)")
    else:
        print("No valid HD data extracted!")
    
    # Process HC data
    print(f"\nProcessing HC data from {hc_dir}")
    hc_files = sorted([f for f in os.listdir(hc_dir) if f.endswith('_similarity_matrix.csv')])
    print(f"Found {len(hc_files)} HC similarity matrices")
    
    hc_data = []
    hc_subjects = []
    for fname in hc_files:
        subject_id = extract_subject_id(fname)
        if subject_id is None:
            print(f"  Warning: Could not extract subject ID from {fname}")
            continue
        
        fpath = os.path.join(hc_dir, fname)
        try:
            regional_msn = extract_regional_msn(fpath)
            if len(regional_msn) != len(roi_names):
                print(f"  Warning: {fname} has {len(regional_msn)} ROIs, expected {len(roi_names)}, skipping")
                continue
            hc_data.append(regional_msn)
            hc_subjects.append(subject_id)
        except Exception as e:
            print(f"  Error processing {fname}: {e}")
    
    # Save HC regional MSN
    if len(hc_data) > 0:
        hc_df = pd.DataFrame(hc_data, columns=roi_names)
        hc_df.insert(0, 'subject', hc_subjects)
        hc_out = os.path.join(out_dir, 'HC_regional_msn.csv')
        hc_df.to_csv(hc_out, index=False)
        print(f"Saved HC regional MSN: {hc_out} ({len(hc_df)} subjects x {len(roi_names)} ROIs)")
    else:
        print("No valid HC data extracted!")
    
    # Save ROI names
    roi_out = os.path.join(out_dir, 'roi_names_308_actual.txt')
    with open(roi_out, 'w') as f:
        for name in roi_names:
            f.write(f"{name}\n")
    print(f"\nSaved ROI names: {roi_out}")
    
    print(f"\n=== Extraction complete ===")
    print(f"Output directory: {out_dir}")
    print(f"  - HD_regional_msn.csv ({len(hd_data) if len(hd_data) > 0 else 0} subjects)")
    print(f"  - HC_regional_msn.csv ({len(hc_data) if len(hc_data) > 0 else 0} subjects)")
    print(f"  - roi_names_308_actual.txt ({len(roi_names)} ROIs)")
    print(f"\nROI names are now actual brain region names (e.g., lh.bankssts_part1)")
    print(f"not generic ROI_1...ROI_308 anymore!")

if __name__ == '__main__':
    main()

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Extract subject-level regional MSN summaries for HD participants from standardized MSN matrices.
# Input source/location: HD MSN matrices and atlas/region labels under the configured project data root.
# Output location: HD regional MSN tables in the configured results directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Discover HD matrices; validate matrix shape; calculate region-wise MSN summaries; attach region labels; export subject-by-region results.
# Log location: No dedicated log file; progress is written to standard output.
# =============================================================================
