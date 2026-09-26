#!/usr/bin/env python3
"""
Extract regional MSN values from HC308 individual MSN matrices.
For each subject's 308x308 MSN matrix, calculate regional MS by averaging each row (excluding diagonal).
This matches the method used for HD/mHD/preHD groups.
Then combine all subjects into a single CSV file.
"""

import os
import numpy as np
import pandas as pd
import glob

# Define paths
data_root = os.environ.get('HD_MSN_DATA_ROOT', os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data'))
input_dir = os.path.join(data_root, 'HC308')
output_dir = os.path.join(data_root, 'HC_individual_regional_ms')

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# Find all CSV files in HC308 directory
csv_files = sorted(glob.glob(os.path.join(input_dir, '*_similarity_matrix.csv')))
print(f"\nFound {len(csv_files)} CSV files in {input_dir}")

if len(csv_files) == 0:
    print("ERROR: No CSV files found!")
    exit(1)

# Dictionary to store regional MS values for each subject
all_subjects_data = {}

# Process each subject's MSN matrix
for csv_file in csv_files:
    # Extract subject ID from filename
    filename = os.path.basename(csv_file)
    subject_id = filename.replace('_similarity_matrix.csv', '')
    
    print(f"\nProcessing: {subject_id}")
    
    try:
        # Read the MSN matrix (308x308)
        # First column is the row index, so index_col=0
        msn_matrix = pd.read_csv(csv_file, index_col=0)
        
        print(f"  Matrix shape: {msn_matrix.shape}")
        
        # Verify it's a 308x308 matrix
        if msn_matrix.shape[0] != 308 or msn_matrix.shape[1] != 308:
            print(f"  WARNING: Expected 308x308 matrix, got {msn_matrix.shape}. Skipping...")
            continue
        
        # Calculate regional MS by averaging each row (excluding diagonal)
        # This matches the method used for HD/mHD/preHD groups
        msn_array = msn_matrix.values
        
        # Calculate regional MSN for each region (row-wise mean excluding diagonal)
        regional_ms = []
        n = msn_array.shape[0]
        for i in range(n):
            row = msn_array[i, :]
            # Exclude the diagonal element (self-correlation)
            mask = np.arange(n) != i
            regional_ms.append(np.mean(row[mask]))
        
        regional_ms = np.array(regional_ms)
        
        print(f"  Regional MS values range: [{regional_ms.min():.6f}, {regional_ms.max():.6f}]")
        
        # Store the regional MS values
        all_subjects_data[f'HC_{subject_id}'] = regional_ms
        
    except Exception as e:
        print(f"  ERROR processing {subject_id}: {e}")
        continue

# Convert to DataFrame
print(f"\n{'='*80}")
print(f"Combining data from {len(all_subjects_data)} subjects...")

if len(all_subjects_data) == 0:
    print("ERROR: No valid data collected!")
    exit(1)

# Create DataFrame with Region_Index as rows and subjects as columns
df = pd.DataFrame(all_subjects_data)

# Add Region_Index column (1-based indexing)
df.index = range(1, len(df) + 1)
df.index.name = 'Region_Index'

print(f"Combined data shape: {df.shape} (rows=regions, columns=subjects)")
print(f"Number of regions: {df.shape[0]}")
print(f"Number of subjects: {df.shape[1]}")

# Save to CSV file
output_file = os.path.join(output_dir, 'all_subjects_regional_msn.csv')
df.to_csv(output_file)
print(f"\nOutput saved to: {output_file}")

# Print summary statistics
print(f"\n{'='*80}")
print("Summary Statistics:")
print(f"{'='*80}")
for col in df.columns[:5]:  # Show first 5 subjects as examples
    print(f"{col}:")
    print(f"  Mean: {df[col].mean():.6f}")
    print(f"  Std:  {df[col].std():.6f}")
    print(f"  Min:  {df[col].min():.6f}")
    print(f"  Max:  {df[col].max():.6f}")

print(f"\nOverall statistics across all subjects:")
print(f"  Mean of means: {df.mean(axis=1).mean():.6f}")
print(f"  Global min: {df.min().min():.6f}")
print(f"  Global max: {df.max().max():.6f}")

print(f"\n{'='*80}")
print("Processing completed successfully!")
print(f"{'='*80}")

# =============================================================================
# SCRIPT DOCUMENTATION
# Purpose: Extract subject-level regional MSN summaries for healthy controls from standardized MSN matrices.
# Input source/location: HC MSN matrices and atlas/region labels under the configured project data root.
# Output location: HC regional MSN tables in the configured results directory.
# Input/output notes: Inputs are de-identified/public analysis resources; outputs contain derived analysis products only.
# Main steps: Discover HC matrices; validate matrix shape; calculate region-wise MSN summaries; attach region labels; export subject-by-region results.
# Log location: No dedicated log file; progress is written to standard output.
# =============================================================================
