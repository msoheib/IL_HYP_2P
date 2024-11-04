import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os
import argparse
from tqdm import tqdm
import csv

def load_and_extract_rois(file_path, iscell_path):
    stat = np.load(file_path, allow_pickle=True)
    iscell = np.load(iscell_path)
    
    print(f"Loaded data type: {type(stat)}")
    print(f"Loaded data shape: {stat.shape}")
    print(f"Loaded iscell shape: {iscell.shape}")
    
    rois = []
    iscell_indices = []
    for i, (roi, is_cell) in enumerate(zip(stat, iscell)):
        if is_cell[0] and isinstance(roi, dict) and 'ypix' in roi and 'xpix' in roi:
            ypix, xpix = roi['ypix'], roi['xpix']
            centroid = (np.mean(ypix), np.mean(xpix))
            bbox = (np.min(ypix), np.max(ypix), np.min(xpix), np.max(xpix))
            rois.append((centroid, bbox))
            iscell_indices.append(i)
    return rois, iscell_indices

def translate_indices(original_indices, iscell_indices):
    return [iscell_indices.index(i) if i in iscell_indices else -1 for i in original_indices]

def load_manually_matched_rois(file_path):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        return [(int(row[0]), int(row[1])) for row in reader]

def save_translated_matches_to_csv(matches, pre_iscell_indices, post_iscell_indices, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Pre_Session_Index', 'Post_Session_Index'])
        for pre_idx, post_idx in matches:
            translated_pre = translate_indices([pre_idx], pre_iscell_indices)[0]
            translated_post = translate_indices([post_idx], post_iscell_indices)[0]
            if translated_pre != -1 and translated_post != -1:
                writer.writerow([translated_pre + 1, translated_post + 1])  # Adding 1 to convert from 0-based to 1-based indexing
    print(f"Translated matches saved to {output_file}")

def drop_columns_with_nan_rows(df1, df2):
    # Ensure both DataFrames have the same columns
    common_columns = df1.columns.intersection(df2.columns)
    df1 = df1[common_columns]
    df2 = df2[common_columns]
    
    # Create a mask for columns that have NaN in any row in either DataFrame
    mask = (df1.isna() | df2.isna()).any()
    
    # Drop those columns from both DataFrames
    df1_cleaned = df1.loc[:, ~mask]
    df2_cleaned = df2.loc[:, ~mask]
    
    return df1_cleaned, df2_cleaned

def main():
    parser = argparse.ArgumentParser(description='Translate manually matched ROIs using iscell filtering.')
    parser.add_argument('--manual_matches', type=str, required=True, help='Path to the manually matched ROIs CSV file')
    parser.add_argument('--output', type=str, default='translated_matched_rois.csv', help='Output CSV file name (default: translated_matched_rois.csv)')
    args = parser.parse_args()

    pre_file = os.path.join('pre', 'stat.npy')
    post_file = os.path.join('post', 'stat.npy')
    pre_iscell_file = os.path.join('pre', 'iscell.npy')
    post_iscell_file = os.path.join('post', 'iscell.npy')
    manual_matches_file = args.manual_matches
    output_file = args.output

    print("Loading pre-session data:")
    pre_rois, pre_iscell_indices = load_and_extract_rois(pre_file, pre_iscell_file)
    print("\nLoading post-session data:")
    post_rois, post_iscell_indices = load_and_extract_rois(post_file, post_iscell_file)

    print(f"\nNumber of pre-session ROIs after iscell filtering: {len(pre_rois)}")
    print(f"Number of post-session ROIs after iscell filtering: {len(post_rois)}")

    manual_matches = load_manually_matched_rois(manual_matches_file)
    print(f"Loaded {len(manual_matches)} manually matched ROIs")

    save_translated_matches_to_csv(manual_matches, pre_iscell_indices, post_iscell_indices, output_file)

if __name__ == "__main__":
    main()