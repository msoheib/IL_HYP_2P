import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt


def match_neurons(f_pre_path, f_post_path, iscell_pre_path, iscell_post_path, match_rois_path):
    # Load data
    f_pre = np.load(f_pre_path)
    f_post = np.load(f_post_path)
    iscell_pre = np.load(iscell_pre_path)
    iscell_post = np.load(iscell_post_path)
    match_rois = pd.read_csv(match_rois_path)

    # Create masks for valid cells
    pre_mask = iscell_pre[:, 0] == 1
    post_mask = iscell_post[:, 0] == 1

    # Create mappings from original to filtered indices
    pre_mapping = {orig: filt for filt, orig in enumerate(np.where(pre_mask)[0])}
    post_mapping = {orig: filt for filt, orig in enumerate(np.where(post_mask)[0])}

    # Apply mappings to match_rois
    match_rois['Filtered_Pre_Index'] = match_rois['Pre_Session_Index'].map(pre_mapping)
    match_rois['Filtered_Post_Index'] = match_rois['Post_Session_Index'].map(post_mapping)

    # Remove unmatched or invalid cells
    valid_matches = match_rois.dropna()

    # Extract matched F traces
    matched_f_pre = f_pre[pre_mask][valid_matches['Filtered_Pre_Index'].astype(int)]
    matched_f_post = f_post[post_mask][valid_matches['Filtered_Post_Index'].astype(int)]

    return valid_matches, matched_f_pre, matched_f_post



def process_matched_neurons(matched_rois_path, iscell_pre_path, iscell_post_path):
    # Load the data
    matched_rois = pd.read_csv(matched_rois_path)
    iscell_pre = np.load(iscell_pre_path)
    iscell_post = np.load(iscell_post_path)

    # Compute indices where first column is 1 (iscell status)
    iscell_pre_indices = np.where(iscell_pre[:, 0] == 1)[0]
    iscell_post_indices = np.where(iscell_post[:, 0] == 1)[0]

    # Create mappings from original indices to new filtered indices
    iscell_pre_mapping = {original_index: new_index for new_index, original_index in enumerate(iscell_pre_indices)}
    iscell_post_mapping = {original_index: new_index for new_index, original_index in enumerate(iscell_post_indices)}

    # Apply mappings to matched_rois
    matched_rois['Filtered_Pre_Session_Index'] = matched_rois['Pre_Session_Index'].map(iscell_pre_mapping)
    matched_rois['Filtered_Post_Session_Index'] = matched_rois['Post_Session_Index'].map(iscell_post_mapping)

    # Remove rows where mapping couldn't be applied (NaN) or resulted in zero
    final_matches = matched_rois.dropna(subset=['Filtered_Pre_Session_Index', 'Filtered_Post_Session_Index'])
    final_matches = final_matches[
        (final_matches['Filtered_Pre_Session_Index'] != 0) & 
        (final_matches['Filtered_Post_Session_Index'] != 0)
    ]

    # Adjust indices to be 0-based
    final_matches['Filtered_Pre_Session_Index'] -= 1
    final_matches['Filtered_Post_Session_Index'] -= 1

    # Get the lists of detected ROIs for each session
    rois_pre = final_matches['Filtered_Pre_Session_Index'].tolist()
    rois_post = final_matches['Filtered_Post_Session_Index'].tolist()

    return final_matches, rois_pre, rois_post

def plot_matched_neurons_subset(final_matches, n_samples=50):
    """
    Plot a subset of matched neurons to visually verify the matching.
    
    Parameters:
    final_matches (pd.DataFrame): DataFrame containing the matched and filtered neuron indices
    n_samples (int): Number of samples to plot (default 50)
    """
    # Sample a subset of matches
    sample = final_matches.sample(n=min(n_samples, len(final_matches)))
    
    # Create the scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(sample['Filtered_Pre_Session_Index'], sample['Filtered_Post_Session_Index'], alpha=0.6)
    plt.xlabel('Pre-Session Neuron Index')
    plt.ylabel('Post-Session Neuron Index')
    plt.title(f'Subset of {n_samples} Matched Neurons')
    
    # Add a diagonal line for reference
    max_index = max(sample['Filtered_Pre_Session_Index'].max(), sample['Filtered_Post_Session_Index'].max())
    plt.plot([0, max_index], [0, max_index], 'r--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def convert_mat_to_csv(mat_file_path, output_csv_path):
    """
    Converts the 'cell_to_index_map' dataset from a MATLAB v7.3 file to a CSV file.
    
    Parameters:
        mat_file_path (str): Path to the .mat file.
        output_csv_path (str): Path where the CSV file will be saved.
    """
    with h5py.File(mat_file_path, 'r') as file:
        # Extract the 'cell_to_index_map' data
        cell_to_index_map = file['cell_registered_struct']['cell_to_index_map'][:]
    
    # Convert the data to a DataFrame
    df = pd.DataFrame(cell_to_index_map.T, columns=["Pre_Session_Index", "Post_Session_Index"])
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"CSV file has been saved to: {output_csv_path}")