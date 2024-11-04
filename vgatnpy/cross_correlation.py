import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import fisher_exact
from scipy.signal import correlate
import helper_functions as hf

def analyze_correlations(pre_data, post_data, correlate_with='pupil_size', name_of_v="Pupil Size", from_files=False, threshold=0.1, alpha=0.05, n_bootstrap=10000):
    """
    Analyze and compare zero-lag correlations from pre and post sessions with a Prism-like plot style,
    including confidence intervals, significance test, significance bar, and individual data points.
    
    Args:
    pre_data (str or pd.DataFrame): Path to pre session CSV file or pre session DataFrame.
    post_data (str or pd.DataFrame): Path to post session CSV file or post session DataFrame.
    from_files (bool): If True, pre_data and post_data are expected to be file paths. If False, they are DataFrames.
    threshold (float): Correlation threshold to consider as significant.
    alpha (float): Significance level for the statistical test.
    n_bootstrap (int): Number of bootstrap samples for CI calculation.
    
    Returns:
    None, but plots and saves a bar chart comparing the proportions of significant correlations with individual data points.
    """
    # Load data from files if necessary
    if from_files:
        pre_df = pd.read_csv(pre_data)
        post_df = pd.read_csv(post_data)
    else:
        pre_df = pre_data
        post_df = post_data
    
    # Identify ROI columns (assuming they are numeric)
    roi_columns_pre = [col for col in pre_df.columns if col.isnumeric()]
    roi_columns_post = [col for col in post_df.columns if col.isnumeric()]
    
    # Calculate zero-lag correlations
    zero_lag_corr_pre = pre_df[roi_columns_pre].corrwith(pre_df[correlate_with])
    zero_lag_corr_post = post_df[roi_columns_post].corrwith(post_df[correlate_with])
    
    # Function to calculate ratio of significant correlations
    def calc_significant_ratio(corr_series, threshold):
        return (abs(corr_series) > threshold).mean()
    
    # Calculate ratios
    ratio_significant_pre = calc_significant_ratio(zero_lag_corr_pre, threshold)
    ratio_significant_post = calc_significant_ratio(zero_lag_corr_post, threshold)
    
    # Bootstrap function
    def bootstrap_ci(data, func, n_bootstrap):
        bootstrap_results = [func(data.sample(n=len(data), replace=True)) for _ in range(n_bootstrap)]
        return np.percentile(bootstrap_results, [2.5, 97.5])
    
    # Calculate CIs
    pre_ci = bootstrap_ci(zero_lag_corr_pre, lambda x: calc_significant_ratio(x, threshold), n_bootstrap)
    post_ci = bootstrap_ci(zero_lag_corr_post, lambda x: calc_significant_ratio(x, threshold), n_bootstrap)
    
    # Perform Fisher's exact test
    table = [[sum(abs(zero_lag_corr_pre) > threshold), sum(abs(zero_lag_corr_pre) <= threshold)],
             [sum(abs(zero_lag_corr_post) > threshold), sum(abs(zero_lag_corr_post) <= threshold)]]
    _, p_value = fisher_exact(table)
    
    # Set the style
    sns.set_style("ticks")
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(4, 7))
    
    # Create the bar plot
    bars = ax.bar(['Pre-Session', 'Post-Session'], [ratio_significant_pre, ratio_significant_post], 
                  yerr=[[ratio_significant_pre - pre_ci[0], ratio_significant_post - post_ci[0]], 
                        [pre_ci[1] - ratio_significant_pre, post_ci[1] - ratio_significant_post]], 
                  color=['#6baed6', '#fd8d3c'], capsize=5, 
                  edgecolor='black', linewidth=1, width=0.6)
    
    # Add individual data points
    ax.scatter([bars[0].get_x() + bars[0].get_width() / 2] * len(zero_lag_corr_pre),
               abs(zero_lag_corr_pre), color='black', zorder=3, alpha=0.7, s=30)
    ax.scatter([bars[1].get_x() + bars[1].get_width() / 2] * len(zero_lag_corr_post),
               abs(zero_lag_corr_post), color='black', zorder=3, alpha=0.7, s=30)
    
    # Add significance line and p-value/ns
    y_max = max(max(abs(zero_lag_corr_pre)), max(abs(zero_lag_corr_post)))
    y_line = y_max * 1.1
    y_text = y_max * 1.15
    ax.plot([0, 0, 1, 1], [y_line, y_line + y_max * 0.02, y_line + y_max * 0.02, y_line], 
            lw=1.5, c='black')
    
    if p_value < alpha:
        significance_text = f'p = {p_value:.4f}'
    else:
        significance_text = 'ns'
    
    ax.text(0.5, y_text, significance_text, ha='center', va='bottom')
    
    # Customize the plot
    ax.set_title(f'Correlation with {name_of_v}', fontweight='bold')
    ax.set_ylabel('Ratio of ROIs with Significant Correlation')
    ax.set_xlabel('')
    
    # Remove top and right spines
    sns.despine()
    
    # Adjust y-axis to start from 0
    ax.set_ylim(0, y_max * 1.2)
    
    plt.tight_layout()
    
    # Save the plot in the current folder
    plt.savefig(f'correlation_with_{name_of_v.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_cross_correlations(df, correlate_with='pupil_size', name_of_v="Pupil Size"):
    """
    Plot the normalized cross-correlations between 'speed' and each ROI column with zero lag in the center.
    Also plot the mean trace of all correlations in red, handling NaN values.
    
    Args:
    df (pandas.DataFrame): DataFrame containing the data.
    
    Returns:
    None, but displays the cross-correlations plot.
    """

    roi_columns = hf.get_digit_columns(df)
    cross_correlations = {}
    
    for col in roi_columns:
        # Check for NaN values
        if df[correlate_with].isna().any() or df[col].isna().any():
            #print(f"Warning: NaN values found in 'speed' or '{col}'. These will be ignored in the correlation calculation.")
            continue
        
        # Remove NaN values for correlation calculation
        valid_data = df[[correlate_with, col]].dropna()
        
        if len(valid_data) < 2:
            #print(f"Error: Not enough valid data points for '{col}'. Skipping this column.")
            continue
        valid_data_col = valid_data[correlate_with]
        # Calculate the cross-correlation
        corr = correlate(valid_data_col, valid_data[col], mode='full')
        
        # Normalize the correlation
        n = len(valid_data)
        norm = np.sqrt(np.dot(valid_data_col, valid_data_col) * np.dot(valid_data[col], valid_data[col]))
        
        if norm == 0:
            print(f"Warning: Zero norm encountered for '{col}'. This may lead to NaN values.")
            corr = np.full_like(corr, np.nan)
        else:
            corr = corr / norm
        
        cross_correlations[col] = corr

    if not cross_correlations:
        print("Error: No valid correlations could be calculated. Check your data for NaN values or constant columns.")
        return

    sns.set(style='whitegrid')

    plt.figure(figsize=(7, 7))
    
    # Calculate the mean trace, ignoring NaN values
    all_correlations = np.array(list(cross_correlations.values()))
    mean_trace = np.nanmean(all_correlations, axis=0)
    
    if np.isnan(mean_trace).all():
        print("Error: Mean trace is all NaN. Unable to plot.")
        return
    
    lags = np.arange(-len(df) + 1, len(df))
    lags = lags/30
    
    for col, corr in cross_correlations.items():
        plt.plot(lags, corr, label=col, lw=0.5, color='gray', alpha=0.7)
    
    # Plot the mean trace in red
    plt.plot(lags, mean_trace, label='Mean', lw=2, color='red')

    plt.title(f'Cross-Correlation between {name_of_v} and ROI')
    plt.xlabel('Lag (seconds)')
    plt.ylabel('Cross-Correlation (zero-time)')
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)

    plt.xticks(fontsize=12)
    plt.grid(False, linestyle='--', linewidth=0.5, color='gray')
    
    # Add legend
    #plt.legend(loc='best', fontsize=10)
    
    # Set y-axis limits to [-1, 1]
    plt.ylim(-1, 1)
    
    plt.tight_layout()
    
    # Save the plot as PNG
    plt.savefig(f'cross_correlation_{name_of_v.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_cross_correlations_around_stimuli(df, time_window, correlate_with='pupil_size', name_of_v="Pupil Size", plot_title=None):
    """
    Plot the normalized cross-correlations between 'correlate_with' and each ROI column with zero lag in the center,
    using data selected around all stimuli orientations.
    
    Args:
    df (pandas.DataFrame): DataFrame containing the data.
    time_window (int): Number of time points to include before and after each stimulus onset.
    correlate_with (str): Column name to correlate with ROIs.
    name_of_v (str): Name of the variable to correlate with (for plot title).
    plot_title (str, optional): Custom plot title. If None, a default title will be used.
    
    Returns:
    None, but displays the cross-correlations plot and saves it as a PNG file.
    """
    roi_columns = hf.get_digit_columns(df)
    cross_correlations = {}
    
    # Find the stimuli column (assuming it follows the pattern 'degrees_X')
    stimuli_column = next((col for col in df.columns if col.startswith('degrees_')), None)
    if stimuli_column is None:
        print("Error: No stimuli column found. Ensure your CSV file contains a column named 'degrees_X' where X is a number.")
        return
    
    # Find indices where stimuli change (stimulus onset)
    stimuli_indices = df[df[stimuli_column] != df[stimuli_column].shift()].index
    
    # Function to get data around stimuli
    def get_data_around_stimuli(series):
        data = []
        for idx in stimuli_indices:
            start = max(0, idx - time_window)
            end = min(len(df), idx + time_window + 1)
            data.extend(series[start:end])
        return np.array(data)
    
    # Get data for correlate_with column
    correlate_with_data = get_data_around_stimuli(df[correlate_with])
    
    for col in roi_columns:
        # Get data for current ROI column
        roi_data = get_data_around_stimuli(df[col])
        
        # Remove NaN values
        valid_mask = ~np.isnan(correlate_with_data) & ~np.isnan(roi_data)
        valid_correlate_with = correlate_with_data[valid_mask]
        valid_roi_data = roi_data[valid_mask]
        
        if len(valid_correlate_with) < 2:
            print(f"Error: Not enough valid data points for '{col}'. Skipping this column.")
            continue
        
        # Calculate the cross-correlation
        corr = correlate(valid_correlate_with, valid_roi_data, mode='full')
        
        # Normalize the correlation
        n = len(valid_correlate_with)
        norm = np.sqrt(np.dot(valid_correlate_with, valid_correlate_with) * np.dot(valid_roi_data, valid_roi_data))
        
        if norm == 0:
            print(f"Warning: Zero norm encountered for '{col}'. This may lead to NaN values.")
            corr = np.full_like(corr, np.nan)
        else:
            corr = corr / norm
        
        cross_correlations[col] = corr

    if not cross_correlations:
        print("Error: No valid correlations could be calculated. Check your data for NaN values or constant columns.")
        return

    sns.set(style='whitegrid')

    plt.figure(figsize=(10, 7))
    
    # Calculate the mean trace, ignoring NaN values
    all_correlations = np.array(list(cross_correlations.values()))
    mean_trace = np.nanmean(all_correlations, axis=0)
    
    if np.isnan(mean_trace).all():
        print("Error: Mean trace is all NaN. Unable to plot.")
        return
    
    lags = np.arange(-len(correlate_with_data) + 1, len(correlate_with_data))
    lags = lags / 30  # Assuming 30 Hz sampling rate, adjust if different
    
    for col, corr in cross_correlations.items():
        plt.plot(lags, corr, label=col, lw=0.5, color='gray', alpha=0.7)
    
    # Plot the mean trace in red
    plt.plot(lags, mean_trace, label='Mean', lw=2, color='red')

    if plot_title is None:
        plot_title = f'Cross-Correlation between {name_of_v} and ROI\nAround All Stimuli Orientations'
    plt.title(plot_title)
    plt.xlabel('Lag (seconds)')
    plt.ylabel('Cross-Correlation')
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)

    plt.xticks(fontsize=12)
    plt.grid(False, linestyle='--', linewidth=0.5, color='gray')
    
    # Set y-axis limits to [-1, 1]
    plt.ylim(-1, 1)
    
    plt.tight_layout()
    
    # Save the plot as a PNG file in the current directory
    plt.savefig('cross_correlations_plot.png', dpi=300, bbox_inches='tight')
    
    plt.show()


from numpy import arange
x = arange(25).reshape(5, 5)
rwb_cmap = sns.diverging_palette(220, 20, as_cmap=True)


def plot_roi_correlation_matrix(df):
    """
    Plot the correlation matrix of all ROI columns, after dropping columns with all zeros or NaN values.
    
    Args:
    df (pandas.DataFrame): DataFrame containing the ROI data.
    
    Returns:
    None, but displays the correlation matrix plot.
    """
    
    # Get ROI columns
    roi_columns = hf.get_digit_columns(df)
    
    # Create a new DataFrame with only ROI columns
    roi_df = df[roi_columns].copy()
    
    # Drop columns with all zeros
    roi_df = roi_df.loc[:, (roi_df != 0).any(axis=0)]
    
    # Drop columns with any NaN values
    roi_df = roi_df.dropna(axis=1)
    
    # Check if we have any columns left
    if roi_df.empty:
        print("Error: No valid ROI columns left after dropping zeros and NaNs.")
        return
    
    # Calculate the correlation matrix
    corr_matrix = roi_df.corr()
    
    # Set up the matplotlib figure
    plt.figure(figsize=(9, 7.5))
    from numpy import arange
    x = arange(25).reshape(5, 5)
    rwb_cmap = sns.diverging_palette(220, 20, as_cmap=True)
    # Create a heatmap
    sns.heatmap(corr_matrix, cmap =rwb_cmap, vmin=-1, vmax=1, center=0)
    
    plt.title("Correlation Between ROIs", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    
    # Print information about dropped columns
    dropped_columns = set(roi_columns) - set(roi_df.columns)
    if dropped_columns:
        print(f"Dropped columns: {', '.join(dropped_columns)}")
    print(f"Remaining columns: {len(roi_df.columns)}")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import helper_functions as hf

def plot_roi_correlation_comparison(df1, df2, alpha=0.05):
    """
    Plot the correlation matrices of two DataFrames side by side, print correlation statistics,
    and create a bar plot comparing the percentage of significant correlations.
    
    Args:
    df1, df2 (pandas.DataFrame): DataFrames containing the ROI data.
    alpha (float): Significance level for correlation tests.
    
    Returns:
    None, but displays plots and prints statistics.
    """
    
    def process_df(df):
        roi_columns = hf.get_digit_columns(df)
        roi_df = df[roi_columns].copy()
        roi_df = roi_df.loc[:, (roi_df != 0).any(axis=0)]
        roi_df = roi_df.dropna(axis=1)
        return roi_df
    
    roi_df1 = process_df(df1)
    roi_df2 = process_df(df2)
    
    if roi_df1.empty or roi_df2.empty:
        print("Error: No valid ROI columns left after dropping zeros and NaNs.")
        return
    
    def compute_corr_stats(corr_matrix):
        n = corr_matrix.shape[0]
        tri_k = n * (n-1) // 2  # number of elements in upper triangle
        
        # Compute p-values
        p_values = np.zeros_like(corr_matrix)
        for i in range(n):
            for j in range(i+1, n):  # Only upper triangle
                r = corr_matrix.iloc[i, j]
                t = r * np.sqrt((n-2) / (1-r**2))
                p_values[i, j] = stats.t.sf(np.abs(t), n-2)*2
        
        # Count significant correlations in upper triangle
        sig_corrs = np.sum(p_values[np.triu_indices(n, k=1)] < alpha)
        percent_sig = (sig_corrs / tri_k) * 100
        
        # Calculate mean and median correlation from upper triangle
        corr_values = corr_matrix.values[np.triu_indices(n, k=1)]
        mean_corr = np.mean(corr_values)
        median_corr = np.median(corr_values)
        
        return sig_corrs, tri_k, percent_sig, mean_corr, median_corr
    
    corr_matrix1 = roi_df1.corr()
    corr_matrix2 = roi_df2.corr()
    
    sig_corrs1, total_corrs1, percent_sig1, mean_corr1, median_corr1 = compute_corr_stats(corr_matrix1)
    sig_corrs2, total_corrs2, percent_sig2, mean_corr2, median_corr2 = compute_corr_stats(corr_matrix2)
    
    # Chi-square test for percentage of significant correlations
    observed = np.array([[sig_corrs1, total_corrs1 - sig_corrs1],
                         [sig_corrs2, total_corrs2 - sig_corrs2]])
    chi2, p_value_chi2 = stats.chi2_contingency(observed)[:2]
    
    # Plotting correlation matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))
    
    rwb_cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix1, cmap=rwb_cmap, vmin=-1, vmax=1, center=0, ax=ax1)
    ax1.set_title("Correlation Matrix 1", fontsize=16)
    
    sns.heatmap(corr_matrix2, cmap=rwb_cmap, vmin=-1, vmax=1, center=0, ax=ax2)
    ax2.set_title("Correlation Matrix 2", fontsize=16)
    
    plt.tight_layout()
    plt.show()
    
    # GraphPad Prism-like theme adjustments
    plt.style.use('seaborn-white')  # Use a clean background
    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.35
    index = np.arange(2)

    # Pastel colors
    colors = ['#aec7e8', '#98df8a']  # Pastel blue and green

    bars = plt.bar(index, [percent_sig1, percent_sig2], bar_width,
                alpha=0.9, color=colors, label=['Pre Stress', 'Post Stress'], edgecolor='grey')

    # Make the plot resemble GraphPad Prism's style
    plt.xlabel('Datasets', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage', fontsize=12, fontweight='bold')
    plt.title('Comparison of percentage of ROIs with Significant Correlations', fontsize=14, fontweight='bold')
    plt.xticks(index, ('Dataset 1', 'Dataset 2'), fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend(frameon=False, fontsize=11)

    # Customize the axes and grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('grey')
    ax.spines['bottom'].set_color('grey')
    ax.yaxis.grid(True, linestyle='--', which='major', color='lightgrey', alpha=0.7)  # Light horizontal grid lines

    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"Dataset 1:")
    print(f"  Percentage of significant correlations: {percent_sig1:.2f}%")
    print(f"  Mean correlation: {mean_corr1:.4f}")
    print(f"  Median correlation: {median_corr1:.4f}")
    print(f"\nDataset 2:")
    print(f"  Percentage of significant correlations: {percent_sig2:.2f}%")
    print(f"  Mean correlation: {mean_corr2:.4f}")
    print(f"  Median correlation: {median_corr2:.4f}")
    
    # Compute differences
    diff_percent_sig = percent_sig2 - percent_sig1
    diff_mean_corr = mean_corr2 - mean_corr1
    diff_median_corr = median_corr2 - median_corr1
    
    # Fisher's r-to-z transformation for testing difference between correlations
    z1 = np.arctanh(mean_corr1)
    z2 = np.arctanh(mean_corr2)
    n1 = corr_matrix1.shape[0]
    n2 = corr_matrix2.shape[0]
    se_diff = np.sqrt(1/(n1-3) + 1/(n2-3))
    z = (z2 - z1) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    print(f"\nDifferences (Dataset 2 - Dataset 1):")
    print(f"  Difference in percentage of significant correlations: {diff_percent_sig:.2f}%")
    print(f"  Difference in mean correlation: {diff_mean_corr:.4f}")
    print(f"  Difference in median correlation: {diff_median_corr:.4f}")
    print(f"  P-value for difference in mean correlation: {p_value:.4f}")
    print(f"  P-value for difference in percentage of significant correlations: {p_value_chi2:.4f}")


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import helper_functions as hf

def plot_roi_correlation_comparison(df1, df2, alpha=0.05):
    """
    Plot the correlation matrices of two DataFrames side by side, print correlation statistics,
    and create a bar plot comparing the percentage of significant correlations.
    
    Args:
    df1, df2 (pandas.DataFrame): DataFrames containing the ROI data.
    alpha (float): Significance level for correlation tests.
    
    Returns:
    None, but displays plots and prints statistics.
    """
    
    def process_df(df):
        roi_columns = hf.get_digit_columns(df)
        roi_df = df[roi_columns].copy()
        roi_df = roi_df.loc[:, (roi_df != 0).any(axis=0)]
        roi_df = roi_df.dropna(axis=1)
        return roi_df
    
    roi_df1 = process_df(df1)
    roi_df2 = process_df(df2)
    
    if roi_df1.empty or roi_df2.empty:
        print("Error: No valid ROI columns left after dropping zeros and NaNs.")
        return
    
    def compute_corr_stats(corr_matrix):
        n = corr_matrix.shape[0]
        tri_k = n * (n-1) // 2  # number of elements in upper triangle
        
        # Compute p-values
        p_values = np.zeros_like(corr_matrix)
        for i in range(n):
            for j in range(i+1, n):  # Only upper triangle
                r = corr_matrix.iloc[i, j]
                t = r * np.sqrt((n-2) / (1-r**2))
                p_values[i, j] = stats.t.sf(np.abs(t), n-2)*2
        
        # Count significant correlations in upper triangle
        sig_corrs = np.sum(p_values[np.triu_indices(n, k=1)] < alpha)
        percent_sig = (sig_corrs / tri_k) * 100
        
        # Calculate mean and median correlation from upper triangle
        corr_values = corr_matrix.values[np.triu_indices(n, k=1)]
        mean_corr = np.mean(corr_values)
        median_corr = np.median(corr_values)
        
        return sig_corrs, tri_k, percent_sig, mean_corr, median_corr
    
    corr_matrix1 = roi_df1.corr()
    corr_matrix2 = roi_df2.corr()
    
    sig_corrs1, total_corrs1, percent_sig1, mean_corr1, median_corr1 = compute_corr_stats(corr_matrix1)
    sig_corrs2, total_corrs2, percent_sig2, mean_corr2, median_corr2 = compute_corr_stats(corr_matrix2)
    
    # T-test for proportion of significant correlations
    def proportion_ttest(count1, nobs1, count2, nobs2):
        p1 = count1 / nobs1
        p2 = count2 / nobs2
        se = np.sqrt(p1 * (1 - p1) / nobs1 + p2 * (1 - p2) / nobs2)
        t = (p1 - p2) / se
        df = nobs1 + nobs2 - 2
        p_value = 2 * (1 - stats.t.cdf(np.abs(t), df))
        return t, p_value

    t_stat, p_value_ttest = proportion_ttest(sig_corrs1, total_corrs1, sig_corrs2, total_corrs2)
    
    # Plotting correlation matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))
    
    rwb_cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    sns.heatmap(corr_matrix1, cmap=rwb_cmap, vmin=-1, vmax=1, center=0, ax=ax1)
    ax1.set_title("Correlation Matrix Pre-stress", fontsize=16)
    
    sns.heatmap(corr_matrix2, cmap=rwb_cmap, vmin=-1, vmax=1, center=0, ax=ax2)
    ax2.set_title("Correlation Matrix Post-stress", fontsize=16)
    
    plt.tight_layout()
    plt.savefig('correlation_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Set the style
    sns.set_style("ticks")
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'
    
    # Create the bar plot
    fig, ax = plt.subplots(figsize=(4, 7))
    
    bars = ax.bar(['Pre-stress', 'Post-stress'], [percent_sig1, percent_sig2], 
                  color=['#6baed6', '#fd8d3c'], capsize=5, 
                  edgecolor='black', linewidth=1, width=0.6)
    
    # Add individual data points
    corr_values1 = [abs(corr) for corr in corr_matrix1.values.flatten() if corr != 1]
    corr_values2 = [abs(corr) for corr in corr_matrix2.values.flatten() if corr != 1]
    
    ax.scatter([bars[0].get_x() + bars[0].get_width() / 2] * len(corr_values1),
               corr_values1,
               color='black', zorder=3, alpha=0.7, s=30)
    ax.scatter([bars[1].get_x() + bars[1].get_width() / 2] * len(corr_values2),
               corr_values2,
               color='black', zorder=3, alpha=0.7, s=30)
    
    # Add significance bar if the difference is significant
    y_max = max(percent_sig1, percent_sig2)
    y_line = y_max * 1.1
    y_text = y_max * 1.15
    ax.plot([0, 0, 1, 1], [y_line, y_line + y_max * 0.02, y_line + y_max * 0.02, y_line], 
            lw=1.5, c='black')
    
    if p_value_ttest < alpha:
        significance_text = f'p = {p_value_ttest:.4f}'
    else:
        significance_text = 'ns'
    
    ax.text(0.5, y_text, significance_text, ha='center', va='bottom')
    
    # Customize the plot
    ax.set_title('Ratio of ROIs with Significant Correlations', fontweight='bold')
    ax.set_ylabel('Percentage')
    ax.set_xlabel('')
    
    # Remove top and right spines
    sns.despine()
    
    # Adjust y-axis to start from 0
    ax.set_ylim(0, y_max * 1.2)
    
    plt.tight_layout()
    plt.savefig('significant_correlations_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print(f"Dataset 1:")
    print(f"  Percentage of significant correlations: {percent_sig1:.2f}%")
    print(f"  Mean correlation: {mean_corr1:.4f}")
    print(f"  Median correlation: {median_corr1:.4f}")
    print(f"\nDataset 2:")
    print(f"  Percentage of significant correlations: {percent_sig2:.2f}%")
    print(f"  Mean correlation: {mean_corr2:.4f}")
    print(f"  Median correlation: {median_corr2:.4f}")
    
    # Compute differences
    diff_percent_sig = percent_sig2 - percent_sig1
    diff_mean_corr = mean_corr2 - mean_corr1
    diff_median_corr = median_corr2 - median_corr1
    
    # Fisher's r-to-z transformation for testing difference between correlations
    z1 = np.arctanh(mean_corr1)
    z2 = np.arctanh(mean_corr2)
    n1 = corr_matrix1.shape[0]
    n2 = corr_matrix2.shape[0]
    se_diff = np.sqrt(1/(n1-3) + 1/(n2-3))
    z = (z2 - z1) / se_diff
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    print(f"\nDifferences (Dataset 2 - Dataset 1):")
    print(f"  Difference in percentage of significant correlations: {diff_percent_sig:.2f}%")
    print(f"  Difference in mean correlation: {diff_mean_corr:.4f}")
    print(f"  Difference in median correlation: {diff_median_corr:.4f}")
    print(f"  P-value for difference in mean correlation: {p_value:.4f}")
    print(f"  P-value for difference in percentage of significant correlations (t-test): {p_value_ttest:.10f}")
