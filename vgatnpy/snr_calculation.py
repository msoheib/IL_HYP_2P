import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon, ttest_rel
from scipy import stats

def calculate_snr(pre_df, post_df):
    rois = [col for col in post_df.columns if col.isdigit()]  # ROI columns
    orientations = [col for col in post_df.columns if "degrees" in col]  # Stimulus orientation columns
    time_column = 'time' if 'time' in post_df.columns else None
    if not time_column:
        return "No time column found"

    # Determine the frame rate using the time column
    fps = round(1 / post_df[time_column].diff().mean())
    pre_stimulus_frames = 2 * fps  # 2 seconds before stimulus
    post_stimulus_frames = 2 * fps  # 2 seconds after stimulus

    snr_data = {}

    for orientation in orientations:
        snr_data[orientation] = {}
        stimulus_indices = post_df[post_df[orientation] == 1].index

        for roi in rois:
            # Gather pre-stimulus and post-stimulus data for each trial, based on orientation
            pre_stimulus_values = [pre_df.loc[max(idx - pre_stimulus_frames, 0): idx - 1, roi] for idx in stimulus_indices]
            post_stimulus_values = [post_df.loc[idx: idx + post_stimulus_frames - 1, roi] for idx in stimulus_indices]

            # Calculate the mean of the squares of the post-stimulus response
            mean_squared = np.mean([np.mean(trial) ** 2 for trial in post_stimulus_values])

            # Calculate the standard deviation of the pre-stimulus response
            std_deviation = np.std([np.mean(trial) for trial in pre_stimulus_values])

            # Calculate SNR
            snr = mean_squared / std_deviation if std_deviation != 0 else np.inf
            snr_data[orientation][roi] = snr

    return snr_data

def compute_and_save_snr(pre_stimulus_file, post_stimulus_file, output_file):
    # Load the data from the provided file paths
    pre_df = pd.read_csv(pre_stimulus_file)
    post_df = pd.read_csv(post_stimulus_file)
    
    # Calculate SNR using the previously defined function
    snr_data = calculate_snr(pre_df, post_df)
    
    # Convert the SNR data dictionary to a DataFrame for easier handling
    snr_df = pd.DataFrame(snr_data)
    
    # Save the DataFrame to a CSV file
    snr_df.to_csv(output_file, index_label='ROI')
    
    return output_file


def compare_snr_sessions(df1, df2):
    """
    Compares pairwise significant differences in SNR between two sessions using the Wilcoxon signed-rank test,
    handling cases where differences may be zero for all elements.
    
    Args:
    df1 (pd.DataFrame): DataFrame containing SNR values for the first session.
    df2 (pd.DataFrame): DataFrame containing SNR values for the second session.
    
    Returns:
    pd.DataFrame: A DataFrame containing the test statistics and p-values for each ROI,
                  and a note if the differences are zero for all elements.
    """
    results = []
    for col in df1.columns:
        if col in df2.columns:
            # Extract the data for the ROI from both sessions
            data1 = df1[col].dropna()
            data2 = df2[col].dropna()
            
            # Ensuring both data arrays have the same length
            min_length = min(len(data1), len(data2))
            data1 = data1.iloc[:min_length]
            data2 = data2.iloc[:min_length]
            
            if len(data1) > 0 and len(data2) > 0:
                if np.all(data1 - data2 == 0):
                    # All differences are zero
                    results.append({'ROI': col, 'Test Statistic': 'N/A', 'P-value': 'N/A', 'Note': 'No variation in differences'})
                else:
                    # Perform the Wilcoxon signed-rank test
                    try:
                        stat, p_value = wilcoxon(data1, data2)
                        results.append({'ROI': col, 'Test Statistic': stat, 'P-value': p_value, 'Note': ''})
                    except Exception as e:
                        results.append({'ROI': col, 'Test Statistic': 'Error', 'P-value': 'Error', 'Note': str(e)})
            else:
                results.append({'ROI': col, 'Test Statistic': None, 'P-value': None, 'Note': 'Insufficient data'})
        else:
            results.append({'ROI': col, 'Test Statistic': None, 'P-value': 'Column not in both datasets', 'Note': 'Missing column'})
    
    return pd.DataFrame(results)

def filter_values(df):
    """
    Filters out values greater than 250 from each column in the DataFrame.
    """
    return df.apply(lambda x: x[x <= 250])

def plot_snr_comparison(pre_df, post_df):
    # Define ROI columns
    rois = [col for col in pre_df.columns if col.isdigit()]

    # Calculate means and standard deviations
    pre_means = pre_df[rois].mean()
    post_means = post_df[rois].mean()
    pre_std = pre_df[rois].std()
    post_std = post_df[rois].std()

    # Compute individual SNR
    individual_pre_snr = pre_means / pre_std
    individual_post_snr = post_means / post_std

    # Compute overall mean SNR and confidence intervals for plotting
    mean_pre_snr, ci_pre_snr = mean_confidence_interval(individual_pre_snr)
    mean_post_snr, ci_post_snr = mean_confidence_interval(individual_post_snr)

    # Set the style
    sns.set_style("ticks")
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'

    # Create the plot
    fig, ax = plt.subplots(figsize=(4, 7))

    # Create the bar plot
    bars = ax.bar(['Pre-Session', 'Post-Session'], [mean_pre_snr, mean_post_snr], 
                  yerr=[ci_pre_snr, ci_post_snr], color=['#6baed6', '#fd8d3c'], 
                  capsize=5, edgecolor='black', linewidth=1, width=0.6)

    # Add individual data points
    ax.scatter([bars[0].get_x() + bars[0].get_width() / 2] * len(individual_pre_snr),
               individual_pre_snr, color='black', zorder=3, alpha=0.7, s=30)
    ax.scatter([bars[1].get_x() + bars[1].get_width() / 2] * len(individual_post_snr),
               individual_post_snr, color='black', zorder=3, alpha=0.7, s=30)

    # Perform statistical test
    stat, p_value = stats.wilcoxon(individual_pre_snr, individual_post_snr)

    # Add significance line and p-value/ns
    y_max = max(max(individual_pre_snr), max(individual_post_snr))
    y_line = y_max * 1.1
    y_text = y_max * 1.15
    ax.plot([0, 0, 1, 1], [y_line, y_line + y_max * 0.02, y_line + y_max * 0.02, y_line],
            lw=1.5, c='black')

    if p_value < 0.05:
        significance_text = f'p = {p_value:.4f}'
    else:
        significance_text = 'ns'

    ax.text(0.5, y_text, significance_text, ha='center', va='bottom')

    # Customize the plot
    ax.set_title('Mean SNR', fontweight='bold')
    ax.set_ylabel('SNR')
    ax.set_xlabel('')

    # Remove top and right spines
    sns.despine()

    # Adjust y-axis to start from 0
    ax.set_ylim(0, y_max * 1.2)

    plt.tight_layout()

    # Save the plot in the current folder
    plt.savefig('snr_comparison.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h


def xplot_snr_comparison(pre_df, post_df):
    # Define ROI columns
    rois = [col for col in pre_df.columns if col.isdigit()]

    # Calculate means and standard deviations
    pre_means = pre_df[rois].mean()
    post_means = post_df[rois].mean()
    pre_std = pre_df[rois].std()
    post_std = post_df[rois].std()

    # Compute individual SNR
    individual_pre_snr = pre_means / pre_std
    individual_post_snr = post_means / post_std

    # Compute overall mean SNR for plotting
    overall_pre_snr = individual_pre_snr.mean()
    overall_post_snr = individual_post_snr.mean()
    overall_snr_comparison_df = pd.DataFrame({
        'Overall Mean SNR': [overall_pre_snr, overall_post_snr]
    }, index=['Pre-Session', 'Post-Session'])

    # Plotting
    plt.style.use('seaborn-whitegrid')
    fig, ax = plt.subplots(figsize=(4, 7))
    overall_snr_comparison_df.plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e'], legend=False, zorder=2, width=0.6, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.scatter(x=np.zeros(len(individual_pre_snr)), y=individual_pre_snr, color='#1f77b4', edgecolor='black', zorder=3, label='Pre-Session SNR')
    ax.scatter(x=np.ones(len(individual_post_snr)), y=individual_post_snr, color='#ff7f0e', edgecolor='black', zorder=3, label='Post-Session SNR')

    # Connect the same ROI SNRs with lines
    for i in range(len(individual_pre_snr)):
        ax.plot([0, 1], [individual_pre_snr[i], individual_post_snr[i]], color='gray', linestyle='-', marker='', zorder=1)

    # Setting titles and labels
    ax.set_title('Overall Mean SNR Comparison with Individual Points and Connectors', fontsize=14, fontweight='bold')
    ax.set_ylabel('SNR', fontsize=12)
    ax.set_xlabel('Session', fontsize=12)
    plt.xticks([0, 1], ['Pre-Session', 'Post-Session'], fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend()



    # Customize the grid and axes to be less pronounced
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()

def handle_input(input_data):
    if isinstance(input_data, pd.DataFrame):
        return input_data
    elif isinstance(input_data, str):
        return pd.read_csv(input_data)
    else:
        raise ValueError("Input must be a DataFrame or a filepath to a CSV.")

def process_and_visualize_snr(pre_input, post_input):
    # Handle inputs whether they are file paths or DataFrames
    pre_df = handle_input(pre_input)
    post_df = handle_input(post_input)
    
    # Calculate SNR
    snr_data = calculate_snr(pre_df, post_df)
    
    # Convert the SNR data dictionary to a DataFrame
    snr_df = pd.DataFrame(snr_data)
    
    # Assume output file path
    output_file = 'snr_output.csv'
    snr_df.to_csv(output_file, index_label='ROI')
    
    # Plot the SNR comparison
    plot_snr_comparison(pre_df, post_df)

    print(f"SNR results saved to {output_file}")



def plot_snr_comparison(pre_df, post_df):
    # Define ROI columns
    rois = [col for col in pre_df.columns if col.isdigit()]

    # Calculate means and standard deviations
    pre_means = pre_df[rois].mean()
    post_means = post_df[rois].mean()
    pre_std = pre_df[rois].std()
    post_std = post_df[rois].std()

    # Compute individual SNR
    individual_pre_snr = pre_means / pre_std
    individual_post_snr = post_means / post_std

    # Compute overall mean SNR for plotting
    overall_pre_snr = individual_pre_snr.mean()
    overall_post_snr = individual_post_snr.mean()

    # Set the style
    sns.set_style("ticks")
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'

    # Setup plot
    fig, ax = plt.subplots(figsize=(4, 7))

    # Calculate confidence intervals
    def bootstrap_ci(data, n_bootstrap=1000, ci=95):
        bootstrap_means = np.array([np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)])
        lower_bound = np.percentile(bootstrap_means, (100 - ci) / 2)
        upper_bound = np.percentile(bootstrap_means, 100 - ((100 - ci) / 2))
        return (lower_bound, upper_bound)

    pre_ci = bootstrap_ci(individual_pre_snr)
    post_ci = bootstrap_ci(individual_post_snr)

    # Create the bar plot
    bars = ax.bar(['Pre-Session', 'Post-Session'], [overall_pre_snr, overall_post_snr], 
                  yerr=[[overall_pre_snr - pre_ci[0], overall_post_snr - post_ci[0]], 
                        [pre_ci[1] - overall_pre_snr, post_ci[1] - overall_post_snr]], 
                  color=['#6baed6', '#fd8d3c'], capsize=5, 
                  edgecolor='black', linewidth=1, width=0.6)

    # Add individual data points
    ax.scatter([bars[0].get_x() + bars[0].get_width() / 2] * len(individual_pre_snr),
               individual_pre_snr, color='black', zorder=3, alpha=0.7, s=30)
    ax.scatter([bars[1].get_x() + bars[1].get_width() / 2] * len(individual_post_snr),
               individual_post_snr, color='black', zorder=3, alpha=0.7, s=30)

    # Perform statistical test
    try:
        stat, p_value = wilcoxon(individual_pre_snr, individual_post_snr)
    except ValueError:
        stat, p_value = ttest_rel(individual_pre_snr, individual_post_snr)

    # Add significance line and p-value/ns
    y_max = max(max(individual_pre_snr), max(individual_post_snr))
    y_line = y_max * 1.1
    y_text = y_max * 1.15
    ax.plot([0, 0, 1, 1], [y_line, y_line + y_max * 0.02, y_line + y_max * 0.02, y_line], 
            lw=1.5, c='black')

    if p_value < 0.05:
        significance_text = f'p = {p_value:.4f}'
    else:
        significance_text = 'ns'

    ax.text(0.5, y_text, significance_text, ha='center', va='bottom')

    # Customize the plot
    ax.set_title('Mean SNR', fontweight='bold')
    ax.set_ylabel('SNR')
    ax.set_xlabel('')

    # Remove top and right spines
    sns.despine()

    # Adjust y-axis to start from 0
    ax.set_ylim(0, y_max * 1.2)

    plt.tight_layout()

    # Save the plot in the current folder
    plt.savefig('snr_comparison.png', dpi=300, bbox_inches='tight')

    plt.show()