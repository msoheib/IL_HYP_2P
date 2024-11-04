#####################################################
#Open mean by angle and calculate gosi for each roi
#####################################################
import pandas as pd
import helper_functions
import pandas as pdb
import helper_functions
from scipy.optimize import curve_fit

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, wilcoxon


##########################################################################
############ 		ROUTINE		    ######################################
##########################################################################

# openpathtracesexp= 'meanByangle.csv'
# column_names = helper_functions.generate_column_names(0, 92)
def calculate_gosi(input_data, column_names = helper_functions.generate_column_names(0, 50)):
    

    if isinstance(input_data, str):
        # If input_data is a string, assume it's a file path and read the CSV
        input_data = pd.read_csv(input_data, delimiter=",", header=0, decimal='.', engine='python')
    
    column_names = [col for col in input_data.columns if col.isdigit()]
    for col in input_data.columns:
        if col.isdigit():
            input_data[col] = input_data[col].astype(int)
    
    gosi_results = pd.DataFrame()
    gosi_tuples = []
    
    for r in column_names:
        # print(type(r))
        # print(input_data['degrees'].values)
        gosival = helper_functions.calculate_gOSI(input_data[r], input_data['degrees'].values)
        gosival = (r, gosival[0], gosival[1])
        gosi_tuples.append(gosival)
    
    gosi_results = pd.DataFrame(gosi_tuples, columns=['roi', 'gOSI_index', 'prefer_ori_degrees'])
    
    return gosi_results



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway

def bootstrap_ci(data, num_samples=10000, ci=95):
    medians = []
    n = len(data)
    for _ in range(num_samples):
        sample = np.random.choice(data, n, replace=True)
        medians.append(np.median(sample))
    lower = np.percentile(medians, (100 - ci) / 2)
    upper = np.percentile(medians, 100 - (100 - ci) / 2)
    return lower, upper

def confidence_interval(data):
    mean = np.mean(data)
    sem = np.std(data) / np.sqrt(len(data))
    ci = 1.96 * sem
    return mean, ci


# pre_session_path = 'pre_session_gosi.csv'
# post_session_path = 'post_session_gosi.csv'




def bootstrap_ci(data, num_samples=10000, ci=95):
    medians = []
    n = len(data)
    for _ in range(num_samples):
        sample = np.random.choice(data, n, replace=True)
        medians.append(np.median(sample))
    lower = np.percentile(medians, (100 - ci) / 2)
    upper = np.percentile(medians, 100 - (100 - ci) / 2)
    return lower, upper

# def plot_gosi(pre_session_path, post_session_path):
#     pre_session_data = pd.read_csv(pre_session_path)
#     post_session_data = pd.read_csv(post_session_path)
    
#     # Remove rows with inf values
#     pre_session_data = pre_session_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['gOSI_index'])
#     post_session_data = post_session_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['gOSI_index'])
    
#     pre_iqr_data = pre_session_data[
#         (pre_session_data['gOSI_index'] >= pre_session_data['gOSI_index'].quantile(0.25)) & 
#         (pre_session_data['gOSI_index'] <= pre_session_data['gOSI_index'].quantile(0.75))
#     ]
#     post_iqr_data = post_session_data[
#         (post_session_data['gOSI_index'] >= post_session_data['gOSI_index'].quantile(0.25)) & 
#         (post_session_data['gOSI_index'] <= post_session_data['gOSI_index'].quantile(0.75))
#     ]
    
#     pre_median = np.median(pre_session_data['gOSI_index'])
#     post_median = np.median(post_session_data['gOSI_index'])
    
#     pre_median_ci = bootstrap_ci(pre_session_data['gOSI_index'])
#     post_median_ci = bootstrap_ci(post_session_data['gOSI_index'])
    
#     f_stat, p_value_anova = f_oneway(pre_iqr_data['gOSI_index'], post_iqr_data['gOSI_index'])
#     print(f"ANOVA p-value: {p_value_anova}")
    
#     labels = ['Pre Session', 'Post Session']
#     median_values = [pre_median, post_median]
#     x = range(len(labels))
    
#     fig, ax = plt.subplots(figsize=(4, 7))
    
#     ax.scatter([0] * len(pre_iqr_data['gOSI_index']), pre_iqr_data['gOSI_index'], color='blue', alpha=0.5, marker='x')
#     ax.scatter([1] * len(post_iqr_data['gOSI_index']), post_iqr_data['gOSI_index'], color='red', alpha=0.5, marker='x')
    
#     ax.bar(x, median_values, width=0.4, align='center', color='teal', edgecolor='black', linewidth=2, alpha=0.5, label='Median')
    
#     ax.errorbar(x, median_values, yerr=[
#         [median_values[0] - pre_median_ci[0], median_values[1] - post_median_ci[0]], 
#         [pre_median_ci[1] - median_values[0], post_median_ci[1] - median_values[1]]
#     ], fmt='none', ecolor='black', capsize=5)
    
#     max_y = max(pre_iqr_data['gOSI_index'].max(), post_iqr_data['gOSI_index'].max())
#     significance_y = max_y + 0.22
#     ax.plot([0, 1], [significance_y] * 2, color='black', linewidth=1.5)
#     ax.plot([0, 0], [max_y + 0.05, significance_y], color='black', linewidth=1.5)
#     ax.plot([1, 1], [max_y + 0.05, significance_y], color='black', linewidth=1.5)
#     ax.text(0.5, significance_y, '*', ha='center', va='bottom', color='black', fontsize=16)
    
#     ax.set_ylabel('GOSI Index')
#     ax.set_title('Median GOSI Index Before and After Stress')
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels)
#     #ax.legend()
    
#     sns.despine(trim=True)
#     ax.grid(True, linestyle='--', alpha=0.7)
#     ax.set_axisbelow(True)
    
#     plt.tight_layout()
#     plt.show()

# Example usage:
def iqr_plot_gosi(pre_session_path, post_session_path):
    pre_session_data = pd.read_csv(pre_session_path)
    post_session_data = pd.read_csv(post_session_path)
    
    # Remove rows with inf values
    pre_session_data = pre_session_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['gOSI_index'])
    post_session_data = post_session_data.replace([np.inf, -np.inf], np.nan).dropna(subset=['gOSI_index'])
    
    # Ensure data alignment by merging datasets on a common identifier
    combined_data = pd.merge(pre_session_data, post_session_data, on='roi', suffixes=('_pre', '_post'))
    
    # Extract aligned gOSI_index values
    pre_values = combined_data['gOSI_index_pre']
    post_values = combined_data['gOSI_index_post']
    
    # Filter to IQR range
    pre_iqr_data = combined_data[
        (pre_values >= pre_values.quantile(0.25)) & 
        (pre_values <= pre_values.quantile(0.75))
    ]
    post_iqr_data = combined_data[
        (post_values >= post_values.quantile(0.25)) & 
        (post_values <= post_values.quantile(0.75))
    ]
    
    pre_median = np.median(pre_iqr_data['gOSI_index_pre'])
    post_median = np.median(post_iqr_data['gOSI_index_post'])
    
    pre_median_ci = bootstrap_ci(pre_iqr_data['gOSI_index_pre'])
    post_median_ci = bootstrap_ci(post_iqr_data['gOSI_index_post'])
    
    # Perform Wilcoxon signed-rank test
    stat, p_value_wilcoxon = wilcoxon(pre_values, post_values)
    print(f"Wilcoxon p-value: {p_value_wilcoxon}")
    
    labels = ['Pre Session', 'Post Session']
    median_values = [pre_median, post_median]
    x = range(len(labels))
    
    fig, ax = plt.subplots(figsize=(4, 7))
    
    ax.scatter([0] * len(pre_iqr_data['gOSI_index_pre']), pre_iqr_data['gOSI_index_pre'], color='blue', alpha=0.5, marker='x')
    ax.scatter([1] * len(post_iqr_data['gOSI_index_post']), post_iqr_data['gOSI_index_post'], color='red', alpha=0.5, marker='x')
    
    ax.bar(x, median_values, width=0.4, align='center', color='teal', edgecolor='black', linewidth=2, alpha=0.5, label='Median')
    
    ax.errorbar(x, median_values, yerr=[
        [median_values[0] - pre_median_ci[0], median_values[1] - post_median_ci[0]], 
        [pre_median_ci[1] - median_values[0], post_median_ci[1] - median_values[1]]
    ], fmt='none', ecolor='black', capsize=5)
    
    max_y = max(pre_iqr_data['gOSI_index_pre'].max(), post_iqr_data['gOSI_index_post'].max())
    significance_y = max_y + 0.22
    ax.plot([0, 1], [significance_y] * 2, color='black', linewidth=1.5)
    ax.plot([0, 0], [max_y + 0.05, significance_y], color='black', linewidth=1.5)
    ax.plot([1, 1], [max_y + 0.05, significance_y], color='black', linewidth=1.5)
    if p_value_wilcoxon < 0.05:
        ax.text(0.5, significance_y, f'{p_value_wilcoxon}*', ha='center', va='bottom', color='black', fontsize=16)
    else:
        ax.text(0.5, significance_y, f'{p_value_wilcoxon} n.s.', ha='center', va='bottom', color='black', fontsize=16)
    
    ax.set_ylabel('GOSI Index')
    ax.set_title('Median GOSI Index Before and After Stress')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    #ax.legend()
    
    sns.despine(trim=True)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.show()



def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def process_and_plot(mean_angle_path, response_path):
    mean_angle_df = pd.read_csv(mean_angle_path)
    response_df = pd.read_csv(response_path)
    
    #  stimulus angles and mapping
    stimulus_angles = np.array([0, 45, 90, 135, 180, -45, -90, -135])
    angle_to_index = {angle: i for i, angle in enumerate(stimulus_angles)}
    
    # get the closest stimulus angle
    def closest_stimulus_angle(preferred_angle):
        differences = np.abs(stimulus_angles - preferred_angle)
        closest_angle = stimulus_angles[np.argmin(differences)]
        return closest_angle

    response_df['closest_stimulus'] = response_df['prefer_ori_degrees'].apply(closest_stimulus_angle)
    
    # adjust responses
    adjusted_responses = pd.DataFrame(columns=stimulus_angles, index=response_df['roi'])

    for index, row in response_df.iterrows():
        closest_index = angle_to_index[row['closest_stimulus']]
        responses = mean_angle_df.iloc[:, 1:-1].values[:, index]
        adjusted_responses.loc[index] = np.roll(responses, -closest_index)
    
    # gaussian curves
    fit_results = pd.DataFrame(columns=['ROI', 'Amplitude', 'Mean', 'StdDev', 'Width at Half Amplitude'])
    
    for index, row in adjusted_responses.iterrows():
        try:
            popt, _ = curve_fit(gaussian, stimulus_angles, row, p0=[1, 0, 20])
            amplitude, mean, stddev = popt
            width_half_amp = 2.355 * stddev
            fit_results.loc[index] = [index, amplitude, mean, stddev, width_half_amp]
        except:
            fit_results.loc[index] = [index, np.nan, np.nan, np.nan, np.nan]
    
    # plot curves
    plt.figure(figsize=(12, 8))
    x_values = np.linspace(-135, 180, 400)
    
    for _, row in fit_results.dropna().iterrows():
        y_values = gaussian(x_values, row['Amplitude'], row['Mean'], row['StdDev'])
        plt.plot(x_values, y_values, label=f'ROI {int(row["ROI"])}')
    
    plt.title('Gaussian Tuning Curves for ROIs')
    plt.xlabel('Degrees')
    plt.ylabel('Response')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return adjusted_responses, fit_results


def bootstrap_ci(data, n_bootstraps=1000, ci=95):
    bootstraps = np.random.choice(data, size=(n_bootstraps, len(data)), replace=True)
    bootstrap_means = np.median(bootstraps, axis=1)
    lower_bound = np.percentile(bootstrap_means, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrap_means, 100 - ((100 - ci) / 2))
    return (lower_bound, upper_bound)

# def plot_gosi(pre_session_path, post_session_path):
#     pre_session_data = pd.read_csv(pre_session_path)
#     post_session_data = pd.read_csv(post_session_path)

#     # Replace inf values and drop rows with NaN in 'gOSI_index'
#     pre_session_data.replace([np.inf, -np.inf], np.nan, inplace=True)
#     pre_session_data.dropna(subset=['gOSI_index'], inplace=True)
#     post_session_data.replace([np.inf, -np.inf], np.nan, inplace=True)
#     post_session_data.dropna(subset=['gOSI_index'], inplace=True)

#     # Merge datasets on 'roi' identifier
#     combined_data = pd.merge(pre_session_data, post_session_data, on='roi', suffixes=('_pre', '_post'))

#     # Setup plot
#     fig, ax = plt.subplots(figsize=(6, 8))

#     # Scatter plots for pre and post data points
#     ax.scatter([0] * len(combined_data['gOSI_index_pre']), combined_data['gOSI_index_pre'], color='blue', alpha=0.5, marker='x')
#     ax.scatter([1] * len(combined_data['gOSI_index_post']), combined_data['gOSI_index_post'], color='red', alpha=0.5, marker='x')

#     # Draw lines connecting individual points
#     for i in range(len(combined_data)):
#         ax.plot([0, 1], [combined_data.iloc[i]['gOSI_index_pre'], combined_data.iloc[i]['gOSI_index_post']], color='gray', linestyle='-', alpha=0.5)

#     # Bar chart of median values
#     pre_median = np.median(combined_data['gOSI_index_pre'])
#     post_median = np.median(combined_data['gOSI_index_post'])
#     ax.bar([0, 1], [pre_median, post_median], color='teal', width=0.4, edgecolor='black', linewidth=2, alpha=0.7)

#     # Error bars for confidence intervals
#     pre_median_ci = bootstrap_ci(combined_data['gOSI_index_pre'])
#     post_median_ci = bootstrap_ci(combined_data['gOSI_index_post'])
#     ax.errorbar([0, 1], [pre_median, post_median], yerr=[
#         [pre_median - pre_median_ci[0], post_median - post_median_ci[0]], 
#         [pre_median_ci[1] - pre_median, post_median_ci[1] - post_median]
#     ], fmt='none', ecolor='black', capsize=5)

#     ax.set_xticks([0, 1])
#     ax.set_xticklabels(['Pre Session', 'Post Session'])
#     ax.set_ylabel('GOSI Index')
#     ax.set_title('Median GOSI Index Before and After Session')

#     sns.despine()
#     plt.tight_layout()
#     plt.show()


def plot_gosi(pre_session_path, post_session_path):
    pre_session_data = pd.read_csv(pre_session_path)
    post_session_data = pd.read_csv(post_session_path)

    # Replace inf values and drop rows with NaN in 'gOSI_index'
    pre_session_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    pre_session_data.dropna(subset=['gOSI_index'], inplace=True)
    post_session_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    post_session_data.dropna(subset=['gOSI_index'], inplace=True)

    # Merge datasets on 'roi' identifier
    combined_data = pd.merge(pre_session_data, post_session_data, on='roi', suffixes=('_pre', '_post'))

    # Set the style
    sns.set_style("ticks")
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'

    # Setup plot
    fig, ax = plt.subplots(figsize=(4, 7))

    # Calculate medians and confidence intervals
    pre_median = np.median(combined_data['gOSI_index_pre'])
    post_median = np.median(combined_data['gOSI_index_post'])
    pre_ci = bootstrap_ci(combined_data['gOSI_index_pre'])
    post_ci = bootstrap_ci(combined_data['gOSI_index_post'])

    # Create the bar plot
    bars = ax.bar(['Pre Session', 'Post Session'], [pre_median, post_median], 
                  yerr=[[pre_median - pre_ci[0], post_median - post_ci[0]], 
                        [pre_ci[1] - pre_median, post_ci[1] - post_median]], 
                  color=['#6baed6', '#fd8d3c'], capsize=5, 
                  edgecolor='black', linewidth=1, width=0.6)

    # Add individual data points
    ax.scatter([bars[0].get_x() + bars[0].get_width() / 2] * len(combined_data['gOSI_index_pre']),
               combined_data['gOSI_index_pre'], color='black', zorder=3, alpha=0.7, s=30)
    ax.scatter([bars[1].get_x() + bars[1].get_width() / 2] * len(combined_data['gOSI_index_post']),
               combined_data['gOSI_index_post'], color='black', zorder=3, alpha=0.7, s=30)

    # Perform statistical test
    stat, p_value = wilcoxon(combined_data['gOSI_index_pre'], combined_data['gOSI_index_post'])

    # Add significance line and p-value/ns
    y_max = max(max(combined_data['gOSI_index_pre']), max(combined_data['gOSI_index_post']))
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
    ax.set_title('Median GOSI Index', fontweight='bold')
    ax.set_ylabel('GOSI Index')
    ax.set_xlabel('')

    # Remove top and right spines
    sns.despine()

    # Adjust y-axis to start from 0
    ax.set_ylim(0, y_max * 1.2)

    plt.tight_layout()

    # Save the plot in the current folder
    plt.savefig('gosi_index_comparison.png', dpi=300, bbox_inches='tight')

    plt.show()


## gOSI cruve
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * stddev ** 2))

def process_and_plot_gosi_curve(mean_angle_path, response_path):
    mean_angle_df = pd.read_csv(mean_angle_path)
    response_df = pd.read_csv(response_path)
    
    #  stimulus angles and mapping
    stimulus_angles = np.array([0, 45, 90, 135, 180, -45, -90, -135])
    angle_to_index = {angle: i for i, angle in enumerate(stimulus_angles)}
    
    # get the closest stimulus angle
    def closest_stimulus_angle(preferred_angle):
        differences = np.abs(stimulus_angles - preferred_angle)
        closest_angle = stimulus_angles[np.argmin(differences)]
        return closest_angle

    response_df['closest_stimulus'] = response_df['prefer_ori_degrees'].apply(closest_stimulus_angle)
    
    # adjust responses
    adjusted_responses = pd.DataFrame(columns=stimulus_angles, index=response_df['roi'])

    for index, row in response_df.iterrows():
        closest_index = angle_to_index[row['closest_stimulus']]
        responses = mean_angle_df.iloc[:, 1:-1].values[:, index]
        adjusted_responses.loc[index] = np.roll(responses, -closest_index)
    
    # gaussian curves
    fit_results = pd.DataFrame(columns=['ROI', 'Amplitude', 'Mean', 'StdDev', 'Width at Half Amplitude'])
    
    for index, row in adjusted_responses.iterrows():
        try:
            popt, _ = curve_fit(gaussian, stimulus_angles, row, p0=[1, 0, 20])
            amplitude, mean, stddev = popt
            width_half_amp = 2.355 * stddev
            fit_results.loc[index] = [index, amplitude, mean, stddev, width_half_amp]
        except:
            fit_results.loc[index] = [index, np.nan, np.nan, np.nan, np.nan]
    
    # plot curves
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(6, 8))
    x_values = np.linspace(-135, 180, 400)
    
    for _, row in fit_results.dropna().iterrows():
        y_values = gaussian(x_values, row['Amplitude'], row['Mean'], row['StdDev'])
        plt.plot(x_values, y_values, label=f'ROI {int(row["ROI"])}', linewidth=2.5)
    
    plt.title('Gaussian Tuning Curves for ROIs')
    plt.xlabel('Degrees')
    plt.ylabel('Response')
    plt.legend()
    
    sns.despine()
    plt.grid(True, linestyle='--', linewidth=0.5, color='gray')
    plt.tight_layout()
    
    plt.show()    
    return adjusted_responses, fit_results
