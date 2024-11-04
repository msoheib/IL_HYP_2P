import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wilcoxon
from scipy import stats

#needs gosi first
#takes mean angle and gosi csv files and calculates OSI
def calculate_osi(pre_gosi_results_csv, mean_angle_pre_csv, output_csv):

    # Load the datasets
    pre_gosi_results = pd.read_csv(pre_gosi_results_csv)
    mean_angle_pre = pd.read_csv(mean_angle_pre_csv)

    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in mean_angle_pre.columns:
        mean_angle_pre.drop(columns=['Unnamed: 0'], inplace=True)

    # Normalize the index to be just integers
    mean_angle_pre.set_index('degrees', inplace=True)
    mean_angle_pre.index = mean_angle_pre.index.astype(str).str.extract('(\d+)')[0].astype(int)
    mean_angle_pre.columns = map(int, mean_angle_pre.columns)
    
    # Find the closest angle for each ROI and the corresponding response
    def find_closest_angle(angle):
        standard_angles = np.array([0, 45, 90, 135, 180, 225, 270, 315])
        idx = (np.abs(standard_angles - angle)).argmin()
        return standard_angles[idx]

    pre_gosi_results['closest_angle'] = pre_gosi_results['prefer_ori_degrees'].apply(find_closest_angle)

    # Recalculate the OSI using the corrected index format
    osi_data_final = []
    for index, row in pre_gosi_results.iterrows():
        roi = int(row['roi'])  # Ensure ROI is treated as integer for column indexing
        closest_angle = row['closest_angle']
        orthogonal_angle = (closest_angle + 90) % 360  # Using full circle to get the orthogonal angle

        # Get response for preferred and orthogonal angles
        vpref = mean_angle_pre.loc[closest_angle, roi]
        vorth = mean_angle_pre.loc[orthogonal_angle, roi] if orthogonal_angle in mean_angle_pre.index else 0  # Default to 0 if angle is not present

        # Calculate OSI
        osi = (vpref - vorth) / (vpref + vorth) if (vpref + vorth) != 0 else 0  # Avoid division by zero
        osi = max(0, osi)  # Ensure OSI is non-negative
        osi_data_final.append((roi, closest_angle, orthogonal_angle, vpref, vorth, osi))

    # Create a DataFrame for the corrected OSI values
    osi_results_final = pd.DataFrame(osi_data_final, columns=['ROI', 'Closest Angle', 'Orthogonal Angle', 'Preferred Response', 'Orthogonal Response', 'OSI'])
        
    # Save the results to a CSV file
    osi_results_final.to_csv(output_csv, index=False)

#plot the plot as a comparison

def plot_osi_trends(pre_csv, post_csv):
    # Load the pre and post OSI results from the CSV files
    osi_results_pre = pd.read_csv(pre_csv)
    osi_results_post = pd.read_csv(post_csv)

    # Set the style
    sns.set_style("ticks")
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['font.family'] = 'sans-serif'

    # Calculate mean and confidence intervals
    mean_osi_pre = osi_results_pre['OSI'].mean()
    mean_osi_post = osi_results_post['OSI'].mean()
    ci_pre = stats.sem(osi_results_pre['OSI']) * 1.96
    ci_post = stats.sem(osi_results_post['OSI']) * 1.96

    # Create the plot
    fig, ax = plt.subplots(figsize=(4, 7))

    # Create the bar plot
    bars = ax.bar(['Pre', 'Post'], [mean_osi_pre, mean_osi_post], 
                  yerr=[ci_pre, ci_post], color=['#6baed6', '#fd8d3c'], 
                  capsize=5, edgecolor='black', linewidth=1, width=0.6)

    # Add individual data points and lines connecting same ROIs
    for roi in osi_results_pre['ROI'].unique():
        pre_value = osi_results_pre[osi_results_pre['ROI'] == roi]['OSI'].values[0]
        post_value = osi_results_post[osi_results_post['ROI'] == roi]['OSI'].values[0]
        ax.plot(['Pre', 'Post'], [pre_value, post_value], color='gray', linestyle='-', marker='', zorder=1, alpha=0.5)

    ax.scatter(['Pre'] * len(osi_results_pre), osi_results_pre['OSI'], 
               color='black', zorder=3, alpha=0.7, s=30)
    ax.scatter(['Post'] * len(osi_results_post), osi_results_post['OSI'], 
               color='black', zorder=3, alpha=0.7, s=30)

    # Perform statistical test
    stat, p_value = wilcoxon(osi_results_pre['OSI'], osi_results_post['OSI'])

    # Add significance line and p-value/ns
    y_max = max(osi_results_pre['OSI'].max(), osi_results_post['OSI'].max())
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
    ax.set_title('Mean OSI', fontweight='bold')
    ax.set_ylabel('OSI')
    ax.set_xlabel('')

    # Remove top and right spines
    sns.despine()

    # Adjust y-axis to start from 0
    ax.set_ylim(0, y_max * 1.2)

    plt.tight_layout()
    plt.savefig('osi_index_plot.png', dpi=300, bbox_inches='tight')

    plt.show()


def calculate_and_plot(pre_gosi='pre_gosi_results.csv', mean_pre ='meanAngle_pre.csv',post_gosi ="post_gosi_results.csv", mean_post= "meanAngle_post.csv", osi_pre='osi_results_pre.csv',osi_post='osi_results_post.csv'):
    calculate_osi(pre_gosi, mean_pre, osi_pre)
    calculate_osi(post_gosi, mean_post, osi_post)
    plot_osi_trends(osi_pre, osi_post)