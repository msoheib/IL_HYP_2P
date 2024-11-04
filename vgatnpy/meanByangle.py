#####################################################
#Open deltaF and calculate mean for each angle on
#####################################################
import pandas as pd
import helper_functions
import matplotlib.pyplot as plt
import matplotlib.cm as cm
##########################################################################
############ 		ROUTINE		    ######################################
##########################################################################
# def calculate_mean_by_angle(openpathtracesexp='aligned_dff30.csv', targetcolumns=['degrees_0','degrees_45','degrees_90','degrees_135','degrees_180','degrees_225','degrees_270','degrees_315'], Select_columns_toplot=helper_functions.generate_column_names(0,92)):
#     # column_names = helper_functions.generate_column_names(0, 92)
#     column_names = Select_columns_toplot
#     degreesarray=[0,45,90,135,180,225,270,315]

#     #open and read dataframe with tracesRaw
#     if isinstance(openpathtracesexp, pd.DataFrame):
#         deltaF = openpathtracesexp.copy()
#     else:
#         deltaF1 = pd.read_csv(openpathtracesexp, delimiter=",", header=0, decimal='.', engine='python')
#         deltaF = deltaF1.copy()
    
#     deltaF.set_index(['time'], inplace=True) 
#     print("deltaf")
#     print(deltaF)
#     print("deltaf")

#     angles=helper_functions.mean_columns_based_on_binary(deltaF,targetcolumns,column_names)
#     angles=angles.T
#     angles['degrees']=degreesarray

#     print("angles")
#     print(angles)
#     print("angles")
    
#     pre_angles=helper_functions.mean_before_first_one(deltaF,targetcolumns,column_names, n_rows=60)
#     pre_angles=pre_angles.T
#     pre_angles['pre_degrees']=degreesarray
    
    
#     print("pre_angles")
#     print(pre_angles)
#     print(pre_angles.shape)
#     print("pre_angles")
#     # angles['degrees'] = angles['degrees'] - pre_angles['pre_degrees']

#     # angles = angles.drop('pre_degrees', axis=1)

#     angles.to_csv('meanByangle.csv', sep=',', index = True, header=True)

#     #PLOT 
#     helper_functions.plot_polar(angles,'degrees',Select_columns_toplot)
#     plt.show()

#     helper_functions.plot_polar_individual(angles,'degrees',Select_columns_toplot, commonmax=True)
#     plt.show()

#     figAngles=angles[Select_columns_toplot].plot(subplots=False, sharey=False, legend=True, colormap=cm.winter)
#     plt.show()

#calculate_mean_by_angle()



##############
############




#LAST WORKING FUNCTION
# def calculate_mean_by_angle(input, output_path,  Select_columns_toplot=helper_functions.generate_column_names(0,92)):
#     # Load the data
#     if isinstance(input, pd.DataFrame):
#         df = input.copy()
#     else:
#         df = pd.read_csv(input)
    
#     degreesarray=[0,45,90,135,180,225,270,315]

#     # Automatically determine mean columns by checking if the column names are digits
#     mean_columns = [col for col in df.columns if col.isdigit()]
#     Select_columns_toplot = [col for col in df.columns if col.isdigit()]

#     # Define the condition columns as specified previously
#     condition_columns = ['degrees_0','degrees_45','degrees_90','degrees_135','degrees_180','degrees_225','degrees_270','degrees_315']
#     # Function to calculate the differences
#     def mean_columns_comparison(df, condition_columns, mean_columns, rows=60):
#         result = {cond_col: {mean_col: [] for mean_col in mean_columns} for cond_col in condition_columns}
        
#         for cond_col in condition_columns:
#             in_series = False
#             series_start = None
#             for i in range(1, len(df)):
#                 if df[cond_col].iloc[i-1] == 0 and df[cond_col].iloc[i] == 1:
#                     series_start = i
#                     in_series = True
#                 elif df[cond_col].iloc[i-1] == 1 and df[cond_col].iloc[i] == 0 and in_series:
#                     end_series = i
#                     start_zero = max(0, series_start - rows)
#                     end_zero = max(0, series_start)
#                     for mean_col in mean_columns:
#                         mean_preceding = df.loc[start_zero:end_zero-1, mean_col].mean() if start_zero < end_zero else np.nan
#                         mean_series = df.loc[series_start:end_series-1, mean_col].mean()
#                         result[cond_col][mean_col].append(mean_series - mean_preceding)
#                     in_series = False

#         diff_df = pd.DataFrame({(cond_col, mean_col): result[cond_col][mean_col]
#                                 for cond_col in condition_columns for mean_col in mean_columns})
#         return diff_df

#     diff_df = mean_columns_comparison(df, condition_columns, mean_columns)

#     # Reformat the DataFrame to match the desired structure and save as CSV
#     angle_columns = [str(i) for i in range(len(mean_columns))]
#     angle = pd.DataFrame(columns=angle_columns + ['degrees'])

#     for idx, cond_col in enumerate(condition_columns):
#         mean_diff_values = diff_df[cond_col].mean().tolist() if cond_col in diff_df.columns.levels[0] else [np.nan] * len(mean_columns)
#         angle.loc[idx] = mean_diff_values + [cond_col]

#     # Insert the new label column at the beginning of the DataFrame
#     angle.insert(0, '', angle['degrees'])
#     angle['degrees']=degreesarray

#     # angle.drop('degrees', axis=1, inplace=True)
#     # angle.drop('degrees', axis=1, inplace=True)

#     angle.to_csv(output_path, index=False)
#     #PLOT 
#     helper_functions.plot_polar(angle,'degrees',Select_columns_toplot)
#     plt.show()

#     helper_functions.plot_polar_individual(angle,'degrees',Select_columns_toplot, commonmax=True)
#     plt.show()

#     figAngles=angle[Select_columns_toplot].plot(subplots=False, sharey=False, legend=True, colormap=cm.winter)
#     plt.show()
    
#     return angle



# calculate mean by angle
def calculate_mean_by_angle(input, output_path, Select_columns_toplot=None):

    # Load the data
    if isinstance(input, pd.DataFrame):
        df = input.copy()
    else:
        df = pd.read_csv(input)
    
    # Check initial DataFrame columns
    # print("Initial DataFrame columns:", df.columns.tolist())

    degreesarray = [0, 45, 90, 135, 180, 225, 270, 315]

    # Automatically determine mean columns by checking if the column names are digits
    mean_columns = [col for col in df.columns if col.isdigit()]
    # print("Mean columns:", mean_columns)

    if Select_columns_toplot is None:
        Select_columns_toplot = mean_columns

    # print("Select columns to plot:", Select_columns_toplot)

    # Define the condition columns as specified previously
    condition_columns = ['degrees_0', 'degrees_45', 'degrees_90', 'degrees_135', 'degrees_180', 'degrees_225', 'degrees_270', 'degrees_315']
    # print("Condition columns:", condition_columns)

    # Function to calculate the differences
    def mean_columns_comparison(df, condition_columns, mean_columns, rows=60):
        result = {cond_col: {mean_col: [] for mean_col in mean_columns} for cond_col in condition_columns}
        
        for cond_col in condition_columns:
            in_series = False
            series_start = None
            for i in range(1, len(df)):
                if df[cond_col].iloc[i-1] == 0 and df[cond_col].iloc[i] == 1:
                    series_start = i
                    in_series = True
                elif df[cond_col].iloc[i-1] == 1 and df[cond_col].iloc[i] == 0 and in_series:
                    end_series = i
                    start_zero = max(0, series_start - rows)
                    end_zero = max(0, series_start)
                    for mean_col in mean_columns:
                       # mean_preceding = df.loc[start_zero:end_zero-1, mean_col].mean() if start_zero < end_zero else np.nan
                        mean_series = df.loc[series_start:end_series-1, mean_col].mean()
                        result[cond_col][mean_col].append(mean_series)
                        #result[cond_col][mean_col].append(mean_series - mean_preceding)
                    in_series = False
        
        for cond_col in condition_columns:
            for mean_col in mean_columns:
                if len(result[cond_col][mean_col]) == 0:
                    result[cond_col][mean_col].append(np.nan)
        
        diff_df = pd.DataFrame({(cond_col, mean_col): result[cond_col][mean_col]
                                for cond_col in condition_columns for mean_col in mean_columns})
        # print("Difference DataFrame columns:", diff_df.columns)
        return diff_df

    diff_df = mean_columns_comparison(df, condition_columns, mean_columns)

    # Reformat the DataFrame to match the desired structure and save as CSV
    angle_columns = [str(i) for i in range(len(mean_columns))]
    angle = pd.DataFrame(columns=angle_columns + ['degrees'])
    # print("Angle columns:", angle_columns)

    for idx, cond_col in enumerate(condition_columns):
        mean_diff_values = diff_df[cond_col].mean().tolist() if cond_col in diff_df.columns.levels[0] else [np.nan] * len(mean_columns)
        angle.loc[idx] = mean_diff_values + [cond_col]

    # Insert the new label column at the beginning of the DataFrame
    angle.insert(0, '', angle['degrees'])
    angle['degrees'] = degreesarray

    # Check final DataFrame before saving
    # print("Final DataFrame columns:", angle.columns.tolist())
    # print("Final DataFrame head:\n", angle.head())
    
    angle.to_csv(output_path, index=False)

    # Ensure Select_columns_toplot are valid columns
    missing_columns = [col for col in Select_columns_toplot if col not in angle.columns]
    if missing_columns:
        print(f"Warning: The following columns are missing and will not be plotted: {missing_columns}")
        Select_columns_toplot = [col for col in Select_columns_toplot if col in angle.columns]

    # Plot the results
    import helper_functions
    helper_functions.plot_polar(angle, 'degrees', Select_columns_toplot)
    plt.show()

    helper_functions.plot_polar_individual(angle, 'degrees', Select_columns_toplot, commonmax=True)
    plt.show()

    figAngles = angle[Select_columns_toplot].plot(subplots=False, sharey=False, legend=True, colormap=cm.winter)
    plt.show()
    
    return angle


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
    plt.figure(figsize=(6, 5))
    
    # Create a heatmap
    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0, 
                square=True)
    
    plt.title('Correlation Matrix of ROI Columns', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print information about dropped columns
    dropped_columns = set(roi_columns) - set(roi_df.columns)
    if dropped_columns:
        print(f"Dropped columns: {', '.join(dropped_columns)}")
    print(f"Remaining columns: {len(roi_df.columns)}")