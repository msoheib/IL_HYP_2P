import pandas as pd
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as smt


def get_non_digit_columns(df):
    return [col for col in df.columns if not col.isdigit()]

def get_digit_columns(df):
    return [col for col in df.columns if col.isdigit()]

def get_essential_columns(df):
    return df[['time', 'degrees_180', 'degrees_225', 'degrees_135', 'degrees_0', 'degrees_315', 'degrees_90', 'degrees_270', 'degrees_45', 'speed', 'direction', 'pupil_size']]

def get_essential_columns_as_list(df):
    return ['time', 'degrees_180', 'degrees_225', 'degrees_135', 'degrees_0', 'degrees_315', 'degrees_90', 'degrees_270', 'degrees_45', 'speed', 'direction', 'pupil_size']

def generate_column_names(start=0, end=92):
    return [str(i) for i in range(start, end + 1)]

def convert_column_to_int(df):
    for col in df.columns:
        if col.isdigit():
            if df[col].dtype == 'float64':
                df[col] = df[col].astype(int)
            #convert the int to strings digits
            df[col]=df[col].apply(str)
    return df

#to fix make another function that convert list to int if digit and then those int to string
#foatr conv to int dowes not work use list_post = [int(i) for i in list_post]

def convert_list_to_int_str(lst):
    for i in range(len(lst)):
        if isinstance(lst[i], float):
            lst[i] = int(lst[i])
            lst[i] = str(lst[i])
        elif isinstance(lst[i], int):
            lst[i] = str(lst[i])
        elif isinstance(lst[i], str):
            continue
        
    return lst

def max_digit_columns(df):
    # Convert all digit columns to int
    for col in df.columns:
        if col.isdigit():
            df[col] = df[col].astype(int)

    # Find the maximum value among the digit columns
    max_value = df.select_dtypes(include=[np.number]).max().max()

    # Generate a list of columns that are equal to the max value
    max_columns = [col for col in df.columns if df[col].max() == max_value]

    return max_columns

def drop_columns_with_nan_rows(df1, df2):
    # Find columns with NaN in corresponding rows
    mask = (df1.isna() | df2.isna()).any()
    
    # Drop those columns from both DataFrames
    df1_cleaned = df1.loc[:, ~mask]
    df2_cleaned = df2.loc[:, ~mask]
    
    return df1_cleaned, df2_cleaned

def resequence_digit_columns(df):
    """
    Renames columns that are digits to consecutive numbers starting from 0.
    
    Parameters:
    - df: The DataFrame whose digit columns are to be renamed.
    
    Returns:
    - A DataFrame with digit columns renamed.
    """
    # Assuming hf.get_digit_columns() returns a list of column names that are digits
    digit_columns = get_digit_columns(df)
    
    # Calculate the number of digit columns
    max_digit = len(digit_columns)
    
    # Generate new column names
    new_column_names = [str(i) for i in range(max_digit)]
    
    # Create a mapping from old to new column names for digit columns
    mapping = {digit_columns[i]: new_column_names[i] for i in range(max_digit)}
    
    # Rename columns
    df_renamed = df.rename(columns=mapping)
    
    return df_renamed

def customize_pandas_plot(axs):
    # Remove frame from all subplots
    [x.set_frame_on(False) for x in axs.ravel()]
    # Turn on axis for all subplots
    [x.set_axis_on() for x in axs.ravel()]
    # Set title position for all subplots
    [x.title.set_position([0.15, 0.5]) for x in axs.ravel()]
    # Set tick parameters for all subplots
    [x.tick_params(labelsize=10) for x in axs.ravel()]

def mean_columns_based_on_binary(df, condition_columns, mean_columns):
    '''
    Mean of values of mean_columns only when corresponding values in condition_columns are 1.
    
    df (pd.DataFrame): The input dataframe.
    condition_columns (list of str): List of columns with binary values (0 or 1).
    mean_columns (list of str): List of columns whose values will be average.
    
    '''
    result = {cond_col: [] for cond_col in condition_columns}
    for mean_col in mean_columns:
        for cond_col in condition_columns:
            condition_mean = df.loc[df[cond_col] == 1, mean_col].mean()
            result[cond_col].append(condition_mean)
    
    # Convert to DataFrame
    mean_df = pd.DataFrame(result, index=mean_columns)
    
    return mean_df

def mean_before_first_one(df, condition_column, mean_columns, n_rows):
    '''
    Finds the first occurrence of 1 in the condition column and calculates the mean
    for the specified mean columns in the n_rows before the found row.
    
    df (pd.DataFrame): The input dataframe.
    condition_column (str): Column with binary values (0 or 1).
    mean_columns (list of str): Columns whose values will be averaged.
    n_rows (int): Number of rows before the found row to include in the mean calculation.
    '''
    # Find the index of the first occurrence of 1
    first_one_index = df[df[condition_column] == 1].index.min()
    
    # Calculate start index
    start_index = max(first_one_index - n_rows, 0)
    
    # Select the rows from start_index to first_one_index - 1
    relevant_rows = df.loc[start_index:first_one_index-1, mean_columns]
    print(relevant_rows)
    
    # Calculate the mean of the relevant rows
    mean_values = relevant_rows.mean()
    
    return mean_values


def plot_polar(df, angle_column, value_columns, fill=False):
    '''
    Plots a polar chart for specified value columns based on angles in the angle column.

    df: The input dataframe.
    angle_column: The column name representing angles.
    value_columns: List of column names whose values will be plotted.
    '''

    angles = np.deg2rad(df[angle_column].values)  # Convert angles to radians
    angles_extended = np.concatenate((angles, [angles[0]]))  # Close the loop
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for value_column in value_columns:
        values = df[value_column].values
        values_extended = np.concatenate((values, [values[0]]))  # Close the loop
        
        
        ax.plot(angles_extended, values_extended, linewidth=2, label=value_column)
        if (fill==True):
            ax.fill(angles_extended, values_extended, alpha=0.25, label=value_column)
    
    # Add labels and title
    ax.set_xticks(angles)
    ax.set_xticklabels([f'{int(a)}°' for a in np.rad2deg(angles)])
    ax.set_yticklabels([])
    plt.title('VEP')
    plt.legend(loc='upper right')
    
    plt.show()

def plot_polar_individual(df, angle_column, value_columns, commonmax=True):
    '''
    Plots a polar chart for specified value columns based on angles in the angle column.
    df: The input dataframe.
    angle_column: The column name representing angles.
    value_columns: List of column names whose values will be plotted.
    commonmax: True to scale to max of all columns
    '''
    
    angles = np.deg2rad(df[angle_column].values)  # Convert angles to radians
    max=df[value_columns].values.max()
    for value_column in value_columns:
        values = df[value_column].values
        angles_extended = np.concatenate((angles, [angles[0]]))  # Close the loop
        values_extended = np.concatenate((values, [values[0]]))  # Close the loop
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.fill(angles_extended, values_extended, alpha=0.25)
        ax.plot(angles_extended, values_extended, linewidth=2)
        
        # Add labels and title
        if commonmax:
            # ax.set_ylim(0,max)
            ax.set_ylim(0,10)
        ax.set_xticks(angles)
        ax.set_xticklabels([f'{int(a)}°' for a in np.rad2deg(angles)])
        plt.title(f'Polar Plot for {value_column}')
        plt.show()
        
def bandpassfiltnewColumns(data, freq, cutofflow, cutoffhigh, butterord):
    '''
    data: The input data to filter.
    freq: Sampling frequency.
    cutofflow: Low cutoff frequency.
    cutoffhigh: High cutoff frequency.
    butterord: Order of the Butterworth filter.
     '''
    
    fs = freq 
    
    # Create a DataFrame to store the results
    result = pd.DataFrame(index=data.index, columns=data.columns)
    
    for column in data.columns:
        col_data = data[column]
        
        # Drop NaN values and get indices
        original_index = col_data.index
        col_data = col_data.dropna()
        new_index = col_data.index
        
        # Normalizing frequencies
        wlow = cutofflow / (fs / 2)
        whigh = cutoffhigh / (fs / 2)
        
        # Design Butterworth bandpass filter
        b, a = signal.butter(butterord, [wlow, whigh], btype='bandpass')
        
        # Apply the filter
        output = signal.filtfilt(b, a, col_data)
        
        # Add the mean back to the filtered signal
        output = output + col_data.mean()
        
        # Create a Series to hold the filtered data
        filtered_series = pd.Series(np.nan, index=original_index)
        filtered_series[new_index] = output
        
        # Store the filtered data in the result DataFrame
        result[column] = filtered_series
    
    return result

def DeltaF_percentile_columns(data, Hz, percentile):
    '''
    Calculates the deltaF for each column in the DataFrame based on a given percentile as baseline.
    data: The input data.
    Hz: The sampling frequency.
    percentile: The percentile to be considered as baseline (0 to 1).
    '''
    # Create a DataFrame to store the deltaF results
    deltaF_df = pd.DataFrame(index=data.index, columns=data.columns)
    
    for column in data.columns:
        col_data = data[column]
        
        # Drop NaN values (if needed)
        col_data = col_data.dropna()
        
        # Calculate the F0 as the specified percentile
        F0 = col_data.quantile(percentile, interpolation='lower')
        print(f"F0 for column {column}: {F0}")
        
        # Calculate deltaF
        deltaF = (col_data - F0) * 100 / F0
        
        # Store the deltaF values in the result DataFrame
        deltaF_df[column] = deltaF
    
    # Create an array of time in seconds
    x_seconds = np.arange(len(deltaF_df)) / Hz
    
    return deltaF_df, x_seconds

def sum_columns_based_on_binary(df, condition_columns, sum_columns):
    '''
    Sums values of sum_columns only when corresponding values in condition_columns are 1.
    
    df: The input dataframe.
    condition_columns: List of columns with binary values (0 or 1).
    sum_columns: List of columns whose values will be summed.
    
    '''
    result = {cond_col: [] for cond_col in condition_columns}
    for sum_col in sum_columns:
        for cond_col in condition_columns:
            condition_sum = df.loc[df[cond_col] == 1, sum_col].sum()
            result[cond_col].append(condition_sum)
    
    sum_df = pd.DataFrame(result, index=sum_columns)
    
    return sum_df

def calculate_gOSI(y, stimuli_list):
    ### Mark Mazurek. 2014
    '''Calculate the global orientation selectivity index (gOSI) and preferred orientation.
    gOSI = |ΣR(θ)e^(2iθ)| / ΣR(θ)
    where R(θ) is the response at orientation θ.
    Parameters
    ----------
    y : 1d array
    Response values at each orientation, use raw or normalized y.
    stimuli_list : 1d array
    List of orientations in degrees.
    Returns
    -------
    gOSI : float
    Global orientation selectivity index.
    prefer_ori : float
    Preferred orientation in degrees.
    '''
    angle_list = stimuli_list / 180 * np.pi
    z = np.sum(y* np.exp(2j * angle_list)) / np.sum(y)
    gOSI = abs(z)
    prefer_ori = np.angle(z)%(2*np.pi)*180/2/np.pi
    return gOSI, prefer_ori

def correlate_columns_with_target(data, target_column, selected_columns):
    datacopy=data.copy(deep=True)
    # Calculate correlations
    correlations = datacopy[selected_columns].apply(lambda x: x.corr(datacopy[target_column]))
    # Convert to DataFrame
    correlation_df = pd.DataFrame(correlations, columns=['Correlation'])
    
    return correlation_df

def compute_all_cross_correlations(data, base_col, maxlag):
    """
    Compute cross-correlation between a base column and all numbered columns in the DataFrame.
    
    Parameters:
        data (DataFrame): The pandas DataFrame containing the data.
        base_col (str): The name of the base column to compare against other columns.
        maxlag (int): The maximum lag to compute the cross-correlation for.
    
    Returns:
        dict: A dictionary containing cross-correlation arrays for each numbered column with the base column.
    """
    cross_correlations = {}
    numbered_cols = [col for col in data.columns if col.isdigit()]

    for col in numbered_cols:
        # Calculate cross-correlation using Statsmodels ccf
        cc_values = smt.ccf(data[base_col], data[col], adjusted=False, fft=False)[:maxlag+1]
        cross_correlations[col] = cc_values

    return cross_correlations

def getonsets(data,selected_columns):
    datacopy=data.copy(deep=True)
    #first derivative
    firstder=datacopy[selected_columns].apply(lambda x : np.diff(x, prepend=0))
    firstder[firstder!=1]=0
    return firstder

def count_rois_in_sessions(csv_path, s1_name="pre", s2_name="post"):
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Assume column names are 'ROI_Session1' and 'ROI_Session2'
    session1_rois = df[s1_name].tolist()
    session2_rois = df[s2_name].tolist()

    return session1_rois, session2_rois