#%%
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.interpolate import interp1d
import xml.etree.ElementTree as ET
import plotly.express as px
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import h5py
import random


#nwb imports for additins to the nwb
from pynwb import NWBHDF5IO
from pynwb.ophys import RoiResponseSeries, DfOverF
import tables

#%%
#nwb imports
#%%
from datetime import datetime
from dateutil import tz
from pathlib import Path
from neuroconv.datainterfaces import Suite2pSegmentationInterface, CsvTimeIntervalsInterface
from neuroconv.tools.roiextractors import roiextractors


import data_import as di #from pycoontrol ulitity functions
#%%
def correct_align_and_add_orientations(session_df, xml_file_path, fluorescence_df):
    #grab photodector times from sesh_df
    photodetector_times = session_df[session_df['name'] == 'photodetector']['time'].values

    #func to find closest photodector time for strt times
    def find_nearest_photodetector_time(reference_time, time_range=300):
        potential_times = photodetector_times[(photodetector_times >= reference_time - time_range) &
                                              (photodetector_times <= reference_time + time_range)]
        return potential_times[np.argmin(np.abs(potential_times - reference_time))] if len(potential_times) > 0 else reference_time

    #same idea but for the stop times
    def find_distant_photodetector_time(reference_time, time_range=500):
        potential_times = photodetector_times[(photodetector_times >= reference_time - time_range) &
                                              (photodetector_times <= reference_time + time_range)]
        return potential_times[np.argmax(np.abs(potential_times - reference_time))] if len(potential_times) > 0 else reference_time

    #clear out nans in 'name' col
    session_df['name'] = session_df['name'].fillna('')

    #get rows that start with degrees_ for orientation
    degree_states = session_df[session_df['name'].str.startswith('degrees_')].copy()

    #make a dict for corrected times for the degree states
    corrected_times_dict = {}

    for _, row in degree_states.iterrows():
        state_name = row['name']
        start_time = row['time']
        stop_time = start_time + row['duration']

        #fix the start&stop based on photodetector
        corrected_start_time = find_nearest_photodetector_time(start_time)
        corrected_stop_time = find_distant_photodetector_time(stop_time)

        #throw it in the dict
        if state_name not in corrected_times_dict:
            corrected_times_dict[state_name] = []
        corrected_times_dict[state_name].append((corrected_start_time, corrected_stop_time))

    #load up the xml to get frame details
    xml = ET.parse(xml_file_path)
    root = xml.getroot()

    sequenceElement = root.find('Sequence')
    frameElements = [frameElement for frameElement in sequenceElement.iter('Frame')]

    frame_period = float(frameElements[1].get('relativeTime')) - float(frameElements[0].get('relativeTime'))
    duration = float(frameElements[-1].get('relativeTime')) + frame_period - float(frameElements[0].get('relativeTime'))
    frames = len(frameElements)
    frame_relative_time = np.arange(0, duration, frame_period)

    #figure out the frame nums for the corrected times
    key_frames = {}
    for state, times in corrected_times_dict.items():
        key_frames[state] = []
        for start_time, stop_time in times:
            start = start_time / 1000  #sec conversion
            end = stop_time / 1000     #sec conversion
            closest_start = min(frame_relative_time, key=lambda x: abs(x - start))
            closest_end = min(frame_relative_time, key=lambda x: abs(x - end))
            start_frame = np.where(frame_relative_time == closest_start)[0][0]
            end_frame = np.where(frame_relative_time == closest_end)[0][0]
            key_frames[state].append((start_frame, end_frame))

    #add orientation info to fluorescence df
    deltaF_F = fluorescence_df.copy()
    #add time col from relative time
    deltaF_F['time'] = frame_relative_time[:len(deltaF_F)]

    for angle, frame_start_stops in key_frames.items():
        deltaF_F[f'{angle}'] = 0
        for frame_start_stop in frame_start_stops:
            start = frame_start_stop[0]
            end = frame_start_stop[1] + 1
            deltaF_F.loc[start:end, f'{angle}'] = 1
    
    return deltaF_F

#%%
#def correct_align_and_add_orientations_folder(folder_path, window_size=30, percentile=10, video_fps=20, method='percentile'):

def correct_align_and_add_orientations_folder(*, folder_path, window_size=30, video_fps=20, method='percentile', stable_frames=300, step_size=10, sliding_size=300, percentile_s=10):

    #find files by pattern
    txt_file_path = glob.glob(os.path.join(folder_path, '*.txt'))[0]
    xml_file_path = glob.glob(os.path.join(folder_path, '*.xml'))[0]
    speed_file_path = glob.glob(os.path.join(folder_path, '*_Speed.pca'))[0]
    direction_file_path = glob.glob(os.path.join(folder_path, '*_Direction.pca'))[0]


    #paths for the numpy files
    F_path = os.path.join(folder_path, 'F.npy')
    print(F_path)
    iscell_path = os.path.join(folder_path, 'iscell.npy')
    spks_path = os.path.join(folder_path, 'spks.npy')
    ops_path = os.path.join(folder_path, 'ops.npy')
    Fneu_path = os.path.join(folder_path, 'Fneu.npy')
    pupil_csv_path = os.path.join(folder_path, 'pupil_output.csv')
    
    #pycontrol files
    session_df = di.session_dataframe(txt_file_path)

    #load npy arrays from paths
    F = np.load(F_path)
    iscell = np.load(iscell_path)
    spks = np.load(spks_path)
    Fneu = np.load(Fneu_path)
    #print(ops_path)
    ops = np.load(ops_path, allow_pickle=True)
    ops = ops.item()
    print("done ops loading")
    print(ops_path)

    speed_np = di.load_analog_data(speed_file_path)
    direction_np = di.load_analog_data(direction_file_path)
    
    #gen the fluorescence DataFrame using the delta_fifty_window function
    #fluorescence_data = delta_fify_method_folder_three(F=F, iscell=iscell, spks=spks, Fneu=Fneu, window_size=window_size, percentile_s=percentile, method=method, stable_frames=stable_frames, step_size=step_size, sliding_size=sliding_window_size)
    #fluorescence_df = pd.DataFrame(fluorescence_data)
    
    #v4 of the dff calcucaltion from the dec meeting
    fluorescence_data, frames_rate = dff_baseline_sliding(F=F, iscell=iscell, spks=spks, Fneu=Fneu, sliding_size=sliding_size, percentile_s=percentile_s, window_size=window_size, step_size=step_size)
    fluorescence_df = pd.DataFrame(fluorescence_data)
    
    #grab photodector times from sesh_df
    photodetector_times = session_df[session_df['name'] == 'photodetector']['time'].values

    #func to find closest photodector time for strt times
    def find_nearest_photodetector_time(reference_time, time_range=300):
        potential_times = photodetector_times[(photodetector_times >= reference_time - time_range) &
                                              (photodetector_times <= reference_time + time_range)]
        return potential_times[np.argmin(np.abs(potential_times - reference_time))] if len(potential_times) > 0 else reference_time

    #same idea but for the stop times
    def find_distant_photodetector_time(reference_time, time_range=500):
        potential_times = photodetector_times[(photodetector_times >= reference_time - time_range) &
                                              (photodetector_times <= reference_time + time_range)]
        return potential_times[np.argmax(np.abs(potential_times - reference_time))] if len(potential_times) > 0 else reference_time

    #clear out nans in 'name' col
    session_df['name'] = session_df['name'].fillna('')

    #get rows that start with degrees_ for orientation
    degree_states = session_df[session_df['name'].str.startswith('degrees_')].copy()

    #make a dict for corrected times for the degree states
    corrected_times_dict = {}

    for _, row in degree_states.iterrows():
        state_name = row['name']
        start_time = row['time']
        stop_time = start_time + row['duration']

        #fix the start&stop based on photodetector
        corrected_start_time = find_nearest_photodetector_time(start_time)
        corrected_stop_time = find_distant_photodetector_time(stop_time)

        #throw it in the dict
        if state_name not in corrected_times_dict:
            corrected_times_dict[state_name] = []
        corrected_times_dict[state_name].append((corrected_start_time, corrected_stop_time))

    #load up the xml to get frame details
    xml = ET.parse(xml_file_path)
    root = xml.getroot()

    sequenceElement = root.find('Sequence')
    frameElements = [frameElement for frameElement in sequenceElement.iter('Frame')]

    frame_period = float(frameElements[1].get('relativeTime')) - float(frameElements[0].get('relativeTime'))
    duration = float(frameElements[-1].get('relativeTime')) + frame_period - float(frameElements[0].get('relativeTime'))
    frames = len(frameElements)
    frame_relative_time = np.arange(0, duration, frame_period)

    fs = 1 / frame_period
    ops['fs'] = fs
    print("done ops")
    #save the ops np array as npy file
    print(ops_path)
    np.save(ops_path, ops)
    print("done ops saving")



    #figure out the frame nums for the corrected times
    key_frames = {}
    for state, times in corrected_times_dict.items():
        key_frames[state] = []
        for start_time, stop_time in times:
            start = start_time / 1000  #sec conversion
            end = stop_time / 1000     #sec conversion
            closest_start = min(frame_relative_time, key=lambda x: abs(x - start))
            closest_end = min(frame_relative_time, key=lambda x: abs(x - end))
            start_frame = np.where(frame_relative_time == closest_start)[0][0]
            end_frame = np.where(frame_relative_time == closest_end)[0][0]
            key_frames[state].append((start_frame, end_frame))

    #add orientation info to fluorescence df
    deltaF_F = fluorescence_df.copy()
    #add time col from relative time
    deltaF_F['time'] = frame_relative_time[:len(deltaF_F)]

    for angle, frame_start_stops in key_frames.items():
        deltaF_F[f'{angle}'] = 0
        for frame_start_stop in frame_start_stops:
            start = frame_start_stop[0]
            end = frame_start_stop[1] + 1
            deltaF_F.loc[start:end, f'{angle}'] = 1
    
    align_movement_data_to_fluorescence(deltaF_F, speed_np, direction_np)

    upsample_and_append_pupil(deltaF_F, xml_file_path, pupil_csv_path, video_fps=video_fps)
    deltaF_F.columns = deltaF_F.columns.astype(str)
    return deltaF_F


#%%

def simple_align_and_add_orientations_folder(*, folder_path, window_size=30, video_fps=20, method='percentile', stable_frames=300, step_size=10, sliding_size=300, percentile_s=10):

    # Find files by pattern
    txt_file_path = glob.glob(os.path.join(folder_path, '*.txt'))[0]
    xml_file_path = glob.glob(os.path.join(folder_path, '*.xml'))[0]
    speed_file_path = glob.glob(os.path.join(folder_path, '*_Speed.pca'))[0]
    direction_file_path = glob.glob(os.path.join(folder_path, '*_Direction.pca'))[0]

    # Paths for the numpy files
    F_path = os.path.join(folder_path, 'F.npy')
    iscell_path = os.path.join(folder_path, 'iscell.npy')
    spks_path = os.path.join(folder_path, 'spks.npy')
    ops_path = os.path.join(folder_path, 'ops.npy')
    Fneu_path = os.path.join(folder_path, 'Fneu.npy')
    pupil_csv_path = os.path.join(folder_path, 'pupil_output.csv')
    
    # Load session data
    #session_df = pd.read_csv(txt_file_path, delimiter='\t')  # Adjust as necessary for your data format

    session_df = di.session_dataframe(txt_file_path)
    # Load numpy arrays
    F = np.load(F_path, allow_pickle=True)
    iscell = np.load(iscell_path, allow_pickle=True)
    spks = np.load(spks_path, allow_pickle=True)
    Fneu = np.load(Fneu_path, allow_pickle=True)
    ops = np.load(ops_path, allow_pickle=True).item()

    # Load analog data

    speed_np = di.load_analog_data(speed_file_path)
    direction_np = di.load_analog_data(direction_file_path)
    
    # Calculate fluorescence data
    fluorescence_data, frames_rate = dff_baseline_sliding(F=F, iscell=iscell, spks=spks, Fneu=Fneu, sliding_size=sliding_size, percentile_s=percentile_s, window_size=window_size, step_size=step_size)
    fluorescence_df = pd.DataFrame(fluorescence_data)
    
    # Get rows that start with degrees_ for orientation
    session_df['name'] = session_df['name'].fillna('')

    degree_states = session_df[session_df['name'].str.startswith('degrees_')].copy()

    # Use XML to get frame details
    xml = ET.parse(xml_file_path)
    root = xml.getroot()
    sequenceElement = root.find('Sequence')
    frameElements = [frameElement for frameElement in sequenceElement.iter('Frame')]
    frame_period = float(frameElements[1].get('relativeTime')) - float(frameElements[0].get('relativeTime'))
    duration = float(frameElements[-1].get('relativeTime')) + frame_period - float(frameElements[0].get('relativeTime'))
    frames = len(frameElements)
    frame_relative_time = np.arange(0, duration, frame_period)

    # Set fs in ops and save
    fs = 1 / frame_period
    ops['fs'] = fs
    np.save(ops_path, ops)

    # Determine the frame numbers for the times
    key_frames = {}
    for _, row in degree_states.iterrows():
        state_name = row['name']
        start_time = row['time'] / 1000  # Convert ms to seconds
        end_time = start_time + row['duration'] / 1000  # Convert ms to seconds

        start_frame = np.searchsorted(frame_relative_time, start_time, side='left')
        end_frame = np.searchsorted(frame_relative_time, end_time, side='right') - 1

        if state_name not in key_frames:
            key_frames[state_name] = []
        key_frames[state_name].append((start_frame, end_frame))

    # Add orientation info to fluorescence dataframe
    for angle, frame_start_stops in key_frames.items():
        fluorescence_df[f'{angle}'] = 0
        for start, end in frame_start_stops:
            fluorescence_df.loc[start:end, f'{angle}'] = 1
    
    return fluorescence_df

# %%
def align_movement_data_to_fluorescence(fl_df, speed_npy, direction_npy):
    #first cols timestamp, second is the data
    speed_timestamps = speed_npy[:, 0]
    speed_samples = speed_npy[:, 1]
    direction_timestamps = direction_npy[:, 0]
    direction_samples = direction_npy[:, 1]
    
    #gotta resample to match the df rate
    num_target_samples = int(fl_df.shape[0])
    
    #resample the speed and dir
    resampled_speed = resample(speed_samples, num_target_samples)
    resampled_direction = resample(direction_samples, num_target_samples)
    
    #stick the new cols on df
    fl_df['speed'] = resampled_speed
    fl_df['direction'] = resampled_direction

    return fl_df

# %%
#parses frame deetails from the xml
def get_frame_details(xml_file_path):
    xml = ET.parse(xml_file_path)
    root = xml.getroot()
    sequenceElement = root.find('Sequence')
    frameElements = [frameElement for frameElement in sequenceElement.iter('Frame')]
    frame_period = float(frameElements[1].get('relativeTime')) - float(frameElements[0].get('relativeTime'))
    duration = float(frameElements[-1].get('relativeTime')) + frame_period - float(frameElements[0].get('relativeTime'))
    frames = len(frameElements)
    frame_relative_time = np.arange(0, duration, frame_period)
    return {"frame_period": frame_period, "session_duration": duration, "total_frames": frames, "frame_relative_time": frame_relative_time}

#upsamples and appends pupil data, gotta match the frames
def upsample_and_append_pupil_depr(target_csv_df, xml_file_path, pupil_csv_path, video_fps, delay=0.03):
    #load pupil data
    pupil_df = pd.read_csv(pupil_csv_path)
    #assuming pupil's in 'processedPupil'
    original_data = pupil_df['processedPupil'].values
    original_rate = video_fps  #video fps
    original_timestamps = np.arange(0, len(original_data)/original_rate, 1/original_rate)
    
    #get frame info frm xml
    frame_details = get_frame_details(xml_file_path)
    target_rate = 1 / frame_details['frame_period']  #target fps
    
    #interp to fit the target
    interpolation_function = interp1d(original_timestamps, original_data, kind='linear', fill_value="extrapolate")
    target_timestamps = frame_details['frame_relative_time'] + delay
    upsampled_data = interpolation_function(target_timestamps)
    
    #slap it onto the df
    target_csv_df['pupil_size'] = upsampled_data
    
    return target_csv_df


# accounts for the difference in index after interpolation
#%%
def upsample_and_append_pupil(target_csv_df, xml_file_path, pupil_csv_path, video_fps, fill_initial_30ms_with_mean=True):
    # Load the pupil CSV
    pupil_df = pd.read_csv(pupil_csv_path)
    # Assuming the pupil data is in a column named 'processedPupil'
    original_data = pupil_df['processedPupil'].values
    original_rate = video_fps  # Original sampling rate of the video
    original_timestamps = np.arange(0, len(original_data)/original_rate, 1/original_rate)
    
    # Get frame details from the XML file
    frame_details = get_frame_details(xml_file_path)
    target_rate = 1 / frame_details['frame_period']  # Target sampling rate
    
    # Calculate the number of points equivalent to 30ms in the target rate
    num_points_30ms = int(np.ceil(0.03 * target_rate))
    
    # Calculate the mean of the original pupil data
    mean_pupil_size = np.mean(original_data)
    
    # If we want to fill the initial 30ms with the mean value
    if fill_initial_30ms_with_mean:
        # Fill the first equivalent of 30ms with the mean value
        original_data[:num_points_30ms] = mean_pupil_size
    
    # Interpolate the pupil data to the target rate
    interpolation_function = interp1d(original_timestamps, original_data, kind='linear', fill_value="extrapolate")
    target_timestamps = frame_details['frame_relative_time']
    print(target_timestamps)
    upsampled_data = interpolation_function(target_timestamps)
    
    # Adjust the length of upsampled_data to match target_csv_df
    if len(upsampled_data) > len(target_csv_df):
        # If upsampled_data is longer, trim it
        upsampled_data = upsampled_data[:len(target_csv_df)]
    elif len(upsampled_data) < len(target_csv_df):
        # If upsampled_data is shorter, pad it with the mean value
        mean_value = np.mean(upsampled_data)
        padding_length = len(target_csv_df) - len(upsampled_data)
        padding = np.full(padding_length, mean_value)
        upsampled_data = np.concatenate([upsampled_data, padding])

    # Append the upsampled data to the target dataframe
    target_csv_df['pupil_size'] = upsampled_data
    
    return target_csv_df
# %%
#takes npy files, does deltaF/F%
def delta_fify_window(F, iscell, spks, Fneu, window_size=30, percentile=10):
    #only take cells marked '1'
    Fofiscell = F[iscell[:, 0] == 1, :]
    Fneuofiscell = Fneu[iscell[:, 0] == 1, :]
    Spksofiscell = spks[iscell[:, 0] == 1, :]
    #correct for neuropil
    correctedFofiscell = Fofiscell - 0.7 * Fneuofiscell
    correctedFofiscell = np.transpose(correctedFofiscell)

    #make dFF array
    dFF = np.zeros(correctedFofiscell.shape)
    Spksofiscell = np.transpose(Spksofiscell)
    Fofiscell = np.transpose(Fofiscell)

    #loop thru and calc dFF
    for i in range(Fofiscell.shape[1]):
        #smooth it out with running avg
        filteredtrace = np.convolve(correctedFofiscell[:, i], np.ones(window_size) / window_size, mode='same')
        #baseline as 10th percentile
        bl = np.percentile(filteredtrace, percentile)
        dFF[:, i] = (filteredtrace - bl) / bl
    return dFF


#sliding window dff calcucaltion code with center of the window #after meetign in dec23

def dff_baseline_sliding(F, iscell, Fneu, spks, window_size=30, sliding_size=1800, percentile_s=10, step_size=10):
    #fucn to fill the zeros with last value for padding
    def fill_zeros_with_last(arr):
        mask = arr > 0
        arr = np.where(mask, arr, np.maximum.accumulate(mask * arr))
        return arr
    #only take cells marked '1'
    Fofiscell = F[iscell[:, 0] == 1, :]
    Fneuofiscell = Fneu[iscell[:, 0] == 1, :]
    
    #correct for neuropil
    correctedFofiscell = Fofiscell - 0.7 * Fneuofiscell
    correctedFofiscell = np.transpose(correctedFofiscell)

    #make dFF array
    dFF = np.zeros(correctedFofiscell.shape)
    baseline = None

    #loop thru and calc dFF
    for i in range(correctedFofiscell.shape[1]):
        neuron_trace = np.convolve(correctedFofiscell[:, i], np.ones(window_size) / window_size, mode='same')
        baseline = np.zeros_like(neuron_trace)

        #print(range(len(neuron_trace) - sliding_size + 1))
        
        #baseline as sliding window
        half_sliding_size = sliding_size // 2
        for j in range(half_sliding_size, len(neuron_trace) - half_sliding_size + 1, step_size):
            #print(j)
            start = max(0, j - half_sliding_size)
            end = min(len(neuron_trace), j + half_sliding_size)
            baseline[start:end] = np.percentile(neuron_trace[start:end], percentile_s)
            #print(baseline[start:end])


        pad_width = sliding_size // 2

        ####*** fill the remiahng with prior values but dsiregard the last value for next calcution

        neuron_trace_padded = np.pad(neuron_trace, (pad_width, pad_width), mode='edge') #
 
        baseline_padded = np.pad(baseline, (pad_width, pad_width), mode='edge')
        baseline_padded = fill_zeros_with_last(baseline_padded)

        dFF[:, i] = (neuron_trace_padded[pad_width:-pad_width] - baseline_padded[pad_width:-pad_width]) / baseline_padded[pad_width:-pad_width]

    return dFF, baseline


def dff_baseline_sliding(F, iscell, Fneu, spks, window_size=30, sliding_size=1800, percentile_s=10, step_size=10):
    # Function to fill the zeros with last value for padding
    def fill_zeros_with_last(arr):
        mask = arr > 0
        arr = np.where(mask, arr, np.maximum.accumulate(mask * arr))
        return arr

    # Only take cells marked '1'
    Fofiscell = F[iscell[:, 0] == 1, :]
    Fneuofiscell = Fneu[iscell[:, 0] == 1, :]
    
    # Correct for neuropil
    correctedFofiscell = Fofiscell - 0.7 * Fneuofiscell
    correctedFofiscell = np.transpose(correctedFofiscell)

    # Make dFF array
    dFF = np.zeros(correctedFofiscell.shape)
    baseline = None

    # Loop through and calculate dFF
    for i in range(correctedFofiscell.shape[1]):
        neuron_trace = np.convolve(correctedFofiscell[:, i], np.ones(window_size) / window_size, mode='same')
        baseline = np.zeros_like(neuron_trace)

        # Baseline as sliding window
        half_sliding_size = sliding_size // 2
        for j in range(half_sliding_size, len(neuron_trace) - half_sliding_size + 1, step_size):
            start = max(0, j - half_sliding_size)
            end = min(len(neuron_trace), j + half_sliding_size)
            baseline[start:end] = np.percentile(neuron_trace[start:end], percentile_s)

        pad_width = sliding_size // 2

        # Fill the remaining with prior values but disregard the last value for next calculation
        neuron_trace_padded = np.pad(neuron_trace, (pad_width, pad_width), mode='edge')
        baseline_padded = np.pad(baseline, (pad_width, pad_width), mode='edge')
        baseline_padded = fill_zeros_with_last(baseline_padded)

        dFF[:, i] = (neuron_trace_padded[pad_width:-pad_width] - baseline_padded[pad_width:-pad_width]) / baseline_padded[pad_width:-pad_width]

        # Rescale the responses between 0 and 100
        dFF[:, i] = (dFF[:, i] - np.min(dFF[:, i])) / (np.max(dFF[:, i]) - np.min(dFF[:, i])) * 100

    return dFF, baseline

#%% test
def delta_fify_method_update(session_folder="None", F=None, iscell=None, spks=None, Fneu=None, method ="percentile", step_size=10, percentile_s=10, stable_frames=300, window_size=30, sliding_size=1800):
    """
    Calculate the delta F/F% values for each cell in a given session.

    Parameters:
        session_folder (str): The path to the folder containing the session data.
        method (str, optional): The method to calculate the baseline. Defaults to 'percentile'.
        percentile (int, optional): The percentile used to calculate the baseline when method is 'percentile'. Defaults to 10.
        stable_frames (int, optional): The number of stable frames used to calculate the baseline when method is 'stable_baseline'. Defaults to 300.
        window_size (int, optional): The size of the window used for smoothing the data. Defaults to 30.

    Returns:
        np.ndarray: The delta F/F% values for each cell in the session.
    """

    # Only take cells marked '1'
    Fofiscell = F[iscell[:, 0] == 1, :]
    Fneuofiscell = Fneu[iscell[:, 0] == 1, :]
    Spksofiscell = spks[iscell[:, 0] == 1, :]
    
    # Correct for neuropil
    correctedFofiscell = Fofiscell - 0.7 * Fneuofiscell
    correctedFofiscell = np.transpose(correctedFofiscell)

    # Make dFF array
    dFF = np.zeros(correctedFofiscell.shape)

    # Loop through and calculate dFF
    for i in range(correctedFofiscell.shape[1]):
        # Smooth it out with running avg
        filteredtrace = np.convolve(correctedFofiscell[:, i], np.ones(window_size) / window_size, mode='same')

        if method == 'percentile':
            # Baseline as specified percentile
            bl = np.percentile(filteredtrace, percentile_s)
        elif method == 'stable_baseline':
            # Baseline as mean of the first 'stable_frames' frames
            bl = np.mean(filteredtrace[:stable_frames])

        elif method == 'sliding':
            #neuron_trace = correctedFofiscell[:, i]
            neuron_trace = np.convolve(correctedFofiscell[:, i], np.ones(window_size) / window_size, mode='same')
            baseline = np.zeros_like(neuron_trace)

            pad_width = sliding_size // 2

            #pad_width = (pad_width, 0)
            neuron_trace = np.pad(neuron_trace, (pad_width, pad_width), mode='edge') #
            baseline = np.pad(baseline, (pad_width, pad_width), mode='edge') 

            #baseline as sliding window
            half_sliding_size = sliding_size // 2
            for j in range(half_sliding_size, len(neuron_trace) - half_sliding_size + 1, step_size):
                # print(j)
                start = max(0, j - half_sliding_size)
                end = min(len(neuron_trace), j + half_sliding_size)
                baseline[start:end] = np.percentile(neuron_trace[start:end], percentile_s)

            filteredtrace = neuron_trace[pad_width:-pad_width]
            
            bl = baseline[pad_width:-pad_width]
        else:
            raise ValueError("Invalid method. Choose 'percentile' or 'stable_baseline'.")

        dFF[:, i] = (filteredtrace - bl) / bl

    return dFF

#%%
def delta_fify_method_folder_three(session_folder="None", F=None, iscell=None, spks=None, Fneu=None, method ="percentile", percentile_s=10, stable_frames=300, window_size=30, sliding_size=1800, step_size=10):
    
    # Only take cells marked '1'
    Fofiscell = F[iscell[:, 0] == 1, :]
    Fneuofiscell = Fneu[iscell[:, 0] == 1, :]
    
    # Correct for neuropil
    correctedFofiscell = Fofiscell - 0.7 * Fneuofiscell
    correctedFofiscell = np.transpose(correctedFofiscell)

    # Make array for final data (dFF or smoothed raw fluorescence)
    final_data = np.zeros(correctedFofiscell.shape)

    # Loop through and calculate dFF or return smoothed raw fluorescence
    for i in range(correctedFofiscell.shape[1]):
        # Smooth it out with running avg
        
        if window_size == 0:
            Fofiscell_raw = np.transpose(Fofiscell)
            smoothed_trace = Fofiscell_raw[:, i]
        else:
            smoothed_trace = np.convolve(correctedFofiscell[:, i], np.ones(window_size) / window_size, mode='same')

        if method == 'percentile':
            # Baseline as specified percentile
            bl = np.percentile(smoothed_trace, percentile_s)
            final_data[:, i] = (smoothed_trace - bl) / bl
        elif method == 'stable_baseline':
            # Baseline as mean of the specified frames, or all frames if stable_frames is -1
            baseline_frames = smoothed_trace if stable_frames == -1 else smoothed_trace[:stable_frames]

            bl = np.mean(baseline_frames)
            final_data[:, i] = (smoothed_trace - bl) / bl
        elif method == 'raw':
            # Return smoothed raw fluorescence
            final_data[:, i] = smoothed_trace
        elif method == 'sliding':
            final_data, baseline_discard = dff_baseline_sliding_2(F=F, iscell=iscell, spks=spks, Fneu=Fneu, window_size=30, percentile_s=20, step_size=10, sliding_size=1800)
            break
            #final_data[:, i] = (neuron_trace[pad_width:-pad_width] - baseline[pad_width:-pad_width]) / baseline[pad_width:-pad_width]

    return final_data

#%%
def plot_stimulus_responses(csv_file_path):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load the data
    fluorescence_data = pd.read_csv(csv_file_path)
    fluorescence_data.columns = fluorescence_data.columns.astype(str)

    # Assuming uniform frame rate
    time_per_frame = fluorescence_data['time'][1] - fluorescence_data['time'][0]
    frame_rate = 1 / time_per_frame  # Frames per second if time is in seconds

    # Calculate the number of frames for 2 seconds before and 4 seconds after the stimulus presentation
    frames_before = int(2 * frame_rate)
    frames_after = int(4 * frame_rate)

    # Define stimuli
    stimuli = ['degrees_0', 'degrees_45', 'degrees_90', 'degrees_135', 'degrees_180', 'degrees_225', 'degrees_270', 'degrees_315']

    # Initialize a figure
    plt.figure(figsize=(20, 15))

    for i, stimulus in enumerate(stimuli, 1):
        stimulus_times = fluorescence_data.index[fluorescence_data[stimulus] == 1].tolist()
        
        # Aggregate ROI responses for each stimulus
        aggregated_responses = []
        for time in stimulus_times:
            start = max(time - frames_before, 0)
            end = min(time + frames_after, len(fluorescence_data))
            # Mean across all ROI columns (first 100 columns are assumed to be ROIs)
            response = fluorescence_data.iloc[start:end, :100].mean(axis=1)
            aggregated_responses.append(response)

        # Calculate the mean of aggregated responses across all similar stimuli presentations
        if aggregated_responses:
            mean_response = pd.concat(aggregated_responses, axis=1).mean(axis=1)
            time_axis = np.linspace(-2, 4, len(mean_response))
            
            plt.subplot(4, 2, i)
            plt.plot(time_axis, mean_response)
            plt.title(f'Mean Response for {stimulus}')
            plt.xlabel('Time from stimulus onset (s)')
            plt.ylabel('Mean ΔF/F% (%)')

    plt.tight_layout()
    plt.suptitle('Aggregated Mean Responses of All ROIs to Different Stimuli')
    plt.show()


#test code
# center of the window

def dff_baseline_sliding_folder(folder_path, window_size=30, sliding_size=1800, percentile_s=10, step_size=10):
    
    ##paths for the numpy files
    F_path = os.path.join(folder_path, 'F.npy')
    iscell_path = os.path.join(folder_path, 'iscell.npy')
    spks_path = os.path.join(folder_path, 'spks.npy')
    Fneu_path = os.path.join(folder_path, 'Fneu.npy')    

    #load npy arrays from paths
    F = np.load(F_path)
    iscell = np.load(iscell_path)
    spks = np.load(spks_path)
    Fneu = np.load(Fneu_path)
    
    #only take cells marked '1'
    Fofiscell = F[iscell[:, 0] == 1, :]
    Fneuofiscell = Fneu[iscell[:, 0] == 1, :]
    
    #correct for neuropil
    correctedFofiscell = Fofiscell - 0.7 * Fneuofiscell
    correctedFofiscell = np.transpose(correctedFofiscell)

    #print(correctedFofiscell.shape)
     #make dFF array
    dFF = np.zeros(correctedFofiscell.shape)

    baseline = None

    #loop thru and calc dFF
    for i in range(correctedFofiscell.shape[1]):
        #neuron_trace = correctedFofiscell[:, i]
        neuron_trace = np.convolve(correctedFofiscell[:, i], np.ones(window_size) / window_size, mode='same')
        baseline = np.zeros_like(neuron_trace)

        #print(range(len(neuron_trace) - sliding_size + 1))
        pad_width = sliding_size // 2

        # ###*** fill the remiahng with prior values but dsiregard the last value for next calcution
        #if j + sliding_size < len(neuron_trace):

        #pad_width = (pad_width, 0)
        neuron_trace = np.pad(neuron_trace, (pad_width, pad_width), mode='edge') #
        #baseline[j+sliding_size:] = baseline[j+sliding_size-1]
        #baseline[j+sliding_size:] = np.pad(baseline, (pad_width, 0), mode='edge') 
        
        baseline = np.pad(baseline, (pad_width, pad_width), mode='edge') 
        #print(f"neuron trace shape {neuron_trace.shape}, baseline shape {baseline.shape}")

         #baseline as sliding window
        half_sliding_size = sliding_size // 2
        for j in range(half_sliding_size, len(neuron_trace) - half_sliding_size + 1, step_size):
            # print(j)
            start = max(0, j - half_sliding_size)
            end = min(len(neuron_trace), j + half_sliding_size)
            baseline[start:end] = np.percentile(neuron_trace[start:end], percentile_s)
            # print(start,end)
            # print(baseline.shape)

        #print(f"wihtout padding {neuron_trace[900:-900].shape}")
        # print(f"{baseline.shape}")
        # plt.plot(neuron_trace, color="blue")
        # plt.plot(baseline, color = "yellow")
        # plt.show()
        # alculate dFF
        dFF[:, i] = (neuron_trace[pad_width:-pad_width] - baseline[pad_width:-pad_width]) / baseline[pad_width:-pad_width]
    
    return dFF, baseline


#%%
def dff_baseline_sliding_2(F=None, iscell=None, spks=None, Fneu=None, window_size=30, sliding_size=1800, percentile_s=10, step_size=10):

    #only take cells marked '1'
    Fofiscell = F[iscell[:, 0] == 1, :]
    Fneuofiscell = Fneu[iscell[:, 0] == 1, :]
    
    #correct for neuropil
    correctedFofiscell = Fofiscell - 0.7 * Fneuofiscell
    correctedFofiscell = np.transpose(correctedFofiscell)

    #print(correctedFofiscell.shape)
     #make dFF array
    dFF = np.zeros(correctedFofiscell.shape)

    baseline = None

    #loop thru and calc dFF
    for i in range(correctedFofiscell.shape[1]):
        #neuron_trace = correctedFofiscell[:, i]
        neuron_trace = np.convolve(correctedFofiscell[:, i], np.ones(window_size) / window_size, mode='same')
        baseline = np.zeros_like(neuron_trace)

        #print(range(len(neuron_trace) - sliding_size + 1))
        pad_width = sliding_size // 2

        # ###*** fill the remiahng with prior values but dsiregard the last value for next calcution
        #if j + sliding_size < len(neuron_trace):

        #pad_width = (pad_width, 0)
        neuron_trace = np.pad(neuron_trace, (pad_width, pad_width), mode='edge') #
        #baseline[j+sliding_size:] = baseline[j+sliding_size-1]
        #baseline[j+sliding_size:] = np.pad(baseline, (pad_width, 0), mode='edge') 
        
        baseline = np.pad(baseline, (pad_width, pad_width), mode='edge') 
        #print(f"neuron trace shape {neuron_trace.shape}, baseline shape {baseline.shape}")

         #baseline as sliding window
        half_sliding_size = sliding_size // 2
        for j in range(half_sliding_size, len(neuron_trace) - half_sliding_size + 1, step_size):
            # print(j)
            start = max(0, j - half_sliding_size)
            end = min(len(neuron_trace), j + half_sliding_size)
            baseline[start:end] = np.percentile(neuron_trace[start:end], percentile_s)
            # print(start,end)
            # print(baseline.shape)

        #print(f"wihtout padding {neuron_trace[900:-900].shape}")
        # print(f"{baseline.shape}")
        # plt.plot(neuron_trace, color="blue")
        # plt.plot(baseline, color = "yellow")
        # plt.show()
        # alculate dFF
        dFF[:, i] = (neuron_trace[pad_width:-pad_width] - baseline[pad_width:-pad_width]) / baseline[pad_width:-pad_width]
    
    return dFF, baseline

#%%
def delta_fify_window_folder(session_folder, window_size=30, percentile=10):
    # Function to load .npy files from a given folder
    def load_session_data(folder):
        F = np.load(os.path.join(folder, 'F.npy'))
        iscell = np.load(os.path.join(folder, 'iscell.npy'))
        Fneu = np.load(os.path.join(folder, 'Fneu.npy'))
        spks = np.load(os.path.join(folder, 'spks.npy'))
        return F, iscell, Fneu, spks

    # Load data for each session
    F, iscell, Fneu, spks = load_session_data(session_folder)

     #only take cells marked '1'
    Fofiscell = F[iscell[:, 0] == 1, :]
    Fneuofiscell = Fneu[iscell[:, 0] == 1, :]
    Spksofiscell = spks[iscell[:, 0] == 1, :]
    #correct for neuropil
    correctedFofiscell = Fofiscell - 0.7 * Fneuofiscell
    correctedFofiscell = np.transpose(correctedFofiscell)

    #make dFF array
    dFF = np.zeros(correctedFofiscell.shape)
    Spksofiscell = np.transpose(Spksofiscell)
    Fofiscell = np.transpose(Fofiscell)

    #loop thru and calc dFF
    for i in range(Fofiscell.shape[1]):
        #smooth it out with running avg
        filteredtrace = np.convolve(correctedFofiscell[:, i], np.ones(window_size) / window_size, mode='same')
        #baseline as 10th percentile
        bl = np.percentile(filteredtrace, percentile)
        dFF[:, i] = (filteredtrace - bl) / bl
    return dFF


#%% CHECK THIS
def delta_fify_method_folder(session_folder, method='percentile', percentile=10, stable_frames=300):
    # Function to load .npy files from a given folder
    def load_session_data(folder):
        F = np.load(os.path.join(folder, 'F.npy'))
        iscell = np.load(os.path.join(folder, 'iscell.npy'))
        Fneu = np.load(os.path.join(folder, 'Fneu.npy'))
        spks = np.load(os.path.join(folder, 'spks.npy'))
        return F, iscell, Fneu, spks

    # Load data for each session
    F, iscell, Fneu, spks = load_session_data(session_folder)

    # Only take cells marked '1'
    Fofiscell = F[iscell[:, 0] == 1, :]
    Fneuofiscell = Fneu[iscell[:, 0] == 1, :]
    Spksofiscell = spks[iscell[:, 0] == 1, :]
    
    # Correct for neuropil
    correctedFofiscell = Fofiscell - 0.7 * Fneuofiscell
    correctedFofiscell = np.transpose(correctedFofiscell)

    # Make dFF array
    dFF = np.zeros(correctedFofiscell.shape)

    # Loop through and calculate dFF
    for i in range(correctedFofiscell.shape[1]):
        if method == 'percentile':
            # Baseline as specified percentile
            bl = np.percentile(correctedFofiscell[:, i], percentile)
        elif method == 'stable_baseline':
            # Baseline as mean of the first 'stable_frames' frames
            bl = np.mean(correctedFofiscell[:stable_frames, i])
        else:
            raise ValueError("Invalid method. Choose 'percentile' or 'stable_baseline'.")

        dFF[:, i] = (correctedFofiscell[:, i] - bl) / bl

    return dFF

#%%

def extract_stimuli_timestamps_modified(dataframe):
    """
    Modified function to extract the timestamps of stimuli from the given DataFrame using the 'time' column.

    Parameters:
    dataframe (pd.DataFrame): DataFrame containing dF/F% values, stimuli information, and time column.

    Returns:
    pd.DataFrame: DataFrame with columns 'stimuli', 'start_time_index', 'stop_time_index', 'start_time', 'stop_time'.
    """
    # Identifying columns related to stimuli (degrees_x)
    stimuli_columns = [col for col in dataframe.columns if 'degrees_' in str(col)]
    # Creating a new DataFrame to store the extracted timestamps
    stimuli_df = pd.DataFrame(columns=["stimuli", "start_time_index", "stop_time_index", "start_time", "stop_time"])

    # Iterating over each stimuli column to extract start and stop times
    for stimuli in stimuli_columns:
        # Identifying the start (rising edge) and stop (falling edge) of the stimuli
        start_indices = dataframe.index[dataframe[stimuli].diff() == 1].tolist()
        stop_indices = dataframe.index[dataframe[stimuli].diff() == -1].tolist()

        # Adjust for stimuli starting at the beginning or ending at the end of the recording
        if start_indices and start_indices[0] > stop_indices[0]:
            start_indices.insert(0, 0)
        if stop_indices and (len(stop_indices) < len(start_indices)):
            stop_indices.append(dataframe.shape[0] - 1)

        # Extracting time values for each start and stop index
        for start, stop in zip(start_indices, stop_indices):
            start_time = dataframe.loc[start, 'time']
            stop_time = dataframe.loc[stop, 'time']
            stimuli_df = stimuli_df.append({"stimuli": stimuli, 
                                            "start_time_index": start, 
                                            "stop_time_index": stop, 
                                            "start_time": start_time, 
                                            "stop_time": stop_time}, 
                                           ignore_index=True)

    return stimuli_df

#%%

def convert_s2p_to_nwb(folder_path, nwbfile_path, csv_timepoints_path, dffcsv_path):
    """
    Convert suite2p data from the suite2p folder continaing plane files to NWB file.

    Args:
        folder_path (str): The path to the folder containing the data. eg. r"H:\stress\wt\TSeries-07312023-2113-1gp-001\suite2p"
        nwbfile_path (str): The path to save the converted NWB file. eg. r"H:\stress\wt\TSeries-07312023-2113-1gp-001\suite2p\output_file5.nwb"

    Returns:
        None
    """
    #ops_file_path = os.path.join(folder_path, "ops.npy")
    ops_file_path = folder_path + "\\plane0\\ops.npy"
    ops = np.load(ops_file_path, allow_pickle=True).item()
    #ops = ops.item()
    fs = ops['fs']
    #ops['fs'] = 1/frame_period

    interface = Suite2pSegmentationInterface(folder_path=folder_path, verbose=True)

    metadata = interface.get_metadata()
    
    #-------------------------------------#
    dffcsv = pd.read_csv(dffcsv_path)
    nptimestamps = np.array(dffcsv['time'])

    interface.set_aligned_timestamps(nptimestamps)

    #-------------------------------------#
    session_start_time = datetime(2020, 1, 1, 12, 30, 0, tzinfo=tz.gettz("US/Pacific"))
    
    metadata["NWBFile"].update(session_start_time=session_start_time)
    metadata["TwoPhotonSeries"].update(rate=fs, conversion=1.0)
    
    interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)
    
    csv_interface = CsvTimeIntervalsInterface(file_path=csv_timepoints_path, verbose=True)
    csv_interface.run_conversion(nwbfile_path=nwbfile_path, metadata=metadata)


#%%
def extract_stimuli_timestamps(dataframe):
    """
    Modified function to extract the timestamps of stimuli from the given DataFrame using the 'time' column.

    Parameters:
    dataframe (pd.DataFrame): DataFrame containing dF/F% values, stimuli information, and time column.

    Returns:
    pd.DataFrame: DataFrame with columns 'stimuli', 'start_time_index', 'stop_time_index', 'start_time', 'stop_time'.
    """
    # Identifying columns related to stimuli (degrees_x)
    stimuli_columns = [col for col in dataframe.columns if 'degrees_' in col]

    # Creating a new DataFrame to store the extracted timestamps
    stimuli_df = pd.DataFrame(columns=["stimuli", "start_time_index", "stop_time_index", "start_time", "stop_time"])

    # Iterating over each stimuli column to extract start and stop times
    for stimuli in stimuli_columns:
        # Identifying the start (rising edge) and stop (falling edge) of the stimuli
        start_indices = dataframe.index[dataframe[stimuli].diff() == 1].tolist()
        stop_indices = dataframe.index[dataframe[stimuli].diff() == -1].tolist()

        # Adjust for stimuli starting at the beginning or ending at the end of the recording
        if start_indices and start_indices[0] > stop_indices[0]:
            start_indices.insert(0, 0)
        if stop_indices and (len(stop_indices) < len(start_indices)):
            stop_indices.append(dataframe.shape[0] - 1)

        # Extracting time values for each start and stop index
        for start, stop in zip(start_indices, stop_indices):
            start_time = dataframe.loc[start, 'time']
            stop_time = dataframe.loc[stop, 'time']
            stimuli_df = stimuli_df.append({"stimuli": stimuli, 
                                            "start_time_index": start, 
                                            "stop_time_index": stop, 
                                            "start_time": start_time, 
                                            "stop_time": stop_time}, 
                                           ignore_index=True)

    return stimuli_df

#%%

def write_dff_to_nwb(csv_file_path, nwb_file_path):
    """
    Filter and write data from a CSV file to an NWB file.

    Args:
        csv_file_path (str): The path to the CSV file.
        nwb_file_path (str): The path to the NWB file.

    Returns:
        None
    """
    tables.file._open_files.close_all()

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file_path)
    dataf = df[[col for col in df.columns if str(col).isdigit()]]
    dataf = df.to_numpy().T  # roiss as rows & frames as columns

    timestamps = np.array(df['time'])

    # Extract the IDs you want to filter by
    #ids_to_filter = df['id'].values

    # Load the NWB file
    with NWBHDF5IO(nwb_file_path, 'r+') as io:
        nwbfile = io.read()

        ophys_module = nwbfile.get_processing_module('ophys')

        fluorescence = ophys_module.get_data_interface('Fluorescence')

        # Create the new RoiResponseSeries
        dff_series = RoiResponseSeries(name='DfOverF',
                                    data=dataf,
                                    unit='Relative Change',  # Replace with appropriate unit
                                    timestamps=timestamps,
                                    rois=fluorescence.roi_response_series['RoiResponseSeries'].rois)  # Assuming existing ROIs can be reused

        # Create the new RoiResponseSeries
        dff_series_F = RoiResponseSeries(name='fDfOverF',
                                    data=dataf,
                                    unit='Relative Change',  # Replace with appropriate unit
                                    timestamps=timestamps,
                                    rois=fluorescence.roi_response_series['RoiResponseSeries'].rois)  # Assuming existing ROIs can be reused

        # Add the new RoiResponseSeries to the Fluorescence module
        #fluorescence.add_roi_response_series(roi_response_series=dff_series_F)


        # Write the changes back to th    # Create a DfOverF object with the filtered data
        df_over_f = DfOverF(roi_response_series=dff_series)

        ophys_module = nwbfile.processing['ophys']

        #ophys_module.add_data_interface(df_over_f)
        
        io.write(nwbfile)

#%%

def load_mat_file(mat_file_path):
    with h5py.File(mat_file_path, 'r') as file:
        cell_to_index_map = file['cell_registered_struct/cell_to_index_map'][:]
        return cell_to_index_map.T - 1  # Adjust for Python's 0-indexing

#%%
def rearrange_rows(df, order):
    # Convert the order array to a list of integers
    order = list(map(int, order))
    
    # Check if all elements in order exist in the DataFrame's index
    if set(order).issubset(set(df.index)):
        # Reindex the DataFrame
        df = df.reindex(order)
    else:
        print("Some indices in the order array do not exist in the DataFrame.")
    
    return df

#%%
# Create a function to select a column from cell_to_index_map and map it to filtered data
def get_the_match_columns_names(F_npy_path, iscell_npy_path, session2_F_file, session2_iscell_file, mat_file_path):
    """
    Processes the given .npy and .mat files and creates a DataFrame with additional columns 'match1' and 'match2', 
    with specific conditions on filling values.

    Args:
    - F_npy_path (str): Path to the F.npy file.
    - iscell_npy_path (str): Path to the iscell.npy file.
    - mat_file_path (str): Path to the .mat file.

    Returns:
    - pd.DataFrame: DataFrame with columns 'Findex', 'iscell_col1', 'iscell_col2', 'iscell_index', 'match1', and 'match2',
      with specific conditions for 'match1' and 'match2'.
    """
    # Load the .npy files
    F_data = np.load(F_npy_path)
    iscell_data = np.load(iscell_npy_path)

    # Create a DataFrame with a single column ('Findex') for the indices of F.npy
    F_indices = np.arange(len(F_data))
    df = pd.DataFrame(F_indices, columns=['Findex'])

    # Add the columns of iscell.npy to the DataFrame
    df = pd.concat([df, pd.DataFrame(iscell_data, columns=['iscell_col1', 'iscell_col2'])], axis=1)

    # Calculate and add the 'iscell_index1' column with NaN for non-1 entries in iscell_col1
    iscell_index = np.where(iscell_data[:, 0] == 1, np.cumsum(iscell_data[:, 0]) - 1, np.nan)
    df['iscell_index1'] = iscell_index

    # Load the .mat file and extract the columns of cell_to_index_map, adjusting for the specific condition
    with h5py.File(mat_file_path, 'r') as file:
        cell_to_index_map = file['cell_registered_struct/cell_to_index_map'][()]
    match1 = cell_to_index_map[0, :len(F_data)] - 1  # Subtract 1 from each element
    match2 = cell_to_index_map[1, :len(F_data)] - 1  # Same for the second column

    # Applying condition for match1 and match2
    df['match1'] = np.where((match1 >= 0) & (match2 >= 0), match1, np.nan)
    df['match2'] = np.where((match1 >= 0) & (match2 >= 0), match2, np.nan)

    df1 = df.copy()

    df2 = process_F_and_iscell(session2_F_file, session2_iscell_file)

    combined_df = pd.concat([df1, df2], axis=1)

    seived_df = combined_df[['iscell_index1', 'match1', 'match2', 'iscell_index2']].dropna()
    # if match1's values are in iscell_index1 and match2 in iscell_index2
    #seived_df = seived_df[seived_df['match1'].isin(seived_df['iscell_index1']) & seived_df['match2'].isin(seived_df['iscell_index2'])]

    return seived_df

#ignore this but do not delete:: helper function for get_the_match_columns_names
#%%
def process_F_and_iscell(F_npy_path, iscell_npy_path):
    """
    Processes the given F.npy and iscell.npy files and creates a DataFrame with specific columns.

    Args:
    - F_npy_path (str): Path to the F.npy file.
    - iscell_npy_path (str): Path to the iscell.npy file.

    Returns:
    - pd.DataFrame: DataFrame with columns 'Findex', 'iscell_col1', 'iscell_col2', and 'iscell_index'.
    """
    # Load the .npy files
    F_data = np.load(F_npy_path)
    iscell_data = np.load(iscell_npy_path)

    # Create a DataFrame with a single column ('Findex') for the indices of F.npy
    F_indices = np.arange(len(F_data))
    df = pd.DataFrame(F_indices, columns=['Findex2'])

    # Add the columns of iscell.npy to the DataFrame
    df = pd.concat([df, pd.DataFrame(iscell_data, columns=['iscell2', 'iscell21'])], axis=1)

    # Calculate and add the 'iscell_index' column with NaN for non-1 entries in iscell_col1
    iscell_index = np.where(iscell_data[:, 0] == 1, np.cumsum(iscell_data[:, 0]) - 1, np.nan)
    df['iscell_index2'] = iscell_index

    return df

#%%
#given the df from of the matches get the meatched neurons from the csv 
def filter_dffcsv_by_matches(csv_location, df_of_list, session=1):
    # Import CSV file
    df = pd.read_csv(csv_location)
    session = str(session)
    column_name = 'iscell_index' + session
    list_to_filter = df_of_list[column_name].tolist()
    list_to_filter = list(map(str, map(int, list_to_filter)))
    # Drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # List of columns to always keep
    keep_columns = ['time', 'degrees_180', 'degrees_225', 'degrees_135', 'degrees_0', 'degrees_315', 'degrees_90', 'degrees_270', 'degrees_45', 'speed', 'direction', 'pupil_size']

    # List of columns to filter out
    filter_columns = list_to_filter  # replace with your list of columns

    # Filter out columns
    df = df[[col for col in df.columns if col in filter_columns or col in keep_columns]]

    order_column_name = 'match' + session
    order_columns = df_of_list[order_column_name].tolist()
    order_columns = list(map(str, map(int, order_columns)))

    #check if columns in order_columns are in df
    order_columns = [col for col in order_columns if col in df.columns]

    #print columns not found in df
    not_found_columns = [col for col in order_columns if col not in df.columns]
    if not_found_columns:
        print("Columns not found in DataFrame: ", not_found_columns)
        
    # Remove rows where match2 is not in iscell_index2
    #df = df[df['match2'].isin(df['iscell_index2'])]


    df = df[order_columns + keep_columns]

    return df

# %%
###################################
#plots


def plot_raster_subset(data, num_neurons=None, start_time=None, end_time=None, vmin=None, vmax=None):
    """
    Plots a raster plot of neural activity for all neurons or a subset of the first n neurons.

    :param data: DataFrame containing neural activity with neurons as columns and time as rows.
    :param num_neurons: The number of first neurons to plot (if None, plots all neurons).
    :param start_time: Time to start the plot (optional).
    :param end_time: Time to end the plot (optional).
    """
    # Define the columns to exclude
    excluded_columns = ['pupil_size', 'speed', 'direction', 'time']
    excluded_columns += [col for col in data.columns if 'Unnamed' in str(col)]
    # Get a list of all other columns that aren't in the excluded list
    neuron_columns = [col for col in data.columns if col not in excluded_columns and str(col).isdigit()]

    # Limit the number of neurons to plot if specified
    if num_neurons is None:
        num_neurons = len(neuron_columns)
    neuron_columns_to_plot = neuron_columns[:num_neurons]

    # If start or end time is not specified, use the entire range of time
    if start_time is None:
        start_time = data['time'].min()
    if end_time is None:
        end_time = data['time'].max()

    # Select the subset of data to plot
    time_mask = (data['time'] >= start_time) & (data['time'] <= end_time)
    neural_data = data.loc[time_mask, neuron_columns_to_plot]

    # Determine color scale range based on filtered neuron data
    #vmin, vmax = neural_data.min().min(), neural_data.quantile(0.99).max()
    if vmin is None or vmax is None:
        vmin, vmax = neural_data.min().min(), neural_data.quantile(0.99).max()
    else:
        vmin = vmin
        vmax = vmax

    # Plot the raster
    fig, ax = plt.subplots(figsize=(15, 10))
    cax = ax.imshow(neural_data.T, aspect='auto', origin='lower', cmap='viridis', vmin=vmin, vmax=vmax,
                    extent=[start_time, end_time, 0, num_neurons])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron Index')
    ax.set_title('Raster Plot of ΔF/F% Activity')
    fig.colorbar(cax, label='ΔF/F%')
    plt.show()
#%%
#plot the raster plots with plotly
def plot_raster_plotly(data, num_neurons=None, start_time=None, end_time=None, color_scale_min = None, color_scale_max = None):
    """
    Plots a raster plot of neural activity using Plotly for an interactive plot.

    :param data: DataFrame containing neural activity with neurons as columns and time as rows.
    :param num_neurons: The number of first neurons to plot (if None, plots all neurons).
    :param start_time: Time to start the plot (optional).
    :param end_time: Time to end the plot (optional).
    """
    # Define the columns to exclude
    excluded_columns = ['pupil_size', 'speed', 'direction', 'time']
    excluded_columns += [col for col in data.columns if 'Unnamed' in str(col)]
    # Get a list of all other columns that aren't in the excluded list
    neuron_columns = [col for col in data.columns if col not in excluded_columns and str(col).isdigit()]

    # Limit the number of neurons to plot if specified
    if num_neurons is None:
        num_neurons = len(neuron_columns)
    neuron_columns_to_plot = neuron_columns[:num_neurons]

    # If start or end time is not specified, use the entire range of time
    if start_time is None:
        start_time = data['time'].min()
    if end_time is None:
        end_time = data['time'].max()

    # Select the subset of data to plot
    time_mask = (data['time'] >= start_time) & (data['time'] <= end_time)
    neural_data = data.loc[time_mask, neuron_columns_to_plot]

    # Determine color scale range based on filtered neuron data
    if color_scale_min is None or color_scale_max is None:
        # color_scale_min = neural_data.min().min()
        # color_scale_max = neural_data.quantile(0.99).max()
        color_scale_min = 0
        color_scale_max = 100
    else:
        color_scale_min = color_scale_min
        color_scale_max = color_scale_min

    # Create the figure with the range_color argument
    fig = px.imshow(neural_data.T,
                    labels=dict(x="Time (s)", y="Neuron Index", color="ΔF/F%"),
                    x=data['time'][time_mask],
                    aspect="auto",
                    color_continuous_scale='Viridis',
                    origin='lower',
                    range_color=(color_scale_min, color_scale_max),  # Set the color scale range
                    width=750,
                    height=500
                    )
    fig.update_layout(title='Raster Plot of ΔF/F% Activity', xaxis_title='Time (s)', yaxis_title='Neuron Index')
    fig.update_xaxes(side="bottom")
    return fig

#%%
# code for the trasnsients

def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def calculate_transient_metrics(data, neuron, baseline_range):
    """
    Calculate metrics for calcium transients of a given neuron.

    :param data: DataFrame with time as rows and neurons as columns.
    :param neuron: Neuron index or name.
    :param baseline_range: Tuple indicating the start and end of the baseline period.
    :return: Dictionary with CV, tau, and rise time.
    """
    metrics = {}

    # Calculate CV of baseline
    baseline_data = data.loc[baseline_range[0]:baseline_range[1], neuron]
    cv_baseline = np.std(baseline_data) / np.mean(baseline_data)
    metrics['cv_baseline'] = cv_baseline

    # Identify calcium transients
    # You may need to adjust the threshold based on your specific data
    threshold = np.mean(baseline_data) + 2 * np.std(baseline_data)
    transient_indices = data[neuron] > threshold

    # Find transient peaks and calculate rise time and tau
    rise_times = []
    decay_constants = []
    for start, end in find_transient_periods(transient_indices):
        transient_data = data.loc[start:end, neuron]
        peak_time = transient_data.idxmax()
        rise_time = peak_time - start
        rise_times.append(rise_time)

        # Fit exponential decay to the decay phase
        decay_phase = transient_data.loc[peak_time:end]
        decay_time = decay_phase.index - peak_time
        popt, _ = curve_fit(exponential_decay, decay_time, decay_phase, p0=(1, 1e-2, 1), maxfev=20000)
        decay_constants.append(popt[1])  # b in the exponential decay function is the decay constant

    metrics['mean_rise_time'] = np.mean(rise_times)
    metrics['mean_decay_constant'] = np.mean(decay_constants)

    return metrics

def find_transient_periods(transient_indices):
    """
    Find start and end indices of transient periods.

    :param transient_indices: Boolean series indicating where transients occur.
    :return: List of tuples with start and end indices.
    """
    transient_periods = []
    start = None
    for i, is_transient in enumerate(transient_indices):
        if is_transient and start is None:
            start = i
        elif not is_transient and start is not None:
            transient_periods.append((start, i))
            start = None
    if start is not None:
        transient_periods.append((start, len(transient_indices)))
    return transient_periods


#%%
#heatmap of calcicum intensity for orientaions 
def plot_orientation_heatmap_ordered(data, response_type='mean', vmin=None, vmax=None):
    """
    Plots a heatmap of the mean or peak neural response to each orientation in a specified order.

    :param data: DataFrame containing neural activity and stimulus presentation information.
    :param response_type: 'mean' for the mean response, 'peak' for the peak response.
    """
    # Dictionary to rename the orientation columns
    orientation_rename = {
        'degrees_225': 'degrees_-45',
        'degrees_270': 'degrees_-90',
        'degrees_315': 'degrees_-135',
        'degrees_0': 'degrees_-180'
    }
    data = data.rename(columns=orientation_rename)

    # Filter out columns that are not neural data based on their names
    neural_data_columns = [col for col in data.columns if isinstance(col, int) or col.isdigit()]

    # Define the order of orientations
    ordered_orientations = [
        'degrees_45', 'degrees_-45', 'degrees_90', 'degrees_-90',
        'degrees_135', 'degrees_-135', 'degrees_180', 'degrees_-180'
    ]

    # Calculate the mean or peak response for each neuron to each orientation
    responses = []
    for orientation in ordered_orientations:
        # Find the frames where this orientation was presented
        presentation_frames = data[data[orientation] == 1][neural_data_columns]

        if response_type == 'peak':
            # Calculate the peak response during the presentation of this orientation
            orientation_response = presentation_frames.max()
        else:
            # Calculate the mean response during the presentation of this orientation
            orientation_response = presentation_frames.mean()

        responses.append(orientation_response)

    # Construct a DataFrame from the responses
    response_matrix = pd.DataFrame(responses, index=ordered_orientations).T

    # Determine the color scale range based on the response matrix
    #vmin, vmax = response_matrix.min().min(), response_matrix.max().max()
    #vmin, vmax = response_matrix.min().min(), response_matrix.quantile(0.99).max()
    if vmin is None or vmax is None:
        vmin, vmax = response_matrix.min().min(), response_matrix.max().max()
    else:
        vmin = vmin
        vmax = vmax

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(response_matrix, annot=False, fmt=".2f", cmap='viridis', vmin=vmin, vmax=vmax, cbar_kws={'label': f'{response_type.capitalize()} ΔF/F%'})
    plt.xlabel('Orientation')
    plt.ylabel('Neuron Index')
    plt.title(f'ΔF/F% Responses to Orientations ({response_type.capitalize()})')
    plt.xticks(np.arange(len(ordered_orientations)) + 0.5, ordered_orientations, rotation=45)
    plt.show()


#%%

def plot_orientation_heatmap_plotly(data, response_type='mean', vmin=None, vmax=None):
    """
    Plots an interactive heatmap of the mean or peak neural response to each orientation using Plotly.

    :param data: DataFrame containing neural activity and stimulus presentation information.
    :param response_type: 'mean' for the mean response, 'peak' for the peak response.
    """
    # Dictionary to rename the orientation columns
    orientation_rename = {
        'degrees_225': 'degrees_-45',
        'degrees_270': 'degrees_-90',
        'degrees_315': 'degrees_-135',
        'degrees_0': 'degrees_-180'
    }
    data = data.rename(columns=orientation_rename)

    # Filter out columns that are not neural data based on their names
    neural_data_columns = [col for col in data.columns if isinstance(col, int) or col.isdigit()]

    # Define the order of orientations
    ordered_orientations = [
        'degrees_45', 'degrees_-45', 'degrees_90', 'degrees_-90',
        'degrees_135', 'degrees_-135', 'degrees_180', 'degrees_-180'
    ]

    # Calculate the mean or peak response for each neuron to each orientation
    responses = []
    for orientation in ordered_orientations:
        # Find the frames where this orientation was presented
        presentation_frames = data[data[orientation] == 1][neural_data_columns]

        if response_type == 'peak':
            # Calculate the peak response during the presentation of this orientation
            orientation_response = presentation_frames.max()
        else:
            # Calculate the mean response during the presentation of this orientation
            orientation_response = presentation_frames.mean()

        responses.append(orientation_response)

    # Construct a DataFrame from the responses
    response_matrix = pd.DataFrame(responses, index=ordered_orientations).T

    # Determine the color scale range based on the response matrix
    #vmin, vmax = response_matrix.min().min(), response_matrix.max().max()
    #vmin, vmax = response_matrix.min().min(), response_matrix.quantile(0.99).max()
    if vmin is None or vmax is None:
        vmin, vmax = response_matrix.min().min(), response_matrix.max().max()
    else:
        vmin = vmin
        vmax = vmax
    
    # Create the figure using Plotly
    fig = go.Figure(data=go.Heatmap(
        z=response_matrix,
        x=ordered_orientations,
        y=response_matrix.index,
        colorscale='Viridis',
        colorbar=dict(title=f'{response_type.capitalize()} ΔF/F%'),
        zmin=vmin,  # Set the minimum value for the color scale
        zmax=vmax,  # Set the maximum value for the color scale
    ))

    # Update the layout
    fig.update_layout(
        title=f'ΔF/F% Responses to Orientations ({response_type.capitalize()} Response)',
        xaxis_title='Orientation',
        yaxis_title='Neuron Index',
        width=750,
        height=600
    )

    fig.show()
    return fig

#%%
def plot_polar_tuning_curve_with_response(data, neuron_index=None, response_type='mean'):
    """
    Plots a polar orientation tuning curve for a specified neuron or a random neuron,
    with a custom sequence of orientations on the polar plot and closes the loop.
    It allows choosing between the mean and peak response.

    :param data: DataFrame containing neural activity and stimulus presentation information.
    :param neuron_index: Index of the neuron to plot. If None, a random neuron is chosen.
    :param response_type: 'mean' for the mean response, 'peak' for the peak response.
    """
    # Custom sequence of orientations and their corresponding angles on the polar plot
    orientation_angles = {
        'degrees_45': np.deg2rad(45),
        'degrees_90': np.deg2rad(90),
        'degrees_135': np.deg2rad(135),
        'degrees_180': np.deg2rad(180),
        'degrees_-45': np.deg2rad(225),
        'degrees_-90': np.deg2rad(270),
        'degrees_-135': np.deg2rad(315),
        'degrees_-180': np.deg2rad(0)
    }

    # Rename columns according to the new orientation names
    data = data.rename(columns={
        'degrees_225': 'degrees_-45',
        'degrees_270': 'degrees_-90',
        'degrees_315': 'degrees_-135',
        'degrees_0': 'degrees_-180'
    })

    # If no neuron index is specified, choose a random neuron
    if neuron_index is None:
        neuron_index = np.random.choice(range(1, data.shape[1] - len(orientation_angles) - 3))
    
    # Calculate the response (mean or peak) for the selected neuron to each orientation
    responses = []
    for orientation, angle in orientation_angles.items():
        # Find the frames where this orientation was presented
        presentation_frames = data[data[orientation] == 1]
        # Calculate the response during the presentation of this orientation
        if response_type == 'peak':
            response = presentation_frames.iloc[:, neuron_index].max()
        else:
            response = presentation_frames.iloc[:, neuron_index].mean()
        responses.append(response)
    
    # Create a polar plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Plot the tuning curve with the custom sequence of orientations and close the loop
    angles = list(orientation_angles.values()) + [np.deg2rad(45)]
    responses = responses + [responses[0]]
    
    # Plot the tuning curve
    ax.plot(angles, responses, 'b-')
    ax.fill(angles, responses, 'b', alpha=0.1)
    
    # Set the orientation of the 0 degree to the top
    ax.set_theta_zero_location('N')
    # Set the direction of angles to be clockwise
    ax.set_theta_direction(-1)
    
    # Set the labels for the angles
    ax.set_xticks(list(orientation_angles.values()))
    ax.set_xticklabels([orientation.replace('degrees_', '') + '°' for orientation in orientation_angles.keys()])
    
    # Add title and labels
    ax.set_title(f'Polar Orientation Tuning Curve (Neuron {neuron_index}) - {response_type.capitalize()} Response')
    plt.show()

#%%
#polar tuning curve for a single neuron
def plotly_polar_tuning_curve(data, neuron_index=None, response_type='mean'):
    """
    Plots a polar orientation tuning curve using Plotly for a specified neuron or a random neuron,
    with a custom sequence of orientations and the option to choose between mean and peak response.

    :param data: DataFrame containing neural activity and stimulus presentation information.
    :param neuron_index: Index of the neuron to plot. If None, a random neuron is chosen.
    :param response_type: 'mean' for the mean response, 'peak' for the peak response.
    """
    # Custom sequence of orientations and their corresponding angles on the polar plot
    orientation_angles = {
        'degrees_45': 45,
        'degrees_90': 90,
        'degrees_135': 135,
        'degrees_180': 180,
        'degrees_-45': 225,
        'degrees_-90': 270,
        'degrees_-135': 315,
        'degrees_-180': 0
    }

    # Rename columns according to the new orientation names
    data = data.rename(columns={
        'degrees_225': 'degrees_-45',
        'degrees_270': 'degrees_-90',
        'degrees_315': 'degrees_-135',
        'degrees_0': 'degrees_-180'
    })

    # If no neuron index is specified, choose a random neuron
    if neuron_index is None:
        neuron_index = np.random.choice(range(1, data.shape[1] - len(orientation_angles) - 3))
    
    # Calculate the response (mean or peak) for the selected neuron to each orientation
    responses = []
    for orientation, angle in orientation_angles.items():
        # Find the frames where this orientation was presented
        presentation_frames = data[data[orientation] == 1]
        # Calculate the response during the presentation of this orientation
        if response_type == 'peak':
            response = presentation_frames.iloc[:, neuron_index].max()
        else:
            response = presentation_frames.iloc[:, neuron_index].mean()
        responses.append(response)
    
    # Include the first point at the end to close the loop
    angles = list(orientation_angles.values()) + [orientation_angles['degrees_45']]
    responses = responses + [responses[0]]
    
    # Create a polar plot using Plotly
    fig = go.Figure(go.Scatterpolar(
        r=responses,
        theta=angles,
        fill='toself',
        name=f'Neuron {neuron_index}'
    ))

    # Update layout to adjust the appearance
    fig.update_layout(
        title=f'Polar Orientation Tuning Curve (Neuron {neuron_index}) - {response_type.capitalize()} Response',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[min(responses), max(responses)]
            ),
            angularaxis=dict(
                direction='clockwise',
                thetaunit='degrees',
                dtick=45
            )
        ),
        showlegend=False
    )
    
    fig.show()
    return fig


#%%
def plotly_polar_tuning_curve_grid(data, specific_neurons=None, num_neurons=4, response_type='mean'):
    # Custom sequence of orientations and their corresponding angles on the polar plot
    orientation_angles = {
        'degrees_45': 45,
        'degrees_90': 90,
        'degrees_135': 135,
        'degrees_180': 180,
        'degrees_-45': 225,
        'degrees_-90': 270,
        'degrees_-135': 315,
        'degrees_-180': 0
    }

    # Rename columns according to the new orientation names
    data = data.rename(columns={
        'degrees_225': 'degrees_-45',
        'degrees_270': 'degrees_-90',
        'degrees_315': 'degrees_-135',
        'degrees_0': 'degrees_-180'
    })

    # Determine neurons to plot
    if specific_neurons is not None:
        neurons_to_plot = specific_neurons
    else:
        neurons_to_plot = np.random.choice(range(1, data.shape[1] - len(orientation_angles) - 3), num_neurons, replace=False)
    
    # Calculate number of rows and columns for the subplot grid
    num_rows = int(np.ceil(len(neurons_to_plot) ** 0.5))
    num_cols = int(np.ceil(len(neurons_to_plot) / num_rows))

    # Create subplots
    fig = make_subplots(rows=num_rows, cols=num_cols, specs=[[{'type': 'polar'}] * num_cols] * num_rows)

    for i, neuron_index in enumerate(neurons_to_plot, start=1):
        # Calculate the response for each orientation
        responses = []
        for orientation, angle in orientation_angles.items():
            presentation_frames = data[data[orientation] == 1]
            if response_type == 'peak':
                response = presentation_frames.iloc[:, neuron_index].max()
            else:
                response = presentation_frames.iloc[:, neuron_index].mean()
            responses.append(response)

        # Include the first point at the end to close the loop
        angles = list(orientation_angles.values()) + [orientation_angles['degrees_45']]
        responses = responses + [responses[0]]

        # Determine row and column for the subplot
        row = int(np.ceil(i / num_cols))
        col = i - (row - 1) * num_cols

        # Add the plot to the subplot
        fig.add_trace(go.Scatterpolar(
            r=responses,
            theta=angles,
            fill='toself',
            name=f'Neuron {neuron_index}'
        ), row=row, col=col)

    # Update layout
    fig.update_layout(
        title='Polar Orientation Tuning Curves',
        height=300 * num_rows,
        width=300 * num_cols,
        showlegend=False
    )

    fig.show()
    return fig



#%%
def gaussian(x, A, x0, sigma, B):
    return A * np.exp(-(x - x0)**2 / (2 * sigma**2)) + B





#%%

#rank neurons by SNR

def rank_neurons_by_snr(df, snr_threshold=1):
    """
    Rank neurons by signal-to-noise ratio (SNR) and filter out neurons below a given threshold.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the neuron data.
        snr_threshold (float, optional): The threshold SNR value. Neurons with SNR below this value will be filtered out. Defaults to 1.

    Returns:
        Tuple[pandas.Series, pandas.Index]: A tuple containing two objects:
            - ranked_neurons (pandas.Series): A Series object containing the ranked neurons based on SNR.
            - filtered_ranked_neurons (pandas.Index): An Index object containing the indices of the ranked neurons that have SNR greater than the threshold.
    """
    # Load the data

    neuron_columns = [col for col in df.columns if isinstance(col, int) or col.isdigit()]
    neuron_data = df[neuron_columns]
    # Calculating SNR for each neuron (Mean of signal / Standard deviation of noise)
    # Here, the entire recording is assumed to be the signal
    snr = neuron_data.mean() / neuron_data.std()

    # Ranking neurons based on SNR (Higher SNR means higher rank)
    ranked_neurons = snr.sort_values(ascending=False)

    # Return the ranked neurons columns that are greater than 1
    filtered_ranked_neurons = snr[snr > snr_threshold].index

    non_neuron_columns = df.columns.difference(neuron_columns)

    ranked_columns_list = filtered_ranked_neurons.append(non_neuron_columns)
    print(ranked_columns_list)
    df_only_ranked_neurons = df[ranked_columns_list]

    return df_only_ranked_neurons, ranked_columns_list, filtered_ranked_neurons


#find preffered orientation of the each neuron first

#find the othrogonal oreintation of each neuron

#
#%%
def fit_and_plot_gaussian_tuning_curves(data, n_neurons=None, response_type='mean'):
    """
    Fits a Gaussian tuning curve to the responses of the first n neurons and plots them.

    :param data: DataFrame containing neural activity and stimulus presentation information.
    :param n_neurons: Number of neurons to plot. If None, fits all neurons.
    :param response_type: 'mean' or 'peak' to determine the type of response to use for fitting.
    """
    
    # Rename columns according to the new orientation names
    data = data.rename(columns={
        'degrees_225': 'degrees_-45',
        'degrees_270': 'degrees_-90',
        'degrees_315': 'degrees_-135',
        'degrees_0': 'degrees_-180'
    })
   
    # Custom sequence of orientations and their corresponding angles   
    orientation_angles = {
        'degrees_45': np.deg2rad(45),
        'degrees_90': np.deg2rad(90),
        'degrees_135': np.deg2rad(135),
        'degrees_180': np.deg2rad(180),
        'degrees_-45': np.deg2rad(225),
        'degrees_-90': np.deg2rad(270),
        'degrees_-135': np.deg2rad(315),
        'degrees_-180': np.deg2rad(0)
    }
    
    # If n_neurons is not specified, fit all neurons
    # if n_neurons is None:
    #     n_neurons = len(data.columns) - len(orientation_angles) - 3  # Subtract non-neuron columns
    
    neural_data = [col for col in data.columns if isinstance(col, int) or col.isdigit()]

    if n_neurons is None:
        n_neurons = len([col for col in data.columns if isinstance(col, int) or col.isdigit()])

    # Fit the Gaussian tuning curve for each neuron and plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for i in range(n_neurons):
        neuron_responses = []
        # Get the response for each orientation
        for orientation, angle in orientation_angles.items():
            responses = data.loc[data[orientation] == 1, i]
            if response_type == 'peak':
                response = responses.max()
            else:  # default to mean if anything else is specified
                response = responses.mean()
            neuron_responses.append(response)

        # Perform the Gaussian fit
        try:
            popt, _ = curve_fit(gaussian, list(orientation_angles.values()), neuron_responses)
        except RuntimeError:
            # If the curve fitting does not converge, skip this neuron
            print(f"Could not fit a Gaussian for neuron {i}")
            continue
        
        # Generate data for the curve
        x_curve = np.linspace(min(list(orientation_angles.values())), max(list(orientation_angles.values())), 100)
        y_curve = gaussian(x_curve, *popt)
        
        # Convert radians to degrees for plotting
        x_curve_degrees = np.rad2deg(x_curve)

        # Plot the fitted curve
        ax.plot(x_curve_degrees, y_curve, label=f'Neuron {i+1}')
    
    # Setting the xticks to match the orientation angles in degrees
    ax.set_xticks(np.rad2deg(list(orientation_angles.values())))
    ax.set_xticklabels([str(np.rad2deg(angle)) + '°' for angle in list(orientation_angles.values())])

    # Adding labels and title
    ax.set_xlabel('Orientation (degrees)')
    ax.set_ylabel('Response (a.u.)')
    ax.set_title('Gaussian Tuning Curves of Neurons')
    ax.legend(loc='upper right')

    plt.show()

#%%
#Plot tuning curve normalized
# Adjust the function to remove the extra space between -45 and 45 on the x-axis and restore the previous axis limits
def plot_tuning_curve(aligned_df, neuron_index=None):
    # If no specific neuron index is provided, select a random one from the numerically named columns
    if neuron_index is None:
        neuron_index = np.random.choice([col for col in aligned_df.columns if col.isdigit()])
    
    # Rename 'degrees_0' to 'degrees_-180' to match the request
    # if 'degrees_0' in aligned_df.columns:
    #     aligned_df = aligned_df.rename(columns={'degrees_0': 'degrees_-180'})
    
    # Define the order of orientations based on the requested order, excluding 0 degrees
    ordered_orientations = [ 'degrees_225', 'degrees_270', 'degrees_315','degrees_0', 
                            'degrees_45', 'degrees_90', 'degrees_135', 'degrees_180']
    
    # Extract the responses for the ordered orientations
    orientation_degrees = np.array([-180, -135, -90, -45, 45, 90, 135, 180])
    responses = aligned_df[ordered_orientations].loc[:, ordered_orientations].mean().values
    
    # Normalize the responses to the preferred orientation
    max_response = responses.max()
    normalized_responses = responses / max_response
    
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(orientation_degrees, normalized_responses, 'o-', label=f'Neuron {neuron_index}')
    
    # Aesthetics
    plt.xlabel('Stimulus Orientation (deg)')
    plt.ylabel('Normalized Response')
    plt.title(f'Neuron {neuron_index} Tuning Curve')
    plt.legend()
    
    # Define x-axis ticks to match the orientations without extra spaces, except for the space between -45 and 45
    x_ticks = [-180, -135, -90, -45, 45, 90, 135, 180]
    plt.xticks(x_ticks)
    
    plt.grid(True)
    
    # Show plot
    plt.show()
    
    return neuron_index  # Return the selected neuron index for reference

#%%




#%%

#psth calculation from the dfe
def calculate_and_plot_psth(data, pre_event_time, post_event_time, bin_size):
    """
    Calculate and plot the Peristimulus Time Histogram (PSTH) for each orientation.

    :param data: DataFrame containing neural activity, time, and orientation data.
    :param pre_event_time: Time before the stimulus onset to include in the PSTH (in seconds).
    :param post_event_time: Time after the stimulus onset to include in the PSTH (in seconds).
    :param bin_size: Size of each bin in the PSTH (in seconds).
    """
    orientations = [45, 90, 135, 180, -45, -90, -135, -180]
    time = data['time']
    bin_edges = np.arange(-pre_event_time, post_event_time + bin_size, bin_size)
    
    # Create a figure for the PSTH plots
    fig, axes = plt.subplots(len(orientations), 1, figsize=(10, 20), sharex=True)
    
    for i, orientation in enumerate(orientations):
        # Find the times when the orientation was presented
        stimulus_times = time[data[f'degrees_{orientation}'] == 1]
        
        # Initialize an array to hold the binned responses
        binned_responses = np.zeros(len(bin_edges) - 1)
        
        # For each stimulus time, bin the neural responses
        for stimulus_time in stimulus_times:
            # Find the relevant time window
            window_mask = (time >= stimulus_time - pre_event_time) & (time < stimulus_time + post_event_time)
            window_times = time[window_mask]
            window_responses = data.loc[window_mask, data.columns.difference(['time', 'pupil_size', 'speed', 'direction'])]
            
            # Bin the responses and sum them up
            digitized = np.digitize(window_times - stimulus_time, bin_edges) - 1
            for bin_index in range(len(binned_responses)):
                binned_responses[bin_index] += window_responses.iloc[digitized == bin_index].sum().sum()
        
        # Normalize the responses
        binned_responses /= len(stimulus_times)
        
        # Plot the PSTH for this orientation
        axes[i].bar(bin_edges[:-1], binned_responses, width=bin_size, align='edge')
        axes[i].set_title(f'Orientation {orientation}°')
        axes[i].set_ylabel('Response (a.u.)')
    
    axes[-1].set_xlabel('Time relative to stimulus onset (s)')
    plt.tight_layout()
    plt.show()


#%%

# def calculate_and_plot_psth(data, pre_event_time, post_event_time, bin_size):
#     # Updated orientation names to match your dataset
#     orientations = {
#         'degrees_45': 45, 'degrees_90': 90, 'degrees_135': 135, 'degrees_180': 180, 
#         'degrees_225': -45, 'degrees_270': -90, 'degrees_315': -135, 'degrees_0': -180
#     }

#     time = data['time']
#     bin_edges = np.arange(-pre_event_time, post_event_time + bin_size, bin_size)

#     fig, axes = plt.subplots(len(orientations), 1, figsize=(10, 20), sharex=True)
    
#     for i, (orientation, angle) in enumerate(orientations.items()):
#         stimulus_times = time[data[orientation] == 1]
#         binned_responses = np.zeros(len(bin_edges) - 1)

#         for stimulus_time in stimulus_times:
#             window_mask = (time >= stimulus_time - pre_event_time) & (time < stimulus_time + post_event_time)
#             window_times = time[window_mask]
#             window_responses = data.loc[window_mask, data.columns.difference(['time', 'pupil_size', 'speed', 'direction'])]

#             digitized = np.digitize(window_times - stimulus_time, bin_edges) - 1
#             for bin_index in range(len(binned_responses)):
#                 binned_responses[bin_index] += window_responses.iloc[digitized == bin_index].sum().sum()

#         binned_responses /= len(stimulus_times)
#         axes[i].bar(bin_edges[:-1], binned_responses, width=bin_size, align='edge')
#         axes[i].set_title(f'Orientation {angle}°')
#         axes[i].set_ylabel('Response (a.u.)')

#     axes[-1].set_xlabel('Time relative to stimulus onset (s)')
#     plt.tight_layout()
#     plt.show()

#%%
def calculate_and_plot_mean_trace(data, pre_event_time, post_event_time):
    """
    Calculate and plot the mean trace of neural activity around each stimulus presentation for each orientation.

    :param data: DataFrame containing neural activity, time, and orientation data.
    :param pre_event_time: Time before the stimulus onset to include in the trace (in seconds).
    :param post_event_time: Time after the stimulus onset to include in the trace (in seconds).
    """
    orientations = {
        'degrees_45': 45, 'degrees_90': 90, 'degrees_135': 135, 'degrees_180': 180, 
        'degrees_225': -45, 'degrees_270': -90, 'degrees_315': -135, 'degrees_0': -180
    }

    time = data['time']
    time_window = np.arange(-pre_event_time, post_event_time, np.median(np.diff(time)))

    fig, axes = plt.subplots(len(orientations), 1, figsize=(10, 20), sharex=True)

    for i, (orientation, angle) in enumerate(orientations.items()):
        stimulus_times = time[data[orientation] == 1]
        mean_responses = np.zeros(len(time_window))

        for stimulus_time in stimulus_times:
            window_mask = (time >= stimulus_time - pre_event_time) & (time < stimulus_time + post_event_time)
            window_responses = data.loc[window_mask, data.columns.difference(['time', 'pupil_size', 'speed', 'direction'])]
            aligned_responses = window_responses.set_index(time[window_mask] - stimulus_time)
            mean_responses += aligned_responses.mean(axis=1).reindex(time_window, fill_value=0)

        mean_responses /= len(stimulus_times)
        axes[i].plot(time_window, mean_responses)
        axes[i].set_title(f'Orientation {angle}°')
        axes[i].set_ylabel('Mean Response (a.u.)')

    axes[-1].set_xlabel('Time relative to stimulus onset (s)')
    plt.tight_layout()
    plt.show()

#%%
def calculate_and_plot_mean_trace_plotly_2(data, pre_event_time, post_event_time):
    orientations = {
        'degrees_45': 45, 'degrees_90': 90, 'degrees_135': 135, 'degrees_180': 180, 
        'degrees_225': -45, 'degrees_270': -90, 'degrees_315': -135, 'degrees_0': -180
    }

    time = data['time']
    time_window = np.arange(-pre_event_time, post_event_time, np.median(np.diff(time)))

    # Create a subplot for each orientation
    fig = make_subplots(rows=len(orientations), cols=1, shared_xaxes=True)

    for i, (orientation, angle) in enumerate(orientations.items(), start=1):
        stimulus_times = data[data[orientation] == 1]['time'].values
        neuron_columns = data.columns.difference(['time', 'pupil_size', 'speed', 'direction'])

        mean_responses = np.zeros((len(time_window), len(neuron_columns)))

        for stimulus_time in stimulus_times:
            window_mask = (time >= stimulus_time - pre_event_time) & (time < stimulus_time + post_event_time)
            window_time = time[window_mask]
            relative_time = window_time - stimulus_time

            for j, neuron in enumerate(neuron_columns):
                neuron_responses = data.loc[window_mask, neuron]
                for k, rel_time in enumerate(time_window):
                    # Find the index of the nearest time point to rel_time in relative_time
                    nearest_idx = np.argmin(np.abs(relative_time - rel_time))
                    mean_responses[k, j] += neuron_responses.iloc[nearest_idx]

        # Average across all stimulus occurrences
        mean_responses /= len(stimulus_times)

        # Average across all neurons and plot
        mean_neuron_responses = np.mean(mean_responses, axis=1)
        fig.add_trace(go.Scatter(x=time_window, y=mean_neuron_responses, mode='lines', name=f'Orientation {angle}°'), row=i, col=1)

    # Update layout
    fig.update_layout(height=2000, width=800, title_text="Mean Neural Response for Each Orientation", showlegend=False)
    fig.update_xaxes(title_text="Time relative to stimulus onset (s)", row=len(orientations), col=1)
    fig.update_yaxes(title_text="Mean Response (DF/F%)")

    fig.show()


#%%
def calculate_and_plot_mean_trace_plotly_vectorized(data, pre_event_time, post_event_time):
    orientations = {
        'degrees_45': 45, 'degrees_90': 90, 'degrees_135': 135, 'degrees_180': 180, 
        'degrees_225': -45, 'degrees_270': -90, 'degrees_315': -135, 'degrees_0': -180
    }

    time = data['time']
    time_window = np.arange(-pre_event_time, post_event_time, np.median(np.diff(time)))

    # Create a subplot for each orientation
    fig = make_subplots(rows=len(orientations), cols=1, shared_xaxes=True)

    neuron_columns = data.columns.difference(['time', 'pupil_size', 'speed', 'direction'])

    # Precompute the indices for all time windows
    all_indices = []
    for stimulus_time in data['time'][data['degrees_45'] == 1]:  # Use any orientation to compute indices
        window_mask = (time >= stimulus_time - pre_event_time) & (time < stimulus_time + post_event_time)
        relative_time = time[window_mask] - stimulus_time
        nearest_indices = np.argmin(np.abs(relative_time.values[:, None] - time_window), axis=0)
        all_indices.append(nearest_indices)

    for i, (orientation, angle) in enumerate(orientations.items(), start=1):
        stimulus_times = data[data[orientation] == 1]['time'].values
        mean_responses = np.zeros((len(time_window), len(neuron_columns)))

        for nearest_indices in all_indices:
            # Get responses for all neurons at these indices for each stimulus time
            window_mask = (time >= stimulus_times[0] - pre_event_time) & (time < stimulus_times[0] + post_event_time)
            neuron_responses = data.loc[window_mask, neuron_columns].values
            mean_responses += neuron_responses[nearest_indices, :]

        # Average across all stimulus occurrences
        mean_responses /= len(stimulus_times)

        # Average across all neurons and plot
        mean_neuron_responses = np.mean(mean_responses, axis=1)
        fig.add_trace(go.Scatter(x=time_window, y=mean_neuron_responses, mode='lines', name=f'Orientation {angle}°'), row=i, col=1)

    # Update layout
    fig.update_layout(height=2000, width=800, title_text="Mean Neural Response for Each Orientation", showlegend=False)
    fig.update_xaxes(title_text="Time relative to stimulus onset (s)", row=len(orientations), col=1)
    fig.update_yaxes(title_text="Mean Response (DF/F%)")

    fig.show()


def calculate_and_plot_mean_trace_plotly_vectorized_3(data, pre_event_time, post_event_time):
    orientations = ['degrees_45', 'degrees_90', 'degrees_135', 'degrees_180', 'degrees_225', 'degrees_270', 'degrees_315', 'degrees_0']
    time_window = np.arange(-pre_event_time, post_event_time, 1/30.3)

    # Create a subplot for each orientation
    fig = make_subplots(rows=len(orientations), cols=1, shared_xaxes=True)

    neuron_columns = data.columns.difference(['Unnamed: 0', 'speed', 'direction', 'pupil_size'] + orientations)

    for i, orientation in enumerate(orientations, start=1):
        stimulus_indices = data.index[data[orientation] == 1].tolist()
        mean_responses = np.zeros((len(time_window), len(neuron_columns)))

        for stimulus_index in stimulus_indices:
            # Get the window of frames around the stimulus
            start_index = max(stimulus_index - pre_event_time, 0)
            end_index = min(stimulus_index + post_event_time, len(data))
            window_responses = data.loc[start_index:end_index, neuron_columns].values
            
            # If window_responses is shorter than time_window, pad with zeros
            if window_responses.shape[0] < len(time_window):
                padding = np.zeros((len(time_window) - window_responses.shape[0], window_responses.shape[1]))
                window_responses = np.vstack((window_responses, padding))

            mean_responses += window_responses

        mean_responses /= len(stimulus_indices)

        # Plot mean responses for this orientation
        mean_neuron_responses = mean_responses.mean(axis=1)
        fig.add_trace(go.Scatter(x=time_window, y=mean_neuron_responses, mode='lines', name=orientation), row=i, col=1)

    # Update layout
    fig.update_layout(height=2000, width=800, title_text="Mean Neural Response for Each Orientation", showlegend=False)
    fig.update_xaxes(title_text="Time relative to stimulus onset (s)", row=len(orientations), col=1)
    fig.update_yaxes(title_text="Mean Response (DF/F%)")

    return fig


#%%
#ptsh histogram plot with plotly
def calculate_and_plot_psth_plotly(data, pre_event_time, post_event_time, bin_size):
    orientations = {
        'degrees_45': 45, 'degrees_90': 90, 'degrees_135': 135, 'degrees_180': 180, 
        'degrees_225': -45, 'degrees_270': -90, 'degrees_315': -135, 'degrees_0': -180
    }

    time = data['time']
    bin_edges = np.arange(-pre_event_time, post_event_time + bin_size, bin_size)

    # Create a subplot for each orientation
    fig = make_subplots(rows=len(orientations), cols=1, shared_xaxes=True)

    for i, (orientation, angle) in enumerate(orientations.items(), start=1):
        stimulus_times = time[data[orientation] == 1]
        binned_responses = np.zeros(len(bin_edges) - 1)

        for stimulus_time in stimulus_times:
            window_mask = (time >= stimulus_time - pre_event_time) & (time < stimulus_time + post_event_time)
            window_times = time[window_mask]
            window_responses = data.loc[window_mask, data.columns.difference(['time', 'pupil_size', 'speed', 'direction'])]

            digitized = np.digitize(window_times - stimulus_time, bin_edges) - 1
            for bin_index in range(len(binned_responses)):
                binned_responses[bin_index] += window_responses.iloc[digitized == bin_index].sum().sum()

        binned_responses /= len(stimulus_times)

        # Add bar plot to the subplot
        fig.add_trace(go.Bar(x=bin_edges[:-1], y=binned_responses, name=f'Orientation {angle}°'), row=i, col=1)

    # Update layout
    fig.update_layout(height=2000, width=800, title_text="PSTH for Each Orientation", showlegend=False)
    fig.update_xaxes(title_text="Time relative to stimulus onset (s)", row=len(orientations), col=1)
    fig.update_yaxes(title_text="Response (a.u.)")

    fig.show()

#%%
#ptsh tracing plot with plotly
def calculate_and_plot_mean_trace_plotly(data, pre_event_time, post_event_time):
    orientations = {
        'degrees_45': 45, 'degrees_90': 90, 'degrees_135': 135, 'degrees_180': 180, 
        'degrees_225': -45, 'degrees_270': -90, 'degrees_315': -135, 'degrees_0': -180
    }

    time = data['time']
    time_window = np.arange(-pre_event_time, post_event_time, np.median(np.diff(time)))

    # Create a subplot for each orientation
    fig = make_subplots(rows=len(orientations), cols=1, shared_xaxes=True)

    for i, (orientation, angle) in enumerate(orientations.items(), start=1):
        stimulus_times = time[data[orientation] == 1]
        print(f"Stimulus times for {orientation}: {stimulus_times}")  # Debug line

        mean_responses = np.zeros(len(time_window))

        for stimulus_time in stimulus_times:
            window_mask = (time >= stimulus_time - pre_event_time) & (time < stimulus_time + post_event_time)
            window_responses = data.loc[window_mask, data.columns.difference(['time', 'pupil_size', 'speed', 'direction'])]
            print(f"Window responses for stimulus time {stimulus_time}: {window_responses}")  # Debug line

            aligned_responses = window_responses.set_index(time[window_mask] - stimulus_time)
            mean_responses += aligned_responses.mean(axis=1).reindex(time_window, fill_value=0)
            print(f"Mean responses after adding aligned responses: {mean_responses}")  # Debug line

        mean_responses /= len(stimulus_times)
        print(f"Final mean responses for {orientation}: {mean_responses}")  # Debug line

        # Add line plot to the subplot
        fig.add_trace(go.Scatter(x=time_window, y=mean_responses, mode='lines', name=f'Orientation {angle}°'), row=i, col=1)

    # Update layout
    fig.update_layout(height=2000, width=800, title_text="Mean Neural Response for Each Orientation", showlegend=False)
    fig.update_xaxes(title_text="Time relative to stimulus onset (s)", row=len(orientations), col=1)
    fig.update_yaxes(title_text="Mean Response (DF/F%)")

    fig.show()


#%%
def plot_orientation_means(df, frame_rate, orientations, pre_stimulus=1, post_stimulus=3, max_frames=None):
    """
    Plots the mean fluorescence for all orientations in a grid.
    
    Parameters:
    df (DataFrame): The dataframe containing the fluorescence data.
    frame_rate (int): The frame rate of the recording.
    orientations (list): A list of orientations to plot.
    pre_stimulus (int): Seconds before the stimulus to include in the plot.
    post_stimulus (int): Seconds after the stimulus to include in the plot.
    max_frames (int): Maximum number of frames after the stimulus to include in the plot.
    """
    # Calculate the number of frames before and after the stimulus
    pre_frames = int(frame_rate * pre_stimulus)
    post_frames = int(frame_rate * post_stimulus)
    
    # Create a figure with subplots
    num_orientations = len(orientations)
    fig, axes = plt.subplots(num_orientations, 1, figsize=(12, num_orientations * 4), sharex=True)
    
    for i, orientation in enumerate(orientations):
        # Get the indices of the stimulus events for the orientation
        stimulus_indices = df[df['stimulus'] == orientation].index
        mean_tracings = []
        for idx in stimulus_indices:
            # Define the time window around the stimulus
            start_idx = max(idx - pre_frames, 0)
            end_idx = idx + post_frames
            if max_frames:
                end_idx = min(end_idx, idx + max_frames)
            # Extract the mean fluorescence for all neurons
            mean_tracing = df.loc[start_idx:end_idx].mean(axis=1)
            mean_tracings.append(mean_tracing)
        
        # Concatenate all mean tracings into a single DataFrame
        all_mean_tracings = pd.concat(mean_tracings, axis=1)
        
        # Calculate the grand mean across all events
        grand_mean_tracing = all_mean_tracings.mean(axis=1)
        
        # Plot the grand mean fluorescence
        sns.lineplot(ax=axes[i], data=grand_mean_tracing)
        axes[i].axvline(x=pre_frames, color='r', linestyle='--')
        axes[i].set_title(f'Orientation: {orientation}')
        axes[i].set_ylabel('Mean Fluorescence (dF/F%)')
    
    plt.xlabel('Frames')
    plt.tight_layout()
    plt.show()

#%%
def plot_neuron_stimulus_response(df, frames_before=30, frames_after=300, neuron_number=None):
    """
    Plots the response of a specified or randomly selected neuron to various stimuli in the DataFrame,
    accurately marking the onset of the stimulus.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing neuron activity and stimuli information.
    neuron_number (int, optional): The number of the neuron to plot. If None, a neuron is randomly selected.
    """
    # Identify the stimulus columns, ensuring column names are treated as strings
    stimulus_columns = [col for col in df.columns if 'degrees_' in str(col)]

    # Selecting a neuron, ensuring column names are treated as strings
    neuron_columns = [col for col in df.columns if str(col).isdigit()]
    selected_neuron = neuron_number if neuron_number is not None else random.choice(neuron_columns)

    # Creating a figure for the plots
    fig, axes = plt.subplots(len(stimulus_columns), 1, figsize=(10, 2 * len(stimulus_columns)))
    fig.suptitle(f'Tracing of Neuron {selected_neuron} with Accurately Marked Stimuli Onset Points')

    # Plotting the tracing for each stimulus with accurate onset points
    for i, stimulus in enumerate(stimulus_columns):
        # Find indices where the stimulus was presented
        stimulus_indices = df.index[df[stimulus] == 1].tolist()

        # Identifying the start of each stimulus occurrence
        starts = [stimulus_indices[0]]
        for j in range(1, len(stimulus_indices)):
            if stimulus_indices[j] - 1 != stimulus_indices[j - 1]:
                starts.append(stimulus_indices[j])

        # Randomly select one of the occurrences of the stimulus being presented
        selected_start_index = random.choice(starts)

        # Calculate the actual start and end points for plotting
        plot_start = max(selected_start_index - frames_before, 0)
        plot_end = min(selected_start_index + frames_after, len(df))
        rows = df.loc[plot_start:plot_end, selected_neuron]

        # Adjust the x-axis to align with the DataFrame's index
        adjusted_onset_index = selected_start_index - plot_start

        # Plot the tracing for this neuron and stimulus
        axes[i].plot(rows.index, rows)
        axes[i].axvline(rows.index[adjusted_onset_index], color='red', linestyle='--', label='Stimulus Onset')
        axes[i].annotate(f'Onset at {selected_start_index}', xy=(rows.index[adjusted_onset_index], rows[selected_start_index]), 
                         xytext=(rows.index[adjusted_onset_index] + 10, rows[selected_start_index]),
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         horizontalalignment='right', verticalalignment='top')
        axes[i].set_title(f'Stimulus: {stimulus}')
        axes[i].set_xlabel('Row Number')
        axes[i].set_ylabel('dF/F%')
        axes[i].legend()

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


#%%
def plotly_neuron_traces(df, neuron_number=None, frames_before=30, frames_after=300, fps=30.3):
    """
    Plots the response of a specified or randomly selected neuron to various stimuli in the DataFrame,
    in a facet grid layout, accurately marking the onset of the stimulus, using Plotly for interactive plots.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing neuron activity and stimuli information.
    neuron_number (int, optional): The number of the neuron to plot. If None, a neuron is randomly selected.
    """
    # Identify the stimulus columns, ensuring column names are treated as strings
    stimulus_columns = [col for col in df.columns if 'degrees_' in str(col)]
    stimulus_columns = [col for col in df.columns if 'degrees_' in str(col)]
    
    # Selecting a neuron, ensuring column names are treated as strings
    neuron_columns = [col for col in df.columns if str(col).isdigit()]
    selected_neuron = neuron_number if neuron_number is not None else random.choice(neuron_columns)    

    # Creating a figure for the plots with subplots
    rows = len(stimulus_columns)
    fig = make_subplots(rows=rows, cols=1, subplot_titles=stimulus_columns)

    # Plotting the tracing for each stimulus with accurate onset points
    for i, stimulus in enumerate(stimulus_columns, start=1):
        # Find indices where the stimulus was presented
        stimulus_indices = df.index[df[stimulus] == 1].tolist()

        # Identifying the start of each stimulus occurrence
        starts = [stimulus_indices[0]]
        for j in range(1, len(stimulus_indices)):
            if stimulus_indices[j] - 1 != stimulus_indices[j - 1]:
                starts.append(stimulus_indices[j])

        # Randomly select one of the occurrences of the stimulus being presented
        selected_start_index = random.choice(starts)

        # Calculate the actual start and end points for plotting
        plot_start = max(selected_start_index - frames_before, 0)
        plot_end = min(selected_start_index + frames_after, len(df))
        rows = df.loc[plot_start:plot_end, selected_neuron]

        # Adjust the x-axis to align with the DataFrame's index
        adjusted_onset_index = selected_start_index - plot_start

        # Add trace for this neuron and stimulus in the appropriate subplot
        fig.add_trace(go.Scatter(x=rows.index, y=rows, mode='lines', name=f'Stimulus: {stimulus}'), row=i, col=1)

        fig.add_vrect(x0=rows.index[adjusted_onset_index], x1=rows.index[adjusted_onset_index]+60, fillcolor="LightSalmon", opacity=0.5, layer="below", line_width=0, row=i, col=1)

    # Update layout
    fig.update_layout(height=2000, width=800, title_text=f'Tracing of Neuron {selected_neuron} with Orientation Onset')
    fig.show()
    return fig


#%%
def plotly_neuron_traces_dict(dfs_dict, neuron_number=None, frames_before=30, frames_after=300, fps=30.3):
    """
    Plots the response of a specified or randomly selected neuron to various conditions in a dictionary of DataFrames,
    in a facet grid layout, using Plotly for interactive plots.

    Parameters:
    dfs_dict (dict of pandas.DataFrame): The dictionary containing neuron activity DataFrames.
    neuron_number (int, optional): The number of the neuron to plot. If None, a neuron is randomly selected.
    """
    # Creating a figure for the plots with subplots
    rows = len(dfs_dict)
    fig = make_subplots(rows=rows, cols=1, subplot_titles=list(dfs_dict.keys()))

    # Plotting the tracing for each DataFrame in the dictionary
    for i, (condition, df) in enumerate(dfs_dict.items(), start=1):
        # Selecting a neuron, ensuring column names are treated as strings
        neuron_columns = [col for col in df.columns if str(col).isdigit()]
        selected_neuron = neuron_number if neuron_number is not None else random.choice(neuron_columns)    

        # Assume each DataFrame in the dictionary represents a different condition/stimulus
        # Plotting the neuron's response in the corresponding subplot
        rows = df.loc[:, selected_neuron]

        # Adjust the x-axis to align with the DataFrame's index
        adjusted_onset_index = frames_before

        # Add trace for this neuron in the appropriate subplot
        fig.add_trace(go.Scatter(x=rows.index, y=rows, mode='lines', name=f'Condition: {condition}'), row=i, col=1)

        # Highlighting the stimulus onset period (assuming fixed duration for each condition)
        fig.add_vrect(x0=rows.index[adjusted_onset_index], x1=rows.index[adjusted_onset_index]+60, fillcolor="LightSalmon", opacity=0.5, layer="below", line_width=0, row=i, col=1)

    # Update layout
    fig.update_layout(height=2000, width=800, title_text=f'Tracing of Neuron {selected_neuron}')
    fig.show()
    return fig

#%%
def get_rows_with_stimulus(data, stimulus_column):
    """
    Get all rows where a specific stimulus is active (column value is 1).

    :param data: DataFrame containing the time series data.
    :param stimulus_column: Column name for the specific stimulus (e.g., 'degrees_45').
    :return: DataFrame containing rows where the stimulus is active.
    """
    # Filtering rows where the stimulus column is 1
    stimulus_active_rows = data[data[stimulus_column] == 1]

    return stimulus_active_rows

#%%
def extract_stimuli_frames(data, stimulus_column, pre_frames):
    """
    Get all rows where a specific stimulus is active, and include a specified number of rows before each stimulus onset.

    :param data: DataFrame containing the time series data.
    :param stimulus_column: Column name for the specific stimulus (e.g., 'degrees_45').
    :param pre_frames: Number of frames to include before the stimulus starts.
    :return: DataFrame containing rows with the active stimulus and preceding rows.
    """
    stimulus_active_rows = pd.DataFrame()

    # Identifying the start indices of each stimulus instance
    stimulus_start_indices = data.index[data[stimulus_column].diff() == 1].tolist()
    stimulus_stop_indices = data.index[data[stimulus_column].diff() == -1].tolist()

    # Loop through each start index and get preceding rows
    for start_index in stimulus_start_indices:
        # Get range from pre_frames before start_index to start_index (inclusive)
        frame_range = range(max(start_index - pre_frames, 0), start_index + 1)
        stimulus_active_rows = stimulus_active_rows.append(data.loc[frame_range])

    return stimulus_active_rows


# %%
def extract_stimuli_frames(data, pre_frames =30, post_frames=240):
    """
    Get all rows where a specific stimulus is active, and include a specified number of rows before each stimulus onset.

    :param data: DataFrame containing the time series data.
    :param stimulus_column: Column name for the specific stimulus (e.g., 'degrees_45').
    :param pre_frames: Number of frames to include before the stimulus starts.
    :return: DataFrame containing rows with the active stimulus and preceding rows.
    """
    orientations = ["degrees_45","degrees_90", "degrees_135", "degrees_180", "degrees_225", "degrees_270", "degrees_315", "degrees_0"]
    frame_range_dict = {}
    df_of_orientations = {}
    for orientation in orientations:
        # Identifying the start indices of each stimulus instance
        stimulus_start_indices = data.index[data[orientation].diff() == 1].tolist()
        stimulus_stop_indices = data.index[data[orientation].diff() == -1].tolist()
        frame_range_list = []
        orientation_df = []
        # Loop through each start index and get preceding rows
        for start_index, stop_index in zip(stimulus_start_indices, stimulus_stop_indices):
            # Get range from pre_frames before start_index to start_index (inclusive)
            data_temp = None
            stop_index = stop_index + post_frames
            frame_range = range(max(start_index - pre_frames, 0), stop_index)
            frame_range_list.append(frame_range)
            data_temp = data.loc[frame_range]
            
            first_cell_value = data_temp["time"].iloc[0]
            
            data_temp['rtime'] = data_temp['time'].apply(lambda x: x - (1 + first_cell_value))
            orientation_df.append(data_temp)
            
        frame_range_dict[orientation] = frame_range_list
        df_of_orientations[orientation] = orientation_df

    return df_of_orientations, frame_range_dict


#%%
def mean_stimuli(data, pre_frames = 30, post_frames = 240):
    orientations = ["degrees_45","degrees_90", "degrees_135", "degrees_180", "degrees_225", "degrees_270", "degrees_315", "degrees_0"]

    all_stimuli_dict, _ = extract_stimuli_frames(data, pre_frames, post_frames)
    dict_of_means = {}
    for orientation in orientations:
        all_stimuli = all_stimuli_dict[orientation]
        print(all_stimuli)

        length = range(0, len(all_stimuli), 2)

        print(length)
        target = len(all_stimuli)-2
        print(target)
        df_sum = None
        
        # Initialize a new DataFrame with the same shape as the first DataFrame, filled with zeros
        df_sum = pd.DataFrame(0, index=all_stimuli[0].index, columns=all_stimuli[0].columns)
        df_sum = df_sum.reset_index()
        # Add each DataFrame to df_sum
        for df in all_stimuli:
            df = df.reset_index()
            df = df.drop("index", axis=1)
            df_sum = df_sum.add(df, fill_value=0)
        
        df_sum = df_sum.div(len(all_stimuli))
        #df_sum = df_sum.dropna()
        dict_of_means[orientation] = df_sum

    return dict_of_means



##very important function to fetch the proper frames for the stimuli
# def fetch_stimuli_data(df, pre_time, post_time, exclude_stimuli=False, specific_stimuli=None):

#     stimuli_columns = [col for col in df.columns if 'degrees' in col]
#     non_stimuli_columns = [col for col in df.columns if 'degrees' not in col]

#     sampling_rate = 1 / df['time'].iloc[1]
#     pre_rows = int(pre_time * sampling_rate)
#     post_rows = int(post_time * sampling_rate)

#     stimuli_data = {}

#     for stimuli in stimuli_columns:
#         if specific_stimuli is not None and int(stimuli.split('_')[1]) not in specific_stimuli:
#             continue

#         onsets = df[df[stimuli] == 1].index
#         trials = np.split(onsets, np.where(np.diff(onsets) != 1)[0] + 1)

#         for trial in trials:
#             onset_index = trial[0]
#             start_index = max(0, onset_index - pre_rows)
#             end_index = min(len(df), onset_index + post_rows)

#             if exclude_stimuli:
#                 df_segment = df.loc[start_index:onset_index-1, non_stimuli_columns]
#             else:
#                 df_segment = df.loc[start_index:end_index, non_stimuli_columns]

#             if stimuli not in stimuli_data:
#                 stimuli_data[stimuli] = []
#             stimuli_data[stimuli].append(df_segment)

#     return stimuli_data

def fetch_stimuli_data(df, pre_time, post_time, exclude_stimuli=False, specific_stimuli=None, transpose=True):
    """
    Fetches rows based on stimuli onset. Trials X Rows X Columns are returned for each stimuli.

    :param df: Pandas DataFrame containing the data.
    :param pre_time: Time in seconds before the stimuli onset.
    :param post_time: Time in seconds after the stimuli onset. If 0, only pre-stimuli data is fetched.
    :param exclude_stimuli: If True, fetches rows that exclude the stimuli.
    :param specific_stimuli: List of stimuli orientations (e.g., [45, 90]) to specifically fetch data for.
    :return: Dictionary with keys being stimuli and values being lists of filtered DataFrames.
    """
    stimuli_columns = [col for col in df.columns if 'degrees' in col]
    non_stimuli_columns = [col for col in df.columns if 'degrees' not in col]

    sampling_rate = 1 / df['time'].iloc[1]
    pre_rows = int(pre_time * sampling_rate)
    post_rows = int(post_time * sampling_rate)

    stimuli_data = {}

    for stimuli in stimuli_columns:
        if specific_stimuli is not None and int(stimuli.split('_')[1]) not in specific_stimuli:
            continue

        onsets = df[df[stimuli] == 1].index
        trials = np.split(onsets, np.where(np.diff(onsets) != 1)[0] + 1)

        trial_data = []

        for trial in trials:
            onset_index = trial[0]
            start_index = max(0, onset_index - pre_rows)
            end_index = min(len(df), onset_index + post_rows)

            if exclude_stimuli:
                df_segment = df.loc[start_index:onset_index-1, non_stimuli_columns]
            else:
                df_segment = df.loc[start_index:end_index, non_stimuli_columns]

            trial_data.append(df_segment.values)

        # Convert the list of numpy arrays to a numpy array and transpose the axes if transpose is True
        stimuli_data[stimuli] = np.transpose(np.array(trial_data), (0, 2, 1)) if transpose else np.array(trial_data)

    return stimuli_data


#%%

def plot_peristimulus_response(data_path=None, df=None, window_size=60, post_stimulus_window_size=360):
    # Load data from CSV if not provided as DataFrame
    if df is None:
        df = pd.read_csv(data_path)

    # Define the pre- and post-stimulus windows
    window_size = window_size  # 60 rows before the stimulus
    post_stimulus_window_size = post_stimulus_window_size  # 360 rows after the stimulus

    # Identify stimulus columns
    stimulus_columns = [str(col) for col in df.columns if 'degrees_' in str(col)]

    # Prepare the plots
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
    axs = axs.flatten()  # Flatten the axis array for easier indexing

    # Loop through each ROI for single trial plots
    for i in range(8):  # Assuming 8 ROIs for the 4x2 grid
        roi_col = str(i)
        for stimulus_type in stimulus_columns:
            # Find indices where this stimulus starts
            stimulus_onset_indices = df.index[df[stimulus_type] == 1].tolist()
            
            if stimulus_onset_indices:
                # Randomly pick one onset index
                if roi_index==None:
                    random_onset_index = np.random.choice(stimulus_onset_indices)
                else:
                    random_onset_index = stimulus_onset_indices[roi_index]

                # Calculate start and end of the interval
                start_index = max(0, random_onset_index - window_size)
                end_index = random_onset_index + post_stimulus_window_size

                # Extract the ROI data for this window
                roi_data = df.iloc[start_index:end_index, df.columns.get_loc(int(roi_col))]
                
                # Adjust x-axis to be relative to stimulus onset
                time_axis = range(-window_size, post_stimulus_window_size)

                # Plot
                axs[i].plot(time_axis, roi_data, label=f'{stimulus_type}')

                # Highlight stimulus period
                axs[i].axvspan(0, 60, color='yellow', alpha=0.3)

        axs[i].set_title(f'ROI {roi_col}')
        axs[i].legend()

    plt.tight_layout()
    plt.show()

    # Aggregate Plots
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
    axs = axs.flatten()  # Flatten the axis array for easier indexing

    # Loop through each ROI for aggregate plots
    for i in range(8):
        roi_col = str(i)
        for stimulus_type in stimulus_columns:
            stimulus_onset_indices = df.index[df[stimulus_type] == 1].tolist()

            if stimulus_onset_indices:
                all_responses = []

                for onset_index in stimulus_onset_indices:
                    start_index = max(0, onset_index - window_size)
                    end_index = onset_index + post_stimulus_window_size
                    roi_data = df.iloc[start_index:end_index, df.columns.get_loc(int(roi_col))]
                    all_responses.append(roi_data.values)

                if all_responses:
                    mean_response = np.mean(np.vstack(all_responses), axis=0)
                    time_axis = range(-window_size, post_stimulus_window_size)
                    axs[i].plot(time_axis, mean_response, label=f'{stimulus_type}')
                    axs[i].axvspan(0, 60, color='yellow', alpha=0.3)

        axs[i].set_title(f'ROI {roi_col}')
        axs[i].legend()

    plt.tight_layout()
    plt.show()
