# %%
# %%
# %%
import pandas as pd
import numpy as np
from datetime import datetime
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.ophys import ImageSegmentation, Fluorescence, RoiResponseSeries
from pynwb.ophys import OpticalChannel, ImagingPlane
from pynwb.device import Device
from pynwb.behavior import PupilTracking, Position, SpatialSeries
from pynwb.file import Subject
from hdmf.common import DynamicTableRegion
from pynwb.validate import validate
from pynwb.epoch import TimeIntervals
import os

# ---------------------------
# Configuration
# ---------------------------

# File paths and names
csv_file_path = r'J:\My Drive\0-Main\1_STRESS\vgatnpy\pre\dFF_vgatnpy_pre.csv'
suite2p_folder_path = r'J:\My Drive\0-Main\1_STRESS\vgatnpy\pre\plane0'  # Adjust to point to 'plane0' folder
nwb_output_path = 'vgat5.nwb'

# Session information
session_description = 'Acute restraint stress experiment on mouse vgat1'
session_identifier = 'vgat1_acuterestraint'
session_start_time = datetime.now().astimezone()  # You can specify a specific datetime if needed

# Subject information
subject_id = 'vgat1'
species = 'Mus musculus'
subject_description = 'VGAT-NPY-Cre mouse for acute restraint stress experiment'
genotype = 'VGAT-NPY-Cre'

# Imaging device information
device_name = 'Bruker Ultima 2p+'
optical_channel_name = 'OpticalChannel'
optical_channel_description = '2P Optical Channel'
emission_lambda = 510.0  # in nanometers
excitation_lambda = 800.0  # in nanometers
imaging_plane_name = 'ImagingPlane'
imaging_plane_description = 'Layer 2/3 of V1 region'
indicator = 'GCaMP8m'
location = 'V1'

# ---------------------------
# Load and Prepare the Data
# ---------------------------

# Load the CSV file
data = pd.read_csv(csv_file_path)

# Calculate frames per second (fps) and prepare timestamps
time_diff = data['time'].diff().iloc[1]
fps = 1 / time_diff
timestamps = data['time'].values

# Extract behavioral data
pupil_size = data['pupil_size'].values
speed = data['speed'].values
direction = data['direction'].values

# Extract visual stimuli data
visual_stimuli = data.filter(like='degrees_')

# ---------------------------
# Create the NWB File
# ---------------------------

# Create NWB file with session and subject information
nwbfile = NWBFile(
    session_description=session_description,
    identifier=session_identifier,
    session_start_time=session_start_time,
    subject=Subject(
        subject_id=subject_id,
        species=species,
        description=subject_description,
        genotype=genotype
    )
)

# ---------------------------
# Set Up Imaging Metadata
# ---------------------------

# Create a device
device = Device(name=device_name)
nwbfile.add_device(device)

# Create an optical channel
optical_channel = OpticalChannel(
    name=optical_channel_name,
    description=optical_channel_description,
    emission_lambda=emission_lambda
)

# Create an imaging plane
imaging_plane = nwbfile.create_imaging_plane(
    name=imaging_plane_name,
    optical_channel=optical_channel,
    description=imaging_plane_description,
    device=device,
    excitation_lambda=excitation_lambda,
    imaging_rate=fps,
    indicator=indicator,
    location=location
)

# ---------------------------
# Read Suite2p Data Directly
# ---------------------------

import numpy as np
import os

# Attempt to load 'Fall.npy'
parent_dir = os.path.dirname(suite2p_folder_path)
fall_path = os.path.join(parent_dir, 'Fall.npy')
if os.path.exists(fall_path):
    fall = np.load(fall_path, allow_pickle=True).item()
    fluorescence_traces = fall['F']  # Shape: (num_rois, num_frames)
    iscell = fall['iscell']
    stat = fall['stat']
    ops = fall['ops']
    print("Loaded 'Fall.npy'")
else:
    print("'Fall.npy' not found, attempting to load 'F.npy'")
    # Load 'F.npy', 'iscell.npy', and 'stat.npy'
    fluorescence_traces = np.load(os.path.join(suite2p_folder_path, 'F.npy'))
    iscell = np.load(os.path.join(suite2p_folder_path, 'iscell.npy'))
    stat = np.load(os.path.join(suite2p_folder_path, 'stat.npy'), allow_pickle=True)
    # Try to load 'ops.npy'
    ops_path = os.path.join(suite2p_folder_path, 'ops.npy')
    if os.path.exists(ops_path):
        ops = np.load(ops_path, allow_pickle=True).item()
    else:
        ops = None

# Verify shapes
print(f"fluorescence_traces.shape: {fluorescence_traces.shape}")
print(f"iscell.shape: {iscell.shape}")
print(f"Number of entries in 'stat': {len(stat)}")

# Create boolean mask for accepted ROIs
accepted_rois_mask = iscell[:, 0].astype(bool)
num_accepted_rois = np.sum(accepted_rois_mask)
print(f"Number of accepted ROIs: {num_accepted_rois}")

# Ensure the mask length matches the number of ROIs
if fluorescence_traces.shape[0] != iscell.shape[0]:
    raise ValueError("Mismatch in number of ROIs between fluorescence traces and 'iscell.npy'.")

# Filter fluorescence traces and stat
fluorescence_traces = fluorescence_traces[accepted_rois_mask, :]  # Shape: (num_accepted_rois, num_frames)
stat = np.array(stat)[accepted_rois_mask]
roi_ids = np.arange(num_accepted_rois)

# Extract image dimensions
if ops is not None:
    image_height = ops['Ly']
    image_width = ops['Lx']
    print(f"Image dimensions from ops: {image_height} x {image_width}")
else:
    # Set default values or extract from another source
    image_height = 512  # Replace with actual value if known
    image_width = 512   # Replace with actual value if known
    print(f"Using default image dimensions: {image_height} x {image_width}")

# Create ROI masks
roi_masks = []
for idx, roi_stat in enumerate(stat):
    ypix = roi_stat['ypix']
    xpix = roi_stat['xpix']
    mask = np.zeros((image_height, image_width), dtype=bool)
    mask[ypix, xpix] = True
    roi_masks.append(mask)

roi_masks = np.array(roi_masks)

# ---------------------------
# Create Imaging Data Structures in NWB
# ---------------------------

# Create an optical physiology processing module
ophys_module = nwbfile.create_processing_module(
    name='ophys',
    description='Optical physiology processing module'
)

# Create ImageSegmentation
image_segmentation = ImageSegmentation()
ophys_module.add(image_segmentation)

# Create PlaneSegmentation
plane_segmentation = image_segmentation.create_plane_segmentation(
    name='PlaneSegmentation',
    description='Segmented ROIs from Suite2p',
    imaging_plane=imaging_plane
)

# Add ROIs to PlaneSegmentation
for mask, roi_id in zip(roi_masks, roi_ids):
    plane_segmentation.add_roi(image_mask=mask, id=roi_id)

# Create a DynamicTableRegion for the ROIs
roi_table_region = DynamicTableRegion(
    name='rois',
    data=roi_ids,
    description='Indices of ROIs in the PlaneSegmentation',
    table=plane_segmentation
)

# Create Fluorescence container
fluorescence = Fluorescence()
ophys_module.add(fluorescence)

# Ensure that the number of frames matches the length of timestamps
if fluorescence_traces.shape[1] != len(timestamps):
    raise ValueError("Number of frames in fluorescence traces does not match number of timestamps.")

# Create RoiResponseSeries with fluorescence traces
rrs = RoiResponseSeries(
    name='Fluorescence',
    data=fluorescence_traces,  # Shape: (num_rois, num_frames)
    rois=roi_table_region,
    unit='arbitrary',
    timestamps=timestamps
)

fluorescence.add_roi_response_series(rrs)

# ---------------------------
# Visual Stimuli Representation using TimeIntervals
# ---------------------------

# Create TimeIntervals for visual stimuli
stimulus_intervals = TimeIntervals(
    name='StimulusPresentations',
    description='Intervals of visual stimulus presentations'
)

# Add custom columns for stimulus metadata
stimulus_intervals.add_column(
    name='orientation',
    description='Orientation of the grating in degrees'
)
stimulus_intervals.add_column(
    name='spatial_frequency',
    description='Spatial frequency in cycles per degree'
)
stimulus_intervals.add_column(
    name='contrast',
    description='Contrast percentage'
)

# Iterate over visual stimuli columns to extract intervals
for column in visual_stimuli.columns:
    orientation = int(column.split('_')[1])  # Extract orientation from column name
    series = visual_stimuli[column].values
    edges = np.diff(np.concatenate(([0], series)))  # Detect stimulus on/off edges
    starts = np.where(edges == 1)[0]
    stops = np.where(edges == -1)[0]

    # Handle cases where the stimulus stays on until the end
    if len(stops) < len(starts):
        stops = np.append(stops, len(series) - 1)

    for start_idx, stop_idx in zip(starts, stops):
        stimulus_intervals.add_interval(
            start_time=timestamps[start_idx],
            stop_time=timestamps[stop_idx],
            orientation=orientation,
            spatial_frequency=np.nan,  # Replace with actual values if available
            contrast=np.nan            # Replace with actual values if available
        )

# Add the TimeIntervals to the NWB file
nwbfile.add_time_intervals(stimulus_intervals)

# ---------------------------
# Behavioral Data Representation
# ---------------------------

# Create PupilTracking for pupil size data
pupil_tracking = PupilTracking(name='PupilTracking')
pupil_ts = TimeSeries(
    name='PupilSize',
    data=pupil_size,
    unit='pixels',
    timestamps=timestamps,
    description='Pupil size over time'
)
pupil_tracking.add_timeseries(pupil_ts)
nwbfile.add_acquisition(pupil_tracking)

# Create Position for speed and direction data
position = Position(name='Position')

# Speed data
speed_series = SpatialSeries(
    name='Speed',
    data=speed,
    reference_frame='Origin at start position',
    unit='cm/s',
    timestamps=timestamps,
    description='Speed of the animal over time'
)
position.add_spatial_series(speed_series)

# Direction data
direction_series = SpatialSeries(
    name='Direction',
    data=direction,
    reference_frame='Origin at start position',
    unit='degrees',
    timestamps=timestamps,
    description='Direction of movement over time'
)
position.add_spatial_series(direction_series)

nwbfile.add_acquisition(position)

# ---------------------------
# Save and Validate the NWB File
# ---------------------------

# Save the NWB file
with NWBHDF5IO(nwb_output_path, 'w') as io:
    io.write(nwbfile)

print(f"NWB file saved at {nwb_output_path}")

# Validate the NWB file
with NWBHDF5IO(nwb_output_path, 'r') as io:
    validation_errors = validate(io)
    if validation_errors:
        print("Validation errors found:")
        for error in validation_errors:
            print(error)
    else:
        print("NWB file is valid!")

# ---------------------------
# Test Reading the NWB File
# ---------------------------

with NWBHDF5IO(nwb_output_path, 'r') as io:
    read_nwbfile = io.read()
    print("Successfully read the NWB file.")

    # Access the 'ophys' processing module
    ophys_module = read_nwbfile.processing['ophys']

    # Access ImageSegmentation and PlaneSegmentation
    image_seg = ophys_module.data_interfaces['ImageSegmentation']
    plane_seg = image_seg.plane_segmentations['PlaneSegmentation']

    # Count the number of ROIs
    roi_count = len(plane_seg.id[:])
    print(f"File contains {roi_count} ROIs.")

    # Access Fluorescence data
    fluorescence = ophys_module.data_interfaces['Fluorescence']
    fluorescence_series = fluorescence.roi_response_series['Fluorescence']
    print(f"Fluorescence data shape: {fluorescence_series.data.shape}")

# %%
