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

# ---------------------------
# Configuration
# ---------------------------

# File paths and names
file_path = r'J:\My Drive\0-Main\1_STRESS\vgatnpy\pre\dFF_vgatnpy_pre.csv'
nwb_output_path = 'vgat4.nwb'

# ---------------------------
# Load and Prepare the Data
# ---------------------------

# Load the CSV file
data = pd.read_csv(file_path)

# Calculate frames per second (fps) and prepare timestamps
time_diff = data['time'].diff().iloc[1]
fps = 1 / time_diff
timestamps = data['time'].values

# Extract dF/F traces
dff_columns = data.columns[data.columns.str.isdigit() | data.columns.str.isnumeric()]
dff_traces = data[dff_columns].values

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
    session_description='Acute restraint stress experiment on mouse vgat1',
    identifier='vgat1_acuterestraint',
    session_start_time=datetime.now().astimezone(),
    subject=Subject(
        subject_id='vgat1',
        species='Mus musculus',
        description='VGAT-NPY-Cre mouse for acute restraint stress experiment',
        genotype='VGAT-NPY-Cre'
    )
)

# ---------------------------
# Set Up Imaging Metadata
# ---------------------------

# Create a device
device = Device(name='Bruker Ultima 2p+')
nwbfile.add_device(device)

# Create an optical channel
optical_channel = OpticalChannel(
    name='OpticalChannel',
    description='2P Optical Channel',
    emission_lambda=510.0  # In nanometers
)

# Create an imaging plane
imaging_plane = nwbfile.create_imaging_plane(
    name='ImagingPlane',
    optical_channel=optical_channel,
    description='Layer 2/3 of V1 region',
    device=device,
    excitation_lambda=800.0,  # In nanometers
    imaging_rate=fps,
    indicator='GCaMP8m',
    location='V1'
)

# Note: We omit TwoPhotonSeries since raw imaging data is unavailable

# ---------------------------
# Optical Physiology Data
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
    description='Segmented ROIs for the imaging plane',
    imaging_plane=imaging_plane
)

# Add ROIs to PlaneSegmentation
# Replace the following with actual segmentation masks
roi_count = dff_traces.shape[1]
image_height = 512  # Example image height
image_width = 512   # Example image width
roi_masks = [np.random.randint(0, 2, size=(image_height, image_width)) for _ in range(roi_count)]

for mask in roi_masks:
    plane_segmentation.add_roi(image_mask=mask)

# Create a DynamicTableRegion for the ROIs
roi_table_region = DynamicTableRegion(
    name='rois',
    data=list(range(roi_count)),
    description='Indices of ROIs in the PlaneSegmentation',
    table=plane_segmentation
)

# Create Fluorescence container
fluorescence = Fluorescence()
ophys_module.add(fluorescence)

# Create RoiResponseSeries with corrected unit
rrs = RoiResponseSeries(
    name='DfOverF',
    data=dff_traces,
    rois=roi_table_region,
    unit='ratio',  # Updated unit to 'ratio'
    timestamps=timestamps
)

fluorescence.add_roi_response_series(rrs)

# ---------------------------
# Visual Stimuli Representation
# ---------------------------

from pynwb.epoch import TimeIntervals

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

# Iterate over your visual stimuli columns to extract intervals
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
    validation_errors = validate(io=io)
    if validation_errors:
        print("Validation errors found:")
        for error in validation_errors:
            print(error)
    else:
        print("NWB file is valid!")

# ---------------------------
# Test Reading the NWB File
# ------------------------
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

    # Access DfOverF data
    fluorescence = ophys_module.data_interfaces['Fluorescence']
    dff_series = fluorescence.roi_response_series['DfOverF']
    print(f"DfOverF data shape: {dff_series.data.shape}")
