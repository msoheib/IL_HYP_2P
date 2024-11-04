#%%
import pynapple as nap
import pandas as pd
import numpy as np
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
import scienceplots

# %%
# Load NWB file using Pynapple
file_path = 'vgat4.nwb'  # Replace with the actual path to your NWB file
nwb_data = nap.load_file(file_path)

# %%
dff = nwb_data['DfOverF']

# Investigate the structure of dff
print("DFF object attributes:", dir(dff))
print("DFF object type:", type(dff))
print("DFF shape:", dff.shape)
print("DFF index type:", type(dff.index))
print("DFF index (first 5):", dff.index[:5])
print("DFF columns type:", type(dff.columns))
print("DFF columns (first 5):", dff.columns[:5])

# Print the first few rows of the DFF data
print("DFF first 5 rows, first 5 columns:")
print(dff.values[:5, :5])

# %%
# Convert dff to a dictionary of TimeSeries objects
dff_dict = {}
for i, column in enumerate(dff.columns):
    print(f"Processing column {i}: {column}")
    print(f"Column data type: {type(dff[column])}")
    print(f"Column data shape: {dff[column].shape}")
    dff_dict[i] = dff[column]  # dff[column] should already be a Tsd object

# Create TsGroup from the dictionary
dff_group = nap.TsGroup(dff_dict)

# %%
# Access the stimulus intervals table
stimuli_table = nwb_data["StimulusIntervals"]

# Create a dictionary to group intervals by orientation
dict_ep = {}

for _, row in stimuli_table.iterrows():
    orientation = row['orientation']
    start_time = row['start_time']
    stop_time = row['stop_time']
    
    key = f"stim{orientation}"
    if key not in dict_ep:
        dict_ep[key] = []
    dict_ep[key].append((start_time, stop_time))

# Convert lists of tuples to IntervalSets
for key, intervals in dict_ep.items():
    starts, stops = zip(*intervals)
    dict_ep[key] = nap.IntervalSet(start=starts, end=stops)

# Print the resulting dictionary
print(dict_ep)

# %%
# Compute tuning curves
tuning_curves = nap.compute_discrete_tuning_curves(dff_group, dict_ep)

# Set the science plot style
plt.style.use(['science', 'ieee'])

# Convert orientation labels to radians for polar plot and sort them
orientations = [float(key.replace('stim', '')) for key in tuning_curves.index]
sorted_indices = np.argsort(orientations)
orientations = [orientations[i] for i in sorted_indices]
orientations_rad = np.deg2rad(orientations)

# Plot polar tuning curves for a subset of neurons
num_neurons_to_plot = min(15, len(tuning_curves.columns))  # Plot up to 15 neurons
neurons_to_plot = tuning_curves.columns[:num_neurons_to_plot]

plt.figure(figsize=(15, 12))

for i, neuron in enumerate(neurons_to_plot):
    plt.subplot(3, 5, i + 1, projection="polar")
    sorted_tuning = tuning_curves[neuron].iloc[sorted_indices]
    plt.polar(np.concatenate([orientations_rad, [orientations_rad[0]]]),  # Close the circle
              np.concatenate([sorted_tuning, [sorted_tuning.iloc[0]]]),  # Close the circle
              'b-')  # Blue line
    plt.fill(np.concatenate([orientations_rad, [orientations_rad[0]]]),  # Close the circle
             np.concatenate([sorted_tuning, [sorted_tuning.iloc[0]]]),  # Close the circle
             alpha=0.2)  # Light blue fill
    plt.title(f"Neuron {neuron}", fontsize=10)
    plt.xticks(np.deg2rad([0, 90, 180, 270]), ['0°', '90°', '180°', '270°'])
    plt.yticks([])  # Remove radial ticks for cleaner look

plt.tight_layout()
plt.show()

# Plot mean tuning curve across all neurons (polar)
mean_tuning_curve = tuning_curves.mean(axis=1).iloc[sorted_indices]
std_tuning_curve = tuning_curves.std(axis=1).iloc[sorted_indices]

plt.figure(figsize=(8, 8))
ax = plt.subplot(111, projection="polar")
ax.plot(np.concatenate([orientations_rad, [orientations_rad[0]]]),  # Close the circle
        np.concatenate([mean_tuning_curve, [mean_tuning_curve.iloc[0]]]),  # Close the circle
        'b-', linewidth=2)
ax.fill_between(np.concatenate([orientations_rad, [orientations_rad[0]]]),  # Close the circle
                np.concatenate([mean_tuning_curve - std_tuning_curve, [mean_tuning_curve.iloc[0] - std_tuning_curve.iloc[0]]]),  # Close the circle
                np.concatenate([mean_tuning_curve + std_tuning_curve, [mean_tuning_curve.iloc[0] + std_tuning_curve.iloc[0]]]),  # Close the circle
                alpha=0.2)
plt.title('Mean Tuning Curve Across All Neurons', fontsize=12)
plt.xticks(np.deg2rad([0, 90, 180, 270]), ['0°', '90°', '180°', '270°'])
plt.yticks([])  # Remove radial ticks for cleaner look

plt.tight_layout()
plt.show()

# %%
