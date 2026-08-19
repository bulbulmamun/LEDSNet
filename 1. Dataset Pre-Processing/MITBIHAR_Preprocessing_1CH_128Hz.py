"""
MIT-BIH Arrhythmia Database preprocessing for single-channel (MLII) ECG.

This script is a GitHub-ready, linearized version of the 1-channel preprocessing
code from the provided MITBIH-AR.ipynb notebook. The preprocessing operations
and parameter values are kept the same as in the notebook.

Final output:
    2.MITBIHAR_Filtered_Segmented_1CH_128Hz.h5

HDF5 datasets:
    ECG_Signals : precisely resampled ECG segments
    ECG_Labels  : numeric labels for the 16 selected MIT-BIH annotation classes
"""

import wfdb
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample
import pywt
import h5py
import matplotlib.pyplot as plt


# =============================================================================
# Record list
# =============================================================================

record_names = [
    '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
    '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
    '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
    '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
    '222', '223', '228', '230', '231', '232', '233', '234'
]


# =============================================================================
# Filtering functions
# =============================================================================

# Define High-pass, Low-pass, and Notch filters
def butter_highpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_lowpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def notch_filter(freq, fs, quality_factor=30):
    nyquist = 0.5 * fs
    w0 = freq / nyquist
    b, a = iirnotch(w0, quality_factor)
    return b, a


def apply_filter(data, b, a):
    return filtfilt(b, a, data)


# Perform high-pass, low-pass, and notch filtering
def filter_ecg_signal(ecg_signal, fs):
    # Apply High-pass filter to remove baseline wander
    b_high, a_high = butter_highpass(0.5, fs)
    ecg_filtered = apply_filter(ecg_signal, b_high, a_high)

    # Apply Low-pass filter to remove high-frequency noise
    b_low, a_low = butter_lowpass(40, fs)
    ecg_filtered = apply_filter(ecg_filtered, b_low, a_low)

    # Apply Notch filter to remove powerline interference (50Hz)
    b_notch, a_notch = notch_filter(50, fs)
    ecg_filtered = apply_filter(ecg_filtered, b_notch, a_notch)

    return ecg_filtered


# =============================================================================
# Baseline correction and normalization functions
# These functions are retained from the notebook.
# In the 1-channel pipeline used below, both operations remain disabled,
# exactly as in the provided code.
# =============================================================================

# Baseline correction
def baseline_correction(ecg_signal, wavelet='db6', level=8):
    max_level = pywt.dwt_max_level(
        len(ecg_signal),
        pywt.Wavelet(wavelet).dec_len
    )
    level = min(level, max_level)
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)
    coeffs[0] = np.zeros_like(coeffs[0])  # Remove low-frequency components
    corrected_signal = pywt.waverec(coeffs, wavelet)
    return corrected_signal[:len(ecg_signal)]  # Ensure same length


# Function to normalize ECG signals
def normalize_signals(data):
    normalized_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        patient_data = data[i]
        min_val = patient_data.min()
        max_val = patient_data.max()
        max_abs_val = max(abs(min_val), abs(max_val))
        if max_abs_val != 0:  # Avoid division by zero
            normalized_data[i] = patient_data / max_abs_val
        else:
            normalized_data[i] = patient_data  # No normalization if max_abs_val is zero
    return normalized_data


# =============================================================================
# Segmentation settings
# =============================================================================

# List of valid labels in the MIT-BIH Arrhythmia dataset
valid_labels = [
    'N', 'L', 'R', 'V', '/', 'A', 'f', 'F',
    '!', 'j', 'x', 'a', 'E', 'J', 'e', 'Q'
]

# Number of samples to take before and after the annotation point
SAMPLES_BEFORE = 71
SAMPLES_AFTER = 144
SEGMENT_LENGTH = SAMPLES_BEFORE + SAMPLES_AFTER + 1  # Total length of each segment

# Initialize lists to store the segmented data and corresponding labels
segments = []
segment_labels = []

# PhysioNet directory for MIT-BIH Arrhythmia Database
pn_dir = 'mitdb'

# Sampling frequency
fs = 360  # Most MIT-BIH ECG signals are sampled at 360 Hz


# =============================================================================
# Read records, filter MLII, and segment around MIT-BIH annotations
# =============================================================================

# Loop through each record
for record_name in record_names:
    # Read the header file to check available leads
    record = wfdb.rdheader(record_name, pn_dir=pn_dir)

    # Check if MLII is present in the signal names
    if 'MLII' in record.sig_name:
        ml_ii_column = record.sig_name.index('MLII')  # Get the column index of MLII
        print(
            f"Processing Record {record_name} "
            f"(MLII in column {ml_ii_column})..."
        )

        # Load the record (MLII lead only)
        record = wfdb.rdrecord(
            record_name,
            pn_dir=pn_dir,
            channels=[ml_ii_column]
        )

        # Load the annotations
        annotation = wfdb.rdann(record_name, 'atr', pn_dir=pn_dir)

        # Get the signal data (MLII channel)
        signal = record.p_signal[:, 0]

        # Apply filtering
        filtered_signal = filter_ecg_signal(signal, fs)

        # Apply baseline correction
        corrected_signal = filtered_signal
        # corrected_signal = baseline_correction(filtered_signal)  # Uncomment to apply baseline correction

        # Normalize the signal
        normalized_signal = corrected_signal
        # normalized_signal = normalize_signals(corrected_signal.reshape(1, -1))[0]  # Uncomment to apply normalization

        # Loop through each annotation
        for i in range(len(annotation.sample)):
            sample_point = annotation.sample[i]
            label = annotation.symbol[i]

            # Only process if the label is one of the valid labels
            if label in valid_labels:
                # Ensure that we can take 71 samples before and 144 after the sample point
                if (
                    sample_point >= SAMPLES_BEFORE
                    and sample_point + SAMPLES_AFTER < len(normalized_signal)
                ):
                    # Extract the segment of 216 samples
                    segment = normalized_signal[
                        sample_point - SAMPLES_BEFORE:
                        sample_point + SAMPLES_AFTER + 1
                    ]

                    # Append the segment and its corresponding label
                    segments.append(segment)
                    segment_labels.append(label)
    else:
        print(f"Record {record_name} does not have MLII. Skipping.")


# Convert the lists to NumPy arrays
segments_array = np.array(segments)
labels_array = np.array(segment_labels).reshape(-1, 1)

# Print the shapes of the resulting arrays
print(f'Segments Array Shape: {segments_array.shape}')  # (total_number_of_segments, 216)
print(f'Labels Array Shape: {labels_array.shape}')      # (total_number_of_segments, 1)


# =============================================================================
# Check original annotation-label distribution
# =============================================================================

# Extract the unique labels and their counts
unique_labels, counts = np.unique(labels_array, return_counts=True)

# Print the unique labels and their corresponding counts
print(f"Unique labels: {unique_labels}")
print(f"Counts for each label: {counts}")


# =============================================================================
# Convert annotation symbols to numeric labels
# =============================================================================

# Create a dictionary mapping each label to a numeric value
label_mapping = {
    'N': 0,   # Normal beat
    'L': 1,   # Left bundle branch block beat
    'R': 2,   # Right bundle branch block beat
    'V': 3,   # Premature ventricular contraction
    '/': 4,   # Paced beat
    'A': 5,   # Atrial premature beat
    'f': 6,   # Fusion of paced and normal beat
    'F': 7,   # Fusion of ventricular and normal beat
    '!': 8,   # Ventricular flutter wave
    'j': 9,   # Nodal (junctional) premature beat
    'x': 10,  # Non-conducted beat
    'a': 11,  # Aberrated atrial premature beat
    'E': 12,  # Ventricular escape beat
    'J': 13,  # Nodal (junctional) escape beat
    'e': 14,  # Atrial escape beat
    'Q': 15   # Unclassifiable beat
}

# Convert the labels to numeric using the mapping
numeric_labels = np.array([
    label_mapping[label[0]]
    for label in labels_array
])

# Reshape the result to match the original array shape (number_of_segments, 1)
numeric_labels_array = numeric_labels.reshape(-1, 1)

# Print the resulting numeric labels array
print("Numeric Labels Array:")
print(numeric_labels_array)

# Extract the unique numeric labels and their counts
unique_labels, counts = np.unique(
    numeric_labels_array,
    return_counts=True
)

# Print the unique labels and their corresponding counts
print(f"Unique labels: {unique_labels}")
print(f"Counts for each label: {counts}")


# =============================================================================
# Save 360-Hz segmented data
# =============================================================================

with h5py.File(
    '2.MITBIHAR_Filtered_Segmented_1CH_360Hz.h5',
    'w'
) as h5f:
    h5f.create_dataset('ECG_Signals', data=segments_array)
    h5f.create_dataset('ECG_Labels', data=numeric_labels_array)

print(
    "Arrays saved successfully in "
    "2.MITBIHAR_Filtered_Segmented_1CH_360Hz.h5"
)


# =============================================================================
# Downsampling code retained from the notebook
# =============================================================================

# Assuming `data` is your original dataset
def downsample(data, target_size):
    factor = data.shape[-1] // target_size  # Calculate downsampling factor
    downsampled_data = data[:, ::factor]    # Slice every 'factor' element
    return downsampled_data


# Your original data
data = segments_array

# Downsample using simple slicing
downsampled_data = downsample(data, 77)
print(downsampled_data.shape)


# =============================================================================
# Precise resampling to exactly 77 samples
# This is the array saved in the final 128-Hz HDF5 file.
# =============================================================================

# Downsample using precise resampling to get exactly 77 samples
def precise_downsample(data, target_size):
    # Resample along the last axis to match the target size
    downsampled_data = resample(data, target_size, axis=-1)
    return downsampled_data


# Downsample to exactly 77 samples
precise_downsampled_data = precise_downsample(data, 77)

print(f"Precisely downsampled data shape: {precise_downsampled_data.shape}")


# =============================================================================
# Visualization retained from the notebook
# =============================================================================

# Extract the same sample for visualization
sample_index = 100
lead_index = 0
raw_data = data[sample_index, :]
precise_downsampled = precise_downsampled_data[sample_index, :]

# Plot raw and precisely downsampled data
plt.figure(figsize=(7, 4))

# Plot raw data
plt.plot(raw_data, label="Raw Data")
plt.title("Raw ECG Data (216 samples)")
plt.legend()

# Plot precisely downsampled data
plt.plot(
    precise_downsampled,
    label="Downsampled Data (77 samples)",
    color='orange'
)
plt.title("Downsampled ECG Data (77 samples)")
plt.legend()

plt.savefig(
    '2.MITBIHAR_Filtered_Segmented_1CH_128Hz.pdf',
    format='pdf',
    dpi=1200
)

plt.tight_layout()
plt.show()


# =============================================================================
# Save final 77-sample single-channel dataset
# =============================================================================

with h5py.File(
    '2.MITBIHAR_Filtered_Segmented_1CH_128Hz.h5',
    'w'
) as h5f:
    h5f.create_dataset(
        'ECG_Signals',
        data=precise_downsampled_data
    )
    h5f.create_dataset(
        'ECG_Labels',
        data=numeric_labels_array
    )

print(
    "Arrays saved successfully in "
    "2.MITBIHAR_Filtered_Segmented_1CH_128Hz.h5"
)
