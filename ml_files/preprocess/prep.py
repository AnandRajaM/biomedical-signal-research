import pandas as pd
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MinMaxScaler
import os

# Butterworth filter
def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Applies a low-pass Butterworth filter to the given data.

    :param data: 1D array-like, input signal to filter.
    :param cutoff: float, cutoff frequency in Hz.
    :param fs: int, sampling rate in Hz.
    :param order: int, filter order.
    :return: 1D array-like, filtered signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# Normalize data
def normalize_data(data, feature_range=(0, 1)):
    """
    Normalizes the data to a given range.

    :param data: DataFrame, data to normalize.
    :param feature_range: tuple, (min, max) for normalization.
    :return: DataFrame, normalized data.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Preprocess the entire dataset
def preprocess_data(data, sampling_rate=2000, cutoff_frequency=1000):
    """
    Preprocess the entire dataset by applying filters and normalization.

    :param data: DataFrame, input data containing radar_i, radar_q, and tfm_ecg2.
    :param sampling_rate: int, sampling rate in Hz for filtering.
    :param cutoff_frequency: float, cutoff frequency in Hz for filtering.
    :return: DataFrame, preprocessed data.
    """
    # Apply Butterworth filter to each column
    data["radar_i"] = butter_lowpass_filter(data["radar_i"], cutoff_frequency, sampling_rate)
    data["radar_q"] = butter_lowpass_filter(data["radar_q"], cutoff_frequency, sampling_rate)
    data["tfm_ecg2"] = butter_lowpass_filter(data["tfm_ecg2"], cutoff_frequency, sampling_rate)

    #  Normalize the filtered data
    # data[["radar_i", "radar_q", "tfm_ecg2"]] = normalize_data(data[["radar_i", "radar_q", "tfm_ecg2"]])

    return data

# Load and preprocess all data from CSV files
def load_and_preprocess_data(folder_path, sampling_rate=500, cutoff_frequency=50):
    """
    Loads and preprocesses all data from CSV files in a folder.

    :param folder_path: str, path to the folder containing CSV files.
    :param sampling_rate: int, sampling rate in Hz for filtering.
    :param cutoff_frequency: float, cutoff frequency in Hz for filtering.
    :return: DataFrame, preprocessed data.
    """
    # Get list of CSV files in the folder
    csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Load all files into a single DataFrame
    all_data = []
    for file in csv_files:
        print(f"Loading file: {file}")
        df = pd.read_csv(file)
        all_data.append(df)

    # Concatenate all data into one DataFrame
    full_data = pd.concat(all_data, ignore_index=True)

    # Preprocess the full dataset
    print("Preprocessing data...")
    preprocessed_data = preprocess_data(full_data, sampling_rate, cutoff_frequency)

    return preprocessed_data
