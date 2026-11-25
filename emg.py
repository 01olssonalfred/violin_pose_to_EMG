# Functions for loading and processing emg data (exclusively)

import os
import csv
import numpy as np
from scipy import signal

# load emg data from csv files
def load_emg(emg_path):
    # args:
    # emg_path (str): path to directory of subject folders, each folder
    #                 contains emg data in .csv files.
    #  
    # basic file structure:
    # emg_path
    # - A
    #   - A_MVC.csv
    #   - A_R1.csv
    #   - A_R2.csv
    # - B
    #   - B_MVC.csv
    #   - B_R1.csv
    #   - B_R2.csv
    # ...
    #
    # each csv has delimiter ";" and decimal sign "." 
    #
    # returns:
    # emg_data (list): list of dicts, each dict contains a key "filename",
    #                  a key "time", and a key for each emg channel (muscle).
    #                  The specific muscles are hard-coded in this function (for now)
    #
    print(f"Loading emg data from root directory \"{emg_path}\"")

    def read_row(row_entry):
        # for handling missing float values
        # missing values returned as 0.0
        if row_entry == '':
            return 0.0
        return float(row_entry)

    emg_data = []
    for (root, dirs, files) in os.walk(emg_path):
        for file in files:
            if file.endswith('.csv') and file != 'info.csv':
                print(f"Loading \"{file}\"")
                entry = {}
                entry["filename"] = file
                with open(os.path.join(root, file)) as f:
                    reader = csv.reader(f, delimiter=';')
                    time, upper_trap_rt, mid_delt_rt, pec_major_rt, infraspin_rt, upper_trap_lt, lower_trap_rt, erect_spin_long_rt, rect_abdom_lo_rt = [], [], [], [], [], [], [], [], []
                    for row in list(reader)[4:]: # actual data starts from 5th row
                        time.append(float(row[0]))
                        upper_trap_rt.append(read_row(row[3]))
                        mid_delt_rt.append(read_row(row[4]))
                        pec_major_rt.append(read_row(row[5]))
                        infraspin_rt.append(read_row(row[6]))
                        upper_trap_lt.append(read_row(row[7]))
                        lower_trap_rt.append(read_row(row[8]))
                        erect_spin_long_rt.append(read_row(row[9]))
                        rect_abdom_lo_rt.append(read_row(row[10]))
                entry["time"] = time
                entry["upper_trap_rt"] = upper_trap_rt
                entry["mid_delt_rt"] = mid_delt_rt
                entry["pec_major_rt"] = pec_major_rt
                entry["infraspin_rt"] = infraspin_rt
                entry["upper_trap_lt"] = upper_trap_lt
                entry["lower_trap_rt"] = lower_trap_rt
                entry["erect_spin_long_rt"] = erect_spin_long_rt
                entry["rect_abdom_lo_rt"] = rect_abdom_lo_rt
                emg_data.append(entry)
    return emg_data


def rectify_emg(emg_data):
    # takes all emg data (see load_emg) and makes all values in every channel positive
    for entry in emg_data:
        rectified_data = {}
        for key, timeseries in entry.items():
            if key != "time" and key != "filename":
                entry[key] = list(np.abs(timeseries))
    return emg_data

def bandpass_emg(emg_data: list, freq_range: list[float], sample_rate: float, poles: int = 5):
    # takes all emg_data (see load_emg) and applies a bandpass filter to all emg channels
    # returns the filtered data with identical structure to emg_data
    emg_data_filtered = []
    for entry in emg_data:
        #data needs to be ndarray for signal.sosfiltfilt()
        filter = signal.butter(poles, freq_range, 'bandpass', fs=sample_rate, output='sos')

        filtered_data = {}
        for key, values in entry.items():
            if key not in ['filename', 'time']:
                values = np.asarray(values)
                filtered = signal.sosfiltfilt(filter, values)
                filtered_data[key] = filtered
            else:
                if key == "filename":
                    print(f"Bandpass filtering {entry['filename']}...")        
                filtered_data[key] = values

        emg_data_filtered.append(filtered_data)

    return emg_data_filtered # list of dicts
'''
def rms_sliding_window_emg(emg_data: list, window_size: int, sampling_frequency: int):
    # takes all emg_data (see load_emg) and applies rms with a sliding window to all emg channels

    window_size = int(round((window_size/1000)*sampling_frequency)) # window size to s then x Hz
    
    emg_data_filtered = []
    for entry in emg_data:
        #get window size in nr of samples

        filtered_data = {}
        for key, values in entry.items():

            if key == "filename":
                print(f"Applying RMS to {entry['filename']}...")
                filtered_data[key] = values

            elif key == "time":
                filtered_data[key] = values[:len(values)-window_size]

            else:
                nr_samples = len(values)
                rms_data = []

                for index in range(nr_samples-window_size):
                    #calculate rms for each window, add to rms_data
                    window = values[index:index+window_size]
                    sum = 0
                    for xi in window:
                        sum = sum + xi**2

                    rms = np.sqrt((sum/window_size))
                    rms_data.append(rms)
                filtered_data[key] = rms_data

        emg_data_filtered.append(filtered_data)

    return emg_data_filtered # list of dicts
'''

# using convolve instead of nested for loops (much faster)
def rms_sliding_window_emg(emg_data: list, window_size: int, sampling_frequency: int):
    # takes all emg_data (see load_emg) and applies rms with a sliding window to all emg channels
    # window_size is given in milliseconds

    window_size = int(round((window_size / 1000) * sampling_frequency))
    kernel = np.ones(window_size) / window_size

    emg_data_filtered = []
    for entry in emg_data:
        print(f"Applying RMS to {entry['filename']}...")
        filtered_data = {}
        n_out = len(entry["time"]) - window_size  # same output length as original

        for key, values in entry.items():
            if key == "filename":
                filtered_data[key] = values

            elif key == "time":
                filtered_data[key] = values[:n_out]

            else:
                values = np.asarray(values)
                # moving average of the squared signal, then sqrt -> RMS
                # ensure same shape as input! edge effects don't really matter here,
                # but timing does (sync to video/pose data)
                mean_square = np.convolve(values**2, kernel, mode="same") # "same" ensure same shape as input
                filtered_data[key] = np.sqrt(mean_square[:n_out])

        emg_data_filtered.append(filtered_data)

    return emg_data_filtered  # list of dicts

# preprocess emg
def make_mvc_stacks(emg_data, peak_value_window=200, sample_rate=1500):
    # returns a dict with an mvc stack for each subject (e.g. "A", "J", ...).
    # Each MVC stack is a dict with the subject's MVC value for each muscle. 
    # MVC stacks are used to normalize data (see normalize_emg).
    #
    # example: {"A": {"upper_trap_rt": 5.503091122, "mid_delt_rt": 24.71023326, ...}, "B": ...}
    #
    # MVC is calculated from "peak value" method, where we find the
    # highest mean value over a specified window for each channel
    # using a moving average filter (convolution).
    #
    # NOTE: create the MVC stack AFTER preprocessing, such as bandpass and rms smoothing
    #
    # args: 
    # - emg_data (list): list of dicts with file name, timestamp and emg channels (see load_emg)
    # - peak_value_window (float): legth of window for finding the highest mean value (peak window)
    #                              for each emg channel, in milliseconds. Longer windows will lead 
    #                              to lower MVC values but more robustness against outliers.
    # - sample_rate: sampling frequency of emg data in Hz 
    # 
    window_size = int(round((peak_value_window / 1000) * sample_rate))
    kernel = np.ones(window_size) / window_size # for example [1/3, 1/3, 1/3] if window_size=3    

    mvc_stacks = {}

    # loop through subjects
    for entry in emg_data:
        filename = entry["filename"]

        if not filename.endswith("MVC.csv"): # assumes the correct MVC file name ends with "MVC"
            continue

        subject = filename[0] # assumes file name always starts with subject letter (e.g. "K", "L")

        subject_mvc_stack = {}
        # loop through emg channels (muscles)
        for key, values in entry.items():
            if key in ["filename", "time"]:
                continue
            window_means = np.convolve(np.asarray(values), kernel, mode="valid")
            subject_mvc_stack[key] = np.max(window_means) # key is the muscle

        mvc_stacks[subject] = subject_mvc_stack

    return mvc_stacks


def normalize_emg(emg_data, mvc_stacks):
    # takes all emg data (see load_emg) and normalizes emg channels against the MVC value
    # for its corresponding subject from the emg stack, based on file name (e.g. "A", "B",...) 
    emg_data_normalized = []

    for entry in emg_data:
        normalized_data = {}

        subject = entry["filename"][0] # assumes first letter in file name corresponds to subject

        for key, value in entry.items():
            if key not in ["filename", "time"]:
                normalized_data[key] = np.asarray(value) / mvc_stacks[subject][key]
            else:
                normalized_data[key] = value

        emg_data_normalized.append(normalized_data)

    return emg_data_normalized        



if __name__ == "__main__":
    emg_path = r"C:\Users\01ols\Desktop\Fiol Experiment Projekt HT26\Goteborg_1\emg\K"

    emg_data = load_emg(emg_path)
    print(emg_data[0]["upper_trap_rt"][:20])
    emg_data = rectify_emg(emg_data)
    print(emg_data[0]["upper_trap_rt"][:20])

    # order: bandpass -> rms -> normalization
    emg_data = bandpass_emg(emg_data, freq_range=[20, 500], sample_rate=1500)
    print(emg_data[0]["upper_trap_rt"][:20])
    emg_data = rms_sliding_window_emg(emg_data, window_size=75, sampling_frequency=1500)

    mvc_stacks = make_mvc_stacks(emg_data, peak_value_window=200, sample_rate=1500)
    print(mvc_stacks)
    emg_data = normalize_emg(emg_data, mvc_stacks)
    print(emg_data[0]["upper_trap_rt"][:20])