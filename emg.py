import os
from tqdm import tqdm
import csv


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
    #                  a key "time", and a key for each muscle timeseries.
    #                  The specific muscles are hard-coded in this function (for now)
    #
    emg_data = []
    for subject in tqdm(os.listdir(emg_path), unit=" subject"):
        for filename in subject:
            if filename.endswith('.csv'):
                sample = {}
                sample["filename"] = filename
                with open(os.path.join(subject, filename), 'r') as f:
                    reader = csv.reader(f, delimiter=';')

                    time, upper_trap_rt, mid_delt_rt, pec_major_rt, infraspin_rt, upper_trap_lt, lower_trap_rt, erect_spin_long_rt, rect_abdom_lo_rt = [], [], [], [], [], [], [], [], []
                    for row in list(reader)[4:]: # actual data starts from 5th row
                        time.append(float(row[0]))
                        upper_trap_rt.append(float(row[3]))
                        mid_delt_rt.append(float(row[4]))
                        pec_major_rt.append(float(row[5]))
                        infraspin_rt.append(float(row[6]))
                        upper_trap_lt.append(float(row[7]))
                        lower_trap_rt.append(float(row[8]))
                        erect_spin_long_rt.append(float(row[9]))
                        rect_abdom_lo_rt.append(float(row[10]))
                sample["time"] = time
                sample["upper_trap_rt"] = upper_trap_rt
                sample["mid_delt_rt"] = mid_delt_rt
                sample["pec_major_rt"] = pec_major_rt
                sample["infraspin_rt"] = infraspin_rt
                sample["upper_trap_lt"] = upper_trap_lt
                sample["lower_trap_rt"] = lower_trap_rt
                sample["erect_spin_long_rt"] = erect_spin_long_rt
                sample["rect_abdom_lo_rt"] = rect_abdom_lo_rt
                emg_data.append(sample)
        return emg_data

# preprocess emg
def normalize_emg(emg_data):
    pass