# Functions for processing pose and emg data jointly, and for creating and splitting the dataset

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, random_split

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, KFold, LeaveOneGroupOut


def segment_and_pair_data(emg_data, pose_data, segment_length):
    '''
    Args:
        emg_data:
            List of dicts containing EMG data channels (e.g. "upper_trap_rt"), see load_emg in emg.py

            Each dict has one key "filename" with a string value.
            All other values are lists of floats.
            All lists are exactly the same length.
            Pose and emg data both start at t=0 and are already synchronized.

        pose_data:
            Same structure as emg_data, but for pose data, see load_pose in pose.py

            Rather than emg channels, the keys in the dict are axes of tracked body parts (e.g. "right_shoulder_3D_x")
            Pose files are matched to EMG files by filename (e.g. A_R1.csv <-> A_R1_blazepose.csv).
            EMG recordings with no matching pose file (e.g. A_MVC.csv) are skipped.
            Order of the two lists does not matter.
            Pose and emg data both start at t=0 and are already synchronized.

        segment_length:
            Length of each segment in seconds.

    Returns:
        pose_emg_segment_pairs:
            List of dictionaries. Each dictionary contains:
                - emg_filename
                - pose_filename
                - start_time
                - end_time
                - emg_segment
                - pose_segment

            emg_segment and pose_segment are dictionaries containing
            all the original data channels for that time window.
    '''

    pose_emg_segment_pairs = []

    for emg_entry in emg_data:
        emg_stem = emg_entry["filename"].split(".")[0] # "A_R1.csv" -> "A_R1"

        # find the pose recording for this EMG recording (None for MVC files).
        pose_entry = None
        for candidate in pose_data:
            if candidate["filename"].startswith(emg_stem + "_"):
                pose_entry = candidate
                break

        if pose_entry is None:
            continue

        emg_time = np.asarray(emg_entry["time"])
        pose_time = np.asarray(pose_entry["time"])

        # Determine common end time
        end_time = min(emg_time[-1], pose_time[-1])
        # both emg and pose start at 0

        num_segments = int(end_time // segment_length)

        for segment_idx in range(num_segments):

            segment_start = segment_idx * segment_length
            segment_end = segment_start + segment_length

            # boolean masks that are True for samples within this segment
            matching_times_emg = (emg_time >= segment_start) & (emg_time < segment_end)
            matching_times_pose = (pose_time >= segment_start) & (pose_time < segment_end)

            # segment EMG
            emg_segment = {}
            for key, value in emg_entry.items():
                if key == "filename":
                    continue
                emg_segment[key] = np.asarray(value)[matching_times_emg]

            # segment pose
            pose_segment = {}
            for key, value in pose_entry.items():
                if key == "filename":
                    continue
                pose_segment[key] = np.asarray(value)[matching_times_pose]

            # store paired segment
            pose_emg_segment_pairs.append({
                "emg_filename": emg_entry["filename"],
                "pose_filename": pose_entry["filename"],
                "start_time": segment_start,
                "end_time": segment_end,
                "emg_segment": emg_segment,
                "pose_segment": pose_segment,
            })

    return pose_emg_segment_pairs


# custom dataset class
# input shape: (n_pose_frames, n_pose_channels)
# output shape: (1, n_emg_channels) (average muscle activation for that segment in uV)
class PoseToEMGDataset(Dataset):
    def __init__(self, pose_emg_segment_pairs, pose_segment_length):
        self.pose_segment_length = pose_segment_length

        # drop windows with too few pose frames
        # (float rounding at window edges can give 14 instead of 15)
        pairs = []
        for pair in pose_emg_segment_pairs:
            if len(pair["pose_segment"]["time"]) >= pose_segment_length:
                pairs.append(pair)
        print(f"Kept {len(pairs)}/{len(pose_emg_segment_pairs)} segments ({(len(pairs)/len(pose_emg_segment_pairs)*100):.2f}%)")

        # get key names
        first_pair = pairs[0]
        # ["nose_3D_x", "nose_3D_y", "nose_3D_z", "left_eye_3D_z", ...]
        pose_keys = []
        for key in first_pair["pose_segment"]:
            if key != "time" and key != "filename" and key != "frame":
                pose_keys.append(key)
        self.pose_keys = pose_keys

        # ["mid_delt_rt", "upper_trap_rt", ...]
        emg_keys = []
        for key in first_pair["emg_segment"]:
            if key != "time" and key != "filename":
                emg_keys.append(key)
        self.emg_keys = emg_keys

        # inputs: (N, pose_segment_length, n_pose_channels), crop the occasional 16th frame
        # stack along last dimension n_pose_channels (axis=-1)
        '''pose = np.stack([
            np.stack([pair["pose_segment"][key][:pose_segment_length] for key in self.pose_keys], axis=-1)
            for pair in pairs
        ])'''

        pose_segments = []
        for pair in pairs:
            channels_segments = []
            for key in self.pose_keys:
                channels_segments.append(pair["pose_segment"][key][:pose_segment_length])
            pose_segments.append(np.stack(channels_segments, axis=-1))
        pose = np.stack(pose_segments)


        def downsample_EMG_to_frames(x, n_frames=15):
            # mean within each frame bin
            downsampled_frames = []
            for bin in np.array_split(x, n_frames):
                mean = bin.mean()
                downsampled_frames.append(mean)
            return np.array(downsampled_frames)

        # targets: (N, pose_segment_length, n_emg_channels), EMG downsampled to 15
        # frames by taking average over 15 bins within each segment, matching the
        # length of the pose segment
        # stack along n_emg_channels
        '''emg = np.stack([
            np.stack([downsample_EMG_to_frames(pair["emg_segment"][key], pose_segment_length) for key in self.emg_keys], axis=-1)
            for pair in pairs
        ])'''
        pose_segments = []
        for pair in pairs:
            channels_segments = []
            for key in self.emg_keys:
                channels_segments.append(downsample_EMG_to_frames(pair["emg_segment"][key], pose_segment_length))
            pose_segments.append(np.stack(channels_segments, axis=-1))
        emg = np.stack(pose_segments)


        # create tensors
        self.pose_tensor = torch.from_numpy(pose).float()
        self.emg_tensor = torch.from_numpy(emg).float()

        # keep filenames so you can split by recording
        filenames = []
        for pair in pairs:
            filenames.append(pair["emg_filename"])
        self.filenames = filenames

    #def normalize(self, pose_mean, pose_std, emg_mean, emg_std):

    def __len__(self):
        return len(self.emg_tensor)

    def __getitem__(self, idx):
        return self.pose_tensor[idx], self.emg_tensor[idx]



