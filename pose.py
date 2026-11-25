import os
import csv
import re

BODY_PARTS = [
    "nose", "left_eye", "right_eye",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

# regular expression that only matches desired body parts (and only 3D pose)
# captures body part and axis (x, y or z)
COL_PATTERN = re.compile(rf"^({'|'.join(BODY_PARTS)})_3D([xyz])$")

def load_pose(pose_path):
    pose_data = []

    for filename in os.listdir(pose_path):
        print(f"Loading pose data from {filename}...")
        with open (os.path.join(pose_path, filename), 'r') as f:
            reader = csv.reader(f, delimiter=';')
            header = next(reader)

            col_keys = {} # maps column index to body part
            for i, name in enumerate(header):
                match = COL_PATTERN.match(name.strip())
                if match:
                    body_part, axis = match.groups()
                    col_keys[i] = f"{body_part}_3D_{axis}"

            # initialize pose data entry with filename and empty lists
            entry = {"filename": filename, "frame": [], "time": []}
            entry.update({key: [] for key in col_keys.values()})

            # fill empty lists
            for row in reader:
                entry["frame"].append(row[0])
                entry["time"].append(float(row[1].replace(',', '.'))) # dumbass blazepose writes comma instead of dot
                for i, key in col_keys.items():
                    entry[key].append(float(row[i].replace(',', '.')))

            pose_data.append(entry)
    return pose_data

if __name__ == "__main__":
    pose_path = r"C:\Users\01ols\Desktop\Fiol Experiment Projekt HT26\Goteborg_1\pose"
    pose_data = load_pose(pose_path)

    print(pose_data[0]["right_ankle_3D_z"][:20])
    


