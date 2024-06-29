"""
Processes experiment data by converting quaternions to Euler angles and saving updates.

This module reads CSV files containing quaternion data, converts them to Euler angles,
and saves the updated files back to disk. Ensure data is already in CSV format before
running.

Author: Adam Jackson
Date: 2024-05-09
Usage:
    Run directly from the command line to process all designated files in specified directories.
"""

import os
import pandas as pd
import numpy as np
from data_param import GROUPS, TASKS, GROUPS_TYPES, RAW_DATA_FOLDER_PATH, SUBFOLDER
from scipy.spatial.transform import Rotation

# Constants for directory names and groups of data
QUATERNION_DATA_TYPES = ['Pos', 'Vel', 'Acc']

def process_files():
    """
    Process all designated CSV data files within the directory structure.

    Iterates over defined group and task combinations, reads the CSV files,
    and converts quaternion data to Euler angles. Skips non-existent paths and
    non-CSV files.
    """
    
    for group in GROUPS:
        for task in TASKS:
            path = os.path.join(RAW_DATA_FOLDER_PATH, group, SUBFOLDER, task)
            if not os.path.exists(path):
                print(f"Path does not exist: {path}")
                continue
            for group_type in GROUPS_TYPES:
                file_prefix = f'Meta_{group}_{task}_{group_type}_'
                for file_name in os.listdir(path):
                    if file_name.startswith(file_prefix) and file_name.endswith(".csv") and not file_name.endswith("RPY.csv"):
                        convert_quat_to_eul(os.path.join(path, file_name))
                        
def convert_quat_to_eul(file_path):
    """
    Convert quaternion data in a file to Euler angles and save the updated file.

    Args:
        file_path: A string path to the CSV file containing quaternion data.

    Converts quaternion data to Euler angles 'zyx', and saves it back to a new
    file appending '_RPY' before the '.csv' extension. Prints the path of the
    saved file.
    """
    data = pd.read_csv(file_path)
    for data_type in QUATERNION_DATA_TYPES:
        if f'{data_type}_W' in data.columns:
            quaternions = np.stack([
                data[f'{data_type}_I'].values,
                data[f'{data_type}_J'].values,
                data[f'{data_type}_K'].values,
                data[f'{data_type}_W'].values
            ], axis=1)
            euler_angles = Rotation.from_quat(quaternions).as_euler('ZYX', degrees=False)
            data[f'{data_type}_Phi'] = euler_angles[:, 2]
            data[f'{data_type}_Tht'] = euler_angles[:, 1]
            data[f'{data_type}_Psi'] = euler_angles[:, 0]

    updated_file_path = file_path.replace(".csv", "_RPY.csv")
    data.to_csv(updated_file_path, index=False)
    print(f"Updated data saved to {updated_file_path}")

if __name__ == '__main__':
    process_files()