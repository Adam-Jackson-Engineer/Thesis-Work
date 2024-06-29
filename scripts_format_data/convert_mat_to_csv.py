import os
import scipy.io
import numpy as np
from data_param import GROUPS, TASKS, GROUPS_TYPES, RAW_DATA_FOLDER_PATH, SUBFOLDER

# Constants for directory names and groups of data
DATA_ABBREVIATIONS = ['Pos', 'Vel', 'Acc', 'CombFTA_', 'CombFTB_']
KINEMATICS = ['T', 'X', 'Y', 'Z', 'W', 'I', 'J', 'K']
KINETICS = ['T', 'X', 'Y', 'Z', 'Phi', 'Tht', 'Psi']

def create_labels():
    """
    Generates all labels for CSV output based on data abbreviations and their respective kinematic 
    or kinetic components.
    
    Returns:
        list: All labels for CSV column headers.
    """
    full_labels = []
    kinematic_data_types = DATA_ABBREVIATIONS[:3]  # corrected variable name
    for data_type in DATA_ABBREVIATIONS:
        components = KINEMATICS if data_type in kinematic_data_types else KINETICS
        full_labels.extend(f'{data_type}_{component}' for component in components)
    return full_labels

def process_mat_files():
    """
    Processes each .mat file from specified directories, converts to CSV with aligned data arrays.
    
    Each MAT file is read, its arrays are truncated to the shortest length, and combined into a single CSV file.
    """
    full_labels = create_labels()
    for group in GROUPS:
        for task in TASKS:
            path = os.path.join(RAW_DATA_FOLDER_PATH, group, SUBFOLDER, task)
            if not os.path.exists(path):
                print(f"Path does not exist: {path}")
                continue

            for group_type in GROUPS_TYPES:
                file_prefix = f'Meta_{group}_{task}_{group_type}_'
                for file_name in filter(lambda name: name.startswith(file_prefix) and name.endswith(".mat"), os.listdir(path)):
                    convert_mat_file(os.path.join(path, file_name), full_labels)


def convert_mat_file(file_path, full_labels):
    """
    Process and convert a single MAT file and save its data as a CSV file.
    
    Args:
        file_path (str): Full path to the MAT file.
        full_labels (list): Labels for CSV column headers.
    """
    mat_data = scipy.io.loadmat(file_path)
    arrays = []
    for data_type in DATA_ABBREVIATIONS:
        variables = [var for var in mat_data if var.startswith(data_type) and mat_data[var].size > 0]
        arrays.extend(mat_data[var] for var in variables)

    if arrays:
        min_length = min(array.shape[1] for array in arrays)
        preprocessed_arrays = [array[:, :min_length] for array in arrays]
        all_data = np.vstack(preprocessed_arrays).T
        save_path = file_path.replace(".mat", ".csv")
        np.savetxt(save_path, all_data, delimiter=",", header=",".join(full_labels), comments='')
        print(f"Saved combined data to {save_path}")
        
if __name__ == '__main__':
    process_mat_files()
