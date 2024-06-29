import os
import pandas as pd
import numpy as np
import data_param as P
from data_param import GROUPS, RAW_DATA_FOLDER_PATH, SUBFOLDER, TRAINING_DATA_FOLDER_PATH

def get_parameters(tasks_type, groups_type):
    """
    Determines the appropriate tasks and groups based on provided types.

    Args:
        tasks_type (str): Type of tasks to retrieve.
        groups_type (str): Type of groups to retrieve.

    Returns:
        tuple: (list of tasks, list of groups)
    """

    if tasks_type == "all":
        TASKS = P.TASKS
    elif tasks_type == "2d":
        TASKS = P.TASKS_2D
    elif tasks_type == "2d_1dof":
        TASKS = P.TASKS_2D_1DOF
    else:
        TASKS = P.TASKS
    
    if groups_type == "all":
        GROUPS_TYPES = P.GROUPS_TYPES
    elif groups_type == "follower":
        GROUPS_TYPES = P.FOLLOWER_GROUPS_TYPES
    elif groups_type == "lf":
        GROUPS_TYPES = P.LF_GROUPS_TYPE
    elif groups_type == "lff":
        GROUPS_TYPES = P.LFF_GROUPS_TYPE
    else:
        GROUPS_TYPES = P.GROUPS_TYPES

    return TASKS, GROUPS_TYPES

def generate_training_data(TASKS, GROUPS_TYPES, tasks_type, groups_type, completion_time_max, percentiles):
    """
    Generates training data for each cutoff time and specified task and group types.

    Args:
        TASKS (list): The list of tasks to process.
        GROUPS_TYPES (list): The list of group types to consider.
        tasks_type (string): The ID of the task list
        groups_type (string): The ID of the groups list
    """
    print(f"Removing failed attempts (Cutoff above {completion_time_max}s)")
    save_name = f'training_data_{completion_time_max}_sec_{tasks_type}_{groups_type}.csv'
    save_path = os.path.join(TRAINING_DATA_FOLDER_PATH, save_name)

    if os.path.exists(save_path):
        os.remove(save_path)

    loop_and_parse_data(TASKS, GROUPS_TYPES, completion_time_max, save_path)

    completion_time_max = None
    for percentile in percentiles:
        print(f"\nProcessing cutoff percentile:     {percentile}")
        task_percentile_dictionary = make_percentile_dictionary(save_path, percentile, TASKS)
        save_name = f'training_data_{percentile}_perc_{tasks_type}_{groups_type}.csv'
        save_path = os.path.join(TRAINING_DATA_FOLDER_PATH, save_name)
        if os.path.exists(save_path):
            os.remove(save_path)
        loop_and_parse_data(TASKS, GROUPS_TYPES, completion_time_max, save_path, task_percentile_dictionary)

def loop_and_parse_data(TASKS, GROUPS_TYPES, cutoff_time, save_path, percentile_dictionary=None):
    """
    Processes and appends data to the training dataset if it meets the criteria.

    Args:
        tasks (list): Tasks to process.
        groups_types (list): Group types to consider.
        cutoff_time (int): Time threshold to include data.
        save_path (str): Path to save the compiled CSV file.
        cutoff_type (str): Type of cutoff to be used for the data.
    """

    rounds_used = 0
    for group in GROUPS:
        for task in TASKS:
            path = os.path.join(RAW_DATA_FOLDER_PATH,group,SUBFOLDER,task)
            if not os.path.exists(path):
                continue

            if percentile_dictionary is not None:
                cutoff_time = percentile_dictionary[task]

            for group_type in GROUPS_TYPES:
                file_prefix = f'Meta_{group}_{task}_{group_type}_'
                for file_name in os.listdir(path):
                    if file_name.startswith(file_prefix) and file_name.endswith("RPY.csv"):
                        full_path = os.path.join(path,file_name)
                        data = pd.read_csv(full_path)
                        comp_time = data['Pos_T'].iloc[-1] - data['Pos_T'].iloc[0]
                        if comp_time <= cutoff_time:
                            rounds_used += 1
                            append_data(data, group, task, group_type, comp_time, rounds_used, save_path, file_name)

def append_data(data, group, task, group_type, comp_time, rounds_used, save_path, file_name):
    """
    Appends individual dataset to the main CSV file after adding necessary meta information.

    Args:
        data (DataFrame): The data to append.
        group (str): Group identifier.
        task (str): Task identifier.
        group_type (str): Group type.
        comp_time (float): Completion time.
        rounds_used (int): Number of rounds used.
        save_path (str): Path to save the CSV.
    """
    data['Group'] = group
    data['Task'] = task
    data['Type'] = group_type
    data['CompletionTime'] = comp_time
    data['RoundsUsed'] = rounds_used + 1

    task_location = P.TASK_REFERENCE_DICTIONARY[task]
    data['Task_X'] = task_location[0][0]
    data['Task_Y'] = task_location[1][0]
    data['Task_Z'] = task_location[2][0]
    data['Task_Phi'] = task_location[3][0]
    data['Task_Theta'] = task_location[4][0]
    data['Task_Psi'] = task_location[5][0]

    if not os.path.exists(save_path):
        with open(save_path, 'w') as f:
            f.write(",".join(data.columns) + "\n")
    
    data.to_csv(save_path, mode='a', header=False, index=False)
    # print(f'Added data from {file_name}')

def make_percentile_dictionary(save_path, percentile, tasks):
    """
    Loads the completed training data and creates a dictionary of percentile
    completion times based on the percentile
    
    Args:
    save_path (str): The path to the CSV containing all the completion times
    percentile (int): The percentile cutoff for group performance
    tasks (list): List of tasks to calculate percentiles for
    
    Returns:
    percentile_dictionary (dict): A dictionary of tasks to percentile completion times
    """
    data = pd.read_csv(save_path)
    percentile_dictionary = {}

    for task in tasks:
        completion_times = data['CompletionTime'][data['Task'] == task].unique()
        if len(completion_times) > 0:
            percentile_time = np.percentile(completion_times, 100-percentile)
            percentile_dictionary[task] = percentile_time
        else:
            percentile_dictionary[task] = None

    return percentile_dictionary

if __name__ == "__main__":
    tasks_type = "2d"
    groups_type = "follower"
    completion_time_max = 14
    percentiles = np.array([0, 25, 50, 75, 100])
    TASKS, GROUP_TYPES = get_parameters(tasks_type, groups_type)

    generate_training_data(TASKS, GROUP_TYPES, tasks_type, groups_type, completion_time_max, percentiles)