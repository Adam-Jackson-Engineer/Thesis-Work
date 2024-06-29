"""
task_completion_statistics.py

Cycles through the data to find completion statistics.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities import table_parameters as tp

def find_statistics():
    """
    Cycles through the experiment data, finding the completion times for each task and group type.
    Returns:
        tuple: DataFrames containing raw completion times and statistics.
    """
    all_completion_times = {}

    for group in tp.GROUPS:
        for task in tp.TASKS:
            path = os.path.join(tp.RAW_DATA_FOLDER_PATH, group, tp.SUBFOLDER, task)
            if not os.path.exists(path):
                continue
            for group_type in tp.GROUP_TYPES:
                file_prefix = f'Meta_{group}_{task}_{group_type}_'
                file_suffix = "RPY.csv"
                key = f'{task}_{group_type}'
                if key not in all_completion_times:
                    all_completion_times[key] = []
                for file_name in os.listdir(path):
                    if file_name.startswith(file_prefix) and file_name.endswith(file_suffix):
                        full_path = os.path.join(path, file_name)
                        data = pd.read_csv(full_path)
                        completion_time = data['Pos_T'].iloc[-1] - data['Pos_T'].iloc[0]
                        all_completion_times[key].append(completion_time)

    # Create the subfolder for saving the CSV files
    output_folder = os.path.join(os.path.dirname(__file__), 'completion_statistics')
    os.makedirs(output_folder, exist_ok=True)

    # Save the raw completion times to a CSV file
    raw_times_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_completion_times.items()]))
    raw_times_df.to_csv(os.path.join(output_folder, 'raw_completion_times.csv'), index=False)

    # Calculate basic statistics, excluding outliers (tasks registering uncompleted tasks around 14.9)
    stats = {}
    for key, times in all_completion_times.items():
        series = pd.Series(times)
        stats[key] = {
            'mean': series.mean(),
            'median': series.median(),
            'variance': series.var(),
            'percent_completed': np.mean(series < tp.COMPLETED_TIME) * 100,
            'mean_completed': series[series < tp.COMPLETED_TIME].mean()
        }

    # Convert the stats dictionary to a DataFrame
    stats_df = pd.DataFrame(stats).transpose()

    # Save the main statistics DataFrame
    stats_df.to_csv(os.path.join(output_folder, 'completion_times_statistics.csv'), index=True)

    # Save additional statistics for each group type, maintaining task-specific stats
    for group_type in tp.GROUP_TYPES:
        group_type_keys = [key for key in stats_df.index if key.endswith(f'_{group_type}')]
        group_type_df = stats_df.loc[group_type_keys]
        group_type_df.to_csv(os.path.join(output_folder, f'{group_type}_statistics.csv'), index=True)

    # Save statistics for all data excluding group type LL
    all_excluding_ll_keys = [key for key in stats_df.index if not key.endswith('_LL')]
    all_excluding_ll_df = stats_df.loc[all_excluding_ll_keys]
    all_excluding_ll_df.to_csv(os.path.join(output_folder, 'all_excluding_ll_statistics.csv'), index=True)

    return stats_df

if __name__ == "__main__":
    stats_df = find_statistics()
    print(stats_df)
