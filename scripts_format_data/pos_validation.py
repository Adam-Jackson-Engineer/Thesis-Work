"""
This module contains loops through all the csv files, plotting the pos,
vel, acc, CombFTA, and CombFTB. This module is intended to validate that
the conversion from quaternions to roll, pitch, yaw was correct.

Author: Adam Jackson
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from data_param import GROUPS, TASKS, RAW_DATA_FOLDER_PATH, SUBFOLDER

DATA_ABBREVIATIONS = ['Pos','Vel','Acc','CombFTA_','CombFTB_']

def plot_data(data, data_type, file_name):
    """
    Create plots for XYZ coordinates and RPY angles based on available data columns.
    
    Args:
        data (DataFrame): The data loaded from CSV.
        data_type (str): The abbreviation of the data type, e.g., 'Pos'.
        file_name (str): The original file name to be used in plot titles and saving plots.
    """
    t_col = f'{data_type}_T' if f'{data_type}_T' in data.columns else None
    xyz_cols = [f'{data_type}_{axis}' for axis in ['X', 'Y', 'Z'] if f'{data_type}_{axis}' in data.columns]
    rpy_cols = [f'{data_type}_{axis}' for axis in ['Phi', 'Tht', 'Psi'] if f'{data_type}_{axis}' in data.columns]

    if t_col and (xyz_cols or rpy_cols):
        plt.figure(figsize=(12, 6))

        if xyz_cols:
            plt.subplot(1, 2, 1)
            for col in xyz_cols:
                plt.plot(data[t_col], data[col], label=col)
            plt.title(f'{data_type} XYZ for {file_name}')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)

        if rpy_cols:
            plt.subplot(1, 2, 2)
            for col in rpy_cols:
                plt.plot(data[t_col], data[col], label=col)
            plt.title(f'{data_type} RPY for {file_name}')
            plt.xlabel('Time')
            plt.ylabel('Angle (rad)')
            plt.legend()
            plt.grid(True)

        return plt

def save_plot(plt, file_path, data_type):
    """
    Save the plot to a file.

    Args:
        plt (matplotlib.pyplot): The plot object to save.
        file_path (str): The path where the plot will be saved.
        data_type (str): The data type as a suffix for the plot file name.
    """
    plot_save_path = file_path.replace(".csv", f"_{data_type}.png")
    plt.savefig(plot_save_path)
    plt.close()
    print(f"Saved plot to {plot_save_path}")

def loop_and_plot_data():
    """
    Main function to process files and generate plots for each CSV file.
    """
    for group in GROUPS:
        for task in TASKS:
            path = os.path.join(RAW_DATA_FOLDER_PATH, group, SUBFOLDER, task)
            if not os.path.exists(path):
                continue

            for file_name in os.listdir(path):
                if file_name.endswith("RPY.csv"):
                    csv_path = os.path.join(path, file_name)
                    data = pd.read_csv(csv_path)

                    for data_type in DATA_ABBREVIATIONS:
                        plt = plot_data(data, data_type, file_name)
                        if plt:
                            save_plot(plt, csv_path, data_type)

if __name__ == "__main__":
    loop_and_plot_data()