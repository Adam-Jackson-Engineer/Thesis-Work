"""
fix_euler_derivatives.py

Previously the derivatives of the euler angles was found by differentiating the quaternion
then converting that derivative to euler angles, this was a mistake as that is a non-sensical
number. The script goes through and fixes the derivatives to be numerical derivatives of the 

Author: Adam Jackson
Date: June 28, 2024
"""

import os
import pandas as pd
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from data_param import GROUPS, TASKS, GROUPS_TYPES, RAW_DATA_FOLDER_PATH, SUBFOLDER
from scipy.spatial.transform import Rotation

# Constants for directory names and groups of data
EULER_ANGLE_TYPES = ['Pos', 'Vel', 'Acc']

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

                    if file_name.startswith(file_prefix) and file_name.endswith("RPY.csv"):
                        print(file_name)
                        if not os.path.exists(os.path.join(path,file_name.replace(".csv", ".png"))):
                            data = fix_euler_derivatives(os.path.join(path, file_name))
                            plot_euler_derivatives(data, os.path.join(path, file_name))
                            del data

def fix_euler_derivatives(file_path):
    """
    Fix the Euler Derivatives.

    Args:
        file_path: A string path to the CSV file containing all data.

    Takes the numerical derivative of the positional euler data and overwrites
    the current erroneous one.
    """
    data = pd.read_csv(file_path)
    for i in range(len(EULER_ANGLE_TYPES)-1):
        euler_type = EULER_ANGLE_TYPES[i]
        euler_deriv = EULER_ANGLE_TYPES[i+1]
        if f'{euler_type}_Phi' in data.columns:
            # Get the time step of the data
            time_series = data[f'Pos_T'].values
            time_step = mean(np.diff(time_series))

            # Derivative Parameters that match the way it is done in the learned model
            sigma = 0.05
            beta = (2 * sigma - time_step) / (2 * sigma + time_step)

            phi_all = data[f'{euler_type}_Phi'].values
            theta_all = data[f'{euler_type}_Tht'].values
            psi_all = data[f'{euler_type}_Psi'].values


            if euler_type == "Pos":
                phi_all = correct_wrap_around(phi_all)
                theta_all = correct_wrap_around(theta_all)
                psi_all = correct_wrap_around(psi_all)

            phi_d1 = phi_all[0]
            theta_d1 = theta_all[0]
            psi_d1 = psi_all[0]

            phi_dot = 0.0
            theta_dot = 0.0
            psi_dot = 0.0

            for j in range(len(phi_all)):
                phi = phi_all[j]
                theta = theta_all[j]
                psi = psi_all[j]

                phi_dot = beta * phi_dot + (1 - beta) * ((phi - phi_d1) / time_step)
                theta_dot = beta * theta_dot + (1 - beta) * ((theta - theta_d1) / time_step)
                psi_dot = beta * psi_dot + (1 - beta) * ((psi - psi_d1) / time_step)

                phi_d1 = phi
                theta_d1 = theta
                psi_d1 = psi

                data[f'{euler_deriv}_Phi'][j] = phi_dot
                data[f'{euler_deriv}_Tht'][j] = theta_dot
                data[f'{euler_deriv}_Psi'][j] = psi_dot

        else:
            print(f"Error: No Euler Data at: {file_path}")
            input("What do you want to do?")

    # Overwrite the origional file
    data.to_csv(file_path, index=False)
    # print(f"Updated data saved to {file_path}")
    return data

def correct_wrap_around(euler_series):
    """
    Takes in an array of data points and validates that the wrap around are handles, 
    if the data jumps to negative, it will just add it to the top such that the data wraps

    Args:
    euler_series (np.array): The euler series without compensated wrap arounds

    Returns:
    euler_series (np.array): The euler series this wrap arounds fixed such that the derivative is continuous 
    """
    tolerance = np.pi
    euler_prev = euler_series[0]
    for i in range(len(euler_series)):
        euler_step = euler_series[i] - euler_prev
        if abs(euler_step) > tolerance:
            euler_series[i] -= np.sign(euler_prev) * 2 * np.pi
        euler_step = euler_series[i] - euler_prev
        if abs(euler_step) > tolerance:
            input(f"Problem in wrap around at index:    {i} ")
        euler_prev = euler_series[i]
    return euler_series

def plot_euler_derivatives(data, file_path):
    """
    Takes in the data and plots the euler angles and their derivates, then saves them in the file 
    location same as the rest
    
    Arg:
    data (pandas dataframe): All of the raw data
    file_path (str): The path to the csv file where all this data is saved
    """


    time_series = data["Pos_T"].values
    time_series = time_series - time_series[0]
    fig, axes = plt.subplots(1, 3, figsize = (18, 6))
    

    axes[0].plot(time_series, data["Pos_Phi"], label="Phi")
    axes[0].plot(time_series, data["Pos_Tht"], label="Theta")
    axes[0].plot(time_series, data["Pos_Psi"], label="Psi")
    axes[0].set_title("Euler Positions")
    axes[0].legend()

    axes[1].plot(time_series, data["Vel_Phi"], label="Phi")
    axes[1].plot(time_series, data["Vel_Tht"], label="Theta")
    axes[1].plot(time_series, data["Vel_Psi"], label="Psi")
    axes[1].set_title("Euler Velocities")
    axes[1].legend()

    axes[2].plot(time_series, data["Acc_Phi"], label="Phi")
    axes[2].plot(time_series, data["Acc_Tht"], label="Theta")
    axes[2].plot(time_series, data["Acc_Psi"], label="Psi")
    axes[2].set_title("Euler Accelerations")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(file_path.replace(".csv", ".png"))
    plt.close()

if __name__ == '__main__':
    process_files()