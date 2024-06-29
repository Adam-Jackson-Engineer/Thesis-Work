"""
plot_learned_follower_data.py

This script imports data from the log file and generates useful plots from the data to evaluate 
the learned models

Author: Adam Jackson
Data: June 27, 2024
"""

import numpy as np
import matplotlib
import pandas as pd
import os

def main():
    """
    The main script for plotting and saving data
    """
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOTS_FOLDER = "Learned Follower Plots"
    full_plots_folder = os.path.join(CURRENT_DIR, PLOTS_FOLDER)
    if not os.path.exists():
        os.mkdir(full_plots_folder)
    log_file_name = "learned_models_log.csv"
    path_to_data = os.path.join(CURRENT_DIR, "..", log_file_name)

    learned_follower_data = pd.read_csv(path_to_data)

    models = learned_follower_data["Model Name"].values

    for model in models:
        model_save_folder = os.path.join(full_plots_folder, model)
        if os.path.exists(model_save_folder):
            os.rmdir(model_save_folder)
        plot_velocities(model_save_folder)
        plot_forces(model_save_folder)

def plot_velocities():
    """
    Plots all the data for the learned model comparing the simulation velocity with the """
    pass

def plot_forces():
    pass






