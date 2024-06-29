"""
path_generator.py

Generates paths to the task objective.
"""

import os
import numpy as np
import pandas as pd
import random
import utilities.table_parameters as tp
from scipy.spatial.transform import Rotation

class PathGenerator:
    """Generates different paths towards the task objective."""

    def __init__(self, task="REST"):
        """Initializes the PathGenerator with a task and completion time.

        Args:
            task (str): The task for which to generate the path.
        """
        self.task = task
        self.completion_time = tp.TASK_COMPLETION_TIME_DICTIONARY[self.task]
        self.start_position = tp.INITIAL_STATES[:6].copy()
        self.current_position = tp.INITIAL_STATES[:6].copy()
        self.final_position = tp.TASK_REFERENCE_DICTIONARY[self.task]
        self.T_STEP = tp.T_STEP

        # Variables for the look_up function
        self.look_up_path = None
        self.look_up_group_type = "LF"
        self.look_up_completion_time_max = self.completion_time
        self.look_up_data = None
        self.look_up_task_times = None
        self.look_up_current_positions = None
        self.look_up_completion_time = None

    def straight(self, current_time):
        """Generate a path directly towards the objective.

        Args:
            current_time (float): The current simulation time.
        """
        if current_time < self.completion_time:
            self.current_position = (
                self.start_position
                + (self.final_position - self.start_position) * (current_time / self.completion_time)
            )
        else:
            self.current_position = self.final_position.copy()

        return self.current_position
    
    def sequential(self, current_time):
        """Generate where degrees of freedom are sequentially completed"""
        subtasks = self.start_position != self.final_position
        number_of_subtasks = np.count_nonzero(subtasks)
        subtask_indexes = np.where(subtasks)
        if number_of_subtasks > 0 and current_time < self.completion_time:
            time_per_subtask = self.completion_time / number_of_subtasks
            current_subtask = int(current_time / time_per_subtask)
            task_number = subtask_indexes[0][current_subtask]

            # Create an array for the path
            self.current_position[:task_number] = self.final_position[:task_number]
            self.current_position[task_number + 1:] = self.start_position[task_number + 1:]
            self.current_position[task_number] = (
                self.start_position[task_number]
                + (self.final_position[task_number] - self.start_position[task_number]) * (
                        (current_time % time_per_subtask) / time_per_subtask)
            )

        else:
            self.current_position = self.final_position
        return self.current_position

    def average(self):
        """Generate a path that is the average of all other paths."""
        pass

    def look_up(self, current_time, group=None):
        """Look up a path from the real data."""
        if self.look_up_data is None:
            # Find an above-average group
            current_loop = 0
            max_loops = 100
            random_group = False if group is not None else True
            while self.look_up_path == None:
                if random_group:
                    group = random.choice(tp.GROUPS)
                path = os.path.join(tp.RAW_DATA_FOLDER_PATH, group, tp.SUBFOLDER, self.task)
                if not os.path.exists(path):
                    continue
                file_prefix = f'Meta_{group}_{self.task}_{self.look_up_group_type}_'
                file_suffix = "RPY.csv"
                possible_files = [i for i in os.listdir(path) if i.startswith(file_prefix) and i.endswith(file_suffix)]
                if not possible_files:
                    continue
                trial_file = random.choice(possible_files)
                full_path = os.path.join(path, trial_file)
                data = pd.read_csv(full_path)
                completion_time = data['Pos_T'].iloc[-1] - data['Pos_T'].iloc[0]
                
                if completion_time < self.look_up_completion_time_max:
                    self.look_up_path = full_path
                elif current_loop > max_loops:
                    print("Can't find good enough group")
                    return self.final_position
                else:
                    current_loop += 1

            self.look_up_data = pd.read_csv(self.look_up_path)
            self.look_up_task_times = self.look_up_data["Pos_T"].values - self.look_up_data["Pos_T"].values[0]
            self.look_up_completion_time = self.look_up_data["Pos_T"].iloc[-1] - self.look_up_data["Pos_T"].iloc[0]
            self.look_up_current_positions = np.array([
                self.look_up_data["Pos_X"].values,
                self.look_up_data["Pos_Y"].values,
                self.look_up_data["Pos_Z"].values,
                self.look_up_data["Pos_Phi"].values,
                self.look_up_data["Pos_Tht"].values,
                self.look_up_data["Pos_Psi"].values
            ])

        # The length of time to interpolate between the last data point and the real target
        interpolation_time = 1
        if current_time < self.look_up_completion_time:
            index = np.argmin(np.abs(self.look_up_task_times - current_time))
            self.current_position = self.look_up_current_positions[:, index].reshape(-1,1)
            self.current_position[:3] += self.start_position[:3]
        elif current_time < self.look_up_completion_time + interpolation_time:
            # Interpolate data points for a smooth path from the last recorded point to the real target

            last_recorded_position = self.look_up_current_positions[:, -1].reshape(-1,1).copy()
            last_recorded_position[:3] += self.start_position[:3]
            self.current_position = (
                last_recorded_position
                + (self.final_position - last_recorded_position) * ((current_time - self.look_up_completion_time) / interpolation_time)
            )
        else:
            self.current_position = self.final_position

        return self.current_position

    def x_translation():
        """Generate a path where the path prioritizes x translation"""
        # Hypothesized to be too similar to real data (average) to be worth time
        pass

    def smooth(self):
        """Generate a path that is a smooth path to the target."""
        # This feels too similar to straight to be worth time at the moment
        pass
