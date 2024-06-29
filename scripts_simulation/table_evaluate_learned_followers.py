"""
evaluate_learned_followers.py

Cycle through all of the models and record stats on them
Author: Adam Jackson
Data:   June 27, 2024
"""

# Import necessary libraries
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import utilities.table_parameters as tp

from utilities.table_dynamics import TableDynamics
from utilities.path_generator import PathGenerator

# Import leader controllers
from controllers.leader.pid_path_euler_torque_leader_pos import PIDPathEulerTorqueLeaderPos as PIDPathEulerTorqueLeaderPosLeader

# Import follower controllers
from controllers.follower.passive import Passive as PassiveFollower
from controllers.follower.learned_follower import Learned as LearnedFollower

def main(tasks=None, leader_controller=None, follower_controller=None, path=None, model_name=None, log_file_name="learned_model_data.csv"):
    """
    Main function to run the table simulation with a PID controller.
    
    Args:
        tasks (list): List of task names to simulate.
        leader_controller (Class): The controller for the leader for the simulation.
        follower_controller (Class): The controller for the follower for the simulation 
        path (str): Optional path for saving data or plots.
    """
    # Make sure there are tasks and controllers
    if tasks is None:
        tasks = ["REST"]
    if leader_controller is None:
        leader_controller = PIDPathEulerTorqueLeaderPosLeader()
    if follower_controller is None:
        follower_controller = PassiveFollower()

    # Instantiate the table dynamics
    table_dynamics = TableDynamics()
    path_states = None

    # Simulate for each individual task
    for task in tasks:
        # Generate path to save folder
        save_folder_path = generate_model_save_path(path, model_name, task)
        full_log_save_path = os.path.join(save_folder_path, log_file_name)
        if os.path.exists(full_log_save_path):
            os.remove(full_log_save_path)

        if path is not None:
            path_reference = PathGenerator(task=task)
        else:
            path_reference = None

        # Initialize time and output
        current_time = tp.T_START

        # Set the reference states for the current task
        reference_states = tp.TASK_REFERENCE_DICTIONARY[task]

        # Main simulation loop
        while current_time < tp.T_END:
            # Pass the task states into the controller if there isn't a path
            if path is not None:
                # Generate the path according to the input
                path_states = get_path_states(path, current_time, path_reference)
                u_leader = leader_controller.update(path_states, table_dynamics.state)
            else:
                u_leader = leader_controller.update(reference_states, table_dynamics.state)
            u_follower = follower_controller.update(table_dynamics.state, u_leader)
            table_dynamics.update(u_leader, u_follower)

            current_time += tp.T_STEP

            # If the path has reached the task, stop simulation after X seconds
            wait_time = 3
            if path is not None and np.array_equal(path_states, path_reference.final_position) and current_time < tp.T_END - wait_time:
                current_time = tp.T_END - wait_time

            # Log the simulation data
            log_accuracy(full_log_save_path, task, path, current_time, table_dynamics.state, u_leader, u_follower, path_reference, follower_controller)

        # Reset the table dynamics state and controller for the next task
        plot_accuracy(save_folder_path, full_log_save_path)
        table_dynamics.state = tp.INITIAL_STATES.copy()
        leader_controller.reset()

def get_path_states(path_type, current_time, path_reference):
    """
    Generates the states of the path at the current time

    Args:
    path_type (str): The type of path to follow
    current_time (float): The current time in the simulation
    """

    # Get the states from the path reference class
    if path_type == "straight":
        path_states = path_reference.straight(current_time)
    elif path_type == "sequential":
        path_states = path_reference.sequential(current_time)
    elif path_type == "average":
        path_states = path_reference.average(current_time)
    elif path_type == "look_up":
        path_states = path_reference.look_up(current_time)
    else:
        print("Couldn't find path type, defaulting to straight")
        path_states = path_reference.straight(current_time)

    return path_states

def generate_model_save_path(path, model_name, task):
    """
    Makes sure the folder to save the data exists and generates a path for it.
    
    Args:
    path (str): The type of path the controller is following
    model_name (str): The name of the learned follower model
    task (str): The current task
    """
    current_directory = os.path.dirname(os.path.abspath(__file__))
    model_save_path = os.path.join(current_directory, "references", "learned_model_data", path, model_name, task)
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    return model_save_path

def log_accuracy(full_log_save_path, task, path, current_time, states, u_leader, u_follower, path_reference, follower_controller):
    """
    Saves the data from the task being performed to a csv for later use
    
    Args:
    full_log_save_path (str): The save path data to save as a csv
    task (str): The name of the current task
    path (str): The label for the path type
    current_time (float): The current time in the simulation
    states (np.array): The states of the table at the current time
    u_leader (np.array): The leader controller forces in the simulation
    u_follower (np.array): The follower controller forces in the simulation
    path_reference (Class): The class generating the paths to follow
    follower_controller (Class): The controller for the follower
    """

    # The task reference states, the end goal
    reference_states = tp.TASK_REFERENCE_DICTIONARY[task]

    # The current state of the desired trajectory
    if path_reference is None:
        path_states = np.zeros_like(states)
    else:
        path_states = get_path_states(path, current_time, path_reference)

    # Get real data if the path is from real data
    if path != "look_up":
        u_leader_ref = np.zeros_like(u_leader)
        u_follower_ref = np.zeros_like(u_follower)
        vel_ref = np.zeros_like(u_leader)
    else:
        index = np.argmin(np.abs(path_reference.look_up_task_times - current_time))
        u_leader_ref = np.array([[path_reference.look_up_data["CombFTA__X"].values[index]],
                                 [path_reference.look_up_data["CombFTA__Y"].values[index]],
                                 [path_reference.look_up_data["CombFTA__Z"].values[index]],
                                 [path_reference.look_up_data["CombFTA__Phi"].values[index]],
                                 [path_reference.look_up_data["CombFTA__Tht"].values[index]],
                                 [path_reference.look_up_data["CombFTA__Psi"].values[index]]])
        
        u_follower_ref = np.array([[path_reference.look_up_data["CombFTB__X"].values[index]],
                                   [path_reference.look_up_data["CombFTB__Y"].values[index]],
                                   [path_reference.look_up_data["CombFTB__Z"].values[index]],
                                   [path_reference.look_up_data["CombFTB__Phi"].values[index]],
                                   [path_reference.look_up_data["CombFTB__Tht"].values[index]],
                                   [path_reference.look_up_data["CombFTB__Psi"].values[index]]])
        
        vel_ref = np.array([[path_reference.look_up_data["Vel_X"].values[index]],
                            [path_reference.look_up_data["Vel_Y"].values[index]],
                            [path_reference.look_up_data["Vel_Z"].values[index]],
                            [path_reference.look_up_data["Vel_Phi"].values[index]],
                            [path_reference.look_up_data["Vel_Tht"].values[index]],
                            [path_reference.look_up_data["Vel_Psi"].values[index]]])
        
    # Log the data
    if not os.path.exists(full_log_save_path):
        with open(full_log_save_path, 'w') as f:
            f.write("Task,Current Time,Path,Model Name," 
                "X,Y,Z,Phi,Theta,Psi,"
                "X_Vel,Y_Vel,Z_Vel,Phi_dot,Theta_dot,Psi_dot,"
                "X_Ref,Y_Ref,Z_Ref,Phi_Ref,Theta_Ref,Psi_Ref,"
                "X_Path,Y_Path,Z_Path,Phi_Path,Theta_Path,Psi_Path,"
                "X_Comb_A,Y_Comb_A,Z_Comb_A,Phi_Comb_A,Theta_Comb_A,Psi_Comb_A,"
                "X_Comb_B,Y_Comb_B,Z_Comb_B,Phi_Comb_B,Theta_Comb_B,Psi_Comb_B,"
                "X_Vel_look_up,Y_Vel_look_up,Z_Vel_look_up,Phi_dot_look_up,Theta_dot_look_up,Psi_dot_look_up,"
                "X_Comb_A_look_up,Y_Comb_A_look_up,Z_Comb_A_look_up,Phi_Comb_A_look_up,Theta_Comb_A_look_up,Psi_Comb_A_look_up,"
                "X_Comb_B_look_up,Y_Comb_B_look_up,Z_Comb_B_look_up,Phi_Comb_B_look_up,Theta_Comb_B_look_up,Psi_Comb_B_look_up\n")
    
    with open(full_log_save_path, 'a') as f:
        f.write(f"{task},{current_time},{path},{follower_controller.model_name},"
                f"{states[0][0]},{states[1][0]},{states[2][0]},{states[3][0]},{states[4][0]},{states[5][0]},"
                f"{states[6][0]},{states[7][0]},{states[8][0]},{follower_controller.phi_dot},{follower_controller.theta_dot},{follower_controller.psi_dot},"
                f"{reference_states[0][0]},{reference_states[1][0]},{reference_states[2][0]},"
                f"{reference_states[3][0]},{reference_states[4][0]},{reference_states[5][0]},"
                f"{path_states[0][0]},{path_states[1][0]},{path_states[2][0]},"
                f"{path_states[3][0]},{path_states[4][0]},{path_states[5][0]},"
                f"{u_leader[0][0]},{u_leader[1][0]},{u_leader[2][0]},"
                f"{u_leader[3][0]},{u_leader[4][0]},{u_leader[5][0]},"
                f"{u_follower[0][0]},{u_follower[1][0]},{u_follower[2][0]},"
                f"{u_follower[3][0]},{u_follower[4][0]},{u_follower[5][0]},"
                f"{vel_ref[0][0]},{vel_ref[1][0]},{vel_ref[2][0]},"
                f"{vel_ref[3][0]},{vel_ref[4][0]},{vel_ref[5][0]},"
                f"{u_leader_ref[0][0]},{u_leader_ref[1][0]},{u_leader_ref[2][0]},"
                f"{u_leader_ref[3][0]},{u_leader_ref[4][0]},{u_leader_ref[5][0]},"
                f"{u_follower_ref[0][0]},{u_follower_ref[1][0]},{u_follower_ref[2][0]},"
                f"{u_follower_ref[3][0]},{u_follower_ref[4][0]},{u_follower_ref[5][0]}\n")
        
def plot_accuracy(save_folder_path, full_log_save_path):
    """
    Plot the data to be used for analysis
    
    Args:
    save_folder_path (str): The path to save all of the plots
    full_log_save_path (str): The path the data log
    """
    data = pd.read_csv(full_log_save_path)
    current_times = data["Current Time"].values
    for i in range(len(current_times)):
        if i > 0 and current_times[i] - current_times[i-1] > 1:
            current_times[i] = current_times[i-1] + tp.T_STEP
        
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))


    # Translational states (top row)
    axes[0, 0].plot(current_times, data["X"], label="X Position")
    axes[0, 0].plot(current_times, data["Y"], label="Y Position")
    axes[0, 0].plot(current_times, data["Z"], label="Z Position")
    axes[0, 0].plot(current_times, data["X_Path"], label="X Path")
    axes[0, 0].plot(current_times, data["Y_Path"], label="Y Path")
    axes[0, 0].plot(current_times, data["Z_Path"], label="Z Path")
    axes[0, 0].set_title("Translational Position")
    axes[0, 0].legend()

    axes[0, 1].plot(current_times, data["X_Vel"], label="X Velocity")
    axes[0, 1].plot(current_times, data["Y_Vel"], label="Y Velocity")
    axes[0, 1].plot(current_times, data["Z_Vel"], label="Z Velocity")
    axes[0, 1].plot(current_times, data["X_Vel_look_up"], label="X Vel Loop Up")
    axes[0, 1].plot(current_times, data["Y_Vel_look_up"], label="Y Vel Loop Up")
    axes[0, 1].plot(current_times, data["Z_Vel_look_up"], label="Z Vel Loop Up")
    axes[0, 1].set_title("Translational Velocity")
    axes[0, 1].legend()

    axes[0, 2].plot(current_times, data["X_Comb_A"], label="X Leader Force")
    axes[0, 2].plot(current_times, data["Y_Comb_A"], label="Y Leader Force")
    axes[0, 2].plot(current_times, data["Z_Comb_A"], label="Z Leader Force")
    axes[0, 2].plot(current_times, data["X_Comb_A_look_up"], label="X Look Up")
    axes[0, 2].plot(current_times, data["Y_Comb_A_look_up"], label="Y Look Up")
    axes[0, 2].plot(current_times, data["Z_Comb_A_look_up"], label="Z Look Up")
    axes[0, 2].set_title("Translational Leader Forces")
    axes[0, 2].legend()

    axes[0, 3].plot(current_times, data["X_Comb_B"], label="X Follower Force")
    axes[0, 3].plot(current_times, data["Y_Comb_B"], label="Y Follower Force")
    axes[0, 3].plot(current_times, data["Z_Comb_B"], label="Z Follower Force")
    axes[0, 3].plot(current_times, data["X_Comb_B_look_up"], label="X Look Up")
    axes[0, 3].plot(current_times, data["Y_Comb_B_look_up"], label="Y Look Up")
    axes[0, 3].plot(current_times, data["Z_Comb_B_look_up"], label="Z Look Up")
    axes[0, 3].set_title("Translational Follower Forces")
    axes[0, 3].legend()

    # Rotational states (bottom row)
    axes[1, 0].plot(current_times, data["Phi"], label="Phi Position")
    axes[1, 0].plot(current_times, data["Theta"], label="Theta Position")
    axes[1, 0].plot(current_times, data["Psi"], label="Psi Position")
    axes[1, 0].plot(current_times, data["Phi_Path"], label="Phi Path")
    axes[1, 0].plot(current_times, data["Theta_Path"], label="Theta Path")
    axes[1, 0].plot(current_times, data["Psi_Path"], label="Psi Path")
    axes[1, 0].set_title("Rotational Position")
    axes[1, 0].legend()

    axes[1, 1].plot(current_times, data["Phi_dot"], label="Phi Velocity")
    axes[1, 1].plot(current_times, data["Theta_dot"], label="Theta Velocity")
    axes[1, 1].plot(current_times, data["Psi_dot"], label="Psi Velocity")
    axes[1, 1].plot(current_times, data["Phi_dot_look_up"], label="Phi Look Up")
    axes[1, 1].plot(current_times, data["Theta_dot_look_up"], label="Theta Look Up")
    axes[1, 1].plot(current_times, data["Psi_dot_look_up"], label="Psi Look Up")
    axes[1, 1].set_title("Translational Velocity")
    axes[1, 1].legend()

    axes[1, 2].plot(current_times, data["Phi_Comb_A"], label="Phi Leader Force")
    axes[1, 2].plot(current_times, data["Theta_Comb_A"], label="Theta Leader Force")
    axes[1, 2].plot(current_times, data["Psi_Comb_A"], label="Psi Leader Force")
    axes[1, 2].plot(current_times, data["Phi_Comb_A_look_up"], label="Phi Look Up")
    axes[1, 2].plot(current_times, data["Theta_Comb_A_look_up"], label="Theta Look Up")
    axes[1, 2].plot(current_times, data["Psi_Comb_A_look_up"], label="Psi Look Up")
    axes[1, 2].set_title("Translational Leader Forces")
    axes[1, 2].legend()

    axes[1, 3].plot(current_times, data["Phi_Comb_B"], label="Phi Follower Force")
    axes[1, 3].plot(current_times, data["Theta_Comb_B"], label="Theta Follower Force")
    axes[1, 3].plot(current_times, data["Psi_Comb_B"], label="Psi Follower Force")
    axes[1, 3].plot(current_times, data["Phi_Comb_B_look_up"], label="Phi Look Up")
    axes[1, 3].plot(current_times, data["Theta_Comb_B_look_up"], label="Theta Look Up")
    axes[1, 3].plot(current_times, data["Psi_Comb_B_look_up"], label="Psi Look Up")
    axes[1, 3].set_title("Translational Follower Forces")
    axes[1, 3].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder_path, "accuracy_plot.png"))
    plt.close()



if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    path_to_primary_models = os.path.join(tp.TRAINED_MODELS_FOLDER_PATH, "models_primary")
    primary_models = os.listdir(path_to_primary_models)
    path_to_all_models = os.path.join(tp.TRAINED_MODELS_FOLDER_PATH, "models_all")
    all_models = os.listdir(path_to_all_models)

    log_file_name = "learned_models_log.csv"

    if os.path.exists(log_file_name):
        os.remove(log_file_name)
    


    # Leader controller
    leader_path_euler_torque_leader_pos = PIDPathEulerTorqueLeaderPosLeader()

    # Tasks
    tasks = ["REST",
        "TX","TX_N","TY","TY_N","TZ","TZ_N",
        "RX","RX_N","RY","RY_N","RZ","RZ_N",
        "TXY_PP","TXY_NN","TYZ_NP","RXZ_PN",
        "TXY_RZ_NPP","R_leader"
    ]

    PATHS = ['straight', 'sequential', 'look_up']

    for path in PATHS:
        for model_name in primary_models:
            if model_name.endswith(".pth"):
                model_name_split = model_name.split("_")

                reaction_time_type = model_name_split[0]
                output_type = model_name_split[1]
                input_type = model_name_split[2]
                task_type = model_name_split[3]
                cutoff_type = model_name_split[4]
                model_type = model_name_split[5].split(".")[0]

                follower_learned = LearnedFollower(reaction_time_type=reaction_time_type,output_type=output_type,input_type=input_type,
                    task_type=task_type,cutoff_type=cutoff_type,model_type=model_type)

                main(tasks=tasks,leader_controller=leader_path_euler_torque_leader_pos,follower_controller=follower_learned,path=path, model_name=model_name.split(".")[0])
                
                del follower_learned
