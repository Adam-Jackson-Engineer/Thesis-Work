"""
table_simulation_leader_follower.py

Simulates a table in 3D with force signals applied in a dynamical simulation, controlled by a PID controller.
Takes in a leader controller and a follower controller

Author: Adam Jackson
Data:   June 7, 2024
"""

# Import necessary libraries
import os
import sys
import time
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QOpenGLWidget
from PyQt5.QtGui import QGuiApplication
import utilities.table_parameters as tp

from utilities.table_dynamics import TableDynamics
from utilities.table_animation import TableAnimation
from utilities.data_plotter import DataPlotter
from utilities.path_generator import PathGenerator

# Import leader controllers
from controllers.leader.pid_step import PIDStep as PIDStepLeader
from controllers.leader.pid_path import PIDPath as PIDPathLeader
from controllers.leader.pid_path_unsat import PIDPathUnsat as PIDPathUnsatLeader
from controllers.leader.pid_path_rot_vec_torque import PIDPathRotVecTorque as PIDPathRotVecTorqueLeader
from controllers.leader.pid_path_euler_torque import PIDPathEulerTorque as PIDPathEulerTorqueLeader
from controllers.leader.pid_path_euler_torque_leader_pos import PIDPathEulerTorqueLeaderPos as PIDPathEulerTorqueLeaderPosLeader

# Import follower controllers
from controllers.follower.passive import Passive as PassiveFollower
from controllers.follower.mass_damper import MassDamper as MassDamperFollower
from controllers.follower.learned_follower import Learned as LearnedFollower


def main(tasks=None, leader_controller=None, follower_controller=None, plot_data=False, path=None, record_video=False, log_trials=None, log_file_name=None):
    """
    Main function to run the table simulation with a PID controller.
    
    Args:
        tasks (list): List of task names to simulate.
        leader_controller (PIDControllerStepReference): The PID controller for the simulation.
        plot_data (bool): Flag to plot data during the simulation.
        path (str): Optional path for saving data or plots.
        record_video (bool): Flag to record the animation as a video.
        log_trials (bool): Flag to log trial data.
        log_file_name (str): File name to save the log data.
    """
    if tasks is None:
        tasks = ["REST"]
    if leader_controller is None:
        leader_controller = PIDStepLeader()
    if follower_controller is None:
        follower_controller = PassiveFollower()

    # Instantiate the simulation plots and animation
    app = QApplication(sys.argv)
    table_animation = TableAnimation()
    table_animation.show()

    # Instantiate the table dynamics and controller
    table_dynamics = TableDynamics()
    data_plotter = DataPlotter() if plot_data else None
    path_states = None

    # Video recording setup
    if record_video:
        # cv2 variables
        filename = "leader_controlled_animation.avi"
        codec = cv2.VideoWriter_fourcc(*"XVID")
        fps = 60.0
        resolution = (table_animation.width(), table_animation.height())

        video_writer = cv2.VideoWriter(filename, codec, fps, resolution)        

        # The ID and screen for the recording
        screen = QGuiApplication.primaryScreen()
        window_id = table_animation.winId()

    if log_trials:
        logs_per_sec = 10
        time_since_log = 1/logs_per_sec

    for task in tasks:
        if path is not None:
            path_reference = PathGenerator(task=task)
        # Initialize time and output
        current_time = tp.T_START
        y = table_dynamics.h()

        # Main simulation loop
        while current_time < tp.T_END:
            loop_start_time = time.time()
            # Propagate dynamics in between plot samples
            t_next_plot = current_time + tp.T_PLOT

            # Set the reference states based on the current task
            reference_states = tp.TASK_REFERENCE_DICTIONARY[task]

            while current_time < t_next_plot:
                if path is not None:
                    if path == "straight":
                        path_states = path_reference.straight(current_time)
                    elif path == "sequential":
                        path_states = path_reference.sequential(current_time)
                    elif path == "average":
                        path_states = path_reference.average(current_time)
                    elif path == "look_up":
                        path_states = path_reference.look_up(current_time)
                    else:
                        print("Couldn't find path, defaulting to straight")
                        path_states = path_reference.straight(current_time)
                    u_leader = leader_controller.update(path_states, table_dynamics.state)
                else:
                    u_leader = leader_controller.update(reference_states, table_dynamics.state)
                u_follower = follower_controller.update(table_dynamics.state, u_leader)
                y = table_dynamics.update(u_leader, u_follower)

                # Update the data plotter
                if plot_data and data_plotter:
                    data_plotter.update(current_time, reference_states, table_dynamics.state, u_leader, u_follower)

                current_time += tp.T_STEP
                wait_time = 3
                if path is not None and np.array_equal(path_states, path_reference.final_position) and current_time < tp.T_END - wait_time:
                    current_time = tp.T_END - wait_time

                if log_trials:
                    if time_since_log >= (1 / logs_per_sec):
                        log_accuracy(log_file_name, task, current_time, table_dynamics.state, reference_states, path_states)
                        time_since_log = 0
                    time_since_log += tp.T_STEP


            # Update the animation
            table_animation.update_gl(table_dynamics.state, reference_states=reference_states, path_states=path_states, task=task)
            QApplication.processEvents()

            # Record the frame
            if record_video:
                screenshot = screen.grabWindow(window_id)
                image = screenshot.toImage()
                ptr = image.bits()
                ptr.setsize(image.byteCount())
                frame = np.array(ptr).reshape(image.height(), image.width(), 4)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                video_writer.write(frame)

            # Sleep to maintain the simulation rate
            elapsed_time = time.time() - loop_start_time
            sleep_time = max(0, tp.T_PLOT - elapsed_time)
            time.sleep(sleep_time)

        # Reset the table dynamics state and controller for the next task
        table_dynamics.state = tp.INITIAL_STATES.copy()
        leader_controller.reset()

    # Release video writer
    if record_video:
        video_writer.release()
        cv2.destroyAllWindows()
        print("Recording Closed")
    
    app.exec_()

def log_accuracy(log_file_name, task, current_time, states, reference_states, path_states):
    if not os.path.exists(log_file_name):
        with open(log_file_name, 'w') as f:
            f.write("Task, Current Time, X, Y, Z, Phi, Theta, Psi, "
                    "X_Ref_Error, Y_Ref_Error, Z_Ref_Error, Phi_Ref_Error, Theta_Ref_Error, Psi_Ref_Error, "
                    "X_Path_Error, Y_Path_Error, Z_Path_Error, Phi_Path_Error, Theta_Path_Error, Psi_Path_Error\n")

    x = states[0]
    y = states[1]
    z = states[2]
    phi = states[3]
    theta = states[4]
    psi = states[5]

    # Modify the reference states
    if reference_states is not None:
        x_ref = reference_states[0]
        y_ref = reference_states[1]
        z_ref = reference_states[2]
        phi_ref = reference_states[3]
        theta_ref = reference_states[4]
        psi_ref = reference_states[5]

        x_ref_error = x_ref - x
        y_ref_error = y_ref - y
        z_ref_error = z_ref - z
        phi_ref_error = phi_ref - phi
        theta_ref_error = theta_ref - theta
        psi_ref_error = psi_ref - psi
    else:
        x_ref_error = y_ref_error = z_ref_error = 0
        phi_ref_error = theta_ref_error = psi_ref_error = 0

    # Modify the path states
    if path_states is not None:
        x_path = path_states[0]
        y_path = path_states[1]
        z_path = path_states[2]
        phi_path = path_states[3]
        theta_path = path_states[4]
        psi_path = path_states[5]

        x_path_error = x_path - x
        y_path_error = y_path - y
        z_path_error = z_path - z
        phi_path_error = phi_path - phi
        theta_path_error = theta_path - theta
        psi_path_error = psi_path - psi
    else:
        x_path_error = y_path_error = z_path_error = 0
        phi_path_error = theta_path_error = psi_path_error = 0
    
    with open(log_file_name, 'a') as f:
        f.write(f"{task}, {current_time}, "
                f"{x}, {y}, {z}, {phi}, {theta}, {psi}, "
                f"{x_ref_error}, {y_ref_error}, {z_ref_error}, "
                f"{phi_ref_error}, {theta_ref_error}, {psi_ref_error}, "
                f"{x_path_error}, {y_path_error}, {z_path_error}, "
                f"{phi_path_error}, {theta_path_error}, {psi_path_error}\n")

    





if __name__ == "__main__":
    # # Leader controller options
    # leader_step = PIDStepLeader()
    # leader_path = PIDPathLeader()
    # leader_path_unsat = PIDPathUnsatLeader()
    # leader_path_rot_vec_torque = PIDPathRotVecTorqueLeader()
    # leader_path_euler_torque = PIDPathEulerTorqueLeader()
    leader_path_euler_torque_leader_pos = PIDPathEulerTorqueLeaderPosLeader()

    # # Follower controller options
    # follower_passive = PassiveFollower(gravity=False)

    reaction_time_type="SRT"
    output_type="XYS"
    input_type="tao" 
    task_type="2D"
    cutoff_type="10" 
    model_type="LSTM" 
    generate_random_model=False

    follower_learned = LearnedFollower(reaction_time_type=reaction_time_type,
                                       output_type=output_type,
                                       input_type=input_type,
                                       task_type=task_type,
                                       cutoff_type=cutoff_type,
                                       model_type=model_type,
                                       generate_random_model=generate_random_model)
    # input("Imported Learned Follower")


    mass = np.array([[1],[1],[1],[0.25],[0.25],[0.25]])
    damping = np.array([[1],[1],[1],[0.25],[0.25],[0.25]])
    follower_mass_damper = MassDamperFollower(mass=mass, damping=damping)

    tasks = ["REST",
        "TX","TX_N","TY","TY_N","TZ","TZ_N",
        "RX","RX_N","RY","RY_N","RZ","RZ_N",
        "TXY_PP","TXY_NN","TYZ_NP","RXZ_PN",
        "TXY_RZ_NPP","R_leader"
    ]
    path = "straight"
    record_video = False

    main(tasks=tasks, leader_controller=leader_path_euler_torque_leader_pos, follower_controller=follower_learned, path=path, record_video=record_video)