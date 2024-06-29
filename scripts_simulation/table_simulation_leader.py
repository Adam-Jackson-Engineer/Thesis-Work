"""
table_simulation_leader.py

Simulates a table in 3D with force signals applied in a dynamical simulation, controlled by a PID controller.

Author: Adam Jackson
Data:   June 7, 2024
"""

# Import necessary libraries
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
from controllers.leader.pid_step import PIDStep
from controllers.leader.pid_path import PIDPath
from controllers.leader.pid_path_unsat import PIDPathUnsat
from controllers.leader.pid_path_rot_vec_torque import PIDPathRotVecTorque
from controllers.leader.pid_path_euler_torque import PIDPathEulerTorque
from controllers.leader.pid_path_euler_torque_leader_pos import PIDPathEulerTorqueLeaderPos

def main(tasks=None, controller=None, plot_data=False, path=None, record_video=False):
    """
    Main function to run the table simulation with a PID controller.
    
    Args:
        tasks (list): List of task names to simulate.
        controller (PIDControllerStepReference): The PID controller for the simulation.
        plot_data (bool): Flag to plot data during the simulation.
        path (str): Optional path for saving data or plots.
        record_video (bool): Flag to record the animation as a video.
    """
    if tasks is None:
        tasks = ["REST"]
    if controller is None:
        controller = PIDStep()

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

        video_writer = cv2.VideoWriter(filename,codec,fps,resolution)        

        # The ID and screen for the recording
        screen = QGuiApplication.primaryScreen()
        window_id = table_animation.winId()

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
                    u = controller.update(path_states, table_dynamics.state)
                else:
                    u = controller.update(reference_states, table_dynamics.state)
                y = table_dynamics.update(u)

                # Update the data plotter
                if plot_data and data_plotter:
                    data_plotter.update(current_time, reference_states, table_dynamics.state, u)

                current_time += tp.T_STEP
                wait_time = 3
                if path is not None and np.array_equal(path_states, path_reference.final_position) and current_time < tp.T_END - wait_time:
                    current_time = tp.T_END - wait_time

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
        controller.reset()

    # Release video writer
    if record_video:
        video_writer.release()
        cv2.destroyAllWindows()
        print("Recording Closed")
    
    app.exec_()

if __name__ == "__main__":
    # Leader controller options
    controller_step = PIDStep()
    controller_path = PIDPath()
    controller_path_unsat = PIDPathUnsat()
    controller_path_rot_vec_torque = PIDPathRotVecTorque()
    controller_path_euler_torque = PIDPathEulerTorque()
    controller_path_euler_torque_leader_pos = PIDPathEulerTorqueLeaderPos()

    tasks = ["REST",
        "TX","TX_N","TY","TY_N","TZ","TZ_N",
        "RX","RX_N","RY","RY_N","RZ","RZ_N",
        "TXY_PP","TXY_NN","TYZ_NP","RXZ_PN",
        "TXY_RZ_NPP","R_leader"
    ]
    path = "look_up"
    record_video = False

    main(tasks=tasks, controller=controller_path_euler_torque_leader_pos, path=path, record_video=record_video)