"""
table_simulation_dynamic.py

Simulates a table in 3D with force signals applied in a dynamical simulation.
"""

import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication

import utilities.table_parameters as tp
from utilities.signal_generator import SignalGenerator
from utilities.table_animation import TableAnimation
from utilities.table_dynamics import TableDynamics

def main(signal_reference_x=None, signal_reference_y=None, signal_reference_z=None, 
         signal_reference_phi=None, signal_reference_theta=None, signal_reference_psi=None,
         signal_force_x=None, signal_force_y=None, signal_force_z=None, 
         signal_torque_phi=None, signal_torque_theta=None, signal_torque_psi=None):
    """
    Main function to run the dynamic table simulation.
    
    Args:
        signal_reference_x (SignalGenerator, optional): Signal generator for x reference.
        signal_reference_y (SignalGenerator, optional): Signal generator for y reference.
        signal_reference_z (SignalGenerator, optional): Signal generator for z reference.
        signal_reference_phi (SignalGenerator, optional): Signal generator for phi reference.
        signal_reference_theta (SignalGenerator, optional): Signal generator for theta reference.
        signal_reference_psi (SignalGenerator, optional): Signal generator for psi reference.
        signal_force_x (SignalGenerator, optional): Signal generator for x force.
        signal_force_y (SignalGenerator, optional): Signal generator for y force.
        signal_force_z (SignalGenerator, optional): Signal generator for z force.
        signal_torque_phi (SignalGenerator, optional): Signal generator for phi torque.
        signal_torque_theta (SignalGenerator, optional): Signal generator for theta torque.
        signal_torque_psi (SignalGenerator, optional): Signal generator for psi torque.
    """
    # Instantiate the simulation plots and animation
    app = QApplication(sys.argv) 
    table_animation = TableAnimation()
    table_animation.show()

    # Instantiate the table dynamics
    table_dynamimcs = TableDynamics()

    # Initialize the time
    current_time = tp.T_START

    while current_time < tp.T_END:
        loop_start_time = time.time()
        # Propagate the dynamics faster than the simulation
        next_plot_time = current_time + tp.T_PLOT

        while current_time < next_plot_time:
            offset_x = signal_reference_x.square(current_time) if signal_reference_x is not None else 0
            offset_y = signal_reference_y.square(current_time) if signal_reference_y is not None else 0
            offset_z = signal_reference_z.square(current_time) if signal_reference_z is not None else 0
            offset_phi = signal_reference_phi.square(current_time) if signal_reference_phi is not None else 0
            offset_theta = signal_reference_theta.square(current_time) if signal_reference_theta is not None else 0
            offset_psi = signal_reference_psi.square(current_time) if signal_reference_psi is not None else 0

            force_x = signal_force_x.sin(current_time) if signal_force_x is not None else 0
            force_y = signal_force_y.sin(current_time) if signal_force_y is not None else 0
            force_z = signal_force_z.sin(current_time) if signal_force_z is not None else 0
            torque_phi = signal_torque_phi.sin(current_time) if signal_torque_phi is not None else 0
            torque_theta = signal_torque_theta.sin(current_time) if signal_torque_theta is not None else 0
            torque_psi = signal_torque_psi.sin(current_time) if signal_torque_psi is not None else 0

            # Apply the forces to the dynamical table
            u_leader = np.array([[force_x], [force_y], [force_z], [torque_phi], [torque_theta], [torque_psi]])
            y = table_dynamimcs.update(u_leader)

            # Update the table reference
            reference_states = np.array([[tp.X_0 + offset_x],
                                         [tp.Y_0 + offset_y],
                                         [tp.Z_0 + offset_z],
                                         [tp.PHI_0 + offset_phi],
                                         [tp.THETA_0 + offset_theta],
                                         [tp.PSI_0 + offset_psi]])

            current_time += tp.T_STEP

        # Update the animation
        table_animation.update_gl(table_dynamimcs.state, reference_states=reference_states)
        QApplication.processEvents()

        elapsed_time = time.time() - loop_start_time
        sleep_time = max(0, tp.T_PLOT - elapsed_time)
        time.sleep(sleep_time)

    app.exec_()

if __name__ == "__main__":
    # Run the main function with different signal generators
    # main(signal_force_x=tp.signal_force_x)
    main(signal_force_y=tp.signal_force_y)
    # main(signal_force_z=tp.signal_force_z)
    main(signal_torque_phi=tp.signal_torque_phi)
    main(signal_torque_theta=tp.signal_torque_theta)
    main(signal_torque_psi=tp.signal_torque_psi)

    main(signal_reference_x=tp.signal_reference_x)
    main(signal_reference_y=tp.signal_reference_y)
    main(signal_reference_z=tp.signal_reference_z)
    main(signal_reference_phi=tp.signal_reference_phi)
    main(signal_reference_theta=tp.signal_reference_theta)
    main(signal_reference_psi=tp.signal_reference_psi)