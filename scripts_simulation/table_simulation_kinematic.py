"""
table_simulation_kinematic.py

Simulates a table in 3D with positional signals and animations.
"""

import sys
import time
from math import floor

import numpy as np
from PyQt5.QtWidgets import QApplication

import utilities.table_parameters as tp
from utilities.signal_generator import SignalGenerator
from utilities.table_animation import TableAnimation

def main():
    """
    Main function to run the kinematic table simulation.
    """
    # Instantiate the signal generator
    signal_generator = SignalGenerator(amplitude=1, frequency=1 / (tp.T_END / 6 / 2))

    # Instantiate the application and animation
    app = QApplication(sys.argv)
    table_animation = TableAnimation()
    table_animation.show()

    # Initialize the time
    current_time = tp.T_START

    while current_time < tp.T_END:
        loop_start_time = time.time()

        # Generate signal based on the current time
        signal = signal_generator.sin(current_time)

        # Initialize the state of the table
        state = np.array([[tp.X_0], [tp.Y_0], [tp.Z_0], [tp.PHI_0], [tp.THETA_0], [tp.PSI_0]])

        # Determine which state to update based on the time interval
        state_index = floor(current_time / (tp.T_END / 6))
        state[state_index] = state[state_index] + signal

        # Update the animation with the new state
        table_animation.update_gl(state)

        # Increment the time by the plotting interval
        current_time += tp.T_PLOT

        # Process Qt events
        QApplication.processEvents()

        # Calculate elapsed time and sleep to maintain real-time simulation
        elapsed_time = time.time() - loop_start_time
        sleep_time = max(0, tp.T_PLOT - elapsed_time)
        time.sleep(sleep_time)

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
