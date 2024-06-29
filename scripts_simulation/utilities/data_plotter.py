"""
data_plotter.py

Handles the real-time plotting of data for the table dynamics simulation.
"""

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D

plt.ion()  # Enable interactive drawing

class DataPlotter:
    """Class for plotting table dynamics data."""
    
    def __init__(self):
        """Initializes the DataPlotter with multiple subplots."""
        self.num_rows = 3
        self.num_cols = 4

        # Create figure and axes handles with larger figure size
        self.fig, self.ax = plt.subplots(self.num_rows, self.num_cols, sharex=True, figsize=(15, 10))
        self.fig.tight_layout(pad=4.0)

        # Instantiate lists to hold the time and data histories
        self.time_history = []

        # X position, reference, and force history
        self.x_history = []
        self.reference_x_history = []
        self.force_x_history_leader = []
        self.force_x_history_follower = []

        # Y position, reference, and force history
        self.y_history = []
        self.reference_y_history = []
        self.force_y_history_leader = []
        self.force_y_history_follower = []

        # Z position, reference, and force history
        self.z_history = []
        self.reference_z_history = []
        self.force_z_history_leader = []
        self.force_z_history_follower = []

        # Phi position, reference, and force history
        self.phi_history = []
        self.reference_phi_history = []
        self.force_phi_history_leader = []
        self.force_phi_history_follower = []

        # Theta position, reference, and force history
        self.theta_history = []
        self.reference_theta_history = []
        self.force_theta_history_leader = []
        self.force_theta_history_follower = []

        # Psi position, reference, and force history
        self.psi_history = []
        self.reference_psi_history = []
        self.force_psi_history_leader = []
        self.force_psi_history_follower = []

        # Create a handle for every subplot
        self.handle = []
        self.handle.append(MyPlot(self.ax[0, 0], ylabel='x(m)', title='Positional Data'))
        self.handle.append(MyPlot(self.ax[1, 0], ylabel='y(m)'))
        self.handle.append(MyPlot(self.ax[2, 0], xlabel='t(s)', ylabel='z(m)'))

        self.handle.append(MyPlot(self.ax[0, 1], ylabel='x(N)', title='Translational Forces'))
        self.handle.append(MyPlot(self.ax[1, 1], ylabel='y(N)'))
        self.handle.append(MyPlot(self.ax[2, 1], xlabel='t(s)', ylabel='z(N)'))

        self.handle.append(MyPlot(self.ax[0, 2], ylabel=r'$\phi$(deg)', title='Angular Data'))
        self.handle.append(MyPlot(self.ax[1, 2], ylabel=r'$\theta$(deg)'))
        self.handle.append(MyPlot(self.ax[2, 2], xlabel='t(s)', ylabel=r'$\psi$(deg)'))

        self.handle.append(MyPlot(self.ax[0, 3], ylabel=r'$\phi$(Nm)', title='Torsional Forces'))
        self.handle.append(MyPlot(self.ax[1, 3], ylabel=r'$\theta$(Nm)'))
        self.handle.append(MyPlot(self.ax[2, 3], xlabel='t(s)', ylabel=r'$\psi$(Nm)'))


    def update(self, current_time, reference_states, states, u_leader, u_follower=None):
        """Adds to the time and data histories and updates the plots.

        Args:
            current_time (float): Current simulation time.
            reference_states (np.ndarray): Reference states.
            states (np.ndarray): Current states of the system.
            u_leader (np.ndarray): Control inputs from the leader.
            u_follower (np.ndarray, optional): Control inputs from the follower. Defaults to None.
        """
        if u_follower is None:
            u_follower = np.zeros_like(u_leader)
        
        # Update the time history of all plot variables
        self.time_history.append(current_time)

        # X position, reference, and force history
        self.x_history.append(states[0, 0])
        self.reference_x_history.append(reference_states[0, 0])
        self.force_x_history_leader.append(u_leader[0, 0])
        self.force_x_history_follower.append(u_follower[0, 0])
        
        # Y position, reference, and force history
        self.y_history.append(states[1, 0])
        self.reference_y_history.append(reference_states[1, 0])
        self.force_y_history_leader.append(u_leader[1, 0])
        self.force_y_history_follower.append(u_follower[1, 0])

        # Z position, reference, and force history
        self.z_history.append(states[2, 0])
        self.reference_z_history.append(reference_states[2, 0])
        self.force_z_history_leader.append(u_leader[2, 0])
        self.force_z_history_follower.append(u_follower[2, 0])

        # Phi position, reference, and force history
        self.phi_history.append(states[3, 0] * 180 / np.pi)
        self.reference_phi_history.append(reference_states[3, 0] * 180 / np.pi)
        self.force_phi_history_leader.append(u_leader[3, 0])
        self.force_phi_history_follower.append(u_follower[3, 0])

        # Theta position, reference, and force history
        self.theta_history.append(states[4, 0] * 180 / np.pi)
        self.reference_theta_history.append(reference_states[4, 0] * 180 / np.pi)
        self.force_theta_history_leader.append(u_leader[4, 0])
        self.force_theta_history_follower.append(u_follower[4, 0])

        # Psi position, reference, and force history
        self.psi_history.append(states[5, 0] * 180 / np.pi)
        self.reference_psi_history.append(reference_states[5, 0] * 180 / np.pi)
        self.force_psi_history_leader.append(u_leader[5, 0])
        self.force_psi_history_follower.append(u_follower[5, 0])

        # Update the plots with associated histories
        self.handle[0].update(self.time_history, [self.x_history, self.reference_x_history])
        self.handle[1].update(self.time_history, [self.y_history, self.reference_y_history])
        self.handle[2].update(self.time_history, [self.z_history, self.reference_z_history])
        
        self.handle[3].update(self.time_history, [self.force_x_history_leader, self.force_x_history_follower])
        self.handle[4].update(self.time_history, [self.force_y_history_leader, self.force_y_history_follower])
        self.handle[5].update(self.time_history, [self.force_z_history_leader, self.force_z_history_follower])
        
        self.handle[6].update(self.time_history, [self.phi_history, self.reference_phi_history])
        self.handle[7].update(self.time_history, [self.theta_history, self.reference_theta_history])
        self.handle[8].update(self.time_history, [self.psi_history, self.reference_psi_history])

        self.handle[9].update(self.time_history, [self.force_phi_history_leader, self.force_phi_history_follower])
        self.handle[10].update(self.time_history, [self.force_theta_history_leader, self.force_theta_history_follower])
        self.handle[11].update(self.time_history, [self.force_psi_history_leader, self.force_psi_history_follower])

class MyPlot:
    """Class to create each individual subplot."""

    def __init__(self, ax, xlabel='', ylabel='', title='', legend=None):
        """Initializes the subplot.

        Args:
            ax (Axes): Axes handle.
            xlabel (str): Label for the x-axis.
            ylabel (str): Label for the y-axis.
            title (str): Plot title.
            legend (tuple, optional): Tuple of strings that identify the data.
        """
        self.legend = legend
        self.ax = ax
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        self.line_styles = ['-', '--', '-.', ':']
        self.line = []

        self.ax.set_ylabel(ylabel)
        self.ax.set_xlabel(xlabel)
        self.ax.set_title(title)
        self.ax.grid(True)

        self.init = True  

    def update(self, time, data):
        """Adds data to the plot.

        Args:
            time (list): List of time points.
            data (list of lists): List of data series.
        """
        if self.init:
            for i in range(len(data)):
                self.line.append(Line2D(time, data[i],
                                        color=self.colors[i % len(self.colors)],
                                        ls=self.line_styles[i % len(self.line_styles)],
                                        label=self.legend if self.legend else None))
                self.ax.add_line(self.line[i])
            self.init = False
            if self.legend:
                self.ax.legend(handles=self.line)
        else:
            for i in range(len(self.line)):
                self.line[i].set_xdata(time)
                self.line[i].set_ydata(data[i])

        self.ax.relim()
        self.ax.autoscale()

