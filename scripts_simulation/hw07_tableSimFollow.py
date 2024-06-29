import matplotlib.pyplot as plt
import numpy as np
import scripts_simulation.table_parameters as P
from table_dynamics import tableDynamics
from scripts_simulation.leader_pid_controller import ctrlPID
from ctrlFollower import ctrlLearnedFollower
from signal_generator import signalGenerator
from scripts_simulation.table_animation import tableAnimation
from PyQt5.QtWidgets import QApplication
import sys

# instantiate VTOL, controller, and reference classes
tran_ref = signalGenerator(amplitude=1.0, frequency=0.05)
rot_ref = signalGenerator(amplitude=np.pi/2, frequency = 0.05)

# instantiate the simulation plots and animation
app = QApplication(sys.argv)  # Creating a QApplication object
animation = tableAnimation()
animation.show()
table = tableDynamics()
controller_leader = ctrlPID()
follower_timestep = 5
controller_follower = ctrlLearnedFollower(follower_timestep)

for i in range(P.All_tasks.shape[1]):
    table.state = P.states0.copy()
    ref_row = P.All_tasks[:,i]
    ref = ref_row.reshape(-1, 1)
    t = P.t_start  # time starts at t_start
    y = table.h()  # output of system at start of simulation
    if np.array_equal(ref, P.REST_ref):
        time_end = 2.0
    else:
        time_end = P.t_end
    while t < time_end:  # main simulation loop
        # Propagate dynamics in between plot samples
        t_next_plot = t + P.t_plot
        while t < t_next_plot:  # updates control and dynamics at faster simulation rate
            if np.array_equal(ref, P.REST_ref):
                table.state = P.states0.copy()
            u_leader = controller_leader.update(ref, table.state)  # update controller
            u_leader = np.array([[u_leader[0][0]],[u_leader[1][0]],[0.0],[0.0],[0.0],[u_leader[5][0]]])
            # u_leader = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
            u_follower = controller_follower.update(u_leader, table.state)
            # u_follower = np.array([[0.0],[0.0],[0.0],[0.0],[0.0],[0.0]])
            # u_follower = np.array([[u_leader[0][0]],[u_leader[1][0]],[0.0],[0.0],[0.0],[u_leader[5][0]]])
            # u_follower = np.array([[-u_leader[0][0]],[-u_leader[1][0]],[0.0],[0.0],[0.0],[-u_leader[5][0]]])
            
            y = table.update(u_leader, u_follower)  # propagate system
            t = t + P.Ts  # advance time by Ts
        # update animation and data plots
        animation.updateGL(table.state, states_r=ref)
        QApplication.processEvents()
        plt.pause(P.t_plot)  # the pause causes the figure to be displayed during the simulation

