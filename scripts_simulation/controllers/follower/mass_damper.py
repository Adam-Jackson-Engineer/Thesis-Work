"""
mass_damper.py

A follower controller that resists changes in motion and acceleration, acting like a mass damper.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utilities import table_parameters as tp

class MassDamper:
    """Main controller for the impedance follower."""

    def __init__(self, desired_acceleration=None, mass=None, damping=None):
        """Initialize the Impedance controller.

        Args:
            desired_velocity (np.array): Desired constant velocity [x_dot, y_dot, z_dot, omega_x, omega_y, omega_z].
            mass (np.array): mass coefficients [trans_inertia_x, trans_inertia_y, trans_inertia_z, rot_inertial_x, rot_inertial_y, rot_inertial_z].
            damping (np.array): Damping coefficients [b_x, b_y, b_z, b_rot_x, b_rot_y, b_rot_z].
        """
        if desired_acceleration is None:
            desired_acceleration = np.zeros((6, 1))
        if mass is None:
            mass = np.ones((6, 1))
        if damping is None:
            damping = np.ones((6, 1))

        self.desired_acceleration = desired_acceleration
        self.mass = mass
        self.damping = damping

        # Initialize the delay 1 memory term (d1) saving the previous positional states
        self.x_dot_d1 = 0.
        self.y_dot_d1 = 0.
        self.z_dot_d1 = 0.
        self.phi_dot_d1 = 0.
        self.theta_dot_d1 = 0.
        self.psi_dot_d1 = 0.

    def update(self, states, u_leader=None):
        """Update the controller to output the appropriate forces, resisting acceleration.

        Args:
            states: The current states of the system [x, y, z, phi, theta, psi, vx, vy, vz, vphi, vtheta, vpsi].
        """
        # Extract current positions and velocities
        phi, theta, psi = states[3][0], states[4][0], states[5][0]
        x_dot, y_dot, z_dot = states[6][0], states[7][0], states[8][0]
        omega_x, omega_y, omega_z = states[9][0], states[10][0], states[11][0]

        state_rotation = tp.rot_zyx(phi, theta, psi)
        state_follower_offset = state_rotation.T @ tp.FOLLOWER_FORCE_LOCATION

        # Find the velocity at the follower
        x_dot += - state_follower_offset[1][0] * omega_z + state_follower_offset[2][0] * omega_y
        y_dot += state_follower_offset[0][0] * omega_z - state_follower_offset[2][0] * omega_x
        z_dot += - state_follower_offset[0][0] * omega_y + state_follower_offset[1][0] * omega_x

        current_velocity = np.array([[x_dot],
                                     [y_dot],
                                     [z_dot],
                                     [omega_x],
                                     [omega_y],
                                     [omega_z]])
        
        previous_velocity = np.array([[self.x_dot_d1],
                                      [self.y_dot_d1],
                                      [self.z_dot_d1],
                                      [self.phi_dot_d1],
                                      [self.theta_dot_d1],
                                      [self.psi_dot_d1]])
        
        current_accelerations = (current_velocity-previous_velocity) / tp.T_STEP
        
        # When desired acceleration is zero, this is just F = ma
        acceleration_error = self.desired_acceleration - current_accelerations

        force_x = self.mass[0][0] * acceleration_error[0][0] - self.damping[0][0] * x_dot
        force_y = self.mass[1][0] * acceleration_error[1][0] - self.damping[1][0] * y_dot
        force_z = self.mass[2][0] * acceleration_error[2][0] - self.damping[2][0] * z_dot
        torque_rot_x = self.mass[3][0] * acceleration_error[3][0] - self.damping[3][0] * omega_x
        torque_rot_y = self.mass[4][0] * acceleration_error[4][0] - self.damping[4][0] * omega_y
        torque_rot_z = self.mass[5][0] * acceleration_error[5][0] - self.damping[5][0] * omega_z

        tau = np.array([[force_x], 
                        [force_y], 
                        [force_z], 
                        [torque_rot_x], 
                        [torque_rot_y], 
                        [torque_rot_z]])

        return tau

    def reset(self):
        """Reset the controller states."""
        self.x_dot_d1 = 0.
        self.y_dot_d1 = 0.
        self.z_dot_d1 = 0.
        self.phi_dot_d1 = 0.
        self.theta_dot_d1 = 0.
        self.psi_dot_d1 = 0.


