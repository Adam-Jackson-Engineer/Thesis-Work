"""
table_dynamics.py

Handles the dynamics of the table in a 3D gravity environment.
"""

import numpy as np
import utilities.table_parameters as tp

class TableDynamics:
    """
    Main Dynamics class for the table simulation.
    """
    def __init__(self, alpha=0.0, gravity = False):
        """
        Initialize the table dynamics with given parameters.
        
        Args:
            alpha (float): Randomization factor for model parameters.
        """
        # Initial state conditions
        self.state = tp.INITIAL_STATES.copy()
        self.t_step = tp.T_STEP

        # Inertia matrix of table
        self.inertia_matrix = np.array([
            [tp.INERTIA_XX * (1.0 + alpha * (2.0 * np.random.rand() - 1.0)), 0, 0], 
            [0, tp.INERTIA_YY * (1.0 + alpha * (2.0 * np.random.rand() - 1.0)), 0], 
            [0, 0, tp.INERTIA_ZZ * (1.0 + alpha * (2.0 * np.random.rand() - 1.0))]
        ])

        # Torque and force limits for leader and follower
        self.tau_limits_leader = np.array([
            [tp.FORCE_X_MAX_LEADER], 
            [tp.FORCE_Y_MAX_LEADER], 
            [tp.FORCE_Z_MAX_LEADER], 
            [tp.TORQUE_PHI_MAX_LEADER], 
            [tp.TORQUE_THETA_MAX_LEADER], 
            [tp.TORQUE_PSI_MAX_LEADER]
        ])
        
        self.tau_limits_follower = np.array([
            [tp.FORCE_X_MAX_FOLLOWER], 
            [tp.FORCE_Y_MAX_FOLLOWER], 
            [tp.FORCE_Z_MAX_FOLLOWER], 
            [tp.TORQUE_PHI_MAX_FOLLOWER], 
            [tp.TORQUE_THETA_MAX_FOLLOWER], 
            [tp.TORQUE_PSI_MAX_FOLLOWER]
        ])        
        if gravity:
            self.gravity = tp.GRAVITY
        else:
            self.gravity = 0

    def update(self, u_leader, u_follower=None):
        """
        Update the state of the table given leader and optional follower inputs.
        
        Args:
            u_leader (np.ndarray): Leader control inputs.
            u_follower (np.ndarray, optional): Follower control inputs. Defaults to a zero vector.
        
        Returns:
            np.ndarray: The output state.
        """
        if u_follower is None:
            u_follower = np.zeros_like(u_leader)
        u = np.hstack((u_leader, u_follower))
        self.rk4_step(u)  # Propagate the state by one time step
        y = self.h()  # Return the corresponding output
        return y


    def f(self, state, u):
        """
        Compute the state derivatives.
        
        Args:
            state (np.ndarray): Current state of the system.
            u (np.ndarray): Control inputs.
        
        Returns:
            np.ndarray: State derivatives.
        """
        phi = state[3][0]
        theta = state[4][0]
        psi = state[5][0]
        x_dot = state[6][0]
        y_dot = state[7][0]
        z_dot = state[8][0]
        omega_x = state[9][0]
        omega_y = state[10][0]
        omega_z = state[11][0]

        leader_force_x = u[0][0]
        leader_force_y = u[1][0]
        leader_force_z = u[2][0]
        leader_torque_phi = u[3][0]
        leader_torque_theta = u[4][0]
        leader_torque_psi = u[5][0]

        follower_force_x = u[0][1]
        follower_force_y = u[1][1]
        follower_force_z = u[2][1]
        follower_torque_phi = u[3][1]
        follower_torque_theta = u[4][1]
        follower_torque_psi = u[5][1]

        # 12 state equations of motion
        # Derivatives of x, y, and z are already states
        x_dot = x_dot
        y_dot = y_dot
        z_dot = z_dot

        # Derivatives of phi, theta, and psi are functions of phi, theta, psi, omega_x, omega_y, and omega_z
        phi_dot = (omega_x * np.cos(psi) + omega_y * np.sin(psi)) / np.cos(theta)
        theta_dot = -omega_x * np.sin(psi) + omega_y * np.cos(psi)
        psi_dot = omega_x * np.cos(psi) * np.tan(theta) + omega_y * np.sin(psi) * np.tan(theta) + omega_z

        # Derivatives of x_dot, y_dot, and z_dot from F = ma
        x_dot_dot = (leader_force_x + follower_force_x) / tp.TABLE_MASS
        y_dot_dot = (leader_force_y + follower_force_y) / tp.TABLE_MASS
        z_dot_dot = (leader_force_z + follower_force_z) / tp.TABLE_MASS - self.gravity

        # Derivatives of omega_x, omega_y, and omega_z from M = I * w_dot
        # Get the rotation matrix at the current location
        rotation_matrix = tp.rot_zyx(phi, theta, psi)

        # Calculate the inertia matrix at the current configuration in the inertial frame
        current_inertia_matrix = rotation_matrix.T @ self.inertia_matrix @ rotation_matrix

        # Find the location of the leader's forces in the inertial frame
        leader_force_location = rotation_matrix.T @ tp.LEADER_FORCE_LOCATION
        leader_force_location_x = leader_force_location[0][0]
        leader_force_location_y = leader_force_location[1][0]
        leader_force_location_z = leader_force_location[2][0]

        # Find the location of the follower's forces in the inertial frame
        follower_force_location = rotation_matrix.T @ tp.FOLLOWER_FORCE_LOCATION
        follower_force_location_x = follower_force_location[0][0]
        follower_force_location_y = follower_force_location[1][0]
        follower_force_location_z = follower_force_location[2][0]

        # Calculate the moment applied about the inertial x-axis
        moment_phi = (leader_torque_phi + follower_torque_phi + 
                      (leader_force_z * leader_force_location_y - leader_force_y * leader_force_location_z) +
                      (follower_force_z * follower_force_location_y - follower_force_y * follower_force_location_z))
                
        # Calculate the moment applied about the inertial y-axis
        moment_theta = (leader_torque_theta + follower_torque_theta +
                        (leader_force_x * leader_force_location_z - leader_force_z * leader_force_location_x) +
                        (follower_force_x * follower_force_location_z - follower_force_z * follower_force_location_x))
        
        # Calculate the moment applied about the inertial z-axis
        moment_psi = (leader_torque_psi + follower_torque_psi +
                      (leader_force_y * leader_force_location_x - leader_force_x * leader_force_location_y) +
                      (follower_force_y * follower_force_location_x - follower_force_x * follower_force_location_y))

        moment_vector = np.array([[moment_phi], [moment_theta], [moment_psi]])

        omega_dot = np.linalg.inv(current_inertia_matrix) @ moment_vector

        omega_dot_x = omega_dot[0][0]
        omega_dot_y = omega_dot[1][0]
        omega_dot_z = omega_dot[2][0]

        # Build the state derivatives vector
        state_dot = np.array([[x_dot], [y_dot], [z_dot], 
                              [phi_dot], [theta_dot], [psi_dot],
                              [x_dot_dot], [y_dot_dot], [z_dot_dot], 
                              [omega_dot_x], [omega_dot_y], [omega_dot_z]])
        
        return state_dot

    def h(self):
        """
        Return the output state.
        
        Returns:
            np.ndarray: The output state.
        """
        x = self.state[0][0]
        y = self.state[1][0]
        psi = self.state[5][0]
        y = np.array([[x], [y], [psi]])
        return y
    
    def rk4_step(self, u):
        """
        Integrate the ODE using the Runge-Kutta RK4 algorithm.
        
        Args:
            u (np.ndarray): Control inputs.
        """
        f1 = self.f(self.state, u)
        f2 = self.f(self.state + self.t_step / 2 * f1, u)
        f3 = self.f(self.state + self.t_step / 2 * f2, u)
        f4 = self.f(self.state + self.t_step * f3, u)
        self.state += self.t_step / 6 * (f1 + 2 * f2 + 2 * f3 + f4)

        
def saturate(u, limits):
    """
    Saturate the control inputs to within the given limits.
    
    Args:
        u (np.ndarray): Control inputs.
        limits (np.ndarray): Input limits.
    
    Returns:
        np.ndarray: Saturated control inputs.
    """
    for i in range(len(u)):
        if abs(u[i]) > limits[i]:
            u[i] = limits[i] * np.sign(u[i])
    return u