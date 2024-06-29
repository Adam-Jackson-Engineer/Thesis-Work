"""
pid_step_rot_vec_torque.py

A 6 DoF PID controllers for an agent that takes in a reference position, tuned to work
with live trajectories. This version has the torques adjusted to be a single torque PID 
referenced to a rotation vector, then converted to the body, then inertial frame

There are interesting dynamics at play here where since I am applying a torque about a
rotation vector, if the table isn't aligned to go straight to the target, it will oscillate
around the target.

Author: Adam Jackson
Datae: June 7, 2024
"""

import sys
import numpy as np
from scipy.spatial.transform import Rotation

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utilities import table_parameters as tp

class PIDPathRotVecTorque:
    """
    Main controller class for the PID controller.
    """
    def __init__(self):
        """
        Initialize the PID controller.
        """
        # dirty derivative parameters
        self.sigma = 0.05  # cutoff freq for dirty derivative
        self.beta = (2 * self.sigma - tp.T_STEP) / (2 * self.sigma + tp.T_STEP)

        ####################################################
        #       PD Control: Time Design Strategy
        ####################################################
        # Rise Time (tr)
        tr_x = 0.4
        tr_y = 0.5
        tr_z = 1.0
        tr_angle = 1.0

        # Damping Ratios (zeta)
        zeta_x = 0.9
        zeta_y = 0.707
        zeta_z = 0.9
        zeta_angle = 0.707

        # Integrator Gains
        self.ki_x = 0.01
        self.ki_y = 0.01
        self.ki_z = 0.01
        self.ki_angle = 0.01

        # Natural Frequencies (wn)
        wn_x = 2.2 / tr_x
        wn_y = 2.2 / tr_y
        wn_z = 2.2 / tr_z
        wn_angle = 2.2 / tr_angle

        # Desired Characteristic Equations
        delta_cl_d_x = [1, 2 * zeta_x * wn_x, wn_x ** 2.0]
        delta_cl_d_y = [1, 2 * zeta_y * wn_y, wn_y ** 2.0]
        delta_cl_d_z = [1, 2 * zeta_z * wn_z, wn_z ** 2.0]
        delta_cl_d_angle = [1, 2 * zeta_angle * wn_angle, wn_angle ** 2.0]

        # Kp and Kd gains
        self.kp_x = delta_cl_d_x[2] * tp.TABLE_MASS
        self.kd_x = delta_cl_d_x[1] * tp.TABLE_MASS
        
        self.kp_y = delta_cl_d_y[2] * tp.TABLE_MASS
        self.kd_y = delta_cl_d_y[1] * tp.TABLE_MASS

        self.kp_z = delta_cl_d_z[2] * tp.TABLE_MASS
        self.kd_z = delta_cl_d_z[1] * tp.TABLE_MASS

        self.kp_angle = delta_cl_d_angle[2] * np.sqrt(tp.INERTIA_XX**2 + tp.INERTIA_YY**2 + tp.INERTIA_ZZ**2)
        self.kd_angle = delta_cl_d_angle[1] * np.sqrt(tp.INERTIA_XX**2 + tp.INERTIA_YY**2 + tp.INERTIA_ZZ**2)

        # Initialize integrators
        self.integrator_x = 0.
        self.integrator_y = 0.
        self.integrator_z = 0.
        self.integrator_angle = 0.

        # Initialize the delay 1 memory term (d1) saving the previous error
        self.error_x_d1 = 0.
        self.error_y_d1 = 0.
        self.error_z_d1 = 0.
        self.error_angle_d1 = 0.

        # Initialize the positional state derivative terms
        self.x_dot = 0.
        self.y_dot = 0.
        self.z_dot = 0.
        self.angle_dot = 0.

        # Initialize the delay 1 memory term (d1) saving the previous positional states        
        self.x_d1 = tp.X_0
        self.y_d1 = tp.Y_0
        self.z_d1 = tp.Z_0
        self.angle_d1 = tp.PHI_0
        
    def update(self, reference_states, states):
        """
        Update the controller with the reference and table states.
        """
        reference_x = reference_states[0][0]
        reference_y = reference_states[1][0]
        reference_z = reference_states[2][0]
        reference_phi = reference_states[3][0]
        reference_theta = reference_states[4][0]
        reference_psi = reference_states[5][0]

        # We want there to be no angle between the reference and the table
        reference_angle = 0.0

        x = states[0][0]
        y = states[1][0]
        z = states[2][0]
        phi = states[3][0]
        theta = states[4][0]
        psi = states[5][0]

        state_rotation_matrix = tp.rot_zyx(phi, theta, psi)
        reference_rotation_matrix = tp.rot_zyx(reference_phi, reference_theta, reference_psi)

        state_to_reference_rotation_matrix = Rotation.from_matrix(reference_rotation_matrix @ state_rotation_matrix.T)
        state_to_reference_rotation_vector = state_to_reference_rotation_matrix.as_rotvec(degrees=False)

        # Find the magnitude of the rotation vector, which is the angle from the current frame to the new frame
        angle = np.linalg.norm(state_to_reference_rotation_vector)

        # Adjust the range to the angle from 0 to 2pi to be -pi to pi for control
        if angle > np.pi:
            angle -= 2 * np.pi
        elif angle < -np.pi:
            angle += 2 * np.pi

        # Flip the angle direction to now be from the reference to the current frame
        angle = -angle

        # Compute the errors
        error_x = reference_x - x
        error_y = reference_y - y
        error_z = reference_z - z
        error_angle = reference_angle - angle

        # Integrate error
        if self.x_dot < 0.01:
            self.integrator_x = self.integrator_x + (tp.T_STEP / 2) * (error_x + self.error_x_d1)
        if self.y_dot < 0.01:
            self.integrator_y = self.integrator_y + (tp.T_STEP / 2) * (error_y + self.error_y_d1)
        if self.z_dot < 0.01:
            self.integrator_z = self.integrator_z + (tp.T_STEP / 2) * (error_z + self.error_z_d1)
        if self.angle_dot < 0.01:
            self.integrator_angle = self.integrator_angle + (tp.T_STEP / 2) * (error_angle + self.error_angle_d1)

        # Differentiate Positional States
        self.x_dot = self.beta * self.x_dot + (1 - self.beta) * ((x - self.x_d1) / tp.T_STEP)
        self.y_dot = self.beta * self.y_dot + (1 - self.beta) * ((y - self.y_d1) / tp.T_STEP)
        self.z_dot = self.beta * self.z_dot + (1 - self.beta) * ((z - self.z_d1) / tp.T_STEP)
        self.angle_dot = self.beta * self.angle_dot + (1 - self.beta) * ((angle - self.angle_d1) / tp.T_STEP)

        # Forces in the inertial from - PID control - unsaturated
        force_x = self.kp_x * error_x + self.ki_x * self.integrator_x - self.kd_x * self.x_dot
        force_y = self.kp_y * error_y + self.ki_y * self.integrator_y - self.kd_y * self.y_dot
        force_z = self.kp_z * error_z + self.ki_z * self.integrator_z - self.kd_z * self.z_dot
        torque = self.kp_angle * error_angle + self.ki_angle * self.integrator_angle - self.kd_angle * self.angle_dot

        # Find the location of the leader's forces in the inertial frame
        force_location = np.linalg.inv(state_rotation_matrix) @ tp.LEADER_FORCE_LOCATION
        force_location_x = force_location[0][0]
        force_location_y = force_location[1][0]
        force_location_z = force_location[2][0]

        # Torque times the rotation unit vector, (rotation vector/magnitude(angle))
        if angle != 0:
            torque_body_frame = torque*state_to_reference_rotation_vector/angle
        else:
            torque_body_frame = torque*state_to_reference_rotation_vector

        torque_inertial_frame = state_rotation_matrix.T @ torque_body_frame

        moment_from_force_phi = (force_z * force_location_y - force_y * force_location_z)
        moment_from_force_theta = (force_x * force_location_z - force_z * force_location_x)
        moment_from_force_psi = (force_y * force_location_x - force_x * force_location_y)

        force_inertial_frame = np.array([[force_x],[force_y],[force_z]])
        torque_inertial_frame = np.array([[torque_inertial_frame[0] - moment_from_force_phi],
                                          [torque_inertial_frame[1] - moment_from_force_theta],
                                          [torque_inertial_frame[2] - moment_from_force_psi]])

        # Values in the body frame
        force_body_frame = state_rotation_matrix @ force_inertial_frame
        torque_body_frame = state_rotation_matrix @ torque_inertial_frame


        # Saturate in body frame
        force_x_body_frame = saturate(force_body_frame[0][0],tp.FORCE_X_MAX_LEADER, "X")
        force_y_body_frame = saturate(force_body_frame[1][0],tp.FORCE_Y_MAX_LEADER, "Y")
        force_Z_body_frame = saturate(force_body_frame[2][0],tp.FORCE_Z_MAX_LEADER, "Z")
        torque_phi_body_frame = saturate(torque_body_frame[0][0],tp.TORQUE_PHI_MAX_LEADER, "PHI")
        torque_theta_body_frame = saturate(torque_body_frame[1][0],tp.TORQUE_THETA_MAX_LEADER, "THETA")
        torque_psi_body_frame = saturate(torque_body_frame[2][0],tp.TORQUE_PSI_MAX_LEADER, "PSI")
        
        # Saturated Values in the body frame
        force_body_frame_saturated = np.array([[force_x_body_frame],[force_y_body_frame],[force_Z_body_frame]])
        torque_body_frame_saturated = np.array([[torque_phi_body_frame],[torque_theta_body_frame],[torque_psi_body_frame]])

        # Saturated Values in the Inertial Frame
        force_inertial_frame_saturated = state_rotation_matrix.T @ force_body_frame_saturated
        torque_inertial_frame_saturated = state_rotation_matrix.T @ torque_body_frame_saturated

        force_x_inertial_frame_saturated = force_inertial_frame_saturated[0][0]
        force_y_inertial_frame_saturated = force_inertial_frame_saturated[1][0]
        force_z_inertial_frame_saturated = force_inertial_frame_saturated[2][0]
        torque_phi_inertial_frame_saturated = torque_inertial_frame_saturated[0][0]
        torque_theta_inertial_frame_saturated = torque_inertial_frame_saturated[1][0]
        torque_psi_inertial_frame_saturated = torque_inertial_frame_saturated[2][0]

        tau = np.array([[force_x_inertial_frame_saturated], 
                        [force_y_inertial_frame_saturated], 
                        [force_z_inertial_frame_saturated], 
                        [torque_phi_inertial_frame_saturated], 
                        [torque_theta_inertial_frame_saturated], 
                        [torque_psi_inertial_frame_saturated]])

        # Save D1 values
        self.error_x_d1 = error_x
        self.error_y_d1 = error_y
        self.error_z_d1 = error_z
        self.error_angle_d1 = error_angle

        self.x_d1 = x
        self.y_d1 = y
        self.z_d1 = z
        self.angle_d1 = angle

        return tau

    def reset(self):
        # Initialize integrators
        self.integrator_x = 0.
        self.integrator_y = 0.
        self.integrator_z = 0.
        self.integrator_angle = 0.

        # Initialize the delay 1 memory term (d1) saving the previous error
        self.error_x_d1 = 0.
        self.error_y_d1 = 0.
        self.error_z_d1 = 0.
        self.error_angle_d1 = 0.

        # Initialize the positional state derivative terms
        self.x_dot = 0.
        self.y_dot = 0.
        self.z_dot = 0.
        self.angle_dot = 0.

        # Initialize the delay 1 memory term (d1) saving the previous positional states        
        self.x_d1 = tp.X_0
        self.y_d1 = tp.Y_0
        self.z_d1 = tp.Z_0
        self.angle_d1 = tp.PHI_0
 

def saturate(u, limit, saturated_force = None):
    if abs(u) > limit:
        # if saturated_force is not None:
        #     print(f"Force Saturated:    {saturated_force}")
        #     print(f"Limit was:          {limit}")
        #     print(f"Force was:          {u}")
        u = limit * np.sign(u)
    return u

 
