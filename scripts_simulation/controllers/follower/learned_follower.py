"""
learned.py

A follower controller that loads and uses a trained model to generate force outputs.
"""

import os
import sys
import random
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from utilities import table_parameters as tp
from scripts_train_nns import nn_functions as nn_f
from scripts_train_nns import nn_param as nn_p

class Learned:
    """Main controller for the learned follower."""

    def __init__(self, reaction_time_type=None, output_type=None, input_type=None, task_type=None,
                 cutoff_type=None, model_type=None, generate_random_model=False):
        """Initialize the learned controller.

        Args:
            reaction_time_type (str): Length of training data to use ("SRT", "CRT")
            output_type (str): Type of output (["XXX", "YYY", "PSI", "tXY", "XYS"])
            input_type (str): Type of input to use ('vel', 'vat', 'tao', 'vat')
            task_type (str): What types of tasks the model was trained on ('TX', 'TY', 'XY', 'RZ', 'TR', '2D')
            cutoff_type (str): The time required to keep a task in the training set ('05', '08', '10', '12', '14')
            model_type (str): Type of model to use ("CNN" or "LSTM")
            generate_random_model (bool): Whether to pick a random model or a defined model
        """
        self.path_to_primary_models = os.path.join(tp.TRAINED_MODELS_FOLDER_PATH, "models_primary")
        self.path_to_all_models = os.path.join(tp.TRAINED_MODELS_FOLDER_PATH, "models_all")

        self.sigma = 0.05
        self.beta = (2 * self.sigma - tp.T_STEP) / (2 * self.sigma + tp.T_STEP)

        self.reaction_time_type = reaction_time_type or random.choice(nn_p.RT_COMBO_NAMES)
        self.output_type = output_type or random.choice(nn_p.OUTPUT_COMBO_NAMES)
        self.input_type = input_type or random.choice(nn_p.INPUT_COMBO_NAMES)
        self.task_type = task_type or random.choice(nn_p.TASK_COMBO_NAMES)
        self.cutoff_type = cutoff_type or random.choice(nn_p.CUTOFF_NAMES)
        self.model_type = model_type or random.choice(["CNN", "LSTM"])

        if generate_random_model:
            self.random_model_generator()

        model_name = self.generate_model_name()

        if generate_random_model or model_name in os.listdir(self.path_to_primary_models):
            model = self.load_model(model_name, self.path_to_primary_models)
        elif model_name in os.listdir(self.path_to_all_models):
            model = self.load_model(model_name, self.path_to_all_models)
        else:
            print(f"Model not found: {model_name}")
            self.random_model_generator()
            model_name = self.generate_model_name()
            model = self.load_model(model_name, self.path_to_primary_models)

        print(f"\nUsing model:  {model_name}")
        
        self.model_name = model_name
        self.model = model
        self.initialize_derivatives()
        
        # Lists of possible input types
        self.all_model_input_names = [
            'Pos_X', 'Pos_Y', 'Pos_Z', 'Pos_Phi', 'Pos_Theta', 'Pos_Psi',
            'Vel_X', 'Vel_Y', 'Vel_Z', 'Vel_Phi', 'Vel_Theta', 'Vel_Psi',
            'Acc_X', 'Acc_Y', 'Acc_Z', 'Acc_Phi', 'Acc_Theta', 'Acc_Psi',
            'CombFTA__X', 'CombFTA__Y', 'CombFTA__Z',
            'CombFTA__Phi', 'CombFTA__Theta', 'CombFTA__Psi'
        ]
        
        self.all_model_output_names = [
            'CombFTB__X', 'CombFTB__Y', 'CombFTB__Z',
            'CombFTB__Phi', 'CombFTB__Theta', 'CombFTB__Psi'
        ]
                
        self.model_input = np.zeros((self.steps, len(self.input_columns)))
        self.task_step_index = 0        
        
        # Lists of indexes for our models
        model_input_indexes = []
        model_output_indexes = []

        for input_type in self.input_columns:
            model_input_indexes.append(self.all_model_input_names.index(input_type))
        for output_type in self.output_columns:
            model_output_indexes.append(self.all_model_output_names.index(output_type))

        self.model_input_indexes = np.array(model_input_indexes)
        self.model_output_indexes = np.array(model_output_indexes)        

    def random_model_generator(self):
        files_in_path_to_models = os.listdir(self.path_to_primary_models)
        trained_models = [model for model in files_in_path_to_models if model.endswith(".pth")]
        random_model = random.choice(trained_models)
        model_name_split = random_model.split("_")

        self.reaction_time_type = model_name_split[0]
        self.output_type = model_name_split[1]
        self.input_type = model_name_split[2]
        self.task_type = model_name_split[3]
        self.cutoff_type = model_name_split[4]
        self.model_type = model_name_split[5].split(".")[0]

    def generate_model_name(self):
        return "_".join([
            self.reaction_time_type, self.output_type, self.input_type,
            self.task_type, self.cutoff_type, self.model_type
        ]) + ".pth"
    
    def load_model(self, model_name, model_path):
        full_path = os.path.join(model_path, model_name)
        self.output_columns = nn_p.ALL_OUTPUT_COMBO[nn_p.OUTPUT_COMBO_NAMES.index(self.output_type)]
        self.input_columns = nn_p.ALL_INPUT_COMBO[nn_p.INPUT_COMBO_NAMES.index(self.input_type)]
        self.steps = nn_p.ALL_RATE_COMBO[nn_p.RT_COMBO_NAMES.index(self.reaction_time_type)]

        if self.model_type == "CNN":
            model = nn_f.CNN(input_dim=len(self.input_columns), output_dim=len(self.output_columns), step=self.steps)
        elif self.model_type == "LSTM":
            model = nn_f.CNN_LSTM(input_dim=len(self.input_columns), output_dim=len(self.output_columns), step=self.steps)
        else:
            print(f"\nNo model type: {self.model_type}")
            sys.exit()

        model.load_state_dict(torch.load(full_path, map_location=torch.device('cpu')))

        return model

    def initialize_derivatives(self):
        self.phi_dot = 0.0
        self.theta_dot = 0.0
        self.psi_dot = 0.0
        self.x_dot_dot = 0.0
        self.y_dot_dot = 0.0
        self.z_dot_dot = 0.0
        self.phi_dot_dot = 0.0
        self.theta_dot_dot = 0.0
        self.psi_dot_dot = 0.0

        self.phi_d1 = 0.0
        self.theta_d1 = 0.0
        self.psi_d1 = 0.0
        self.x_dot_d1 = 0.0
        self.y_dot_d1 = 0.0
        self.z_dot_d1 = 0.0
        self.phi_dot_d1 = 0.0
        self.theta_dot_d1 = 0.0
        self.psi_dot_d1 = 0.0

    def update(self, states, u_leader):
        """Update the controller to output the appropriate forces, resisting acceleration.

        Args:
            states: The current states of the system [x, y, z, phi, theta, psi, vx, vy, vz, vphi, vtheta, vpsi].
            u_leader: Forces and torques from the leader
        """
        # Extract current positions and velocities
        x, y, z = states[0][0], states[1][0], states[2][0]
        phi, theta, psi = states[3][0], states[4][0], states[5][0]
        x_dot, y_dot, z_dot = states[6][0], states[7][0], states[8][0]
        omega_x, omega_y, omega_z = states[9][0], states[10][0], states[11][0]

        # Forces from the leader in the inertial frame
        force_x_leader, force_y_leader, force_z_leader = u_leader[0][0], u_leader[1][0], u_leader[2][0]
        torque_x_leader, torque_y_leader, torque_z_leader = u_leader[3][0], u_leader[4][0], u_leader[5][0]

        # Differentiate
        self.phi_dot = self.beta * self.phi_dot + (1 - self.beta) * ((phi - self.phi_d1) / tp.T_STEP)
        self.theta_dot = self.beta * self.theta_dot + (1 - self.beta) * ((theta - self.theta_d1) / tp.T_STEP)
        self.psi_dot = self.beta * self.psi_dot + (1 - self.beta) * ((psi - self.psi_d1) / tp.T_STEP)
        
        self.x_dot_dot = self.beta * self.x_dot_dot + (1 - self.beta) * ((x_dot - self.x_dot_d1) / tp.T_STEP)
        self.y_dot_dot = self.beta * self.y_dot_dot + (1 - self.beta) * ((y_dot - self.y_dot_d1) / tp.T_STEP)
        self.z_dot_dot = self.beta * self.z_dot_dot + (1 - self.beta) * ((z_dot - self.z_dot_d1) / tp.T_STEP)

        self.phi_dot_dot = self.beta * self.phi_dot_dot + (1 - self.beta) * ((self.phi_dot - self.phi_dot_d1) / tp.T_STEP)
        self.theta_dot_dot = self.beta * self.theta_dot_dot + (1 - self.beta) * ((self.theta_dot - self.theta_dot_d1) / tp.T_STEP)
        self.psi_dot_dot = self.beta * self.psi_dot_dot + (1 - self.beta) * ((self.psi_dot - self.psi_dot_d1) / tp.T_STEP)

        all_new_data = np.array([
            x, y, z, phi, theta, psi, x_dot, y_dot, z_dot, self.phi_dot, self.theta_dot, self.psi_dot,
            self.x_dot_dot, self.y_dot_dot, self.z_dot_dot, self.phi_dot_dot, self.theta_dot_dot, self.psi_dot_dot,
            force_x_leader, force_y_leader, force_z_leader, torque_x_leader, torque_y_leader, torque_z_leader
        ])
        
        add_new_data = all_new_data[self.model_input_indexes]
        self.model_input = np.delete(np.vstack((self.model_input, add_new_data)), 0, 0)


        output = np.zeros((len(self.output_columns), 1))
        if self.task_step_index > self.steps:
            model_input_tensor = torch.tensor(self.model_input[np.newaxis, :, :], dtype=torch.float32)
            with torch.no_grad():
                output = self.model(model_input_tensor)[0].numpy()
        else:
            self.task_step_index += 1

        all_forces = np.zeros((len(self.all_model_output_names), 1))
        all_forces[self.model_output_indexes] = output.reshape(-1,1)

        force_x_follower = saturate(all_forces[0][0], tp.FORCE_X_MAX_FOLLOWER)
        force_y_follower = saturate(all_forces[1][0], tp.FORCE_Y_MAX_FOLLOWER)
        force_z_follower = saturate(all_forces[2][0], tp.FORCE_Z_MAX_FOLLOWER)
        torque_x_follower = saturate(all_forces[3][0], tp.TORQUE_PHI_MAX_FOLLOWER)
        torque_y_follower = saturate(all_forces[4][0], tp.TORQUE_THETA_MAX_FOLLOWER)
        torque_z_follower = saturate(all_forces[5][0], tp.TORQUE_PSI_MAX_FOLLOWER)

        tau = np.array([
            [force_x_follower],
            [force_y_follower],
            [force_z_follower],
            [torque_x_follower],
            [torque_y_follower],
            [torque_z_follower]
        ])

        self.x_dot_d1 = x_dot
        self.y_dot_d1 = y_dot
        self.z_dot_d1 = z_dot
        self.phi_d1 = phi
        self.theta_d1 = theta
        self.psi_d1 = psi
        self.phi_dot_d1 = self.phi_dot
        self.theta_dot_d1 = self.theta_dot
        self.psi_dot_d1 = self.psi_dot

        return tau

    def reset(self):
        """Reset the controller states."""
        self.x_dot_d1 = 0.0
        self.y_dot_d1 = 0.0
        self.z_dot_d1 = 0.0
        self.phi_d1 = 0.0
        self.theta_d1 = 0.0
        self.psi_d1 = 0.0
        self.phi_dot_d1 = 0.0
        self.theta_dot_d1 = 0.0
        self.psi_dot_d1 = 0.0


def saturate(u, limit):
    """Saturate the input value to the specified limit."""
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u

 