import os
import torch
import torch.nn as nn
import numpy as np
import scripts_simulation.table_parameters as P

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, seq_length):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * (seq_length - 1), 50)
        self.fc2 = nn.Linear(50, output_dim)
        
    def forward(self, x):
        x = x.transpose(1,2)
        x = torch.relu(self.conv1(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ctrlLearnedFollower:
    def __init__(self,timeSteps):
        # Load the trained model
        modelPath = os.path.join("DataGen","trainedModels",f"trained_model_12sec_{timeSteps}steps.pth")
        self.model = self.load_model(modelPath, timeSteps)

        # 2D follower data
        self.timeSteps = timeSteps
        self.history = np.zeros((self.timeSteps,9))

        # dirty derivative parameters
        self.sigma = 0.05  # cutoff freq for dirty derivative
        self.beta = (2 * self.sigma - P.Ts) / (2 * self.sigma + P.Ts)  
        
        self.initialize_derivatives()

        self.delay = 100
        self.happened = 0

    def load_model(self,modelPath, timeSteps):
        input_columns = ['Vel_X', 'Vel_Y', 'Vel_Psi', 'Acc_X', 'Acc_Y', 'Acc_Psi', 'CombFTA__X', 'CombFTA__Y', 'CombFTA__Psi']
        output_columns = ['CombFTB__X', 'CombFTB__Y', 'CombFTB__Psi']

        model = CNN(input_dim=len(input_columns), output_dim=len(output_columns), seq_length=timeSteps)
        model.load_state_dict(torch.load(modelPath))
        return model

    def initialize_derivatives(self):
        self.xddot = 0.0
        self.yddot = 0.0
        self.zddot = 0.0
        self.phiddot = 0.0
        self.thtddot = 0.0
        self.psiddot = 0.0

        self.xdot_d1 = 0.0
        self.ydot_d1 = 0.0
        self.zdot_d1 = 0.0
        self.phidot_d1 = 0.0
        self.thtdot_d1 = 0.0
        self.psidot_d1 = 0.0
        
    def update(self, u_leader, states):
        xdot = states[6][0]
        ydot = states[7][0]
        zdot = states[8][0]
        phidot = states[9][0]
        thtdot = states[10][0]
        psidot = states[11][0]

        Fx = u_leader[0][0]
        Fy = u_leader[1][0]
        Fz = u_leader[2][0]
        Mx = u_leader[3][0]
        My = u_leader[4][0]
        Mz = u_leader[5][0]

        # differentiate
        self.xddot = self.beta * self.xddot + (1 - self.beta) * ((xdot - self.xdot_d1) / P.Ts)
        self.yddot = self.beta * self.yddot + (1 - self.beta) * ((ydot - self.ydot_d1) / P.Ts)
        self.zddot = self.beta * self.zddot + (1 - self.beta) * ((zdot - self.zdot_d1) / P.Ts)
        self.phiddot = self.beta * self.phiddot + (1 - self.beta) * ((phidot - self.phidot_d1) / P.Ts)
        self.thtddot = self.beta * self.thtddot + (1 - self.beta) * ((thtdot - self.thtdot_d1) / P.Ts)
        self.psiddot = self.beta * self.psiddot + (1 - self.beta) * ((psidot - self.psidot_d1) / P.Ts)

        new_data = np.array([xdot, ydot, psidot, self.xddot, self.yddot, self.zddot, Fx, Fy, Mz])
        self.history = np.delete(np.vstack((self.history, new_data)),0,0)

        if self.happened > self.delay:
            history_tensor = torch.tensor(self.history[np.newaxis,:,:], dtype=torch.float32)
            with torch.no_grad():
                output = self.model(history_tensor)
            Fx, Fy, Mz = output[0].numpy()
        else:
            Fx = 0.0
            Fy = 0.0
            Mz = 0.0
            self.happened += 1

        Fx_sat = saturate(Fx, P.Fx_max_F)
        Fy_sat = saturate(Fy, P.Fy_max_F)
        Mz_sat = saturate(Mz, P.Mz_max_F)

        tau = np.array([[Fx_sat], [Fy_sat], [0.0], [0.0], [0.0], [Mz_sat]])

        # Save D1 values
        self.xdot_d1 = xdot
        self.ydot_d1 = ydot
        self.zdot_d1 = zdot
        self.phidot_d1 = phidot
        self.thtdot_d1 = thtdot
        self.psidot_d1 = psidot

        return tau
 

def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u

 
