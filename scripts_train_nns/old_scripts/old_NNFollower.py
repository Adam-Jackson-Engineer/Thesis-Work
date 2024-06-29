import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Max completion time limit
allTimeReq = np.array([5,8,10,12,14])

# All possible task combinations
trans_x_tasks = ['TX','TX_N']
trans_y_tasks = ['TY','TY_N']
trans_xy_tasks = ['TX','TX_N','TY','TY_N','TXY_NN','TXY_PP']
rot_z_tasks = ['RZ','RZ_N']
trans_rot_tasks = ['R_leader','TXY_RZ_NPP']
tasks_2D = ['R_leader','TX','TX_N','TY','TY_N','RZ','RZ_N','TXY_NN','TXY_PP','TXY_RZ_NPP']
allTasksCombo = [trans_x_tasks, trans_y_tasks, trans_xy_tasks, rot_z_tasks, trans_rot_tasks, tasks_2D]

# All input options
vel_only_inputs = ['Vel_X', 'Vel_Y', 'Vel_Psi']
vel_acc_inputs = ['Vel_X', 'Vel_Y', 'Vel_Psi', 'Acc_X', 'Acc_Y', 'Acc_Psi']
tao_inputs = ['CombFTA__X', 'CombFTA__Y', 'CombFTA__Psi']
vel_acc_tao_inputs = ['Vel_X', 'Vel_Y', 'Vel_Psi', 'Acc_X', 'Acc_Y', 'Acc_Psi', 'CombFTA__X', 'CombFTA__Y', 'CombFTA__Psi']
allInputCombo = [vel_only_inputs, vel_acc_inputs, tao_inputs, vel_acc_inputs]

# All possible output options
tao_x_output = ['CombFTB__X']
tao_y_output = ['CombFTB__Y']
tao_psi_output = ['CombFTB__Psi']
tao_xy_output = ['CombFTB__X','CombFTB__Y']
tao_xypsi_output = ['CombFTB__X', 'CombFTB__Y', 'CombFTB__Psi']
allOutputCombo = [tao_x_output, tao_y_output, tao_psi_output, tao_xy_output, tao_xypsi_output]

# All possible time lengths
SRT = 0.25
CRT = 0.8
allRTCombo = [SRT, CRT]

# Load data
req_time = 12
data = pd.read_csv(f"trainingData/TrainingData_{req_time}sec_2D.csv")
print("Data Loaded")

# Define input and output columns
input_columns = vel_acc_tao_inputs
output_columns = ['CombFTB__X', 'CombFTB__Y', 'CombFTB__Psi']


# Function to create sequences
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        sequence_rounds = data.iloc[i:(i+seq_length)]['RoundsUsed']
        next_round = data.iloc[i+seq_length]['RoundsUsed']
        if sequence_rounds.nunique() == 1 and sequence_rounds.iloc[0] == next_round:
            x = data.iloc[i:(i+seq_length)][input_columns].to_numpy()
            y = data.iloc[i+seq_length][output_columns].to_numpy()
            xs.append(x.astype(np.float32))
            ys.append(y.astype(np.float32))
    return np.array(xs), np.array(ys)


# Length of a sequence
seq_length = 10
X, y = create_sequences(data,seq_length)
print("Sequences Created")

del data

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data Split")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Create dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim):
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
    
model = CNN(input_dim=len(input_columns), output_dim=len(output_columns))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_losses = []
val_losses = []

print("Starting Training")
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    train_losses.append(train_loss)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    
    print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')
    
modelFolder = "trainedModels"
modelName = f"trained_model_{req_time}sec_{seq_length}steps.pth"
modelPath = os.path.join(modelFolder, modelName)
torch.save(model.state_dict(), modelPath)

plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Losses')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(modelFolder, f'training_validation_losses_{req_time}sec_{seq_length}steps.png'))  # Save the plot


# P95FxA = np.percentile(np.abs(df['CombFTA__X'].values), 95)
# P95FyA = np.percentile(np.abs(df['CombFTA__Y'].values), 95)
# P95MzA = np.percentile(np.abs(df['CombFTA__Psi'].values), 95)
# P95FxB = np.percentile(np.abs(df['CombFTB__X'].values), 95)
# P95FyB = np.percentile(np.abs(df['CombFTB__Y'].values), 95)
# P95MzB = np.percentile(np.abs(df['CombFTB__Psi'].values), 95)

# print(f'95th Percentile Fx_A: {P95FxA}')
# print(f'95th Percentile Fy_A: {P95FyA}')
# print(f'95th Percentile Mz_A: {P95MzA}')
# print(f'95th Percentile Fx_B: {P95FxB}')
# print(f'95th Percentile Fy_B: {P95FyB}')
# print(f'95th Percentile Mz_B: {P95MzB}')


# MaxFxA = np.max(np.abs(df['CombFTA__X'].values))
# MaxFyA = np.max(np.abs(df['CombFTA__Y'].values))
# MaxMzA = np.max(np.abs(df['CombFTA__Psi'].values))
# MaxFxB = np.max(np.abs(df['CombFTB__X'].values))
# MaxFyB = np.max(np.abs(df['CombFTB__Y'].values))
# MaxMzB = np.max(np.abs(df['CombFTB__Psi'].values))

# print(f'Max Fx_A: {MaxFxA}')
# print(f'Max Ff_A: {MaxFyA}')
# print(f'Max Mz_A: {MaxMzA}')
# print(f'Max Fx_B: {MaxFxB}')
# print(f'Max Ff_B: {MaxFyB}')
# print(f'Max Mz_B: {MaxMzB}')
