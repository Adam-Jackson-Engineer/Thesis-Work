"""
train_nn.py

This script trains a neural network to predict force and torque from input velocity and acceleration data.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Define constants
ALL_TIME_REQ = np.array([5, 8, 10, 12, 14])
TASKS_2D = ['R_leader', 'TX', 'TX_N', 'TY', 'TY_N', 'RZ', 'RZ_N', 'TXY_NN', 'TXY_PP', 'TXY_RZ_NPP']
VEL_ACC_TAO_INPUTS = ['Vel_X', 'Vel_Y', 'Vel_Psi', 'Acc_X', 'Acc_Y', 'Acc_Psi', 'CombFTA__X', 'CombFTA__Y', 'CombFTA__Psi']
OUTPUT_COLUMNS = ['CombFTB__X', 'CombFTB__Y', 'CombFTB__Psi']
SEQ_LENGTH = 10

def create_sequences(data, seq_length):
    """
    Create sequences for training the neural network.

    Args:
        data (pd.DataFrame): The input data.
        seq_length (int): The length of the sequence.

    Returns:
        tuple: Numpy arrays for input (X) and output (y) sequences.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        sequence_rounds = data.iloc[i:(i + seq_length)]['RoundsUsed']
        next_round = data.iloc[i + seq_length]['RoundsUsed']
        if sequence_rounds.nunique() == 1 and sequence_rounds.iloc[0] == next_round:
            x = data.iloc[i:(i + seq_length)][VEL_ACC_TAO_INPUTS].to_numpy()
            y = data.iloc[i + seq_length][OUTPUT_COLUMNS].to_numpy()
            xs.append(x.astype(np.float32))
            ys.append(y.astype(np.float32))
    return np.array(xs), np.array(ys)

def main():
    req_time = 12
    data = pd.read_csv(f"data_training/training_data_{req_time}sec_2d.csv")
    print("Data Loaded")

    X, y = create_sequences(data, SEQ_LENGTH)
    print("Sequences Created")

    del data

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data Split")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    class CNN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(64 * (SEQ_LENGTH - 1), 50)
            self.fc2 = nn.Linear(50, output_dim)

        def forward(self, x):
            x = x.transpose(1, 2)
            x = torch.relu(self.conv1(x))
            x = self.flatten(x)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = CNN(input_dim=len(VEL_ACC_TAO_INPUTS), output_dim=len(OUTPUT_COLUMNS))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses, val_losses = [], []

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

        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}')

    model_folder = "trained_models"
    os.makedirs(model_folder, exist_ok=True)
    model_name = f"trained_model_{req_time}sec_{SEQ_LENGTH}steps.pth"
    model_path = os.path.join(model_folder, model_name)
    torch.save(model.state_dict(), model_path)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(model_folder, f'training_validation_losses_{req_time}sec_{SEQ_LENGTH}steps.png'))  # Save the plot

if __name__ == "__main__":
    main()
