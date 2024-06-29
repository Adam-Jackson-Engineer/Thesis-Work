"""
nn_functions.py

Contains helper functions and classes for neural network training.
"""

import torch
import torch.nn as nn
import numpy as np

def create_sequences(data, tasks, steps, input_cols, output_cols):
    """
    Create sequences for training the neural network.

    Args:
        data (pd.DataFrame): The input data.
        tasks (list): The tasks to filter data.
        steps (int): The number of steps in the sequence.
        input_cols (list): The input columns.
        output_cols (list): The output columns.

    Returns:
        tuple: Numpy arrays for input (X) and output (y) sequences.
    """
    xs, ys = [], []
    printed = False
    for i in range(len(data) - steps):
        if data.iloc[i]['Task'] in tasks:
            sequence_rounds = data.iloc[i:(i + steps)]['RoundsUsed']
            next_round = data.iloc[i + steps]['RoundsUsed']
            if sequence_rounds.nunique() == 1 and sequence_rounds.iloc[0] == next_round:
                x = data.iloc[i:(i + steps)][input_cols].to_numpy()
                y = data.iloc[i + steps][output_cols].to_numpy()
                if not printed:
                    print(x.shape) 
                    printed = True
                xs.append(x.astype(np.float32))
                ys.append(y.astype(np.float32))
    return np.array(xs), np.array(ys)

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, step):
        """
        Initialize the CNN model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            step (int): Sequence length.
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()                
        conv_output_size = ((step - 2) // 2) // 2
        self.fc1 = nn.Linear(32 * conv_output_size, 100)
        self.fc2 = nn.Linear(100, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the CNN model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN_LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, step):
        """
        Initialize the CNN-LSTM model.

        Args:
            input_dim (int): Number of input features.
            output_dim (int): Number of output features.
            seq_length (int): Sequence length.
        """
        super(CNN_LSTM, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size=32, hidden_size=50, num_layers=2, batch_first=True)
        self.fc = nn.Linear(50, output_dim)

    def forward(self, x):
        """
        Forward pass of the CNN-LSTM model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # CNN layers
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        
        # Prepare for LSTM layers
        x = x.transpose(1, 2)
        
        # LSTM layers
        x, (hn, cn) = self.lstm(x)
        
        # Fully connected layer
        x = self.fc(x[:, -1, :])  # Use the last time step's output
        return x