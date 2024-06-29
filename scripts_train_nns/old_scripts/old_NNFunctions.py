import torch
import torch.nn as nn
import numpy as np
    
# Function to create sequences
def create_sequences(data, tasks, steps, input, output):
    xs = []
    ys = []
    for i in range(len(data)-steps):
        if data.iloc[i]['Task'] in tasks:
            sequence_rounds = data.iloc[i:(i+steps)]['RoundsUsed']     # This is checking that all the data is from the same round
            next_round = data.iloc[i+steps]['RoundsUsed']
            if sequence_rounds.nunique() == 1 and sequence_rounds.iloc[0] == next_round:
                x = data.iloc[i:(i+steps)][input].to_numpy()
                y = data.iloc[i+steps][output].to_numpy()
                xs.append(x.astype(np.float32))
                ys.append(y.astype(np.float32))
    return np.array(xs), np.array(ys)

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, step):
        super(CNN, self).__init__()
        # Assuming 9 features and sequence length of 50
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(2)  # Pooling to reduce dimensionality
        self.flatten = nn.Flatten()
        # Adjust the output features of the fully connected layer depending on the pooling and convolution layers' output
        self.fc1 = nn.Linear(32 * step, 100)  # Adjust the size here based on the output size from the last conv/pool layer
        self.fc2 = nn.Linear(100, output_dim)  # Output array of length 3
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1,2)
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x