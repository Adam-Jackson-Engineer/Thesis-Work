"""
train_nn_v2.py

This script iterates over multiple configurations to train neural networks and saves the trained models.
"""

import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import nn_param as P
import nn_functions as F

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

def main():
    model_folder = "trained_models"
    os.makedirs(model_folder, exist_ok=True)

    batch_size = 64
    num_epochs = 250

    for i, time_req in enumerate(P.CUTOFF_TIMES):
        data_file_name = f"training_data_{time_req}sec_2d.csv"
        data_file_path = os.path.join(P.TRAINING_DATA_FOLDER_PATH, data_file_name)
        data = pd.read_csv(data_file_path)
        for j, tasks in enumerate(P.ALL_TASKS_COMBO):
            for k, steps in enumerate(P.ALL_RT_COMBO):
                for m, input in enumerate(P.ALL_INPUT_COMBO):
                    for n, output in enumerate(P.ALL_OUTPUT_COMBO):
                        model_name = (
                            P.RT_COMBO_NAMES[k] + "_" +
                            P.OUTPUT_COMBO_NAMES[n] + "_" +
                            P.INPUT_COMBO_NAMES[m] + "_" +
                            P.TASK_COMBO_NAMES[j] + "_" +
                            P.CUTOFF_NAMES[i]
                        )

                        try:
                            X_data, y_data = F.create_sequences(data, tasks, steps, input, output)
                            X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
                            del X_data, y_data

                            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                            y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
                            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
                            del X_train, y_train, X_val, y_val

                            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                            model = F.CNN(input_dim=len(input), output_dim=len(output), step=steps)
                            criterion = nn.MSELoss()
                            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                            train_losses, val_losses = [], []

                            print(f"\nStarting Training: {model_name}")

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

                            model_name_pth = f"{model_name}.pth"
                            model_path = os.path.join(model_folder, model_name_pth)
                            torch.save(model.state_dict(), model_path)

                            plt.figure(figsize=(10, 5))
                            plt.plot(train_losses, label='Training Loss')
                            plt.plot(val_losses, label='Validation Loss')
                            plt.title('Training and Validation Losses')
                            plt.xlabel('Epochs')
                            plt.ylabel('Loss')
                            plt.legend()
                            plt.savefig(os.path.join(model_folder, f'{model_name}.png'))

                            del train_loader, val_loader, train_dataset, val_dataset
                            del X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor
                            del model, criterion, optimizer, train_losses, val_losses

                        except Exception as e:
                            print(f"Model Failed: {model_name}")
                            print(str(e))

if __name__ == "__main__":
    main()
