"""
train_snn_iteratively.py

This script iterates over multiple configurations to train neural networks and saves the trained models.
"""

import os
import copy
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

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, device):
    """
    Train the given model and return the best model weights.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        criterion: Loss function.
        optimizer: Optimizer.
        num_epochs (int): Number of epochs to train.
        patience (int): Patience for early stopping.
        device: Device to run the training on.

    Returns:
        tuple: (best_model_wts, train_losses, val_losses)
    """
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
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
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
        
        print(f'Epoch {epoch + 1}, Validation Loss: {val_loss}')


    return best_model_wts, train_losses, val_losses

def main(test_set, model_save_path):
    tasks_type = "2d"
    groups_type = "follower"

    batch_size = 64
    num_epochs = 1000
    patience = 10

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}\n")

    os.makedirs(model_save_path, exist_ok=True)

    for i, cutoff_perc in enumerate(P.CUTOFF_PERC):
        data_file_name = f'training_data_{cutoff_perc}_perc_{tasks_type}_{groups_type}.csv'
        data_file_path = os.path.join(P.TRAINING_DATA_FOLDER_PATH, data_file_name)
        print(f"Loading Data: {data_file_name}")
        data = pd.read_csv(data_file_path)
        print(f"Data loaded\n")
        for j, tasks in enumerate(P.ALL_TASKS_COMBO):
            for k, steps in enumerate(P.ALL_RATE_COMBO):
                for m, input_columns in enumerate(P.ALL_INPUT_COMBO):
                    for n, output_columns in enumerate(P.ALL_OUTPUT_COMBO):
                        model_name_base = (
                            P.RT_COMBO_NAMES[k] + "_" +
                            P.OUTPUT_COMBO_NAMES[n] + "_" +
                            P.INPUT_COMBO_NAMES[m] + "_" +
                            P.TASK_COMBO_NAMES[j] + "_" +
                            P.CUTOFF_NAMES[i]
                        )
                        
                        if model_name_base in test_set:
                            full_model_cnn_name = model_name_base + "_CNN.pth"
                            full_model_lstm_name = model_name_base + "_LSTM.pth"

                            if full_model_cnn_name in os.listdir(model_save_path) and full_model_lstm_name in os.listdir(model_save_path):
                                print(f"Models already done:    {model_name_base}")
                                continue

                            print(f"Creating Sequences: {model_name_base}")
                            X_data, y_data = F.create_sequences(data, tasks, steps, input_columns, output_columns)
                            X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
                            print(f"Sequences Created\n")
                            del X_data, y_data

                            X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
                            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
                            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
                            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
                            del X_train, y_train, X_val, y_val

                            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                            for model_class, model_suffix in [(F.CNN, "CNN"), (F.CNN_LSTM, "LSTM")]:
                                model_name = model_name_base + "_" + model_suffix
                                model = model_class(input_dim=len(input_columns), output_dim=len(output_columns), step=steps).to(device)
                                criterion = nn.MSELoss().to(device)
                                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                                print(f"\nStarting Training: {model_name}")
                                best_model_wts, train_losses, val_losses = train_model(
                                    model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, device
                                )

                                model.load_state_dict(best_model_wts)
                                model_name_pth = f"{model_name}.pth"
                                model_path = os.path.join(model_save_path, model_name_pth)
                                torch.save(model.state_dict(), model_path)

                                plt.figure(figsize=(10, 5))
                                plt.plot(train_losses, label='Training Loss')
                                plt.plot(val_losses, label='Validation Loss')
                                plt.title(f'Training and Validation Losses ({model_suffix})')
                                plt.xlabel('Epochs')
                                plt.ylabel('Loss')
                                plt.legend()
                                plt.savefig(os.path.join(model_save_path, f'{model_name}.png'))

                                del model, criterion, optimizer, train_losses, val_losses

                            del train_loader, val_loader, train_dataset, val_dataset
                            del X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor

                            # except Exception as e:
                            #     print(f"Model Failed: {model_name}")
                            #     print(str(e))

if __name__ == "__main__":
    test_set = P.MODEL_PRIMARY_LIST
    model_save_path = P.TRAINED_MODEL_FOLDER_PRIMARY_PATH
    main(test_set, model_save_path)

    test_set = P.MODEL_ALL_LIST
    model_save_path = P.TRAINED_MODEL_FOLDER_ALL_PATH
    main(test_set, model_save_path)