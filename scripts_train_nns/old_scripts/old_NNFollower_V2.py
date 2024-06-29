
import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scripts_train_nns.old_NNParam as P
import scripts_train_nns.old_NNFunctions as F

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

modelFolder = "trainedModels"
os.makedirs(modelFolder, exist_ok=True)

# Training Parameters
batch_size = 64
num_epochs = 100

for i, time_req in enumerate(P.allTimeReq):
    data = pd.read_csv(f"trainingData/TrainingData_{time_req}sec_2D.csv")
    for j, tasks in enumerate(P.allTasksCombo):
        for k, steps in enumerate(P.allRTCombo):
            for m, input in enumerate(P.allInputCombo):
                for n, output in enumerate(P.allOutputCombo):

                    model_name = (P.RTComboNames[k] + "_" + 
                                    P.OutputComboNames[n] + "_" + 
                                    P.InputComboNames[m] + "_" +
                                    P.TaskComboNames[j] + "_" +
                                    P.TimeReqNames[i])

                    try:
                        X_data, y_data = F.create_sequences(data, tasks, steps, input, output)
                        X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
                        del X_data, y_data

                        # Convert to PyTorch tensors
                        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
                        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
                        del X_train, y_train, X_val, y_val

                        # Create dataloaders
                        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                    
                        model = F.CNN(input_dim=len(input), output_dim=len(output), step=steps)
                        criterion = nn.MSELoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                        train_losses = []
                        val_losses = []

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
                            
                            print(f'Epoch {epoch+1}, Validation Loss: {val_loss}')
                            
                        modelName = f"{model_name}.pth"
                        modelPath = os.path.join(modelFolder, modelName)
                        torch.save(model.state_dict(), modelPath)

                        plt.figure(figsize=(10,5))
                        plt.plot(train_losses, label='Training Loss')
                        plt.plot(val_losses, label='Validation Loss')
                        plt.title('Training and Validation Losses')
                        plt.xlabel('Epochs')
                        plt.ylabel('Loss')
                        plt.legend()
                        plt.savefig(os.path.join(modelFolder, f'{model_name}.png'))

                        del train_loader, val_loader, train_dataset, val_dataset
                        del X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor
                        del model, criterion, optimizer, train_losses, val_losses

                    except:
                        print(f"Model Failed: {model_name}")