import sys
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch import optim
from torch.cuda.amp import GradScaler, autocast

# Adjust the path to import your model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from bilstm import BiLSTMModel
print("Imported BiLSTMModel from models")

class CustomDataset(Dataset):
    def __init__(self, data):
        # Extract radar_i, radar_q, and tfm_ecg2
        self.inputs = torch.tensor(data[["radar_i", "radar_q"]].values, dtype=torch.float32)
        self.targets = torch.tensor(data["tfm_ecg2"].values, dtype=torch.float32).unsqueeze(1)  # Shape: (batch_size, 1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx].unsqueeze(0), self.targets[idx]  # Add sequence dimension (1, input_size)

def train_model():
    # Load dataset
    df = pd.read_csv(r'D:\dev\biomedical-signal-research\datasets\csv_files\datasets_subject_01_to_10_scidata\GDN0001\GDN0001_1_Resting.csv')
    print(f"Dataset shape: {df.shape}")

    # Reduce dataset size for initial testing
    df = df.sample(n=100000)  # Randomly sample 10,000 rows for testing

    # Create dataset and dataloader
    dataset = CustomDataset(df)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, pin_memory=True)

    # Define model
    input_size = 2  # radar_i and radar_q
    model = BiLSTMModel(input_size).to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Enable mixed precision training
    scaler = GradScaler()

    print("Starting training...")
    # Training loop
    num_epochs = 10
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # Mixed precision training
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Scale the loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:  # Log progress every 10 batches
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), 'bilstm_model.pth')
    print("Model training complete. Saved as 'bilstm_model.pth'.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train_model()
