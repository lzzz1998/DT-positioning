# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 10:48:30 2025

"""

from settings import APP_CONFIG, UNREAL_CONFIG

CHECKPOINT_PATH = APP_CONFIG['checkpoint_path']
HISTORY_PATH = APP_CONFIG['training_history']
CSV_PATH = APP_CONFIG['preprocessed_csv_file']
NUM_OF_BEACONS = UNREAL_CONFIG['num_of_beacons']

import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 18})

import pandas as pd
df = pd.read_csv(CSV_PATH)
df.head()

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset




X = df[['Beacon1', 'Beacon2', 'Beacon3', 'Beacon4', 'Beacon5', 'Beacon6', 'Beacon7', 'Beacon8', 'Beacon9', 
        'Beacon10', 'Beacon11', 'Beacon12', 'Beacon13', 'Beacon14', 'Beacon15', 'Beacon16', 'Beacon17', 
        'Beacon18', 'Beacon19', 'Beacon20', 'Beacon21', 'Beacon22']].values[0:20000]
#X = df[['Beacon1', 'Beacon2']].values

y = df[['x', 'y', 'z']].values[0:20000]



split_idx = int(20000 * 0.9)  # 90% training, 10% validation
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# Standardize RSSI (input features)
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)  

# Standardize Position (output labels)
scaler_y = StandardScaler()
y_train = scaler_y.fit_transform(y_train)
y_val = scaler_y.transform(y_val)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)  
val_loader = DataLoader(val_dataset, batch_size=128)

import torch.nn as nn
import torch.optim as optim

from einops import rearrange

class UNetLocalization(nn.Module):
    def __init__(self, input_dim=22, output_dim=64):
        super(UNetLocalization, self).__init__()
        
       
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        
        
        self.bottleneck = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU()
        )
        
        
        self.decoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


class Transformer(nn.Module):
    def __init__(self, input_dim=64, num_heads=4, num_layers=2):
        super(Transformer, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, 3)  

    def forward(self, x):
        x = x.unsqueeze(1)  
        x = self.transformer(x)
        x = self.fc(x[:, 0, :])  
        return x

class UNetTransformer(nn.Module):
    def __init__(self):
        super(UNetTransformer, self).__init__()
        self.unet = UNetLocalization()
        self.transformer = Transformer()
        self.fc = nn.Linear(64, 3)  

    def forward(self, x):
        x = self.unet(x)  
        x = self.transformer(x)  
        
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetTransformer().to(device)


criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.0001)


num_epochs = 100
lowest_test_loss = float('inf')  
best_model_state_dict = None

model = UNetTransformer().to(device)
model.load_state_dict(torch.load("best_model.pth"))


model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


preds = []
gts = []

with torch.no_grad():
    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)  # Move to GPU if available
        y_batch = y_batch.to(device)

        # Predict
        y_pred = model(X_batch)

        # Convert predictions and ground truth back to original scale
        y_pred_original = scaler_y.inverse_transform(y_pred.cpu().numpy())
        y_true_original = scaler_y.inverse_transform(y_batch.cpu().numpy())

        # Store
        preds.append(y_pred_original)
        gts.append(y_true_original)

# Convert lists to numpy arrays
preds = np.concatenate(preds, axis=0)
gts = np.concatenate(gts, axis=0)

plt.figure(figsize=(8, 6))

# Plot ground truth
plt.plot(gts[:, 0], gts[:, 1], label="Ground Truth", color="blue", linestyle='dashed', marker='o')

# Plot predictions
plt.plot(preds[:, 0], preds[:, 1], label="Predicted", color="red", marker='x')

plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.title("Predicted vs Ground Truth Trajectory (Top-Down View)")
plt.legend()
plt.grid()
plt.show()






    


