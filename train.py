# train_bigru.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import json
import os
import sys
    
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------
# Argparse: file JSON hasil tuning
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='Path ke tuning_result.json')
args = parser.parse_args()

with open(args.config, 'r') as f:
    all_results = json.load(f)['results']
best_result = max(all_results, key=lambda x: x['val_acc'])
best_params = best_result['params']
print("Hyperparameter terbaik:", best_params)

# ---------------------------
# Load data
# ---------------------------
X = np.load('embedded/bert_embedding.npy')
y_df = pd.read_csv('preprocessed/preprocessed_data.csv')
y = y_df.drop(columns=['Tweet']).values

# Split menjadi train + test
from sklearn.model_selection import train_test_split
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

X_trainval_tensor = torch.tensor(X_trainval, dtype=torch.float32)
y_trainval_tensor = torch.tensor(y_trainval, dtype=torch.float32)

# ---------------------------
# Model
# ---------------------------
class BiGRUModel(nn.Module):
    def __init__(self, units):
        super(BiGRUModel, self).__init__()
        self.gru = nn.GRU(input_size=768, hidden_size=units, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(units * 2, 9)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h = self.gru(x)
        h_concat = torch.cat((h[0], h[1]), dim=1)
        out = self.fc(h_concat)
        return self.sigmoid(out)

# ---------------------------
# Training
# ---------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BiGRUModel(best_params['units']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
criterion = nn.BCELoss()

train_loader = DataLoader(TensorDataset(X_trainval_tensor, y_trainval_tensor),
                          batch_size=best_params['batch_size'], shuffle=True)

print("\nMulai Training...")
for epoch in range(best_params['epochs']):
    model.train()
    epoch_losses, epoch_accs = [], []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = criterion(preds, yb.float())
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())
        preds_binary = (preds > 0.5).float()
        acc = (preds_binary == yb).float().mean().item()
        epoch_accs.append(acc)

    print(f"Epoch {epoch+1}/{best_params['epochs']} - Loss: {np.mean(epoch_losses):.4f} | Accuracy: {np.mean(epoch_accs):.4f}")

# ---------------------------
# Evaluation
# ---------------------------
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=best_params['batch_size'])

model.eval()
test_losses, test_accs = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).squeeze()
        loss = criterion(preds, yb.float())
        test_losses.append(loss.item())
        preds_binary = (preds > 0.5).float()
        acc = (preds_binary == yb).float().mean().item()
        test_accs.append(acc)

print(f"\nEvaluasi Test Set:")
print(f"Test Loss: {np.mean(test_losses):.4f}")
print(f"Test Accuracy: {np.mean(test_accs):.4f}")

# ---------------------------
# Simpan Model
# ---------------------------
save_dir = "./model/"
os.makedirs(save_dir, exist_ok=True)
base_name = "Bi-GRU.pt"
file_path = os.path.join(save_dir, base_name)
counter = 0
while os.path.exists(file_path):
    counter += 1
    name, ext = os.path.splitext(base_name)
    file_path = os.path.join(save_dir, f"{name}({counter}){ext}")

torch.save(model.state_dict(), file_path)
print(f"Model disimpan ke: {file_path}")
