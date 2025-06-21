# train.py
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Setup
print("Menyiapkan data dan model...\n")

X = np.load('app/embedded/bert_embedding.npy')
y_df = pd.read_csv('app/preprocessed/preprocessed_data.csv')
y = y_df.drop(columns=['Tweet']).values

# Split awal: untuk test 20% disimpan
X_train_full, _, y_train_full, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Split lagi untuk train/validation
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.1, random_state=42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Load best params
try:
    with open('app/tuning_result/best_params.json') as f:
        data = json.load(f)
        best_item = max(data, key=lambda x: x['val_acc'])
        best_params = best_item['params']
except Exception as e:
    print("Gagal membaca best_params.json:", e)
    exit(1)

# Model class
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

# Training
print(f"Mulai training dengan hyperparameter: {best_params}", flush=True)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=best_params['batch_size'], shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=best_params['batch_size'], shuffle=False)

model = BiGRUModel(best_params['units']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
criterion = nn.BCELoss()

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(best_params['epochs']):
    model.train()
    epoch_train_loss, epoch_train_acc = [], []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = criterion(preds, yb.float())
        loss.backward()
        optimizer.step()

        epoch_train_loss.append(loss.item())
        preds_binary = (preds > 0.5).float()
        acc = (preds_binary == yb).float().mean().item()
        epoch_train_acc.append(acc)

    # Validation
    model.eval()
    epoch_val_loss, epoch_val_acc = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).squeeze()
            loss = criterion(preds, yb.float())
            epoch_val_loss.append(loss.item())
            preds_binary = (preds > 0.5).float()
            acc = (preds_binary == yb).float().mean().item()
            epoch_val_acc.append(acc)

    avg_train_loss = np.mean(epoch_train_loss)
    avg_val_loss = np.mean(epoch_val_loss)
    avg_train_acc = np.mean(epoch_train_acc)
    avg_val_acc = np.mean(epoch_val_acc)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accs.append(avg_train_acc)
    val_accs.append(avg_val_acc)

    print(f"Epoch {epoch+1}/{best_params['epochs']} - Train Acc: {avg_train_acc:.4f} | Val Acc: {avg_val_acc:.4f} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}", flush=True)

# Simpan model
save_path = 'test/contoh.pt'
torch.save({
    'model_state_dict': model.state_dict(),
    'units': best_params['units']
}, save_path)
print(f"\nTraining selesai. Model disimpan di {save_path}", flush=True)

# Visualisasi
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')
plt.legend()

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()

plt.tight_layout()
plt.savefig('test')
plt.show()
