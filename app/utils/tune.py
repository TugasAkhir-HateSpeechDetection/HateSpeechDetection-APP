import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold

# Load data
# X = np.load('./#SKRIPSI/mini_preprocessed_data_bert_embedding.npy')
# y_df = pd.read_csv('./#SKRIPSI/mini_preprocessed_data.csv')
# y = y_df.drop(columns=['Tweet']).values

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

embedding_path = os.path.join(ROOT_DIR, 'app', 'embedded', 'bert_embedding.npy')
preprocessed_path = os.path.join(ROOT_DIR, 'app', 'preprocessed', 'preprocessed_data.csv')

X = np.load(embedding_path)
y_df = pd.read_csv(preprocessed_path)
y = y_df.drop(columns=['Tweet']).values

X_trainval, _, y_trainval, _ = train_test_split(X, y, test_size=0.2, random_state=42)
X_trainval_tensor = torch.tensor(X_trainval, dtype=torch.float32)
y_trainval_tensor = torch.tensor(y_trainval, dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BiGRUModel(nn.Module):
    def __init__(self, units):
        super().__init__()
        self.gru = nn.GRU(768, units, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(units * 2, 9)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h = self.gru(x)
        return self.sigmoid(self.fc(torch.cat((h[0], h[1]), dim=1)))

def train_model(model, optimizer, criterion, train_loader, val_loader, epochs, fold):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(), yb.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Fold {fold}] Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(train_loader):.4f}", flush=True)

    model.eval()
    accs = []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = (model(xb.to(device)).squeeze() > 0.5).float()
            accs.append((preds == yb.to(device)).float().mean().item())
    acc = np.mean(accs)
    print(f"[Fold {fold}] Validation Accuracy: {acc:.4f}", flush=True)
    return acc

# Hyperparameter space
search_space = {
    'epochs': [20, 30, 40],
    'units': [10, 20, 30, 40, 50],
    'learning_rate': [5e-1, 1e-1, 1e-2],
    'batch_size': [128, 192, 256]
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
n_iterations = 1 #40 



#Per fold-> train acc, train loss, val.acc, val.loss
print("START_TUNING")
for i in range(2):
    params = {k: random.choice(v) for k, v in search_space.items()}
    print(f"ITERATION {i+1}/2 PARAMS: {params}", flush=True)
    val_accs = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_trainval_tensor), 1):
        print(f"  >> Starting Fold {fold}/5...", flush=True)
        train_loader = DataLoader(TensorDataset(X_trainval_tensor[train_idx], y_trainval_tensor[train_idx]), batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(TensorDataset(X_trainval_tensor[val_idx], y_trainval_tensor[val_idx]), batch_size=params['batch_size'])

        model = BiGRUModel(params['units'])
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = nn.BCELoss()

        acc = train_model(model, optimizer, criterion, train_loader, val_loader, params['epochs'], fold)
        val_accs.append(acc)
        print(f"  >> Fold {fold} completed with Accuracy: {acc:.4f}\n", flush=True)

    avg_val_acc = np.mean(val_accs)
    print(f"RESULT {i+1} VAL_ACC: {avg_val_acc:.4f}", flush=True)
    results.append({'iteration': i+1, 'params': params, 'val_acc': avg_val_acc})

# Simpan hasil
results_sorted = sorted(results, key=lambda x: x['val_acc'], reverse=True)[:5]
os.makedirs('./tuning_result', exist_ok=True)
with open('./tuning_result/best_params.json', 'w') as f:
    json.dump(results_sorted, f, indent=4)

print("DONE")
