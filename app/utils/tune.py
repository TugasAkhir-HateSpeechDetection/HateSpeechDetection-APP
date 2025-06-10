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
        super(BiGRUModel, self).__init__()
        self.gru = nn.GRU(input_size=768, hidden_size=units, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(units * 2, 9)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h = self.gru(x)
        h_concat = torch.cat((h[0], h[1]), dim=1)
        out = self.fc(h_concat)
        return self.sigmoid(out)

def train_model(model, optimizer, criterion, train_loader, val_loader, epochs, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb.float())
            loss.backward()
            optimizer.step()
    
    # Evaluation
    def evaluate(loader):
        model.eval()
        losses, accs = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).squeeze()
                loss = criterion(preds, yb.float())
                preds_binary = (preds > 0.5).float()
                acc = (preds_binary == yb).float().mean().item()
                losses.append(loss.item())
                accs.append(acc)
        return np.mean(losses), np.mean(accs)

    train_loss, train_acc = evaluate(train_loader)
    val_loss, val_acc = evaluate(val_loader)
    
    return train_loss, train_acc, val_loss, val_acc

# Hyperparameter space
search_space = {
    'epochs': [20, 30, 40],
    'units': [10, 20, 30, 40, 50],
    'learning_rate': [5e-1, 1e-1, 1e-2],
    'batch_size': [128, 192, 256]
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
n_iterations = 40 

#Per fold-> train acc, train loss, val.acc, val.loss
print("START_TUNING")

for i in range(n_iterations):
    params = {
        'epochs': random.choice(search_space['epochs']),
        'units': random.choice(search_space['units']),
        'learning_rate': random.choice(search_space['learning_rate']),
        'batch_size': random.choice(search_space['batch_size'])
    }

    fold_train_losses, fold_val_losses = [], []
    fold_train_accs, fold_val_accs = [], []

    print(f"\nIterasi {i+1}/{n_iterations} - Params: {params}", flush=True)

    for fold, (train_index, val_index) in enumerate(kf.split(X_trainval_tensor), 1):
        X_train_fold = X_trainval_tensor[train_index]
        y_train_fold = y_trainval_tensor[train_index]
        X_val_fold = X_trainval_tensor[val_index]
        y_val_fold = y_trainval_tensor[val_index]

        train_loader = DataLoader(TensorDataset(X_train_fold, y_train_fold), batch_size=params['batch_size'], shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_fold, y_val_fold), batch_size=params['batch_size'])

        model = BiGRUModel(params['units'])
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = nn.BCELoss()

        train_loss, train_acc, val_loss, val_acc = train_model(
            model, optimizer, criterion, train_loader, val_loader, params['epochs'], device)

        print(f"  Fold {fold} >> Train Acc: {train_acc:.4f} | Train Loss: {train_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}", flush=True)

        fold_train_losses.append(train_loss)
        fold_train_accs.append(train_acc)
        fold_val_losses.append(val_loss)
        fold_val_accs.append(val_acc)

    avg_train_loss = np.mean(fold_train_losses)
    avg_train_acc = np.mean(fold_train_accs)
    avg_val_loss = np.mean(fold_val_losses)
    avg_val_acc = np.mean(fold_val_accs)

    print(f"  >> Avg Train Acc: {avg_train_acc:.4f} | Avg Train Loss: {avg_train_loss:.4f} | "
          f"Avg Val Acc: {avg_val_acc:.4f} | Avg Val Loss: {avg_val_loss:.4f}", flush=True)

    results.append({
        'iteration': i+1,
        'params': params,
        'train_acc': avg_train_acc,
        'train_loss': avg_train_loss,
        'val_acc': avg_val_acc,
        'val_loss': avg_val_loss
    })

# Simpan hasil
results_sorted = sorted(results, key=lambda x: x['val_acc'], reverse=True)[:5]
os.makedirs('./tuning_result', exist_ok=True)
with open('./tuning_result/best_params.json', 'w') as f:
    json.dump(results_sorted, f, indent=4)

print("DONE")
