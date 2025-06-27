import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold

# === Pengaturan SEED ===
USE_SEED = True
SEED = 42

if USE_SEED:
    def set_seed(seed=SEED):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    set_seed(SEED)

    def seed_worker(worker_id):
        worker_seed = SEED + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(SEED)
    print(f"[INFO] SEED diaktifkan dengan nilai {SEED}", flush=True)
else:
    seed_worker = None
    g = None
    print("[INFO] SEED tidak digunakan (hasil tidak reproducible)", flush=True)

# === Load Data ===
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
embedding_path = os.path.join(ROOT_DIR, 'app', 'embedded', 'bert_embedding.npy')
lengths_path = os.path.join(ROOT_DIR, 'app', 'embedded', 'bert_lengths.npy')
preprocessed_path = os.path.join(ROOT_DIR, 'app', 'preprocessed', 'preprocessed_data.csv')

X = np.load(embedding_path)
lengths = np.load(lengths_path)
y_df = pd.read_csv(preprocessed_path)
y = y_df.drop(columns=['Tweet']).values

X_train, _, lengths_train, _, y_train, _ = train_test_split(
    X, lengths, y, test_size=0.2, random_state=SEED if USE_SEED else None
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
lengths_train_tensor = torch.tensor(lengths_train, dtype=torch.int64)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Model ===
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

# === Training Function ===
def train_model(model, optimizer, criterion, train_loader, val_loader, epochs, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        for xb, lb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb.float())
            loss.backward()
            optimizer.step()

    def evaluate(loader):
        model.eval()
        losses, accs = [], []
        with torch.no_grad():
            for xb, lb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).squeeze()
                loss = criterion(preds, yb.float())
                preds_binary = (preds > 0.5).float()
                acc = (preds_binary == yb).float().mean().item()
                losses.append(loss.item())
                accs.append(acc)
        return np.mean(losses), np.mean(accs)

    return evaluate(train_loader) + evaluate(val_loader)

# === Hyperparameter Search Space ===
search_space = {
    'epochs': [20, 30, 40],
    'units': [10, 20, 30, 40, 50],
    'learning_rate': [5e-1, 1e-1, 1e-2],
    'batch_size': [128, 192, 256]
}

# === Generate All Possible Unique Combinations ===
from itertools import product

all_combinations = list(product(
    search_space['epochs'],
    search_space['units'],
    search_space['learning_rate'],
    search_space['batch_size']
))
random.shuffle(all_combinations)  # acak urutan kombinasi
n_iterations = min(5, len(all_combinations))

kf = KFold(n_splits=5, shuffle=True, random_state=SEED if USE_SEED else None)
results = []

print("[INFO] START_TUNING", flush=True)

# === Tuning Loop ===
for i in range(n_iterations):
    epoch, unit, lr, batch = all_combinations[i]
    params = {
        'epochs': epoch,
        'units': unit,
        'learning_rate': lr,
        'batch_size': batch
    }

    fold_train_losses, fold_val_losses = [], []
    fold_train_accs, fold_val_accs = [], []

    print(f"\nIterasi {i+1}/{n_iterations} - Params: {params}", flush=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_tensor), 1):
        X_train_fold = X_train_tensor[train_idx]
        y_train_fold = y_train_tensor[train_idx]

        X_val_fold = X_train_tensor[val_idx]
        y_val_fold = y_train_tensor[val_idx]

        train_loader = DataLoader(
            TensorDataset(X_train_fold, torch.zeros_like(y_train_fold), y_train_fold),
            batch_size=params['batch_size'],
            shuffle=True,
            worker_init_fn=seed_worker if USE_SEED else None,
            generator=g if USE_SEED else None
        )

        val_loader = DataLoader(
            TensorDataset(X_val_fold, torch.zeros_like(y_val_fold), y_val_fold),
            batch_size=params['batch_size'],
            shuffle=False,
            worker_init_fn=seed_worker if USE_SEED else None,
            generator=g if USE_SEED else None
        )

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
        'iteration': i + 1,
        'params': params,
        'train_acc': avg_train_acc,
        'train_loss': avg_train_loss,
        'val_acc': avg_val_acc,
        'val_loss': avg_val_loss
    })

# === Simpan Hasil Tuning ===
results_sorted = sorted(results, key=lambda x: x['val_acc'], reverse=True)
os.makedirs('./tuning_result', exist_ok=True)
with open('./tuning_result/best_params.json', 'w') as f:
    json.dump(results_sorted, f, indent=4)

print("[INFO] DONE", flush=True)