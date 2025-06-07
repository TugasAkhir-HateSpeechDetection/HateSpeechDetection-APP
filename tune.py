# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split, KFold
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# import random
# import os
# import json


# # Load data
# X = np.load('embedded/bert_embedding.npy')
# y_df = pd.read_csv('preprocessed/preprocessed_data.csv')
# # X = np.load('./#SKRIPSI/mini_preprocessed_data_bert_embedding.npy')
# # y_df = pd.read_csv('./#SKRIPSI/mini_preprocessed_data.csv')
# y = y_df.drop(columns=['Tweet']).values

# X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# class BiGRUModel(nn.Module):
#     def __init__(self, units):
#         super(BiGRUModel, self).__init__()
#         self.gru = nn.GRU(input_size=768, hidden_size=units, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(units * 2, 9)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         _, h = self.gru(x)
#         h_concat = torch.cat((h[0], h[1]), dim=1)
#         out = self.fc(h_concat)
#         return self.sigmoid(out)

# def run_tuning():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     X_trainval_tensor = torch.tensor(X_trainval, dtype=torch.float32)
#     y_trainval_tensor = torch.tensor(y_trainval, dtype=torch.float32)

#     search_space = {
#         'epochs': [20, 30, 40],
#         'units': [10, 20, 30, 40, 50],
#         'learning_rate': [5e-1, 1e-1, 1e-2],
#         'batch_size': [128, 192, 256]
#     }

#     n_iterations = 2  # bisa diperbesar jika diperlukan
#     results = []
#     kf = KFold(n_splits=3, shuffle=True, random_state=42)

#     for i in range(n_iterations):
#         params = {
#             'epochs': random.choice(search_space['epochs']),
#             'units': random.choice(search_space['units']),
#             'learning_rate': random.choice(search_space['learning_rate']),
#             'batch_size': random.choice(search_space['batch_size'])
#         }

#         fold_train_accs, fold_val_accs = [], []

#         for train_index, val_index in kf.split(X_trainval_tensor):
#             X_train_fold = X_trainval_tensor[train_index]
#             y_train_fold = y_trainval_tensor[train_index]
#             X_val_fold = X_trainval_tensor[val_index]
#             y_val_fold = y_trainval_tensor[val_index]

#             train_loader = DataLoader(TensorDataset(X_train_fold, y_train_fold), batch_size=params['batch_size'], shuffle=True)
#             val_loader = DataLoader(TensorDataset(X_val_fold, y_val_fold), batch_size=params['batch_size'])

#             model = BiGRUModel(params['units']).to(device)
#             optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
#             criterion = nn.BCELoss()

#             # Train
#             for epoch in range(params['epochs']):
#                 model.train()
#                 for xb, yb in train_loader:
#                     xb, yb = xb.to(device), yb.to(device)
#                     optimizer.zero_grad()
#                     preds = model(xb).squeeze()
#                     loss = criterion(preds, yb.float())
#                     loss.backward()
#                     optimizer.step()

#             # Eval
#             def evaluate(loader):
#                 model.eval()
#                 accs = []
#                 with torch.no_grad():
#                     for xb, yb in loader:
#                         xb, yb = xb.to(device), yb.to(device)
#                         preds = model(xb).squeeze()
#                         preds_binary = (preds > 0.5).float()
#                         acc = (preds_binary == yb).float().mean().item()
#                         accs.append(acc)
#                 return np.mean(accs)

#             train_acc = evaluate(train_loader)
#             val_acc = evaluate(val_loader)

#             fold_train_accs.append(train_acc)
#             fold_val_accs.append(val_acc)

#         results.append({
#             'iteration': i + 1,
#             'params': params,
#             'train_acc': round(np.mean(fold_train_accs), 4),
#             'val_acc': round(np.mean(fold_val_accs), 4)
#         })

#     return results


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
import os
import json

print("🚀 [INFO] Memulai tune.py...")

# Load data
print("📥 [INFO] Memuat embedding dan label...")
X = np.load('embedded/bert_embedding.npy')
y_df = pd.read_csv('preprocessed/preprocessed_data.csv')
y = y_df.drop(columns=['Tweet']).values

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print("✅ [INFO] Data berhasil dimuat. Ukuran X:", X.shape)

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

def run_tuning():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ [INFO] Menggunakan device: {device}")

    X_trainval_tensor = torch.tensor(X_trainval, dtype=torch.float32)
    y_trainval_tensor = torch.tensor(y_trainval, dtype=torch.float32)

    search_space = {
        'epochs': [20, 30, 40],
        'units': [10, 20, 30, 40, 50],
        'learning_rate': [5e-1, 1e-1, 1e-2],
        'batch_size': [128, 192, 256]
    }

    n_iterations = 2  # bisa diperbesar jika diperlukan
    results = []
    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    for i in range(n_iterations):
        print(f"🎯 [INFO] Iterasi tuning ke-{i+1}")
        params = {
            'epochs': random.choice(search_space['epochs']),
            'units': random.choice(search_space['units']),
            'learning_rate': random.choice(search_space['learning_rate']),
            'batch_size': random.choice(search_space['batch_size'])
        }
        print(f"⚙️ [PARAMS] {params}")

        fold_train_accs, fold_val_accs = [], []

        for fold, (train_index, val_index) in enumerate(kf.split(X_trainval_tensor), 1):
            print(f"📁 [FOLD {fold}] Mulai training...")

            X_train_fold = X_trainval_tensor[train_index]
            y_train_fold = y_trainval_tensor[train_index]
            X_val_fold = X_trainval_tensor[val_index]
            y_val_fold = y_trainval_tensor[val_index]

            train_loader = DataLoader(TensorDataset(X_train_fold, y_train_fold), batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(TensorDataset(X_val_fold, y_val_fold), batch_size=params['batch_size'])

            model = BiGRUModel(params['units']).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
            criterion = nn.BCELoss()

            for epoch in range(params['epochs']):
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
                accs = []
                with torch.no_grad():
                    for xb, yb in loader:
                        xb, yb = xb.to(device), yb.to(device)
                        preds = model(xb).squeeze()
                        preds_binary = (preds > 0.5).float()
                        acc = (preds_binary == yb).float().mean().item()
                        accs.append(acc)
                return np.mean(accs)

            train_acc = evaluate(train_loader)
            val_acc = evaluate(val_loader)

            print(f"📈 [FOLD {fold}] Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

            fold_train_accs.append(train_acc)
            fold_val_accs.append(val_acc)

        mean_train_acc = round(np.mean(fold_train_accs), 4)
        mean_val_acc = round(np.mean(fold_val_accs), 4)
        print(f"✅ [ITERASI {i+1}] Val Accuracy Rata-rata: {mean_val_acc}")

        results.append({
            'iteration': i + 1,
            'params': params,
            'train_acc': mean_train_acc,
            'val_acc': mean_val_acc
        })

    # Simpan hasil ke file
    save_directory = "./tuneresult/"
    file_name = "Bi-GRU.pt"
    os.makedirs(save_directory, exist_ok=True)

    save_path = os.path.join(save_directory, file_name)
    output_path = os.path.join(save_directory, "tuning_results.json")

    with open(output_path, 'w') as f:
        json.dump({"results": results}, f, indent=4)

    print(f"💾 [SAVED] Hasil tuning disimpan ke {output_path}")
    print("🎉 [DONE] Semua iterasi selesai.")
    return results


# Panggil langsung jika run sebagai script
if __name__ == "__main__":
    run_tuning()

