import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ============================================
# Pengaturan SEED
# ============================================

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

# ============================================
# Path dan Data Setup
# ============================================

print("[INFO] Menyiapkan data dan model...", flush=True)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
embedding_path = os.path.join(ROOT_DIR, 'app', 'embedded', 'bert_embedding.npy')
lengths_path = os.path.join(ROOT_DIR, 'app', 'embedded', 'bert_lengths.npy')
preprocessed_path = os.path.join(ROOT_DIR, 'app', 'preprocessed', 'preprocessed_data.csv')
eval_dir = os.path.join(ROOT_DIR, 'app', 'evaluation')
os.makedirs(eval_dir, exist_ok=True)

# Load data
X = np.load(embedding_path)
lengths = np.load(lengths_path)
y_df = pd.read_csv(preprocessed_path)
y = y_df.drop(columns=['Tweet']).values

# Split data untuk training dan testing (80/20)
X_train, X_test, lengths_train, lengths_test, y_train, y_test = train_test_split(
    X, lengths, y, test_size=0.2, random_state=SEED if USE_SEED else None
)

# Konversi ke tensor
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
lengths_tensor = torch.tensor(lengths_train, dtype=torch.int64)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# ============================================
# Load Hyperparameter Terbaik dari Tuning
# ============================================

try:
    with open(os.path.join(ROOT_DIR, 'app', 'tuning_result', 'best_params.json')) as f:
        data = json.load(f)
        best_item = max(data, key=lambda x: x['val_acc'])
        best_params = best_item['params']
except Exception as e:
    print("Gagal membaca best_params.json:", e, flush=True)
    exit(1)

print(f"\nMulai training dengan hyperparameter: {best_params}", flush=True)

# ============================================
# Arsitektur Model Bi-GRU
# ============================================

class BiGRUModel(nn.Module):
    def __init__(self, units):
        super(BiGRUModel, self).__init__()
        self.gru = nn.GRU(input_size=768, hidden_size=units, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(units * 2, 9)  # 9 label output multi-label
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        # Packing sequence untuk efisiensi RNN
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.gru(packed)
        h_concat = torch.cat((h[0], h[1]), dim=1)  # Gabung hidden state dari kedua arah
        out = self.fc(h_concat)
        return self.sigmoid(out)

# ============================================
# Inisialisasi Training
# ============================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = DataLoader(
    TensorDataset(X_train_tensor, lengths_tensor, y_train_tensor),
    batch_size=best_params['batch_size'],
    shuffle=True,
    worker_init_fn=seed_worker if USE_SEED else None,
    generator=g if USE_SEED else None
)

model = BiGRUModel(best_params['units']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
criterion = nn.BCELoss()

train_losses, train_accs = [], []

# ============================================
# Proses Training
# ============================================

for epoch in range(best_params['epochs']):
    model.train()
    epoch_train_loss, epoch_train_acc = [], []

    for xb, lengths, yb in train_loader:
        xb, lengths, yb = xb.to(device), lengths.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb, lengths).squeeze()
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

        # Hitung akurasi per batch
        preds_binary = (preds > 0.5).float()
        acc = (preds_binary == yb).float().mean().item()

        epoch_train_loss.append(loss.item())
        epoch_train_acc.append(acc)

    avg_train_loss = np.mean(epoch_train_loss)
    avg_train_acc = np.mean(epoch_train_acc)

    train_losses.append(avg_train_loss)
    train_accs.append(avg_train_acc)

    print(f"Epoch {epoch+1}/{best_params['epochs']} - "
          f"Train Acc: {avg_train_acc:.4f} | Train Loss: {avg_train_loss:.4f}", flush=True)

# ============================================
# Simpan Model Terlatih
# ============================================

model_path = os.path.join(ROOT_DIR, 'app', 'models', 'Bi-GRU.pt')
os.makedirs(os.path.dirname(model_path), exist_ok=True)

torch.save({
    'model_state_dict': model.state_dict(),
    'units': best_params['units']
}, model_path)

print(f"\n[INFO] Training selesai. Model berhasil disimpan", flush=True)

# ============================================
# Simpan Plot Akurasi dan Loss
# ============================================

plot_path = os.path.join(eval_dir, 'training_plot.png')
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(train_accs, label='Train Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train Accuracy per Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses, label='Train Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss per Epoch')
plt.legend()

plt.tight_layout()
plt.savefig(plot_path)
