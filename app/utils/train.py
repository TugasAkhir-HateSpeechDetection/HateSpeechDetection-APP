# train.py
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Setup
print("Menyiapkan data dan model...\n")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

embedding_path = os.path.join(ROOT_DIR, 'app', 'embedded', 'bert_embedding.npy')
preprocessed_path = os.path.join(ROOT_DIR, 'app', 'preprocessed', 'preprocessed_data.csv')

# X = np.load('./#SKRIPSI/mini_preprocessed_data_bert_embedding.npy')
# y_df = pd.read_csv('./#SKRIPSI/mini_preprocessed_data.csv')

X = np.load(embedding_path)
y_df = pd.read_csv(preprocessed_path)
y = y_df.drop(columns=['Tweet']).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
train_loader = None

# Load best params
try:
    with open('./tuning_result/best_params.json') as f:
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
model = BiGRUModel(best_params['units']).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
criterion = nn.BCELoss()

for epoch in range(best_params['epochs']):
    model.train()
    losses, accs = [], []
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb).squeeze()
        loss = criterion(preds, yb.float())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        accs.append((preds > 0.5).float().eq(yb).float().mean().item())

    print(f"Epoch {epoch+1}/{best_params['epochs']} - Loss: {np.mean(losses):.4f} - Acc: {np.mean(accs):.4f}", flush=True)

# Save model
os.makedirs('./models', exist_ok=True)
save_path = './models/Bi-GRU.pt'
torch.save(model.state_dict(), save_path)
print(f"\nTraining selesai. Model disimpan di {save_path}", flush=True)