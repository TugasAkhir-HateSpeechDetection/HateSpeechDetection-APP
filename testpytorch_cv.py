import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import random
import os

# Load data
# X = np.load('./#SKRIPSI/bert_embedding.npy')
# y_df = pd.read_csv('./#SKRIPSI/preprocessed_data.csv')
X = np.load('./#SKRIPSI/mini_preprocessed_data_bert_embedding.npy')
y_df = pd.read_csv('./#SKRIPSI/mini_preprocessed_data.csv')
y = y_df.drop(columns=['Tweet']).values

# Split data into training and test sets
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

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
                loss = criterion(preds, yb.float()  )
                preds_binary = (preds > 0.5).float()
                acc = (preds_binary == yb).float().mean().item()
                losses.append(loss.item())
                accs.append(acc)
        return np.mean(losses), np.mean(accs)

    train_loss, train_acc = evaluate(train_loader)
    val_loss, val_acc = evaluate(val_loader)
    
    return train_loss, train_acc, val_loss, val_acc

# Convert training data to tensors
X_trainval_tensor = torch.tensor(X_trainval, dtype=torch.float32)
y_trainval_tensor = torch.tensor(y_trainval, dtype=torch.float32)

# Hyperparameter space
search_space = {
    'epochs': [20, 30, 40],
    'units': [10, 20, 30, 40, 50],
    'learning_rate': [5e-1, 1e-1, 1e-2],
    'batch_size': [128, 192, 256]
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_iterations = 40
results = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Mulai Random Search Tuning...\n")

for i in range(n_iterations):
    params = {
        'epochs': random.choice(search_space['epochs']),
        'units': random.choice(search_space['units']),
        'learning_rate': random.choice(search_space['learning_rate']),
        'batch_size': random.choice(search_space['batch_size'])
    }

    fold_train_losses, fold_val_losses = [], []
    fold_train_accs, fold_val_accs = [], []

    print(f"\nIterasi {i+1}/{n_iterations} - Params: {params}")

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
              f"Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")

        fold_train_losses.append(train_loss)
        fold_train_accs.append(train_acc)
        fold_val_losses.append(val_loss)
        fold_val_accs.append(val_acc)

    avg_train_loss = np.mean(fold_train_losses)
    avg_train_acc = np.mean(fold_train_accs)
    avg_val_loss = np.mean(fold_val_losses)
    avg_val_acc = np.mean(fold_val_accs)

    print(f"  >> Avg Train Acc: {avg_train_acc:.4f} | Avg Train Loss: {avg_train_loss:.4f} | "
          f"Avg Val Acc: {avg_val_acc:.4f} | Avg Val Loss: {avg_val_loss:.4f}")

    results.append({
        'iteration': i+1,
        'params': params,
        'train_acc': avg_train_acc,
        'train_loss': avg_train_loss,
        'val_acc': avg_val_acc,
        'val_loss': avg_val_loss
    })

# Sorted best hyperparameters
results_sorted = sorted(results, key=lambda x: x['val_acc'], reverse=True)
print("\n5 Kombinasi Hyperparameter Terbaik berdasarkan Val Acc:")
for r in results_sorted[:5]:
    print(f"Iterasi {r['iteration']} - Val Acc: {r['val_acc']:.4f} - Params: {r['params']}")

# Retrain on full training set
best_params = results_sorted[0]['params']
print(f"\nMelatih ulang model dengan hyperparameter terbaik: {best_params}\n")
train_loader = DataLoader(TensorDataset(X_trainval_tensor, y_trainval_tensor), batch_size=best_params['batch_size'], shuffle=True)
model = BiGRUModel(best_params['units'])
optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
criterion = nn.BCELoss()

model.to(device)
for epoch in range(best_params['epochs']):
    model.train()
    epoch_losses = []
    epoch_accuracies = []
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
        epoch_accuracies.append(acc)

    avg_loss = np.mean(epoch_losses)
    avg_acc = np.mean(epoch_accuracies)
    print(f"Epoch {epoch+1}/{best_params['epochs']} - Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.4f}")

# Convert test data to tensors for Evaluation
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=best_params['batch_size'])

# Evaluation
print("\nMengevaluasi model yang sudah dilatih...\n")
model.eval()
test_losses = []
test_accs = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        preds = model(xb).squeeze()
        loss = criterion(preds, yb.float())
        test_losses.append(loss.item())
        preds_binary = (preds > 0.5).float()
        acc = (preds_binary == yb).float().mean().item()
        test_accs.append(acc)
mean_test_loss = np.mean(test_losses)
mean_test_acc = np.mean(test_accs)
print(f"Test Loss: {mean_test_loss:.4f}")
print(f"Test Accuracy: {mean_test_acc:.4f}")

save_directory = "./#SKRIPSI/"
file_name = "Bi-GRU.pt"
os.makedirs(save_directory, exist_ok=True)
save_path = os.path.join(save_directory, file_name)

# Ensure unique filename
counter = 0
current_file_path = save_path
while os.path.exists(current_file_path):
    counter += 1
    name, ext = os.path.splitext(file_name)
    current_file_path = os.path.join(save_directory, f"{name}({counter}){ext}")

torch.save(model.state_dict(), current_file_path)
print(f"Model saved to: {current_file_path}")