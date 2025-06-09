import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import json
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

embedding_path = os.path.join(ROOT_DIR, 'app', 'embedded', 'bert_embedding.npy')
preprocessed_path = os.path.join(ROOT_DIR, 'app', 'preprocessed', 'preprocessed_data.csv')

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

def evaluate_model():
    X_test = np.load(embedding_path)
    y_df = pd.read_csv(preprocessed_path)
    y_test = y_df.drop(columns=['Tweet']).values

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X_test, y_test, test_size=0.2, random_state=42)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Load best params
    try:
        with open('./tuning_result/best_params.json') as f:
            data = json.load(f)
            best_item = max(data, key=lambda x: x['val_acc'])
            best_params = best_item['params']
    except Exception as e:
        print("Gagal membaca best_params.json:", e)
        exit(1)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'Bi-GRU.pt')

    model = BiGRUModel(best_params['units'])
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=best_params['batch_size'])
    criterion = nn.BCELoss()
    
    all_preds = []
    test_losses, test_accs = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb).squeeze()
            loss = criterion(preds, yb.float())
            test_losses.append(loss.item())

            binary_preds = (preds > 0.5).float()
            acc = (binary_preds == yb).float().mean().item()
            test_accs.append(acc)
            all_preds.extend(binary_preds.cpu().numpy())

    y_pred = np.array(all_preds)
    y_true = y_test_tensor.numpy()

    report = classification_report(
        y_true, y_pred,
        target_names=[
            'HS', 'Abusive', 'HS_Individual', 'HS_Group', 'HS_Religion',
            'HS_Race', 'HS_Physical', 'HS_Gender', 'HS_Other'
        ],
        zero_division=0,
        output_dict=True
    )

    return {
        'loss': np.mean(test_losses),
        'accuracy': np.mean(test_accs),
        'report': report
    }
