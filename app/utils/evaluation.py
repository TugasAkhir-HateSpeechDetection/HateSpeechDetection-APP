import os
import torch
import json
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, precision_score, recall_score

class BiGRUModel(nn.Module):
    def __init__(self, units):
        super(BiGRUModel, self).__init__()
        self.gru = nn.GRU(input_size=768, hidden_size=units, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(units * 2, 9)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, h = self.gru(x)
        h_concat = torch.cat((h[0], h[1]), dim=1)
        return self.sigmoid(self.fc(h_concat))

def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data dan model
    with open('./tuning_result/best_params.json') as f:
        best_item = max(json.load(f), key=lambda x: x['val_acc'])
    units = best_item['params']['units']
    batch_size = best_item['params']['batch_size']

    checkpoint = torch.load('./models/Bi-GRU.pt', map_location=device)
    model = BiGRUModel(units)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # X = np.load('./dataset/bert_embedding.npy')
    # y_df = pd.read_csv('./dataset/preprocessed_data.csv')
    
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    embedding_path = os.path.join(ROOT_DIR, 'app', 'embedded', 'bert_embedding.npy')
    preprocessed_path = os.path.join(ROOT_DIR, 'app', 'preprocessed', 'preprocessed_data.csv')

    X = np.load(embedding_path)
    y_df = pd.read_csv(preprocessed_path)
    
    y = y_df.drop(columns=['Tweet']).values
    label_names = y_df.drop(columns=['Tweet']).columns.tolist()

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)),batch_size=batch_size)
    
    # Predict
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(device)).cpu()
            all_preds.append((preds > 0.5).float())
            all_targets.append(yb)
    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()
    
    test_losses, test_accs = [], []
    loss_fn = nn.BCELoss()

    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        with torch.no_grad():
            pred = model(xb)
            loss = loss_fn(pred, yb)
            acc = ((pred > 0.5).float() == yb).float().mean()
            test_losses.append(loss.item())
            test_accs.append(acc.item())

    mean_loss = np.mean(test_losses)
    mean_acc = np.mean(test_accs)
    

    # Save classification report
    recalls = recall_score(y_true, y_pred, average=None, zero_division=0)
    precisions = precision_score(y_true, y_pred, average=None, zero_division=0)
    cms = multilabel_confusion_matrix(y_true, y_pred)
    accuracies = [(cm[0, 0] + cm[1, 1]) / cm.sum() for cm in cms]

    report = []
    for i in range(len(label_names)):
        report.append({
            "label": label_names[i],
            "accuracy": round(accuracies[i], 4),
            "recall": round(recalls[i], 4),
            "precision": round(precisions[i], 4),
        })

    report.append({
        "label": "average",
        "accuracy": round(np.mean(accuracies), 4),
        "recall": round(np.mean(recalls), 4),
        "precision": round(np.mean(precisions), 4),
        "mean_accuracy": round(mean_acc, 4),
        "mean_loss": round(mean_loss, 4)
    })

    os.makedirs('./evaluation', exist_ok=True)
    with open('./evaluation/classification_report.json', 'w') as f:
        json.dump(report, f)

    # Save confusion matrix
    rows, cols = math.ceil(len(label_names) / 3), 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    for i, label in enumerate(label_names):
        cm = cms[i][::-1, ::-1]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Pos', 'Neg'], yticklabels=['Pos', 'Neg'], ax=axes[i])
        axes[i].set_title(label)
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')

    for i in range(len(label_names), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('./evaluation/confusion_matrix.png')
    plt.close()