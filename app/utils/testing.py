import re
import torch
import os
import pandas as pd
import json
from torch import nn
from transformers import BertTokenizer, BertModel
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Load kamus alay
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
csv_path = os.path.join(ROOT_DIR, 'app', 'kamus', 'new_kamusalay.csv')

kamus_df = pd.read_csv(csv_path, encoding='latin1', header=None, names=['slang', 'formal'])
kamus_dict = dict(zip(kamus_df['slang'], kamus_df['formal']))

# Stopword remover
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

# Load BERT tokenizer & model
model_name = "cahya/bert-base-indonesian-522M"
tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)

# Load best params
try:
    with open('./tuning_result/best_params.json') as f:
        data = json.load(f)
        best_item = max(data, key=lambda x: x['val_acc'])
        best_params = best_item['params']
except Exception as e:
    print("Gagal membaca best_params.json:", e)
    exit(1)

#import dari class training, units->best params
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

def standarize_text(text):
    words = text.split()
    return " ".join([kamus_dict.get(w, w) for w in words])

def preprocess_text(text):
    if pd.isnull(text):
        return ''
    text = re.sub(r'^RT\s+', '', text)
    text = re.sub(r'\bUSER\b', '', text)
    text = text.lower()
    text = re.sub(r'(\\x[0-9a-fA-F]{2})+', '', text)
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'[!"$%&\'()*+,\-./:;<=>?@\[\\\]^{|}~_#]', ' ', text)
    text = standarize_text(text)
    text = stopword_remover.remove(text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_bert_embedding(text, max_len=40):
    encoded = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_len)
    with torch.no_grad():
        output = bert_model(**encoded)
    return output.last_hidden_state

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'Bi-GRU.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiGRUModel(best_params['units']).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

label_names = [
    "HS", "Abusive", "HS_Individual", "HS_Group", "HS_Religion",
    "HS_Race", "HS_Physical", "HS_Gender", "HS_Other"
]

def predict_tweet(text):
    clean_text = preprocess_text(text)
    bert_embed = get_bert_embedding(clean_text).to(device)
    with torch.no_grad():
        output = model(bert_embed)
    probabilities = output.cpu().numpy()[0]
    binary = (probabilities >= 0.5).astype(int)
    results = []
    for label, prob, pred in zip(label_names, probabilities, binary):
        results.append({"label": label, "probability": float(prob), "prediction": int(pred)})
    return results
