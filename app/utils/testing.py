import os
import re
import json
import torch
import pandas as pd
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from transformers import BertTokenizer, BertModel
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
KAMUS_PATH = os.path.join(ROOT_DIR, 'app', 'kamus', 'new_kamusalay.csv')
MODEL_PATH = os.path.join(ROOT_DIR, 'app', 'models', 'Bi-GRU.pt')
MODEL_NAME = "cahya/bert-base-indonesian-522M"

LABELS = [
    "HS", "Abusive", "HS_Individual", "HS_Group", "HS_Religion",
    "HS_Race", "HS_Physical", "HS_Gender", "HS_Other"
]

# ============================================
# Preprocessing: Kamus alay dan stopword
# ============================================

kamus_df = pd.read_csv(KAMUS_PATH, encoding='latin1', header=None, names=['slang', 'formal'])
kamus_dict = dict(zip(kamus_df['slang'], kamus_df['formal']))

stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()

# ============================================
# Tokenizer dan Model BERT
# ============================================

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
bert_model = BertModel.from_pretrained(MODEL_NAME)

# ============================================
# Arsitektur BiGRU
# ============================================

class BiGRUModel(nn.Module):
    def __init__(self, units):
        super(BiGRUModel, self).__init__()
        self.gru = nn.GRU(input_size=768, hidden_size=units, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(units * 2, len(LABELS))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths=None):
        if lengths is not None:
            lengths = torch.clamp(lengths, max=x.size(1))
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, h = self.gru(packed)
        else:
            _, h = self.gru(x)
        h_concat = torch.cat((h[0], h[1]), dim=1)
        return self.sigmoid(self.fc(h_concat))

# ============================================
# Fungsi Preprocessing Text
# ============================================

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

# ============================================
# Fungsi Embedding dengan BERT
# ============================================

def get_bert_embedding(text):
    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        output = bert_model(**tokens)
    bert_embed = output.last_hidden_state
    lengths = tokens['attention_mask'].sum(dim=1)
    return bert_embed, lengths

# ============================================
# Load Model Terlatih
# ============================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            units = checkpoint.get('units', 50)
            model_instance = BiGRUModel(units)
            model_instance.load_state_dict(checkpoint['model_state_dict'])
            model_instance.to(device)
            model_instance.eval()
            model = model_instance
            print(f"[INFO] Model berhasil dimuat dari {MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] Gagal memuat model: {e}")
            model = None
    else:
        print(f"[ERROR] File model tidak ditemukan di: {MODEL_PATH}")
        model = None

# ============================================
# Fungsi Prediksi untuk Tweet Baru
# ============================================

def predict_tweet(text):
    if model is None:
        return [{"error": "Model belum tersedia. Silakan latih atau muat model terlebih dahulu."}]
    
    clean_text = preprocess_text(text)
    bert_embed, lengths = get_bert_embedding(clean_text)
    bert_embed = bert_embed.to(device)
    lengths = lengths.to(device)

    with torch.no_grad():
        output = model(bert_embed, lengths)
    
    probabilities = output.cpu().numpy()[0]
    predictions = (probabilities >= 0.5).astype(int)

    return [
        {"label": label, "probability": float(prob), "prediction": int(pred)}
        for label, prob, pred in zip(LABELS, probabilities, predictions)
    ]
