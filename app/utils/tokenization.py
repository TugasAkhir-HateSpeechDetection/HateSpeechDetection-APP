import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

# =========================================
# Inisialisasi Model BERT Bahasa Indonesia
# =========================================

MODEL_NAME = 'cahya/bert-base-indonesian-522M'

# Load tokenizer dan model BERT Indonesia
TOKENIZER = BertTokenizer.from_pretrained(MODEL_NAME)
MODEL = BertModel.from_pretrained(MODEL_NAME)
MODEL.eval()

# Tentukan perangkat (GPU jika tersedia, jika tidak gunakan CPU)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL.to(DEVICE)

# =========================================
# Fungsi: Tokenisasi dan Preview 1 Tweet
# =========================================

def tokenize_one(preprocessed_path, embedding_path):


    # Pastikan file embedding sudah ada
    if not os.path.exists(embedding_path):
        return {
            "error": "Embedding file belum ada. Jalankan proses tokenisasi penuh terlebih dahulu."
        }

    try:
        # Baca file hasil preprocessing
        df = pd.read_csv(preprocessed_path, encoding='utf-8')
        texts = df['Tweet'].astype(str).tolist()

        # Ambil satu sampel tweet secara acak
        sample_text = random.choice(texts)

        # Tokenisasi tweet menggunakan tokenizer BERT
        encoded = TOKENIZER(
            sample_text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=50
        )

        input_ids = encoded['input_ids'].to(DEVICE)
        attention_mask = encoded['attention_mask'].to(DEVICE)

        # Forward pass ke BERT model
        with torch.no_grad():
            outputs = MODEL(input_ids=input_ids, attention_mask=attention_mask)
            embedding = outputs.last_hidden_state[0].cpu().numpy()  # [seq_len, hidden_size]

        # Konversi token dan id untuk ditampilkan
        tokens = TOKENIZER.convert_ids_to_tokens(encoded['input_ids'][0])
        token_ids = input_ids[0].cpu().numpy().tolist()
        attention = attention_mask[0].cpu().numpy().tolist()

        # Buat ringkasan hasil embedding
        return {
            "text": sample_text,
            "tokens": tokens,
            "token_ids": token_ids,
            "attention_mask": attention,
            "embedding_shape": list(embedding.shape),
            "embedding_preview": [list(map(float, embedding[i])) for i in range(50)]
        }

    except Exception as e:
        return {"error": str(e)}
