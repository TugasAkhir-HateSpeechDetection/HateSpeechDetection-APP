import os
import sys
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

# ===============================================
# Inisialisasi Model BERT Bahasa Indonesia
# ===============================================

MODEL_NAME = 'cahya/bert-base-indonesian-522M'

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
model.eval()

# Tentukan perangkat (GPU jika tersedia, jika tidak gunakan CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# ===============================================
# Fungsi: Tokenisasi Batch untuk Semua Tweet
# ===============================================

def tokenize_batch(preprocessed_path, output_embed_path, output_len_path):


    # Cegah pengulangan proses jika file sudah ada
    if os.path.exists(output_embed_path) and os.path.exists(output_len_path):
        print("ALREADY_EXISTS")
        return

    # Load dataset hasil preprocessing
    df = pd.read_csv(preprocessed_path)
    texts = df['Tweet'].astype(str).tolist()

    batch_size = 32
    all_embeddings = []  # Menyimpan semua hasil embedding
    all_lengths = []     # Menyimpan panjang valid token per input

    # Iterasi secara batch
    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing", ncols=100):
        batch_texts = texts[i:i+batch_size]

        # Tokenisasi batch (maks 50 token, padding otomatis)
        encoded = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=50
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        # Forward pass untuk mendapatkan embedding
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state.cpu().numpy()
            lengths = attention_mask.sum(dim=1).cpu().numpy()  # jumlah token bukan padding

        all_embeddings.append(last_hidden)
        all_lengths.append(lengths)

    # Gabungkan hasil semua batch
    final_embeddings = np.concatenate(all_embeddings, axis=0)  # shape: [total_sample, seq_len, hidden]
    final_lengths = np.concatenate(all_lengths, axis=0)        # shape: [total_sample]

    # direktori output
    os.makedirs(os.path.dirname(output_embed_path), exist_ok=True)

    # Simpan ke file .npy
    np.save(output_embed_path, final_embeddings)
    np.save(output_len_path, final_lengths)

    print("DONE")
    
    
if __name__ == '__main__':
    input_path = sys.argv[1]
    output_embed = sys.argv[2]
    output_len = sys.argv[3]
    tokenize_batch(input_path, output_embed, output_len)
