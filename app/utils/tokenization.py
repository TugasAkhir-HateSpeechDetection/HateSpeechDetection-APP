import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import random

MODEL_NAME = 'cahya/bert-base-indonesian-522M'
TOKENIZER = BertTokenizer.from_pretrained(MODEL_NAME)
MODEL = BertModel.from_pretrained(MODEL_NAME)
MODEL.eval()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL.to(DEVICE)

def tokenize_one(preprocessed_path,embedding_path):
    
    if not os.path.exists(embedding_path):
        return {"error": "Embedding file belum ada. Jalankan proses tokenisasi penuh terlebih dahulu."}
    
    try:
        df = pd.read_csv(preprocessed_path, encoding='utf-8')
        texts = df['Tweet'].astype(str).tolist()

        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        model = BertModel.from_pretrained(MODEL_NAME)
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Tokenisasi 1 sample
        sample_text = random.choice(texts)
        encoded = tokenizer(sample_text, return_tensors='pt', padding='max_length', truncation=True, max_length=50)
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embedding = outputs.last_hidden_state[0].cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        token_ids = input_ids[0].cpu().numpy().tolist()
        attention = attention_mask[0].cpu().numpy().tolist()

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