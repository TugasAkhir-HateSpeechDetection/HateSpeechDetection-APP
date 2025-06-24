import os
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm

MODEL_NAME = 'cahya/bert-base-indonesian-522M'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def tokenize_batch(preprocessed_path, output_path):
    if os.path.exists(output_path):
        print("ALREADY_EXISTS")
        return
    
    df = pd.read_csv(preprocessed_path)
    texts = df['Tweet'].astype(str).tolist()
    batch_size = 32
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing", ncols=100):
        batch_texts = texts[i:i+batch_size]
        batch = tokenizer(batch_texts, return_tensors='pt', padding='max_length', truncation=True, max_length=50)
        ids = batch['input_ids'].to(device)
        masks = batch['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids=ids, attention_mask=masks)
            all_embeddings.append(outputs.last_hidden_state.cpu().numpy())
    
    final = np.concatenate(all_embeddings, axis=0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, final)
    print("DONE")

if __name__ == '__main__':
    import sys
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    tokenize_batch(input_path, output_path)
