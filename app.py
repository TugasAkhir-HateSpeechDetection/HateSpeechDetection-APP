from flask import Flask, request, jsonify, send_from_directory, render_template,redirect, url_for,Response
import pandas as pd
import re, os
import numpy as np
import tensorflow as tf
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from transformers import BertTokenizer
from tqdm import tqdm
import subprocess
from tune import run_tuning
import torch
from transformers import BertTokenizer, BertModel
import json
import torch.nn as nn
# from tune_dummy import run_tuning


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PREPROCESSED_FOLDER = 'preprocessed'
TOKENIZED_FOLDER = 'tokenized'
EMBEDDED_FOLDER = 'embedded'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREPROCESSED_FOLDER'] = PREPROCESSED_FOLDER

# ======== PREPROCESSING FUNCTION ========= #
factory = StopWordRemoverFactory()
stopword_remover = factory.create_stop_word_remover()

kamus_df = pd.read_csv("new_kamusalay.csv", encoding='latin1', header=None, names=['slang', 'formal'])
kamus_dict = dict(zip(kamus_df['slang'], kamus_df['formal']))

# === Fungsi Standarisasi ===
def standarize_text(text):
    words = text.split()
    new_words = [kamus_dict.get(word, word) for word in words]
    return ' '.join(new_words)

# === Fungsi Preprocessing ===
def preprocess_text(text):
    if pd.isnull(text):
        return ''

    text = re.sub(r'^RT\s+', '', text)  # Remove Retweet (RT)
    text = re.sub(r'\bUSER\b', '', text)  # Remove USER
    text = text.lower()  # Lowercase
    text = re.sub(r'(\\x[0-9a-fA-F]{2})+', '', text)  # Emoji encoding
    text = re.sub(r'&[a-zA-Z]+;', '', text)  # HTML entity
    text = re.sub(r'\\n', ' ', text)  # Whitespace encoding
    text = re.sub(r'[!"$%&\'()*+,\-./:;<=>?@\[\\\]^{|}~_#]', ' ', text)  # Punctuation
    text = standarize_text(text)
    text = stopword_remover.remove(text)
    text = re.sub(r'\b\d+\b', '', text)  # Numbers
    text = re.sub(r'\b[a-zA-Z]\b', '', text)  # One-letter words
    text = re.sub(r'\s+', ' ', text).strip()  # Multiple spaces
    return text


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file pada request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Hanya file .csv yang diperbolehkan'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_dataset.csv')
    file.save(file_path)

    return jsonify({'status': 'ok'}), 200


@app.route('/get_raw_data')
def get_raw_data():
    try:
        file_path = os.path.join(UPLOAD_FOLDER, 'uploaded_dataset.csv')
        data = pd.read_csv(file_path, encoding='latin1')
        tweets_only = data[['Tweet']].head(10)
        preview = data.head(10).fillna('').to_dict(orient='records')
        return tweets_only.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# === ROUTE PREPROCESS ===
@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        file_path = os.path.join(UPLOAD_FOLDER, 'uploaded_dataset.csv')
        if not os.path.exists(file_path):
            return jsonify({'error': 'Dataset tidak ditemukan'}), 404

        data = pd.read_csv(file_path, encoding='latin1')

        # Drop kolom yang tidak dibutuhkan
        data = data.drop(columns=["HS_Weak", "HS_Moderate", "HS_Strong"], errors='ignore')

        # Hapus duplikat berdasarkan 'Tweet'
        data = data[~data.duplicated(subset=['Tweet'])]

        # Preprocessing
        data['clean_tweet'] = data['Tweet'].apply(preprocess_text)

        # Simpan hasil bersih
        preprocessed_data = data.copy()
        preprocessed_data.drop(columns=['Tweet'], inplace=True)
        preprocessed_data.rename(columns={'clean_tweet': 'Tweet'}, inplace=True)
        cols = ['Tweet'] + [col for col in preprocessed_data.columns if col != 'Tweet']
        preprocessed_data = preprocessed_data[cols]

        # Hitung jumlah kata, filter < 2 kata
        preprocessed_data['word_count'] = preprocessed_data['Tweet'].apply(lambda x: len(str(x).split()))
        preprocessed_data = preprocessed_data[preprocessed_data['word_count'] >= 2]
        preprocessed_data.drop(columns=['word_count'], inplace=True)

        # Simpan hasil akhir
        output_path = os.path.join(PREPROCESSED_FOLDER, 'preprocessed_data.csv')
        preprocessed_data.to_csv(output_path, index=False)

        preview = preprocessed_data[['Tweet']].head(10)
        return preview.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# @app.route('/preprocess', methods=['POST'])
# def preprocess():
#     try:
#         data = pd.read_csv("raw_data.csv", encoding='utf-8', on_bad_lines='skip')  # Pastikan file ini sudah ada
#         data['clean_tweet'] = data['Tweet'].apply(preprocess_text)

#         # Susun ulang kolom
#         preprocessed_data = data.copy()
#         preprocessed_data.drop(columns=['Tweet'], inplace=True)
#         preprocessed_data.rename(columns={'clean_tweet': 'Tweet'}, inplace=True)
#         cols = ['Tweet'] + [col for col in preprocessed_data.columns if col != 'Tweet']
#         preprocessed_data = preprocessed_data[cols]

#         # Simpan ke file jika perlu
#         preprocessed_data.to_csv("preprocessed_data.csv", index=False)

#         return jsonify(preprocessed_data.head(50).to_dict(orient='records'))

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

@app.route('/tokenization')
def tokenization_page():
    return render_template('index.html')

# @app.route('/run_tokenization', methods=['POST'])
# def run_tokenization():
#     try:
#         # --- 1. Cek jika hasil sudah ada, langsung kembalikan dari file ---
#         tokenized_file = os.path.join(TOKENIZED_FOLDER, 'tokenized_data.csv')
#         embedding_file = 'embedded/bert_embedding.npy'
#         raw_file = os.path.join('raw_data', 'raw_data.csv')

#         # Ambil 5 data awal sebelum preprocessing
#         if os.path.exists(raw_file):
#             df_raw = pd.read_csv(raw_file)
#             pre_sample = df_raw[['Tweet']].head(5).reset_index()
#             pre_sample.rename(columns={'index': 'No'}, inplace=True)
#             pre_sample['No'] += 1
#             pre_data = pre_sample.to_dict(orient='records')
#         else:
#             pre_data = []

#         if os.path.exists(tokenized_file) and os.path.exists(embedding_file):
#             df_tokenized = pd.read_csv(tokenized_file)
#             df_tokenized = df_tokenized.head(5).reset_index()
#             df_tokenized.rename(columns={'index': 'No'}, inplace=True)
#             df_tokenized['No'] += 1
#             result = df_tokenized[['No', 'Tweet', 'Tokens', 'Token_IDs']].to_dict(orient='records')
#             return jsonify({'pre_data': pre_data, 'token_data': result})

#         # --- 2. Tokenisasi dan Simpan ---
#         file_path = os.path.join(PREPROCESSED_FOLDER, 'preprocessed_data.csv')
#         df = pd.read_csv(file_path)
#         texts = df['Tweet'].astype(str).tolist()

#         # Load tokenizer dan model IndoBERT
#         model_name = 'cahya/bert-base-indonesian-522M'
#         tokenizer = BertTokenizer.from_pretrained(model_name)
#         model = TFBertModel.from_pretrained(model_name)

#         tokens_list = []
#         token_ids_list = []

#         for text in texts:
#             encoded = tokenizer(text)
#             token_ids = encoded['input_ids']
#             tokens = tokenizer.convert_ids_to_tokens(token_ids)
#             tokens_list.append(tokens)
#             token_ids_list.append(token_ids)

#         df['Tokens'] = tokens_list
#         df['Token_IDs'] = token_ids_list
#         df['Token_Length'] = df['Tokens'].apply(len)

#         os.makedirs(TOKENIZED_FOLDER, exist_ok=True)
#         df.to_csv(tokenized_file, index=False)

#         # --- 3. Embedding jika belum ada ---
#         batch_size = 32
#         bert_embeddings = []

#         for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
#             batch_texts = texts[i:i + batch_size]
#             encoded = tokenizer(batch_texts, return_tensors='tf', padding='max_length', truncation=True, max_length=50)
#             output = model(encoded)
#             bert_embeddings.append(output.last_hidden_state.numpy())

#         final_embeddings = np.concatenate(bert_embeddings, axis=0)
#         os.makedirs('embedded', exist_ok=True)
#         np.save(embedding_file, final_embeddings)

#         # --- 4. Kembalikan hasil 5 data awal ---
#         df_result = df.head(5).reset_index()
#         df_result.rename(columns={'index': 'No'}, inplace=True)
#         df_result['No'] += 1
#         token_data = df_result[['No', 'Tweet', 'Tokens', 'Token_IDs']].to_dict(orient='records')

#         return jsonify({'pre_data': pre_data, 'token_data': token_data})

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

@app.route('/run_tokenization', methods=['POST'])
def run_tokenization():
    try:
        # --- 1. Cek jika hasil sudah ada, langsung kembalikan dari file ---
        tokenized_file = os.path.join(TOKENIZED_FOLDER, 'tokenized_data.csv')
        embedding_file = 'embedded/bert_embedding.npy'
        raw_file = os.path.join('raw_data', 'raw_data.csv')

        # Ambil 5 data awal sebelum preprocessing
        if os.path.exists(raw_file):
            df_raw = pd.read_csv(raw_file)
            pre_sample = df_raw[['Tweet']].head(5).reset_index()
            pre_sample.rename(columns={'index': 'No'}, inplace=True)
            pre_sample['No'] += 1
            pre_data = pre_sample.to_dict(orient='records')
        else:
            pre_data = []

        if os.path.exists(tokenized_file) and os.path.exists(embedding_file):
            df_tokenized = pd.read_csv(tokenized_file)
            df_tokenized = df_tokenized.head(5).reset_index()
            df_tokenized.rename(columns={'index': 'No'}, inplace=True)
            df_tokenized['No'] += 1
            result = df_tokenized[['No', 'Tweet', 'Tokens', 'Token_IDs']].to_dict(orient='records')
            return jsonify({'pre_data': pre_data, 'token_data': result})

        # --- 2. Tokenisasi ---
        file_path = os.path.join(PREPROCESSED_FOLDER, 'preprocessed_data.csv')
        df = pd.read_csv(file_path)
        texts = df['Tweet'].astype(str).tolist()

        # Load tokenizer dan model PyTorch
        model_name = 'cahya/bert-base-indonesian-522M'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
        model.eval()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        tokens_list = []
        token_ids_list = []

        for text in texts:
            encoded = tokenizer(text)
            tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
            token_ids = encoded['input_ids']

            tokens_list.append(tokens)
            token_ids_list.append(token_ids)

        df['Tokens'] = tokens_list
        df['Token_IDs'] = token_ids_list
        df['Token_Length'] = df['Tokens'].apply(len)

        os.makedirs(TOKENIZED_FOLDER, exist_ok=True)
        df.to_csv(tokenized_file, index=False)

        # --- 3. Embedding ---
        from tqdm import tqdm
        import numpy as np

        batch_size = 32
        bert_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(batch_texts, return_tensors='pt', padding='max_length',
                                truncation=True, max_length=40)

            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state.cpu().numpy()
                bert_embeddings.append(embeddings)

        final_embeddings = np.concatenate(bert_embeddings, axis=0)
        os.makedirs('embedded', exist_ok=True)
        np.save(embedding_file, final_embeddings)

        # --- 4. Kembalikan 5 data awal ---
        df_result = df.head(5).reset_index()
        df_result.rename(columns={'index': 'No'}, inplace=True)
        df_result['No'] += 1
        token_data = df_result[['No', 'Tweet', 'Tokens', 'Token_IDs']].to_dict(orient='records')

        return jsonify({'pre_data': pre_data, 'token_data': token_data})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/get_preprocessed', methods=['GET'])
def get_preprocessed():
    try:
        file_path = os.path.join(PREPROCESSED_FOLDER, 'preprocessed_data.csv')
        if not os.path.exists(file_path):
            return jsonify({"error": "File tidak ditemukan"})

        df = pd.read_csv(file_path).head(5)
        pre_data = [{"No": i + 1, "Tweet": row['Tweet']} for i, row in df.iterrows()]
        return jsonify({"pre_data": pre_data})
    except Exception as e:
        return jsonify({"error": str(e)})

# @app.route("/tune", methods=["POST"])
# def tune():
#     try:
#         results = run_tuning()
#         return jsonify({"results": results})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

@app.route("/tune", methods=["POST"])
def tune():
    print("🔧 [INFO] Endpoint /tune dipanggil.")

    try:
        results = run_tuning()
        print("✅ [SUCCESS] Tuning berhasil dijalankan.")
        print("📊 [RESULT] Top 3 results:")
        for r in results:
            print(r)

        return jsonify({"results": results})

    except Exception as e:
        print("❌ [ERROR] Tuning gagal dijalankan.")
        print(f"📄 [DETAIL] {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/train', methods=['GET'])
def train_bigru():
    def generate():
        process = subprocess.Popen(
            ['python', '-u','train.py', '--config', 'tuneresult/tuning_results.json'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in iter(process.stdout.readline, ''):
            yield f"data:{line}\n\n"
        process.stdout.close()
        process.wait()

        # Kirim pesan khusus tanda training selesai
        yield "data:[TRAINING_COMPLETED]\n\n"
    return Response(generate(), mimetype='text/event-stream', headers={"Cache-Control": "no-cache"})

@app.route('/best-params')
def best_params():
    import json
    with open('tuneresult/tuning_results.json', 'r') as f:
        results = json.load(f)['results']
        best = max(results, key=lambda x: x['val_acc'])['params']
    return best

@app.route('/predict', methods=['POST'])
def predict():
    from flask import request, jsonify
    import re
    import json
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn as nn
    from transformers import BertTokenizer, BertModel
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

    # Load kamus alay
    kamus_df = pd.read_csv('new_kamusalay.csv', names=['slang', 'formal'], encoding='latin1')
    kamus_dict = dict(zip(kamus_df['slang'], kamus_df['formal']))

    # Preprocessing tools
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    
    def standarize_text(text):
        words = text.split()
        new_words = [kamus_dict.get(word, word) for word in words]
        return ' '.join(new_words)

    def preprocess_text(text):
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
        return re.sub(r'\s+', ' ', text).strip()

    # Load tokenizer & model
    tokenizer = BertTokenizer.from_pretrained("cahya/bert-base-indonesian-522M")
    bert_model = BertModel.from_pretrained("cahya/bert-base-indonesian-522M")

    def get_bert_embedding(text):
        encoded = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=40)
        with torch.no_grad():
            bert_output = bert_model(**encoded)
        return bert_output.last_hidden_state  # (1, 40, 768)

    class BiGRUModel(nn.Module):
        def __init__(self, units=40):
            super(BiGRUModel, self).__init__()
            self.gru = nn.GRU(input_size=768, hidden_size=units, batch_first=True, bidirectional=True)
            self.fc = nn.Linear(units * 2, 9)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            _, h = self.gru(x)
            h_concat = torch.cat((h[0], h[1]), dim=1)
            return self.sigmoid(self.fc(h_concat))

    # Load model
    model = BiGRUModel()
    model.load_state_dict(torch.load('model/Bi-GRU.pt', map_location=torch.device('cpu')))
    model.eval()

    # Ambil tweet dari frontend
    tweet = request.json.get('text', '')
    cleaned = preprocess_text(tweet)
    embedded = get_bert_embedding(cleaned)

    with torch.no_grad():
        output = model(embedded).squeeze().numpy()

    labels = [
        "HS", "Abusive", "HS_Individual", "HS_Group", "HS_Religion",
        "HS_Race", "HS_Physical", "HS_Gender", "HS_Other"
    ]
    pred_binary = (output >= 0.5).astype(int)

    result = [label for label, val in zip(labels, pred_binary) if val == 1]

    return jsonify({
        'labels': result,
        'probabilities': {label: float(prob) for label, prob in zip(labels, output)}
    })


if __name__ == '__main__':
    app.run(debug=False,threaded=True)