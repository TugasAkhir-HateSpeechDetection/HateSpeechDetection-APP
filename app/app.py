from flask import Flask, render_template, jsonify, send_from_directory, request, Response, send_file
import os
import time
import pandas as pd
import subprocess
import json
import sys

from utils.preprocess import run_preprocessing
from utils.tokenization import tokenize_one
from utils.testing import predict_tweet, load_model
from utils.evaluation import run_evaluation

app = Flask(__name__)

# Folder konfigurasi
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
PREPROCESSED_FOLDER = os.path.join(os.getcwd(), 'preprocessed')
EMBEDDED_FOLDER = os.path.join(os.getcwd(), 'embedded')

# Pastikan direktori tersedia
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)
os.makedirs(EMBEDDED_FOLDER, exist_ok=True)

# Simpan konfigurasi ke Flask
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREPROCESSED_FOLDER'] = PREPROCESSED_FOLDER
app.config['EMBEDDED_FOLDER'] = EMBEDDED_FOLDER

# Variabel global
uploaded_filename = ""
python_executable = sys.executable

# ======================= ROUTING =======================

@app.route('/')
def index():
    return render_template('index.html')

# Upload dataset CSV
@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    global uploaded_filename

    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file pada request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nama file kosong'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Hanya file .csv yang diperbolehkan'}), 400

    timestamp = int(time.time())
    uploaded_filename = f"dataset_{timestamp}.csv"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
    file.save(file_path)

    return jsonify({'status': 'ok', 'filename': uploaded_filename}), 200

# Menampilkan 10 baris pertama dari dataset
@app.route('/show_dataset', methods=['GET'])
def show_dataset():
    global uploaded_filename

    if not uploaded_filename:
        return jsonify({'error': 'Tidak ada dataset yang diupload'}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)

    try:
        df = pd.read_csv(file_path, encoding='latin1')
        df.insert(0, 'No.', range(1, len(df) + 1))
        tweet_col = df.pop('Tweet')
        df.insert(1, 'Tweet', tweet_col)

        head_data = df.head(10).to_dict(orient='records')
        shape = df.shape

        return jsonify({
            'data': head_data,
            'columns': df.columns.tolist(),
            'shape': {'rows': shape[0], 'columns': shape[1]}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Preprocessing tweet
@app.route('/preprocess', methods=['GET'])
def preprocess_dataset():
    global uploaded_filename

    if not uploaded_filename:
        return jsonify({'error': 'Tidak ada dataset diupload'}), 400

    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
        df = pd.read_csv(file_path, encoding='latin1')

        df_original = df[['Tweet']].copy()
        df_original.insert(0, 'No.', range(1, len(df_original) + 1))

        df_processed = run_preprocessing(df)
        df_processed_preview = df_processed[['Tweet']].copy()
        df_processed_preview.insert(0, 'No.', range(1, len(df_processed_preview) + 1))

        output_path = os.path.join(PREPROCESSED_FOLDER, 'preprocessed_data.csv')
        df_processed.to_csv(output_path, index=False)

        return jsonify({
            'original': df_original.head(10).to_dict(orient='records'),
            'processed': df_processed_preview.head(10).to_dict(orient='records'),
            'shape_original': df_original.shape,
            'shape_processed': df_processed.shape
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Tokenisasi data secara batch
@app.route('/start-tokenization')
def start_tokenization():
    preprocessed_path = os.path.join(PREPROCESSED_FOLDER, 'preprocessed_data.csv')
    embedded_path = os.path.join(EMBEDDED_FOLDER, 'bert_embedding.npy')
    lengths_path = os.path.join(EMBEDDED_FOLDER, 'bert_lengths.npy')
    script_path = os.path.join('utils', 'tokenize_batch.py')

    def generate():
        process = subprocess.Popen(
            [python_executable, script_path, preprocessed_path, embedded_path, lengths_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in iter(process.stdout.readline, ''):
            if line.strip() == 'ALREADY_EXISTS':
                yield 'data: ALREADY_EXISTS\n\n'
                break
            yield f'data: {line}\n\n'
        process.stdout.close()

    return Response(generate(), mimetype='text/event-stream')

# Ambil contoh tokenisasi 1 tweet
@app.route('/tokenization-sample')
def tokenization_sample():
    try:
        preprocessed_path = os.path.join(PREPROCESSED_FOLDER, 'preprocessed_data.csv')
        embedded_path = os.path.join(EMBEDDED_FOLDER, 'bert_embedding.npy')
        result = tokenize_one(preprocessed_path, embedded_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Tuning hyperparameter Bi-GRU
@app.route('/start-tuning')
def start_tuning():
    script_path = os.path.join('utils', 'tune.py')

    def generate():
        process = subprocess.Popen(
            [python_executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in process.stdout:
            yield f"data: {line.strip()}\n\n"

    return Response(generate(), mimetype='text/event-stream')

# Mengambil seluruh hasil tuning
@app.route('/get-tuning-result')
def get_tuning_result():
    try:
        with open('./tuning_result/best_params.json') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})

# Mengambil best params hasil tuning
@app.route('/get-best-params')
def get_best_params():
    try:
        with open('./tuning_result/best_params.json') as f:
            data = json.load(f)
            best = max(data, key=lambda x: x['val_acc'])
            return jsonify(best['params'])
    except Exception:
        return jsonify({})

# Latih model Bi-GRU
@app.route('/train-model')
def train_model():
    script_path = os.path.join('utils', 'train.py')

    def generate():
        process = subprocess.Popen(
            [python_executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in iter(process.stdout.readline, ''):
            yield f"data: {line.strip()}\n\n"
        process.stdout.close()
        process.wait()
        load_model()

    return Response(generate(), mimetype='text/event-stream')

# Ambil plot hasil training
@app.route('/get-training-plot')
def get_training_plot():
    plot_path = os.path.join('app', 'evaluation', 'training_plot.png')
    if os.path.exists(plot_path):
        return send_file(plot_path, mimetype='image/png')
    return jsonify({'error': 'File tidak ditemukan'}), 404

# Evaluasi model: confusion matrix & classification report
@app.route('/evaluate-model', methods=['GET'])
def evaluate_model():
    load_model()
    try:
        run_evaluation()
        return jsonify({
            "status": "success",
            "report": "/evaluation/classification_report.json",
            "confusion_matrix": "/evaluation/confusion_matrix.png"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Ambil file evaluasi (gambar/json)
@app.route("/evaluation/<path:filename>")
def evaluation_files(filename):
    return send_from_directory("evaluation", filename)

# Ambil nilai hamming loss
@app.route("/get-hamming-loss")
def get_hamming_loss():
    try:
        with open('./evaluation/hamming_loss.json') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Endpoint prediksi 1 tweet
@app.route('/predict', methods=['POST'])
def predict():
    load_model()
    data = request.json
    tweet_text = data.get('tweet', '')
    if not tweet_text:
        return jsonify({"error": "No tweet text provided"}), 400

    results = predict_tweet(tweet_text)
    return jsonify(results)

# Jalankan server
if __name__ == '__main__':
    app.run(debug=False, threaded=True)
