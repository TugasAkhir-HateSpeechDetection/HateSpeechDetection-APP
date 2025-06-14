from flask import Flask, render_template, jsonify, send_from_directory, request, Response
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

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
PREPROCESSED_FOLDER = os.path.join(os.getcwd(), 'preprocessed')
EMBEDDED_FOLDER = os.path.join(os.getcwd(), 'embedded')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)
os.makedirs(EMBEDDED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREPROCESSED_FOLDER'] = PREPROCESSED_FOLDER
app.config['EMBEDDED_FOLDER'] = EMBEDDED_FOLDER

#Global Variable
uploaded_filename = ""

@app.route('/')
def index():
    return render_template('index.html')

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

    # Unique file naming
    timestamp = int(time.time())
    uploaded_filename = f"dataset_{timestamp}.csv"

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
    file.save(file_path)

    return jsonify({'status': 'ok', 'filename': uploaded_filename}), 200

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
        shape = df.shape  # (rows, columns)

        return jsonify({
            'data': head_data,
            'columns': df.columns.tolist(),
            'shape': {'rows': shape[0], 'columns': shape[1]}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/preprocess', methods=['GET'])
def preprocess_dataset():
    global uploaded_filename
    if not uploaded_filename:
        return jsonify({'error': 'Tidak ada dataset diupload'}), 400

    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_filename)
        df = pd.read_csv(file_path, encoding='latin1')

        # Buat data original (preview)
        df_original = df[['Tweet']].copy()
        df_original.insert(0, 'No.', range(1, len(df_original) + 1))

        # Proses data
        df_processed = run_preprocessing(df)

        # Buat data preview hasil
        df_processed_preview = df_processed[['Tweet']].copy()
        df_processed_preview.insert(0, 'No.', range(1, len(df_processed_preview) + 1))

        # Simpan hasil full (tanpa preview)
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
    
@app.route('/run_tokenization', methods=['POST'])
def run_tokenization():
    try:
        preprocessed_path = os.path.join(PREPROCESSED_FOLDER, 'preprocessed_data.csv')
        raw_path = os.path.join('raw_data', 'raw_data.csv')
        embedded_path = os.path.join(EMBEDDED_FOLDER, 'bert_embedding.npy')

        # Ambil data sebelum tokenisasi
        if os.path.exists(raw_path):
            df_raw = pd.read_csv(raw_path)
            pre_sample = df_raw[['Tweet']].head(5).reset_index()
            pre_sample.rename(columns={'index': 'No'}, inplace=True)
            pre_sample['No'] += 1
            pre_data = pre_sample.to_dict(orient='records')
        else:
            pre_data = []

        if not os.path.exists(preprocessed_path):
            return jsonify({'error': 'Preprocessed file not found'}), 404

        result = tokenize_one(preprocessed_path, embedded_path)

        if 'error' in result:
            return jsonify({'error': result['error']}), 500

        return jsonify({'pre_data': pre_data, 'token_data': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
import sys
python_executable = sys.executable  

@app.route('/start-tuning')
def start_tuning():
    script_path = os.path.join(os.path.dirname(__file__), 'utils', 'tune.py')
    def generate():
        process = subprocess.Popen(
            [python_executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True
        )

        for line in process.stdout:
            yield f"data: {line.strip()}\n\n"

    return Response(generate(), mimetype='text/event-stream')

@app.route('/get-tuning-result')
def get_tuning_result():
    try:
        with open('./tuning_result/best_params.json') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/get-best-params')
def get_best_params():
    try:
        with open('./tuning_result/best_params.json') as f:
            data = json.load(f)
            best = max(data, key=lambda x: x['val_acc'])
            return jsonify(best['params'])
    except Exception as e:
        return jsonify({})
    
@app.route('/train-model')
def train_model():
    script_path = os.path.join(os.path.dirname(__file__), 'utils', 'train.py')

    def generate():
        process = subprocess.Popen(
            [python_executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True  # penting agar tidak binary
        )

        for line in process.stdout:
            yield f"data: {line.strip()}\n\n"

        process.wait()
        load_model()
    return Response(generate(), mimetype='text/event-stream')

@app.route('/predict', methods=['POST'])
def predict():
    load_model()
    data = request.json
    tweet_text = data.get('tweet', '')
    if not tweet_text:
        return jsonify({"error": "No tweet text provided"}), 400

    results = predict_tweet(tweet_text)
    return jsonify(results)

@app.route("/evaluate-model", methods=["GET"])
def evaluate_model():
    try:
        run_evaluation()
        return jsonify({
            "status": "success",
            "report": "/evaluation/classification_report.json",
            "confusion_matrix": "/evaluation/confusion_matrix.png"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/evaluation/<path:filename>")
def evaluation_files(filename):
    return send_from_directory("evaluation", filename)

if __name__ == '__main__':
    app.run(debug=False,threaded=True)