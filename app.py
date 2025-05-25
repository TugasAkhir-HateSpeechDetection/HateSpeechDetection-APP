from flask import Flask, request, jsonify, send_from_directory, render_template
import pandas as pd
import re, os
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PREPROCESSED_FOLDER = 'preprocessed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREPROCESSED_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PREPROCESSED_FOLDER'] = PREPROCESSED_FOLDER

# ======== PREPROCESSING FUNCTION ========= #
factory = StopWordRemoverFactory()
stopword_remover = factory.create_stop_word_remover()

kamus_df = pd.read_csv("new_kamusalay.csv", encoding='latin1', header=None, names=['slang', 'formal'])
kamus_dict = dict(zip(kamus_df['slang'], kamus_df['formal']))

def standarize_text(text):
    words = text.split()
    new_words = [kamus_dict.get(word, word) for word in words]
    return ' '.join(new_words)

def preprocess_text(text):
    if pd.isnull(text):
        return ''

    text = re.sub(r'^RT\s+', '', text)
    text = re.sub(r'\bUSER\b', '', text)
    text = text.lower()
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'(\\x[0-9a-fA-F]{2})+', '', text)
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    text = re.sub(r'\\n', ' ', text)
    text = re.sub(r'(?<=\w)[!"$%&\'()*+,\-./:;<=>?@\[\\\]^{|}~](?=\w)', ' ', text)
    text = re.sub(r'[!"$%&\'()*+,\-./:;<=>?@\[\\\]^{|}~_]', '', text)
    text = standarize_text(text)
    text = stopword_remover.remove(text)
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
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

@app.route('/preprocess', methods=['POST'])
def preprocess():
    try:
        file_path = os.path.join(UPLOAD_FOLDER, 'uploaded_dataset.csv')
        data = pd.read_csv(file_path,encoding='latin1')

        # Proses Preprocessing
        data['clean_tweet'] = data['Tweet'].apply(preprocess_text)

        preprocessed_data = data.copy()
        preprocessed_data.drop(columns=['Tweet'], inplace=True)
        preprocessed_data.rename(columns={'clean_tweet': 'Tweet'}, inplace=True)
        cols = ['Tweet'] + [col for col in preprocessed_data.columns if col != 'Tweet']
        preprocessed_data = preprocessed_data[cols]

        output_path = os.path.join(PREPROCESSED_FOLDER, 'preprocessed_data.csv')
        preprocessed_data.to_csv(output_path, index=False)

        preview = preprocessed_data.head(10).fillna('').to_dict(orient='records')
        tweets_only = preprocessed_data[['Tweet']].head(10)
        return tweets_only.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/tokenize', methods=['GET'])
def tokenize():
    return jsonify({
        'before': ['kalimat satu', 'kalimat dua'],
        'after': [['kalimat', 'satu'], ['kalimat', 'dua']]
    })

@app.route('/train', methods=['GET'])
def train():
    # Simulasi training
    return jsonify({'log': 'Model dilatih dengan akurasi 87%', 'status': 'success'})

@app.route('/tune', methods=['GET'])
def tune():
    return jsonify({'params': {'learning_rate': 0.01, 'batch_size': 32}})

if __name__ == '__main__':
    app.run(debug=True)