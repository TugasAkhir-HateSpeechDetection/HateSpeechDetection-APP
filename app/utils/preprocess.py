import pandas as pd
import re
import os
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Load kamus alay
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
csv_path = os.path.join(ROOT_DIR, 'app', 'kamus', 'new_kamusalay.csv')

kamus_df = pd.read_csv(csv_path, encoding='latin1', header=None, names=['slang', 'formal'])
kamus_dict = dict(zip(kamus_df['slang'], kamus_df['formal']))

factory = StopWordRemoverFactory()
stopword_remover = factory.create_stop_word_remover()

def standarize_text(text):
    words = text.split()
    return ' '.join([kamus_dict.get(w, w) for w in words])

def preprocess_text(text):
    if pd.isnull(text):
        return ''
    text = re.sub(r'^RT\s+|\bUSER\b|\\x[0-9a-fA-F]{2}+', '', text)
    text = re.sub(r'&[a-zA-Z]+;|\\n', ' ', text).lower()
    text = re.sub(r'[!"$%&\'()*+,\-./:;<=>?@\[\\\]^{|}~_#]', ' ', text)
    text = standarize_text(text)
    text = stopword_remover.remove(text)
    text = re.sub(r'\b\d+\b|\b[a-zA-Z]\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def run_preprocessing(df):
    df_clean = df.copy()

    # Drop kolom tidak relevan jika ada
    cols_to_drop = ['HS_Weak', 'HS_Moderate', 'HS_Strong']
    df_clean.drop(columns=[col for col in cols_to_drop if col in df_clean.columns], inplace=True)

    # Drop duplikat (berdasarkan semua kolom, atau bisa disesuaikan hanya pada kolom Tweet)
    df_clean = df_clean.drop_duplicates()

    # Simpan hasil preprocessing di kolom baru
    df_clean['clean_tweet'] = df_clean['Tweet'].apply(preprocess_text)

    # Hapus tweet pendek (kurang dari 2 kata)
    df_clean['word_count'] = df_clean['clean_tweet'].apply(lambda x: len(str(x).split()))
    df_clean = df_clean[df_clean['word_count'] >= 2]

    # Finalisasi: buang kolom asli, rename hasil
    df_clean.drop(columns=['Tweet', 'word_count'], inplace=True)
    df_clean.rename(columns={'clean_tweet': 'Tweet'}, inplace=True)

    # Opsional: pindahkan Tweet ke kolom pertama jika ada label lain
    cols = ['Tweet'] + [col for col in df_clean.columns if col != 'Tweet']
    df_clean = df_clean[cols]

    return df_clean
