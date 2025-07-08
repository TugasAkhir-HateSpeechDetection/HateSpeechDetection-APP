import pandas as pd
import re
import os
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =======================
# Inisialisasi & Persiapan
# =======================

# Ambil path root folder ke kamus alay
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
csv_path = os.path.join(ROOT_DIR, 'app', 'kamus', 'new_kamusalay.csv')

# Baca kamus alay menjadi dictionary
kamus_df = pd.read_csv(csv_path, encoding='latin1', header=None, names=['slang', 'formal'])
kamus_dict = dict(zip(kamus_df['slang'], kamus_df['formal']))

# Inisialisasi stopword remover dari library Sastrawi
factory = StopWordRemoverFactory()
stopword_remover = factory.create_stop_word_remover()

# =======================
# Fungsi Standarisasi Kata
# =======================

def standarize_text(text):
    """
    Mengganti kata tidak baku (alay) berdasarkan kamus.
    """
    words = text.split()
    return ' '.join([kamus_dict.get(w, w) for w in words])

# =======================
# Fungsi Preprocessing
# =======================

def preprocess_text(text):
    """
    Membersihkan teks dari noise dan melakukan normalisasi:
    - Menghapus USER, RT, simbol, angka, huruf satuan
    - Lowercasing
    - Stopword removal
    - Kamus alay
    """
    if pd.isnull(text):
        return ''

    text = re.sub(r'^RT\s+', '', text)                        # Hapus awalan RT
    text = re.sub(r'\bUSER\b', '', text)                      # Hapus kata USER
    text = text.lower()                                       # Huruf kecil semua
    text = re.sub(r'(\\x[0-9a-fA-F]{2})+', '', text)          # Hapus karakter hex
    text = re.sub(r'&[a-zA-Z]+;', '', text)                   # Hapus HTML entity (&gt;, &amp;)
    text = re.sub(r'\\n', ' ', text)                          # Hapus newline literal
    text = re.sub(r'[!"$%&\'()*+,\-./:;<=>?@\[\\\]^{|}~_#]', ' ', text)  # Hapus simbol & tanda baca
    text = standarize_text(text)                              # Ubah kata alay ke formal
    text = stopword_remover.remove(text)                      # Stopword removal
    text = re.sub(r'\b\d+\b', '', text)                       # Hapus angka
    text = re.sub(r'\b[a-zA-Z]\b', '', text)                  # Hapus huruf tunggal
    text = re.sub(r'\s+', ' ', text).strip()                  # Normalisasi spasi

    return text

# =======================
# Fungsi Utama Preprocessing
# =======================

def run_preprocessing(df):
    """
    Proses cleaning dan normalisasi untuk seluruh DataFrame.
    Output berupa DataFrame baru dengan kolom 'Tweet' yang sudah dibersihkan.
    """
    df_clean = df.copy()

    # Drop kolom HS_Weak/Moderate/Strong
    cols_to_drop = ['HS_Weak', 'HS_Moderate', 'HS_Strong']
    df_clean.drop(columns=[col for col in cols_to_drop if col in df_clean.columns], inplace=True)

    # Hapus duplikasi
    df_clean = df_clean.drop_duplicates()

    # Lakukan preprocessing teks ke kolom baru
    df_clean['clean_tweet'] = df_clean['Tweet'].apply(preprocess_text)

    # Hitung jumlah kata, buang yang terlalu pendek (<2 kata)
    df_clean['word_count'] = df_clean['clean_tweet'].apply(lambda x: len(str(x).split()))
    df_clean = df_clean[df_clean['word_count'] >= 2]

    # Finalisasi: hapus kolom tidak perlu, dan ubah nama kolom hasil
    df_clean.drop(columns=['Tweet', 'word_count'], inplace=True)
    df_clean.rename(columns={'clean_tweet': 'Tweet'}, inplace=True)

    # Pastikan kolom Tweet ada di urutan pertama
    cols = ['Tweet'] + [col for col in df_clean.columns if col != 'Tweet']
    df_clean = df_clean[cols]

    return df_clean
