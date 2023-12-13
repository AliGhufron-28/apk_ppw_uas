import streamlit as st
import pandas as pd
# Library
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# Klasifikasi
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import string
from sklearn.decomposition import LatentDirichletAllocation
import pickle

st.set_page_config(
    page_title="Aplikasi Kategori Berita | Klasifikasi Berita Kompas Tv", page_icon="📰")

st.title("Website Applikasi Untuk Prediksi Kategori Berita")
st.caption("app by M. Ali Ghufron")

col1, col2 = st.columns(2)

with col1:

    st.title("Studi Kasus ")

with col2:
    st.image("logo-kompascom-28.png", use_column_width=True)


# proses clean

data_new = pd.read_csv('data_new_berita_kompas.csv')

def tokenizer(text):
  text = text.lower()
  return sent_tokenize(text)
data_new["tokenizing"] = data_new['clean_content'].apply(tokenizer)

# membuat kolom baru dengan nama new_abstrak untuk data baru yang dipunctuation
data_new['clean_content'] = data_new['Content_Artikel'].str.replace('[{}]'.format(string.punctuation), '').str.lower()
# Menghilangkan angka dari kolom 'new_abstrak'
data_new['clean_content'] = data_new['clean_content'].str.replace('\d+', '', regex=True)
# menggabungkan kata
data_new['final_content'] = data_new['tokenizing'].apply(lambda x: ' '.join(x))

# Function Topik
def create_topic_proportion_df(X_summary, k, alpha, beta):
    lda_model = LatentDirichletAllocation(n_components=k, doc_topic_prior=alpha, topic_word_prior=beta)
    
    proporsi_topik_dokumen = lda_model.fit_transform(X_summary)
    
    nama_kolom_topik = [f'Topik {i+1}' for i in range(k)]
    
    proporsi_topik_dokumen_df = pd.DataFrame(proporsi_topik_dokumen, columns=nama_kolom_topik)
    
    return proporsi_topik_dokumen_df

# proses klasifikasi
st.write("""
    ### Want to learn more?
    - Dataset (studi kasus) [kompas.com](https://www.kompas.com/)
    - Github Account [github.com](https://github.com/AliGhufron-28/ppw_regenerate)
    """)
data_final_sm = pd.read_csv('data_final_sm.csv')

vectorizer_summary = TfidfVectorizer()
tfidf_text = vectorizer_summary.fit_transform(data_final_sm['summary']).toarray()
X_summary = create_topic_proportion_df(tfidf_text, 6, 0.1, 0.2)
y = data_final_sm["Category"]

X_train_summary, X_test_summary, y_train_summary, y_test_summary = train_test_split(X_summary, y, test_size=0.3, random_state=42)

# Inisialisasi model Naive Bayes Gaussian
gnb_summary = GaussianNB()
# Melatih model menggunakan data latih
gnb_summary.fit(X_train_summary, y_train_summary)
# Membuat prediksi pada data uji
y_pred_gnb_summary = gnb_summary.predict(X_test_summary)
accuracy = accuracy_score(y_test_summary, y_pred_gnb_summary)
# print(f'Akurasi: {accuracy}')



st.subheader("Masukan Text")

new_data = st.text_area("Masukkan Text Berita", height=250)

hasil = st.button("cek klasifikasi")

if hasil:

    new_data_summary = tokenizer(new_data[0])
    
    tfidf_Xnew_summary = vectorizer_summary.transform([new_data_summary[0]]).toarray()
    topik_tfidf_x = create_topic_proportion_df(tfidf_Xnew_summary, 6, 0.1, 0.2)
    
    pred_gnb_summary = gnb_summary.predict(topik_tfidf_x)

    st.success(f"Prediksi Hasil Klasifikasi : {pred_gnb_summary[0]}")

