import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

# Load Saved Model
model_wardah = joblib.load('warda_UV_final.sav')

# CSS for centering the main content
st.markdown("""
    <style>
        /* Center the entire content */
        .main-content {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        .center-content {
            text-align: center;
        }
        .justify-text {
            text-align: justify;
        }
    </style>
""", unsafe_allow_html=True)

# Begin main content div
st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Judul Halaman WEB (Centered)
st.markdown("""
<h2 class="center-content">SENTIMEN ANALIS PRODUK SKINCARE</h2>
<h4 class="center-content">Oleh: MUTIA ZAHIRMA 21.11.41**</h4>
""", unsafe_allow_html=True)

# Menambahkan gambar
st.image('wowww.jpeg')

# End main content div
st.markdown('</div>', unsafe_allow_html=True)

# Dropdown untuk Tentang Aplikasi
section = st.selectbox(
    'Pilih Kategori',
    (
        'Klik untuk Memilih kategori',
        '1. Tentang Aplikasi',
        '2. Cara Penggunaan',
        '3. Tentang Model',
    )
)

# Konten berdasarkan pilihan dropdown
if section == '1. Tentang Aplikasi':
    st.markdown("""
    <div class="justify-text">
     Tentang Aplikasi

    Aplikasi web ini bertujuan untuk blabla.
    </div>
    """, unsafe_allow_html=True)

elif section == '2. Cara Penggunaan':
    st.markdown("""
     Cara Penggunaan

    1. Masukkan teks ke dalam kotak input.
    2. Klik tombol 'Hasil Deteksi' untuk melihat prediksi.
    3. Prediksi akan muncul di bawah tombol.
    """)

elif section == '3. Tentang Model':
    st.markdown("""
    <div class="justify-text">
     Tentang Model

    Model yang digunakan dalam aplikasi ini adalah model Naïve Bayes.
    </div>
    """, unsafe_allow_html=True)

# Text input for prediction
input_text = st.text_input('Masukkan ulasan produk')

# Button to perform prediction
if st.button('Hasil Deteksi'):
    if input_text:  # Ensure input is not empty
        # Predict the sentiment
        # Predict directly from input text
        prediction = model_wardah.predict([input_text])[0]
        probability = model_wardah.predict_proba(
            [input_text])[0]  # Get prediction probabilities

        # Display result and confidence
        if prediction == 'Positif':
            st.success(f'Sentimen Positif dengan Akurasi {probability[1] * 100:.2f}%')
        else:
            st.error(f'Sentimen Negatif dengan Akurasi {probability[0] * 100:.2f}%')

    else:
        st.warning("Silakan masukkan ulasan sebelum melakukan prediksi.")
