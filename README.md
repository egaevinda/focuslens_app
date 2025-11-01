# 🎯 FocusLens — Klasifikasi Fokus / Bosan / Distraksi (TensorFlow + Streamlit)

Aplikasi machine learning berbasis **Streamlit** untuk mendeteksi kondisi wajah (fokus, bosan, atau distraksi) mahasiswa selama perkuliahan daring (Zoom).  
Model dibuat menggunakan **Convolutional Neural Network (CNN)** dengan framework **TensorFlow** dan dijalankan pada dashboard interaktif berbasis web.

---

## 🚀 Fitur Utama
- 📷 Upload gambar wajah untuk diprediksi (fokus / bosan / distraksi)
- 🎥 Mode kamera realtime (opsional)
- 📊 Tampilan hasil prediksi dan probabilitas
- 🧠 Model CNN dilatih menggunakan dataset wajah mahasiswa saat sesi Zoom
- 🌐 Aplikasi dapat dijalankan di **localhost** atau di **Streamlit Cloud**

---

## 🧠 Arsitektur Model
Model menggunakan 3 kelas target:
- **Fokus**
- **Bosan**
- **Distraksi**

Model CNN dibangun menggunakan TensorFlow 2.17 dengan preprocessing:
- Resize citra wajah ke ukuran `128x128`
- Normalisasi piksel [0,1]
- Data split menjadi train, validation, dan test

---

## ⚙️ Instalasi Lokal
Jika ingin menjalankan di komputer lokal:

```bash
# Clone repository
git clone https://github.com/egaevinda/focuslens_app.git
cd focuslens_app

# Aktifkan environment (Windows)
python -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Jalankan aplikasi Streamlit
python -m streamlit run app_streamlit.py
