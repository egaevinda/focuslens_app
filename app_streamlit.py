import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # biar log TensorFlow tidak terlalu panjang

import streamlit as st
import numpy as np
from PIL import Image
import io
import tensorflow as tf


MODEL_PATH = r"D:\focuslens_app\model_final.h5"  
IMG_SIZE = 128
CLASS_NAMES = ["Fokus", "Bosan", "Distraksi"]
import os, zipfile, pathlib, subprocess


MODEL_DIR = pathlib.Path("model_sm")    
MODEL_ZIP = pathlib.Path("model_sm.zip") 
DRIVE_FILE_ID = "1hF6ZYZY_ecaKZs-JrE39lyT_YLXjXDFU"  

def ensure_model_available():
    """Unduh dan ekstrak model bila belum ada di folder kerja"""
    if MODEL_DIR.is_dir():
        return
    try:
        import gdown
    except Exception:
        subprocess.run(["pip", "install", "-q", "gdown"], check=True)
        import gdown
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    print("Mengunduh model dari Google Drive...")
    gdown.download(url, str(MODEL_ZIP), quiet=False)
    print("Ekstrak model...")
    with zipfile.ZipFile(MODEL_ZIP, "r") as zf:
        zf.extractall(".")
    MODEL_ZIP.unlink(missing_ok=True)
    print("Model siap digunakan.")


ensure_model_available()


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ File model tidak ditemukan di:\n{MODEL_PATH}\nPastikan file model_final.h5 ada di folder tersebut.")
        st.stop()
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.success("✅ Model berhasil dimuat dari lokal!")
        return model
    except Exception as e:
        st.error(f"Gagal memuat model TensorFlow (.h5):\n\n{e}")
        st.stop()

model = load_model()


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Ubah gambar menjadi array siap diprediksi."""
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    return img_array

def predict_image(model, img_array):
    """Prediksi satu gambar."""
    preds = model.predict(np.expand_dims(img_array, axis=0), verbose=0)
    probs = preds[0]
    idx = np.argmax(probs)
    return probs, idx


st.set_page_config(page_title="FocusLens", page_icon="🎯", layout="centered")
st.title("🎯 FocusLens — Klasifikasi Fokus / Bosan / Distraksi (TensorFlow Lokal)")

tab1, tab2 = st.tabs(["📤 Upload Gambar", "📸 Kamera Langsung"])

with tab1:
    uploaded_file = st.file_uploader("Upload gambar wajah (jpg/png)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar Input", use_container_width=True)

        img_array = preprocess_image(image)
        with st.spinner("🔍 Memprediksi..."):
            probs, idx = predict_image(model, img_array)

        st.subheader(f"📊 Hasil Prediksi: **{CLASS_NAMES[idx]}**")
        st.bar_chart({
            "Fokus": [probs[0]],
            "Bosan": [probs[1]],
            "Distraksi": [probs[2]]
        })

with tab2:
    cam_image = st.camera_input("Ambil foto dari kamera")
    if cam_image:
        image = Image.open(io.BytesIO(cam_image.getvalue()))
        st.image(image, caption="Snapshot Kamera", use_container_width=True)

        img_array = preprocess_image(image)
        with st.spinner("🔍 Memprediksi..."):
            probs, idx = predict_image(model, img_array)

        st.subheader(f"📊 Hasil Prediksi Kamera: **{CLASS_NAMES[idx]}**")
        st.bar_chart({
            "Fokus": [probs[0]],
            "Bosan": [probs[1]],
            "Distraksi": [probs[2]]
        })

st.caption("Model dimuat dari lokal (D:\\focuslens_app) menggunakan TensorFlow 2.15 dan tf.keras lama.")

