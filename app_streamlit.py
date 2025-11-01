import os
import io
import zipfile
import pathlib
import subprocess
from typing import Tuple

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf


DRIVE_FILE_ID = "1hF6ZYZY_ecaKZs-JrE39lyT_YLXjXDFU"  

MODEL_DIR = pathlib.Path("model_sm")

MODEL_ZIP = pathlib.Path("model_sm.zip")

IMG_SIZE = 128

LABELS = ["Fokus", "Bosan", "Distraksi"]


def ensure_model_available() -> None:
    """Unduh & ekstrak model SavedModel dari Google Drive jika belum ada."""
    if MODEL_DIR.is_dir():
        return

    if not DRIVE_FILE_ID or DRIVE_FILE_ID == "PASTE_GOOGLE_DRIVE_FILE_ID_DI_SINI":
        st.error(
            "Folder model tidak ditemukan dan `DRIVE_FILE_ID` belum diisi.\n"
            "Isi variabel DRIVE_FILE_ID dengan ID file Google Drive untuk model_sm.zip."
        )
        st.stop()


    try:
        import gdown  
    except Exception:
        subprocess.run(["pip", "install", "-q", "gdown"], check=True)
        import gdown  

    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    with st.spinner("Mengunduh model dari Google Drive..."):
        gdown.download(url, str(MODEL_ZIP), quiet=False)

    with st.spinner("Mengekstrak model..."):
        with zipfile.ZipFile(MODEL_ZIP, "r") as zf:
            zf.extractall(".")

   
    try:
        MODEL_ZIP.unlink(missing_ok=True)
    except Exception:
        pass


# Preprocessing & Prediksi
def preprocess(pil_img: Image.Image) -> np.ndarray:
    """RGB->resize->normalisasi [0,1], return (H,W,3) float32."""
    pil_img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    return arr


def predict_one(model: tf.keras.Model, rgb01: np.ndarray) -> Tuple[np.ndarray, int]:
    probs = model.predict(np.expand_dims(rgb01, 0), verbose=0)[0]
    idx = int(np.argmax(probs))
    return probs, idx


# Load model dengan cache 
@st.cache_resource(show_spinner=False)
def load_model_cached() -> tf.keras.Model:
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"Folder model tidak ditemukan: {MODEL_DIR.resolve()}\n"
            "Pastikan sudah diunduh otomatis dari Google Drive atau commit ke repo."
        )
    
    model = tf.keras.models.load_model(str(MODEL_DIR), compile=False)
    return model


#  Aplikasi Streamlit
def main():
    st.set_page_config(page_title="FocusLens", page_icon="🎯", layout="centered")

   
    ensure_model_available()

   
    st.title("🎯 FocusLens — Klasifikasi **Fokus / Bosan / Distraksi**")
    st.caption("TensorFlow + Streamlit • Input: gambar / kamera • Output: 3 kelas")

    with st.expander("ℹ️ Info runtime", expanded=False):
        st.write(
            f"- TensorFlow: `{tf.__version__}`\n"
            f"- Model folder: `{MODEL_DIR.resolve()}`"
        )

    try:
        with st.spinner("Memuat model..."):
            model = load_model_cached()
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        st.stop()

    tab1, tab2 = st.tabs(["📤 Upload Gambar", "📸 Kamera (snapshot)"])

    with tab1:
        f = st.file_uploader("Pilih gambar (jpg/png)", type=["jpg", "jpeg", "png"])
        if f is not None:
            pil = Image.open(f)
            st.image(pil, caption="Input", use_container_width=True)
            rgb01 = preprocess(pil)
            with st.spinner("Memprediksi..."):
                probs, idx = predict_one(model, rgb01)
            st.subheader(f"Hasil: **{LABELS[idx]}**")
            st.bar_chart({LABELS[i]: float(probs[i]) for i in range(len(LABELS))})

    with tab2:
        snap = st.camera_input("Ambil snapshot wajah")
        if snap is not None:
            pil = Image.open(io.BytesIO(snap.getvalue()))
            st.image(pil, caption="Snapshot", use_container_width=True)
            rgb01 = preprocess(pil)
            with st.spinner("Memprediksi..."):
                probs, idx = predict_one(model, rgb01)
            st.subheader(f"Hasil: **{LABELS[idx]}**")
            st.bar_chart({LABELS[i]: float(probs[i]) for i in range(len(LABELS))})

    st.markdown("---")
    st.caption(
        "Tips: hasil lebih stabil jika wajah jelas, frontal, dan pencahayaan cukup. "
        "Pastikan ukuran input training sama (IMG_SIZE)."
    )


if __name__ == "__main__":
    main()
