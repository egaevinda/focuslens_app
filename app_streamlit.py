import os
import io
import zipfile
import pathlib
import subprocess
from typing import Tuple, Union, Dict, Any

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import keras
from keras.layers import TFSMLayer

DRIVE_FILE_ID = "1hF6ZYZY_ecaKZs-JrE39lyT_YLXjXDFU" 


SAVEDMODEL_DIR = pathlib.Path("model_sm")             
KERAS_FILE = pathlib.Path("model_v3.keras")           
H5_FILE = pathlib.Path("model_legacy.h5")  
MODEL_ZIP = pathlib.Path("model_artifact.zip") 

IMG_SIZE = 128
LABELS = ["Fokus", "Bosan", "Distraksi"]
# ============================================================================


def ensure_model_available():
    
    if SAVEDMODEL_DIR.is_dir() or KERAS_FILE.is_file() or H5_FILE.is_file():
        return

    if not DRIVE_FILE_ID or DRIVE_FILE_ID == "1hF6ZYZY_ecaKZs-JrE39lyT_YLXjXDFU":
        st.error(
            "Model belum tersedia dan DRIVE_FILE_ID belum diisi.\n"
            "Silakan isi variabel DRIVE_FILE_ID dengan ID file Google Drive."
        )
        st.stop()

    try:
        import gdown 
    except Exception:
        subprocess.run(["pip", "install", "-q", "gdown"], check=True)
        import gdown  

    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

    with st.spinner("Mengunduh artefak model dari Google Drive..."):
        gdown.download(url, str(MODEL_ZIP), quiet=False)

    with st.spinner("Mengekstrak artefak model..."):
        with zipfile.ZipFile(MODEL_ZIP, "r") as zf:
            zf.extractall(".")

    try:
        MODEL_ZIP.unlink(missing_ok=True)
    except Exception:
        pass


#  Preprocess & Prediksi 
def preprocess(pil_img: Image.Image) -> np.ndarray:
    """RGB -> resize -> normalisasi [0,1]. Return (H,W,3) float32."""
    pil_img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0
    return arr


def _to_probs(y: Union[np.ndarray, tf.Tensor]) -> np.ndarray:
    """Pastikan output berupa probabilitas 1D np.float64/32 (apply softmax jika perlu)."""
    if isinstance(y, tf.Tensor):
        y = y.numpy()
    y = np.squeeze(y)

    if y.ndim > 1:
        y = np.mean(y, axis=tuple(range(y.ndim - 1)))  
    y = y.astype("float64")
    if (y < 0).any() or (y > 1).any() or not np.isclose(np.sum(y), 1.0, atol=1e-3):
        
        ex = np.exp(y - np.max(y))
        y = ex / np.sum(ex)
    return y


#  Load model dengan cache 
@st.cache_resource(show_spinner=False)
def load_any_model():
    """
    Urutan:
      1) Coba load file .keras
      2) Coba load file .h5
      3) Coba load folder SavedModel lewat Keras (jika kebetulan kompatibel)
      4) Fallback: SavedModel via TFSMLayer (inference-only)
    Return:
      (callable_model, mode) — mode in {"keras", "tfsm"}
    """
    if KERAS_FILE.is_file():
        m = tf.keras.models.load_model(str(KERAS_FILE), compile=False)
        return m, "keras"

    if H5_FILE.is_file():
        m = tf.keras.models.load_model(str(H5_FILE), compile=False)
        return m, "keras"

    if SAVEDMODEL_DIR.is_dir():
        try:
            m = tf.keras.models.load_model(str(SAVEDMODEL_DIR), compile=False)
            return m, "keras"
        except Exception:
           
            m = TFSMLayer(str(SAVEDMODEL_DIR), call_endpoint="serving_default")
            return m, "tfsm"

    raise FileNotFoundError(
        "Tidak menemukan model (.keras / .h5 / folder SavedModel). "
        "Pastikan ZIP berisi salah satu dari ketiganya."
    )


def predict_one(model, mode: str, rgb01: np.ndarray) -> Tuple[np.ndarray, int]:
    x = tf.convert_to_tensor(rgb01[None, ...], dtype=tf.float32)
    y = model(x)  

    if isinstance(y, dict):
        y = next(iter(y.values()))
    elif isinstance(y, (list, tuple)):
        y = y[0]

    probs = _to_probs(y)
    idx = int(np.argmax(probs))
    return probs, idx


#  Aplikasi Streamlit 
def main():
    st.set_page_config(page_title="FocusLens", page_icon="🎯", layout="centered")

    st.title("🎯 FocusLens — Klasifikasi **Fokus / Bosan / Distraksi**")
    st.caption("TensorFlow + Streamlit • Input: gambar / kamera • Output: 3 kelas")

    with st.expander("ℹ️ Info runtime", expanded=False):
        st.write(
            f"- TensorFlow: `{tf.__version__}` • Keras: `{keras.__version__}`\n"
            f"- Python runtime: kompatibel dengan Keras 3 (TF 2.20)\n"
            f"- Artefak model: `.keras` / `.h5` / SavedModel (folder)"
        )

    ensure_model_available()

    # Load model
    try:
        with st.spinner("Memuat model..."):
            model, mode = load_any_model()
        st.success(f"Model siap • mode: **{mode}**")
    except Exception as e:
        st.error(
            "Gagal memuat model: "
            f"{e}\n\nJika pesan menyebut SavedModel legacy, "
            "gunakan fallback ini atau simpan ulang model menjadi `.keras` / `.h5`."
        )
        st.stop()

    
    tab1, tab2 = st.tabs(["📤 Upload Gambar", "📸 Kamera (snapshot)"])

    with tab1:
        f = st.file_uploader("Pilih gambar (jpg/png)", type=["jpg", "jpeg", "png"])
        if f is not None:
            pil = Image.open(f)
            st.image(pil, caption="Input", use_container_width=True)
            rgb01 = preprocess(pil)
            with st.spinner("Memprediksi..."):
                probs, idx = predict_one(model, mode, rgb01)
            st.subheader(f"Hasil: **{LABELS[idx]}**")
            st.bar_chart({LABELS[i]: float(probs[i]) for i in range(len(LABELS))})

    with tab2:
        snap = st.camera_input("Ambil snapshot wajah")
        if snap is not None:
            pil = Image.open(io.BytesIO(snap.getvalue()))
            st.image(pil, caption="Snapshot", use_container_width=True)
            rgb01 = preprocess(pil)
            with st.spinner("Memprediksi..."):
                probs, idx = predict_one(model, mode, rgb01)
            st.subheader(f"Hasil: **{LABELS[idx]}**")
            st.bar_chart({LABELS[i]: float(probs[i]) for i in range(len(LABELS))})

    st.markdown("---")
    st.caption(
        "Catatan: jika output SavedModel bukan probabilitas, sistem akan "
        "mengaplikasikan softmax otomatis agar dapat dipetakan ke 3 kelas."
    )


if __name__ == "__main__":
    main()
