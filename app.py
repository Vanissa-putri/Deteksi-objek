import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os

# ==============================
# KONFIGURASI DASAR
# ==============================
APP_TITLE = "WaterVision — Image Detection & Classification"
YOLO_MODEL_PATH = "model/Vanissa Aulya Putri_Laporan 4.pt"
KERAS_MODEL_PATH = "model/Vanissa Aulya Putri_Laporan 2.h5"
WHATSAPP_NUMBER = "6282245357681"
EMAIL_WATERVISION = "watervision@gmail.com"

# ==============================
# STYLING CSS
# ==============================
def add_css():
    st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(180deg, #d9eefe 0%, #c1e0fb 50%, #eaf6ff 100%);
            background-attachment: fixed;
        }
        .header {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .contact-btn {
            padding: 8px 15px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: bold;
            font-size: 14px;
            margin-right: 5px;
            box-shadow: 0px 3px 6px rgba(0,0,0,0.1);
        }
        .whatsapp { background-color: #25d366; color: white; }
        .gmail { background-color: #d93025; color: white; }
        .card {
            padding: 15px;
            background-color: rgba(255,255,255,0.85);
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

# ==============================
# LOGO GENERATOR (Flat Minimal Water Style)
# ==============================
def generate_logo(size=150):
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    # Circle
    draw.ellipse([10, 10, size-10, size-10], fill=(12, 105, 180))  # deep water blue
    draw.ellipse([25, 25, size-25, size-25], fill=(60, 160, 230))  # lighter blue

    # Wave line
    wave_y = size * 0.65
    draw.rectangle([20, wave_y, size-20, wave_y + 15], fill=(173, 216, 230))

    # WaterVision "W"
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 50)
    except:
        font = ImageFont.load_default()
    draw.text((size/2 - 15, size/2 - 25), "W", fill="white", font=font)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer

# ==============================
# LOAD MODELS (CACHED)
# ==============================
@st.cache_resource
def load_models():
    yolo = YOLO(YOLO_MODEL_PATH) if os.path.exists(YOLO_MODEL_PATH) else None
    keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH) if os.path.exists(KERAS_MODEL_PATH) else None
    return yolo, keras_model

# ==============================
# CEK FORMAT GAMBAR
# ==============================
def allowed_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))

# ==============================
# MAIN APP
# ==============================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    add_css()

    # Header
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image(generate_logo(), width=100)
    with col2:
        st.markdown(f"<div class='header'><h1>{APP_TITLE}</h1></div>", unsafe_allow_html=True)

    # Contact Buttons
    st.markdown(f"""
        <a class="contact-btn whatsapp" href="https://wa.me/{WHATSAPP_NUMBER}" target="_blank">💬 WhatsApp</a>
        <a class="contact-btn gmail" href="mailto:{EMAIL_WATERVISION}" target="_blank">✉ Gmail</a>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Sidebar
    st.sidebar.title("🎯 Mode Aplikasi")
    mode = st.sidebar.radio("Pilih fungsi:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    st.sidebar.title("📌 Cara Penggunaan")
    st.sidebar.write("""
    1. Pilih mode deteksi atau klasifikasi  
    2. Upload gambar berformat JPG/PNG  
    3. Tunggu hasil muncul di layar utama  
    """)

    # Load Models
    yolo, classifier = load_models()

    # Upload
    uploaded = st.file_uploader("Unggah Gambar (hanya JPG/PNG):", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Silakan unggah gambar terlebih dahulu.")
        return
    if not allowed_file(uploaded.name):
        st.error("❌ Format tidak valid. Mohon upload file dengan ekstensi JPG atau PNG.")
        return

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Gambar Diupload", use_column_width=True)

    # Mode Deteksi
    if mode == "Deteksi Objek (YOLO)":
        if yolo is None:
            st.error("Model YOLO tidak ditemukan.")
        else:
            st.subheader("🔍 Hasil Deteksi Objek")
            with st.spinner("Mendeteksi objek..."):
                result = yolo(img)
                result_img = result[0].plot()
                st.image(result_img, use_column_width=True)

    # Mode Klasifikasi
    else:
        if classifier is None:
            st.error("Model Keras tidak ditemukan.")
        else:
            st.subheader("🧠 Hasil Klasifikasi Gambar")
            img_resized = img.resize((224, 224))
            img_array = keras_image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            pred = classifier.predict(img_array)
            class_idx = np.argmax(pred)
            conf = np.max(pred)
            st.success(f"✅ Kelas terdeteksi: **{class_idx}** dengan probabilitas **{conf:.4f}**")

    st.markdown("---")
    st.markdown("© 2025 WaterVision — Powered by AI & Deep Learning")

if __name__ == "__main__":
    main()
