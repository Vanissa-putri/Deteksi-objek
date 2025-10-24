import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
import base64
import os
import time

# ==============================
# CONFIG
# ==============================
APP_TITLE = "💧 WaterVision — Smart Image Detection"
YOLO_MODEL_PATH = "model/Vanissa Aulya Putri_Laporan 4.pt"
KERAS_MODEL_PATH = "model/Vanissa Aulya Putri_Laporan 2.h5"
WHATSAPP_NUMBER = "6282245357681"
EMAIL_WATERVISION = "watervision@gmail.com"
BG_IMAGE = "assets/water_bg.jpg"
BOTOL_IMAGE = "assets/botol_air.jpeg"

# ==============================
# CUSTOM STYLE
# ==============================
def add_custom_css():
    bg = ""
    if os.path.exists(BG_IMAGE):
        with open(BG_IMAGE, "rb") as f:
            bg = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{bg}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #0e1b2b;
            font-family: 'Poppins', sans-serif;
        }}
        h1 {{
            text-align: center;
            font-weight: 700;
            color: #073763;
            text-shadow: 1px 1px 3px rgba(255,255,255,0.7);
        }}
        h2, h3, h4 {{
            color: #0b2545;
        }}
        .stButton>button {{
            background: linear-gradient(90deg, #0099ff, #66ccff);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            font-weight: bold;
            box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        }}
        .stButton>button:hover {{
            background: linear-gradient(90deg, #007acc, #33bbff);
            transform: scale(1.05);
        }}
        .contact-btn {{
            padding: 8px 15px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            font-size: 14px;
            margin-right: 6px;
            box-shadow: 0px 3px 6px rgba(0,0,0,0.25);
            transition: 0.2s ease;
        }}
        .whatsapp {{ background-color: #25d366; color: white; }}
        .gmail {{ background-color: #d93025; color: white; }}
        .contact-btn:hover {{ opacity: 0.85; }}
        .info-box {{
            text-align: center;
            padding: 15px;
            background-color: rgba(255,255,255,0.85);
            border-radius: 10px;
            margin-top: 25px;
            font-weight: 500;
            backdrop-filter: blur(6px);
        }}
        footer {{
            text-align: center;
            color: #ffffff;
            background-color: rgba(0, 0, 0, 0.45);
            border-radius: 8px;
            padding: 8px;
            margin-top: 30px;
        }}
    </style>
    """, unsafe_allow_html=True)

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_models():
    yolo = YOLO(YOLO_MODEL_PATH) if os.path.exists(YOLO_MODEL_PATH) else None
    keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH) if os.path.exists(KERAS_MODEL_PATH) else None
    return yolo, keras_model

# ==============================
# MAIN APP
# ==============================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    add_custom_css()

    st.markdown(f"<h1>{APP_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:18px;'>Empowering Smart Water Quality Detection 🌊</p>", unsafe_allow_html=True)

    # Display bottle image
    if os.path.exists(BOTOL_IMAGE):
        encoded = base64.b64encode(open(BOTOL_IMAGE, "rb").read()).decode()
        st.markdown(f"""
        <div style='display:flex; justify-content:center; align-items:center; margin:15px 0;'>
            <img src='data:image/jpeg;base64,{encoded}' width='250' style='border-radius:12px; box-shadow:0 4px 10px rgba(0,0,0,0.4);'>
        </div>
        """, unsafe_allow_html=True)

    # Contact Buttons
    st.markdown(f"""
    <div style="text-align:center; margin-bottom:20px;">
        <a class="contact-btn whatsapp" href="https://wa.me/{WHATSAPP_NUMBER}" target="_blank">💬 WhatsApp</a>
        <a class="contact-btn gmail" href="mailto:{EMAIL_WATERVISION}" target="_blank">✉ Gmail</a>
    </div>
    """, unsafe_allow_html=True)

    # Mode Selection
    st.sidebar.title("🎯 Pilih Mode Aplikasi")
    mode = st.sidebar.radio("Mode Deteksi:", ["Object Detection (YOLO)", "Image Classification"])

    # Upload Section
    st.markdown("<div style='display:flex; justify-content:center; margin-top:20px;'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("📤 Unggah Gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="📷 Gambar Diupload", use_container_width=True)

        progress = st.progress(0)
        for i in range(0, 101, 20):
            time.sleep(0.05)
            progress.progress(i)

        yolo, classifier = load_models()
        result_img = None

        if "Object" in mode:
            st.subheader("🔍 Hasil Deteksi Objek")
            if yolo:
                with st.spinner("Mendeteksi objek..."):
                    res = yolo(img)
                    result_img = Image.fromarray(res[0].plot())
                    st.image(result_img, use_container_width=True)
            else:
                st.error("⚠ Model YOLO tidak ditemukan!")

        else:
            st.subheader("🧠 Hasil Klasifikasi Gambar")
            if classifier:
                img_resized = img.resize((224, 224))
                arr = keras_image.img_to_array(img_resized)
                arr = np.expand_dims(arr, axis=0) / 255.0
                pred = classifier.predict(arr)
                class_idx = np.argmax(pred)
                conf = np.max(pred)
                st.success(f"✅ Kelas terdeteksi: **{class_idx}** (probabilitas {conf:.2%})")
            else:
                st.error("⚠ Model Keras tidak ditemukan!")

        progress.progress(100)
        st.success("✅ Selesai!")
    else:
        st.markdown("<div class='info-box'>📘 Silakan unggah gambar terlebih dahulu.</div>", unsafe_allow_html=True)

    st.markdown("<footer>© 2025 WaterVision — AI & Deep Learning for a Sustainable Future 💧</footer>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
