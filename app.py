import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
import base64, os, time

# ==============================
# CONFIG
# ==============================
APP_TITLE = "💧 WaterVision — Image Detection & Classification"
YOLO_MODEL_PATH = "model/Vanissa Aulya Putri_Laporan 4.pt"
KERAS_MODEL_PATH = "model/Vanissa Aulya Putri_Laporan 2.h5"
WHATSAPP_NUMBER = "6282245357681"
EMAIL_WATERVISION = "watervision@gmail.com"
BG_IMAGE = "assets/water_bg.jpg"
BOTOL_IMAGE = "assets/botol_air.jpeg"

# ==============================
# CUSTOM STYLE (MODERN UI)
# ==============================
def add_css():
    bg = ""
    if os.path.exists(BG_IMAGE):
        with open(BG_IMAGE, "rb") as f:
            bg = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        html, body, [class*="css"]  {{
            font-family: 'Poppins', sans-serif;
            color: #0e1b2b;
        }}
        .stApp {{
            background: linear-gradient(to bottom, rgba(255,255,255,0.95), rgba(230,244,255,0.9)), 
                        url("data:image/jpg;base64,{bg}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        h1, h2, h3, h4, h5 {{
            color: #004b87;
            font-weight: 700;
            text-align: center;
        }}
        .subtitle {{
            text-align: center;
            color: #1b3c59;
            font-size: 18px;
            font-weight: 500;
        }}
        .contact-btn {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            font-size: 15px;
            margin: 0 5px;
            color: white;
            box-shadow: 0px 4px 8px rgba(0,0,0,0.15);
        }}
        .whatsapp {{ background: #25D366; }}
        .gmail {{ background: #EA4335; }}
        .info-box {{
            text-align: center;
            background: rgba(255,255,255,0.85);
            border-radius: 10px;
            padding: 15px;
            margin-top: 25px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            font-weight: 500;
            color: #0e1b2b;
        }}
        .stProgress > div > div > div > div {{
            background-color: #0078ff;
        }}
    </style>
    """, unsafe_allow_html=True)

# ==============================
# LOAD MODELS
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
    add_css()

    # HEADER
    st.markdown(f"<h1>{APP_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Smart Vision for Every Drop — powered by AI & Deep Learning</p>", unsafe_allow_html=True)

    # HERO IMAGE
    if os.path.exists(BOTOL_IMAGE):
        encoded = base64.b64encode(open(BOTOL_IMAGE, "rb").read()).decode()
        st.markdown(f"""
        <div style='display:flex; justify-content:center; align-items:center; margin:20px 0;'>
            <img src='data:image/jpeg;base64,{encoded}' width='280' style='border-radius:12px; box-shadow:0 4px 10px rgba(0,0,0,0.2);'>
        </div>
        """, unsafe_allow_html=True)

    # CONTACT BUTTONS
    st.markdown(f"""
    <div style="text-align:center; margin-bottom:25px;">
        <a class="contact-btn whatsapp" href="https://wa.me/{WHATSAPP_NUMBER}" target="_blank">💬 WhatsApp</a>
        <a class="contact-btn gmail" href="mailto:{EMAIL_WATERVISION}" target="_blank">📩 Gmail</a>
    </div>
    """, unsafe_allow_html=True)

    # UPLOAD
    st.markdown("<h3 style='text-align:center;'>📤 Upload Image for Detection or Classification</h3>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="📷 Uploaded Image", use_container_width=True)

        progress = st.progress(0)
        for i in range(0, 101, 20):
            time.sleep(0.05)
            progress.progress(i)

        yolo, classifier = load_models()
        result_img = None

        st.markdown("<h3 style='text-align:center; margin-top:20px;'>🧠 Processing...</h3>", unsafe_allow_html=True)

        if yolo:
            res = yolo(img)
            result_img = res[0].plot()
            st.image(result_img, caption="🔍 Detection Result", use_container_width=True)
        else:
            st.warning("⚠ Model YOLO tidak ditemukan.")

        progress.progress(100)
        st.success("✅ Selesai! Gambar berhasil diproses.")
    else:
        st.markdown("<div class='info-box'>📘 Silakan unggah gambar terlebih dahulu untuk memulai deteksi atau klasifikasi.</div>", unsafe_allow_html=True)

    # FOOTER
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#555;'>© 2025 WaterVision | Designed with 💙 by AI</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
