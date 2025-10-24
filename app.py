import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os
import time

# ==============================
# CONFIG
# ==============================
APP_TITLE = "WaterVision — Image Detection & Classification"
YOLO_MODEL_PATH = "model/Vanissa Aulya Putri_Laporan 4.pt"
KERAS_MODEL_PATH = "model/Vanissa Aulya Putri_Laporan 2.h5"
WHATSAPP_NUMBER = "6282245357681"
EMAIL_WATERVISION = "watervision@gmail.com"
BG_IMAGE = "assets/water_bg.jpg"
BOTOL_IMAGE = "assets/botol_air.jpeg"

# ==============================
# STYLE (DARK/LIGHT MODE)
# ==============================
def add_css(dark_mode=False):
    text_color = "#ffffff" if dark_mode else "#0e1b2b"
    bg_overlay = "rgba(0,0,0,0.5)" if dark_mode else "rgba(255,255,255,0.9)"

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
            color: {text_color};
        }}
        h1, h2, h3, h4, h5, h6, p, label, span {{
            color: {text_color} !important;
        }}
        .contact-btn {{
            padding: 8px 15px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: bold;
            font-size: 14px;
            margin-right: 6px;
            box-shadow: 0px 3px 6px rgba(0,0,0,0.2);
        }}
        .whatsapp {{ background-color: #25d366; color: white; }}
        .gmail {{ background-color: #d93025; color: white; }}
        .hero {{
            text-align: center;
            margin-top: 15px;
            margin-bottom: 25px;
        }}
        .info-box {{
            text-align: center;
            padding: 10px;
            background-color: {bg_overlay};
            border-radius: 10px;
            margin-top: 20px;
            font-weight: 500;
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
# APP
# ==============================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    st.sidebar.title("⚙️ Settings")
    lang = st.sidebar.selectbox("🌐 Language", ["English", "Indonesia"])
    dark_mode = st.sidebar.toggle("🌙 Dark Mode", False)
    add_css(dark_mode)

    st.sidebar.markdown("---")
    st.sidebar.title("🎯 Mode Aplikasi")
    mode = st.sidebar.radio("Pilih Mode:", ["Object Detection (YOLO)", "Image Classification"])

    st.sidebar.markdown("### 📘 Panduan")
    if lang == "English":
        st.sidebar.write("1️⃣ Choose mode\n2️⃣ Upload image (JPG/PNG)\n3️⃣ Wait for result")
    else:
        st.sidebar.write("1️⃣ Pilih mode\n2️⃣ Unggah gambar (JPG/PNG)\n3️⃣ Tunggu hasil")

    # HEADER
    st.markdown(f"<h1 style='text-align:center;'>{APP_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Smart Vision for Every Drop 💧</p>", unsafe_allow_html=True)

    # HERO IMAGE (Center)
    if os.path.exists(BOTOL_IMAGE):
        encoded = base64.b64encode(open(BOTOL_IMAGE, "rb").read()).decode()
        st.markdown(f"""
        <div style='display:flex; justify-content:center; align-items:center;'>
            <img src='data:image/jpeg;base64,{encoded}' width='250' style='border-radius:12px; box-shadow:0 4px 10px rgba(0,0,0,0.3);'>
        </div>
        """, unsafe_allow_html=True)

    # CONTACT BUTTONS
    st.markdown(f"""
    <div style="text-align:center; margin-top:10px;">
        <a class="contact-btn whatsapp" href="https://wa.me/{WHATSAPP_NUMBER}" target="_blank">💬 WhatsApp</a>
        <a class="contact-btn gmail" href="mailto:{EMAIL_WATERVISION}" target="_blank">✉ Gmail</a>
    </div>
    """, unsafe_allow_html=True)

    # UPLOAD SECTION (centered)
    st.markdown("<div style='display:flex; justify-content:center; margin-top:30px;'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("📤 Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="📷 Uploaded Image", use_container_width=True)

        progress = st.progress(0)
        for i in range(0, 101, 20):
            time.sleep(0.05)
            progress.progress(i)

        yolo, classifier = load_models()
        result_img = None

        if "Object" in mode:
            st.subheader("🔍 Object Detection Result" if lang == "English" else "🔍 Hasil Deteksi Objek")
            if yolo:
                with st.spinner("Detecting objects..." if lang == "English" else "Mendeteksi objek..."):
                    res = yolo(img)
                    result_img = Image.fromarray(res[0].plot())
                    st.image(result_img, use_container_width=True)
            else:
                st.error("YOLO model not found!" if lang == "English" else "Model YOLO tidak ditemukan!")

        else:
            st.subheader("🧠 Classification Result" if lang == "English" else "🧠 Hasil Klasifikasi Gambar")
            if classifier:
                img_resized = img.resize((224, 224))
                arr = keras_image.img_to_array(img_resized)
                arr = np.expand_dims(arr, axis=0) / 255.0
                pred = classifier.predict(arr)
                class_idx = np.argmax(pred)
                conf = np.max(pred)
                if lang == "English":
                    st.success(f"✅ Predicted class: **{class_idx}** (confidence {conf:.2%})")
                else:
                    st.success(f"✅ Kelas terdeteksi: **{class_idx}** (probabilitas {conf:.2%})")
            else:
                st.error("Keras model not found!" if lang == "English" else "Model Keras tidak ditemukan!")

        progress.progress(100)
        st.success("✅ Done!" if lang == "English" else "✅ Selesai!")
    else:
        message = "📘 Please upload an image first." if lang == "English" else "📘 Silakan unggah gambar terlebih dahulu."
        st.markdown(f"<div class='info-box'>{message}</div>", unsafe_allow_html=True)

    # FOOTER
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:14px;'>© 2025 WaterVision — Powered by AI & Deep Learning</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
