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
BG_IMAGE = "assets/water_bg.png"
HERO_IMAGE = "assets/botol_air.jpeg"

# ==============================
# ADD CSS (DARK/LIGHT MODE + BG)
# ==============================
def add_css(dark_mode=False):
    text_color = "#f5f5f5" if dark_mode else "#0e1b2b"
    box_color = "rgba(30, 30, 30, 0.7)" if dark_mode else "rgba(255,255,255,0.8)"
    bg_image = f"url('{BG_IMAGE}')" if os.path.exists(BG_IMAGE) else "none"

    st.markdown(f"""
    <style>
        .stApp {{
            background-image: {bg_image};
            background-size: cover;
            background-attachment: fixed;
            color: {text_color};
        }}
        h1, h2, h3, h4, h5, h6, p, span, label {{
            color: {text_color} !important;
        }}
        .contact-btn {{
            padding: 8px 15px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: bold;
            font-size: 14px;
            margin-right: 5px;
            box-shadow: 0px 3px 6px rgba(0,0,0,0.2);
        }}
        .whatsapp {{ background-color: #25d366; color: white; }}
        .gmail {{ background-color: #d93025; color: white; }}
        .hero {{
            text-align: center;
            margin-top: 15px;
            margin-bottom: 25px;
        }}
        .card {{
            background-color: {box_color};
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        }}
    </style>
    """, unsafe_allow_html=True)

# ==============================
# LOGO
# ==============================
def generate_logo(size=150):
    img = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([10, 10, size-10, size-10], fill=(12, 105, 180))
    draw.ellipse([25, 25, size-25, size-25], fill=(60, 160, 230))
    wave_y = size * 0.65
    draw.rectangle([20, wave_y, size-20, wave_y + 15], fill=(173, 216, 230))
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
# LOAD MODEL (CACHE)
# ==============================
@st.cache_resource
def load_models():
    yolo = YOLO(YOLO_MODEL_PATH) if os.path.exists(YOLO_MODEL_PATH) else None
    keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH) if os.path.exists(KERAS_MODEL_PATH) else None
    return yolo, keras_model

# ==============================
# UTILITIES
# ==============================
def image_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def download_link(img, filename="detection_result.png"):
    b64 = base64.b64encode(image_to_bytes(img)).decode()
    return f'<a href="data:file/png;base64,{b64}" download="{filename}" class="contact-btn gmail">⬇ Download Result</a>'

# ==============================
# MAIN APP
# ==============================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    # Sidebar
    st.sidebar.title("⚙️ Settings")
    lang = st.sidebar.selectbox("🌐 Language", ["English", "Indonesia"])
    dark_mode = st.sidebar.toggle("🌙 Dark Mode", False)
    add_css(dark_mode)

    st.sidebar.markdown("---")
    st.sidebar.title("🎯 Application Mode")
    mode = st.sidebar.radio("Choose Function:", ["Object Detection (YOLO)", "Image Classification"])
    st.sidebar.title("📌 How to Use")
    if lang == "English":
        st.sidebar.write("1. Choose mode\n2. Upload JPG/PNG image\n3. Wait for result")
    else:
        st.sidebar.write("1. Pilih mode\n2. Upload gambar JPG/PNG\n3. Tunggu hasil muncul")

    # Header
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image(generate_logo(), width=90)
    with col2:
        title = "WaterVision — Image Detection & Classification" if lang == "English" else "WaterVision — Deteksi & Klasifikasi Gambar"
        st.markdown(f"<h1>{title}</h1>", unsafe_allow_html=True)

    st.markdown(f"""
        <a class="contact-btn whatsapp" href="https://wa.me/{WHATSAPP_NUMBER}" target="_blank">💬 WhatsApp</a>
        <a class="contact-btn gmail" href="mailto:{EMAIL_WATERVISION}" target="_blank">✉ Gmail</a>
    """, unsafe_allow_html=True)

    # Hero Section (Botol Air)
    if os.path.exists(HERO_IMAGE):
        st.markdown("<div class='hero'>", unsafe_allow_html=True)
        st.image(HERO_IMAGE, caption="WaterVision — Smart Vision for Every Drop 💧", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Upload Section
    st.markdown("---")
    upload_label = "📤 Upload Image (JPG/PNG):" if lang == "English" else "📤 Unggah Gambar (JPG/PNG):"
    uploaded = st.file_uploader(upload_label, type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="📷 Uploaded Image", use_column_width=True)

        progress = st.progress(0)
        for i in range(0, 101, 15):
            time.sleep(0.05)
            progress.progress(i)

        yolo, classifier = load_models()
        result_img = None

        if mode.startswith("Object"):
            st.subheader("🔍 Object Detection Result" if lang == "English" else "🔍 Hasil Deteksi Objek")
            if yolo:
                with st.spinner("Detecting objects..." if lang == "English" else "Mendeteksi objek..."):
                    res = yolo(img)
                    result_img = Image.fromarray(res[0].plot())
                    st.image(result_img, use_column_width=True)
                    st.markdown(download_link(result_img), unsafe_allow_html=True)
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

        if st.button("🔄 Reset" if lang == "English" else "🔄 Reset Gambar"):
            st.experimental_rerun()

    else:
        st.info("Please upload an image first." if lang == "English" else "Silakan unggah gambar terlebih dahulu.")

    st.markdown("---")
    st.markdown("© 2025 WaterVision — Powered by AI & Deep Learning")

if __name__ == "__main__":
    main()
