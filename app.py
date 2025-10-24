import streamlit as st
from PIL import Image
import base64
import os
import io
import time
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from tensorflow.keras.preprocessing import image as keras_image

# ==============================
# CONFIG
# ==============================
st.set_page_config(page_title="WaterVision — Image Detection & Classification", page_icon="💧", layout="wide")

APP_TITLE = "WaterVision — Image Detection & Classification"
YOLO_MODEL_PATH = "model/Vanissa Aulya Putri_Laporan 4.pt"
KERAS_MODEL_PATH = "model/Vanissa Aulya Putri_Laporan 2.h5"
BG_IMAGE = "assets/water_bg.jpg"
BOTOL_IMAGE = "assets/botol_air.jpeg"
EMAIL_WATERVISION = "watervision@gmail.com"
WHATSAPP_NUMBER = "6282245357681"

# ==============================
# BACKGROUND SETUP
# ==============================
def set_background(image_file):
    if not os.path.exists(image_file):
        return
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    css = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        transition: background 0.5s ease;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_background(BG_IMAGE)

# ==============================
# CUSTOM STYLE (DARK/LIGHT MODE)
# ==============================
def apply_style(dark_mode=False):
    text_color = "#f2f2f2" if dark_mode else "#0e1b2b"
    upload_bg = "rgba(255,255,255,0.1)" if dark_mode else "rgba(255,255,255,0.7)"
    footer_color = "#f2f2f2" if dark_mode else "#0e1b2b"
    box_bg = "rgba(40,40,40,0.7)" if dark_mode else "rgba(255,255,255,0.9)"

    st.markdown(f"""
    <style>
    .title {{
        text-align: center;
        font-size: 2.3rem;
        font-weight: 700;
        margin-top: 1rem;
        color: {text_color};
    }}
    .subtext {{
        text-align: center;
        font-size: 1rem;
        color: {text_color};
        margin-bottom: 1rem;
    }}
    .upload-box {{
        background: {upload_bg};
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #ccc;
        text-align: center;
    }}
    .card {{
        background: {box_bg};
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.3);
    }}
    footer {{
        text-align: center;
        color: {footer_color};
        font-size: 0.9rem;
        margin-top: 4rem;
    }}
    .contact-btn {{
        display: inline-block;
        padding: 8px 15px;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        font-size: 14px;
        margin: 5px;
        color: white;
    }}
    .whatsapp {{ background-color: #25D366; }}
    .gmail {{ background-color: #D93025; }}
    </style>
    """, unsafe_allow_html=True)

# ==============================
# CACHE MODEL
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
    dark_mode = st.sidebar.toggle("🌙 Dark Mode", False)
    apply_style(dark_mode)

    # Sidebar Info
    st.sidebar.title("⚙️ Settings")
    lang = st.sidebar.selectbox("🌐 Language", ["English", "Indonesia"])
    st.sidebar.markdown("---")
    mode = st.sidebar.radio("🎯 Choose Function", ["Object Detection (YOLO)", "Image Classification"])
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Upload gambar JPG/PNG dan tunggu hasil deteksi atau klasifikasi.")

    # Header
    st.markdown(f"<div class='title'>{APP_TITLE}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='subtext'>Smart Vision for Every Drop 💧</div>", unsafe_allow_html=True)

    # Logo Botol Air
    if os.path.exists(BOTOL_IMAGE):
        st.image(BOTOL_IMAGE, width=280, caption="WaterVision", use_container_width=False)

    # Kontak
    st.markdown(f"""
        <div style="text-align:center;">
            <a class="contact-btn whatsapp" href="https://wa.me/{WHATSAPP_NUMBER}" target="_blank">💬 WhatsApp</a>
            <a class="contact-btn gmail" href="mailto:{EMAIL_WATERVISION}" target="_blank">✉ Gmail</a>
        </div>
    """, unsafe_allow_html=True)

    # Upload Gambar
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    upload_label = "📤 Upload Image (JPG/PNG):" if lang == "English" else "📤 Unggah Gambar (JPG/PNG):"
    uploaded = st.file_uploader(upload_label, type=["jpg", "jpeg", "png"])
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="📷 Uploaded Image", use_container_width=True)

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
                    st.image(result_img, use_container_width=True)
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

    # Footer
    st.markdown(f"<footer>© 2025 <b>WaterVision</b> — Powered by <b>AI & Deep Learning</b></footer>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
