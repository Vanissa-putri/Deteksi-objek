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
# KONFIGURASI DASAR
# ==============================
APP_TITLE = "WaterVision — Image Detection & Classification"
YOLO_MODEL_PATH = "model/Vanissa Aulya Putri_Laporan 4.pt"
KERAS_MODEL_PATH = "model/Vanissa Aulya Putri_Laporan 2.h5"
WHATSAPP_NUMBER = "6282245357681"
EMAIL_WATERVISION = "watervision@gmail.com"
HERO_IMAGE = "assets/botol_air.jpg"  # tambahkan gambar estetik di folder assets

# ==============================
# STYLING CSS
# ==============================
def add_css(dark_mode=False):
    if dark_mode:
        bg_color = "#0e1b2b"
        text_color = "#eaf6ff"
    else:
        bg_color = "#d9eefe"
        text_color = "#0e1b2b"

    st.markdown(f"""
    <style>
        .stApp {{
            background: linear-gradient(180deg, {bg_color} 0%, #c1e0fb 50%, #eaf6ff 100%);
            background-attachment: fixed;
            color: {text_color};
        }}
        .header {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .contact-btn {{
            padding: 8px 15px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: bold;
            font-size: 14px;
            margin-right: 5px;
            box-shadow: 0px 3px 6px rgba(0,0,0,0.1);
        }}
        .whatsapp {{ background-color: #25d366; color: white; }}
        .gmail {{ background-color: #d93025; color: white; }}
        .card {{
            padding: 15px;
            background-color: rgba(255,255,255,0.85);
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        }}
        .hero {{
            text-align: center;
            margin-top: 10px;
            margin-bottom: 30px;
        }}
    </style>
    """, unsafe_allow_html=True)

# ==============================
# LOGO GENERATOR
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
# LOAD MODELS (CACHED)
# ==============================
@st.cache_resource
def load_models():
    yolo = YOLO(YOLO_MODEL_PATH) if os.path.exists(YOLO_MODEL_PATH) else None
    keras_model = tf.keras.models.load_model(KERAS_MODEL_PATH) if os.path.exists(KERAS_MODEL_PATH) else None
    return yolo, keras_model

# ==============================
# UTILITAS
# ==============================
def allowed_file(filename):
    return filename.lower().endswith((".jpg", ".jpeg", ".png"))

def image_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

def get_download_link(img, filename="hasil_deteksi.png"):
    img_bytes = image_to_bytes(img)
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}" class="contact-btn gmail">⬇ Download Hasil</a>'
    return href

# ==============================
# MAIN APP
# ==============================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")

    # Sidebar Options
    st.sidebar.title("⚙️ Pengaturan")
    lang = st.sidebar.selectbox("🌐 Pilih Bahasa / Language", ["Indonesia", "English"])
    dark_mode = st.sidebar.toggle("🌙 Dark Mode", False)
    add_css(dark_mode)

    st.sidebar.markdown("---")
    st.sidebar.title("🎯 Mode Aplikasi")
    mode = st.sidebar.radio("Pilih fungsi:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    st.sidebar.title("📌 Cara Penggunaan")
    st.sidebar.write("""
    1. Pilih mode deteksi atau klasifikasi  
    2. Upload gambar berformat JPG/PNG  
    3. Tunggu hasil muncul di layar utama  
    """)

    yolo, classifier = load_models()

    # HEADER
    col1, col2 = st.columns([1, 8])
    with col1:
        st.image(generate_logo(), width=90)
    with col2:
        st.markdown(f"<div class='header'><h1>{APP_TITLE}</h1></div>", unsafe_allow_html=True)

    st.markdown(f"""
        <a class="contact-btn whatsapp" href="https://wa.me/{WHATSAPP_NUMBER}" target="_blank">💬 WhatsApp</a>
        <a class="contact-btn gmail" href="mailto:{EMAIL_WATERVISION}" target="_blank">✉ Gmail</a>
    """, unsafe_allow_html=True)

    # HERO IMAGE
    if os.path.exists(HERO_IMAGE):
        st.markdown("<div class='hero'>", unsafe_allow_html=True)
        st.image(HERO_IMAGE, caption="WaterVision – Smart Vision for Every Drop", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Upload Section
    st.markdown("---")
    uploaded = st.file_uploader("📤 Unggah Gambar (JPG/PNG):", type=["jpg", "jpeg", "png"])

    if uploaded:
        if not allowed_file(uploaded.name):
            st.error("❌ Format tidak valid.")
            return

        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="📷 Gambar Diupload", use_column_width=True)

        progress = st.progress(0)
        for i in range(1, 101, 15):
            time.sleep(0.05)
            progress.progress(i)

        st.subheader("🔎 Proses Analisis Gambar")
        progress.progress(60)

        if mode == "Deteksi Objek (YOLO)":
            if yolo is None:
                st.error("Model YOLO tidak ditemukan.")
            else:
                with st.spinner("Mendeteksi objek..."):
                    result = yolo(img)
                    result_img = Image.fromarray(result[0].plot())
                    st.image(result_img, use_column_width=True)
                    st.markdown(get_download_link(result_img), unsafe_allow_html=True)

        elif mode == "Klasifikasi Gambar":
            if classifier is None:
                st.error("Model Keras tidak ditemukan.")
            else:
                img_resized = img.resize((224, 224))
                img_array = keras_image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array, axis=0) / 255.0
                pred = classifier.predict(img_array)
                class_idx = np.argmax(pred)
                conf = np.max(pred)
                st.success(f"✅ Kelas terdeteksi: **{class_idx}** dengan probabilitas **{conf:.4f}**")

        progress.progress(100)
        st.success("✅ Proses selesai!")

        # Tombol Reset
        if st.button("🔄 Reset Gambar"):
            st.experimental_rerun()

    else:
        st.info("Silakan unggah gambar terlebih dahulu.")

    st.markdown("---")
    st.markdown("© 2025 WaterVision — Powered by AI & Deep Learning")

if __name__ == "__main__":
    main()
