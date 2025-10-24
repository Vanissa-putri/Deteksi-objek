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
APP_TITLE = "WaterVision — Image Detection & Classification"
YOLO_MODEL_PATH = "model/Vanissa Aulya Putri_Laporan 4.pt"
KERAS_MODEL_PATH = "model/Vanissa Aulya Putri_Laporan 2.h5"
WHATSAPP_NUMBER = "6282245357681"
EMAIL_WATERVISION = "watervision@gmail.com"
BG_IMAGE = "assets/water_bg.jpg"
BOTOL_IMAGE = "assets/botol_air.jpeg"

# ==============================
# STYLE (MODERN ESTHETIC)
# ==============================
def add_css():
    bg = ""
    if os.path.exists(BG_IMAGE):
        with open(BG_IMAGE, "rb") as f:
            bg = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

        html, body, [class*="css"] {{
            font-family: 'Poppins', sans-serif;
            color: #0e1b2b;
        }}
        .stApp {{
            background: linear-gradient(to bottom, rgba(255,255,255,0.9), rgba(240,250,255,0.95)),
                        url("data:image/jpg;base64,{bg}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        h1, h2, h3, p, label, span, div {{
            color: #0e1b2b !important;
        }}
        h1 {{
            text-align: center;
            font-weight: 700;
            color: #004b87 !important;
            margin-bottom: -10px;
        }}
        p {{
            text-align: center;
        }}
        .contact-btn {{
            padding: 10px 20px;
            border-radius: 8px;
            font-weight: 600;
            text-decoration: none;
            margin: 5px;
            display: inline-block;
            box-shadow: 0 3px 8px rgba(0,0,0,0.15);
        }}
        .whatsapp {{ background: #25d366; color: white; }}
        .gmail {{ background: #d93025; color: white; }}
        .info-box {{
            text-align: center;
            padding: 10px;
            background-color: rgba(255,255,255,0.85);
            border-radius: 10px;
            margin-top: 20px;
            font-weight: 500;
            box-shadow: 0 3px 10px rgba(0,0,0,0.1);
        }}
        .upload-section {{
            display: flex;
            justify-content: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }}
        .uploaded-img {{
            border-radius: 12px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
            display: block;
            margin: auto;
        }}
        hr {{
            border: 1px solid rgba(0,0,0,0.1);
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
    add_css()

    # Sidebar
    st.sidebar.title("⚙️ Settings")
    lang = st.sidebar.selectbox("🌐 Language", ["English", "Indonesia"])

    st.sidebar.markdown("---")
    st.sidebar.title("🎯 Mode Aplikasi")
    mode = st.sidebar.radio("", ["Object Detection (YOLO)", "Image Classification"])

    st.sidebar.markdown("### 📘 Panduan")
    if lang == "English":
        st.sidebar.write("1️⃣ Choose Mode\n2️⃣ Upload Image (JPG/PNG)\n3️⃣ Wait for Result")
    else:
        st.sidebar.write("1️⃣ Pilih Mode\n2️⃣ Unggah Gambar (JPG/PNG)\n3️⃣ Tunggu Hasil")

    # Header
    st.markdown(f"<h1>{APP_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown("<p>Smart Vision for Every Drop 💧</p>", unsafe_allow_html=True)

    # Gambar Botol Air (center)
    if os.path.exists(BOTOL_IMAGE):
        with open(BOTOL_IMAGE, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <div style='display:flex; justify-content:center;'>
            <img src='data:image/jpeg;base64,{encoded}' width='270' class='uploaded-img'>
        </div>
        """, unsafe_allow_html=True)

    # Tombol Kontak
    st.markdown(f"""
    <div style="text-align:center; margin-top:10px;">
        <a class="contact-btn whatsapp" href="https://wa.me/{WHATSAPP_NUMBER}" target="_blank">💬 WhatsApp</a>
        <a class="contact-btn gmail" href="mailto:{EMAIL_WATERVISION}" target="_blank">✉ Gmail</a>
    </div>
    """, unsafe_allow_html=True)

    # Upload Section (center)
    st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
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

        if "Object" in mode:
            st.subheader("🔍 Object Detection Result" if lang == "English" else "🔍 Hasil Deteksi Objek")
            if yolo:
                with st.spinner("Detecting objects..." if lang == "English" else "Mendeteksi objek..."):
                    res = yolo(img)
                    result_img = Image.fromarray(res[0].plot())
                    st.image(result_img, use_container_width=True)
            else:
                st.error("❌ YOLO model not found!" if lang == "English" else "❌ Model YOLO tidak ditemukan!")

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
                st.error("❌ Keras model not found!" if lang == "English" else "❌ Model Keras tidak ditemukan!")

        progress.progress(100)
        st.success("✅ Done!" if lang == "English" else "✅ Selesai!")
    else:
        msg = "📘 Please upload an image first." if lang == "English" else "📘 Silakan unggah gambar terlebih dahulu."
        st.markdown(f"<div class='info-box'>{msg}</div>", unsafe_allow_html=True)

    # Footer
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:13px;'>© 2025 WaterVision — Powered by AI & Deep Learning</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
