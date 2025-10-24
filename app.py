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
APP_TITLE = "💧 WaterVision — Smart Water Classification"
YOLO_MODEL_PATH = "model/Vanissa Aulya Putri_Laporan 4.pt"
KERAS_MODEL_PATH = "model/Vanissa Aulya Putri_Laporan 2.h5"
BG_IMAGE = "assets/water_bg.jpg"
BOTOL_IMAGE = "assets/botol_air.jpeg"
WHATSAPP_NUMBER = "6282245357681"
EMAIL_WATERVISION = "watervision@gmail.com"


# ==============================
# STYLE
# ==============================
def add_custom_css():
    bg = ""
    if os.path.exists(BG_IMAGE):
        with open(BG_IMAGE, "rb") as f:
            bg = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');
        .stApp {{
            background-image: url("data:image/jpg;base64,{bg}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            font-family: 'Poppins', sans-serif;
        }}
        h1 {{
            text-align: center;
            color: #0a3d62;
            font-weight: 700;
            text-shadow: 1px 1px 3px rgba(255,255,255,0.7);
        }}
        .info-box {{
            text-align: center;
            padding: 15px;
            background-color: rgba(255,255,255,0.85);
            border-radius: 10px;
            margin-top: 25px;
            font-weight: 500;
            backdrop-filter: blur(6px);
        }}
        .about-link {{
            position: fixed;
            top: 15px;
            right: 20px;
            background-color: rgba(0, 153, 255, 0.9);
            color: white !important;
            padding: 8px 14px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            font-size: 14px;
            transition: 0.2s ease;
            box-shadow: 0px 3px 8px rgba(0,0,0,0.3);
        }}
        .about-link:hover {{
            background-color: rgba(0, 120, 200, 1);
        }}
        .done-box {{
            text-align:center;
            background-color: rgba(255,255,255,0.85);
            color: #0066cc;
            font-weight: 600;
            padding: 10px 15px;
            border-radius: 10px;
            width: fit-content;
            margin: 15px auto;
            box-shadow: 0 2px 6px rgba(0,0,0,0.25);
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
# MAIN PAGE
# ==============================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    add_custom_css()

    st.markdown(f"<a class='about-link' href='?page=about'>Tentang Web Ini</a>", unsafe_allow_html=True)

    st.sidebar.title("⚙️ Pengaturan")
    lang = st.sidebar.selectbox("🌐 Bahasa", ["Indonesia", "English"])
    st.sidebar.markdown("---")
    st.sidebar.title("🎯 Mode Aplikasi")
    mode = st.sidebar.radio("", ["Object Detection (YOLO)", "Image Classification"])
    st.sidebar.markdown("### 📘 Panduan")
    if lang == "English":
        st.sidebar.write("1️⃣ Choose mode\n2️⃣ Upload an image (JPG/PNG)\n3️⃣ Wait for the result")
    else:
        st.sidebar.write("1️⃣ Pilih mode\n2️⃣ Unggah gambar (JPG/PNG)\n3️⃣ Tunggu hasilnya")

    st.markdown(f"<h1>{APP_TITLE}</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; font-size:18px;'>Aplikasi untuk mendeteksi dan mengklasifikasikan tingkat air secara cerdas 💧</p>", unsafe_allow_html=True)

    if os.path.exists(BOTOL_IMAGE):
        encoded = base64.b64encode(open(BOTOL_IMAGE, "rb").read()).decode()
        st.markdown(f"""
        <div style='display:flex; justify-content:center; align-items:center; margin:20px 0;'>
            <img src='data:image/jpeg;base64,{encoded}' width='220' style='border-radius:12px; box-shadow:0 4px 10px rgba(0,0,0,0.3);'>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        🌊 Website ini digunakan untuk mendeteksi dan mengklasifikasikan tingkat air ke dalam tiga kelas utama:  
        <b>Full Water</b> 💦, <b>Half Water</b> 💧, dan <b>Overflowing</b> 🚰.
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("📤 Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        st.image(img, caption="📷 Uploaded Image", width=350)  # <<< ukuran sedang
        st.markdown("</div>", unsafe_allow_html=True)

        progress = st.progress(0)
        for i in range(0, 101, 25):
            time.sleep(0.05)
            progress.progress(i)

        yolo, classifier = load_models()

        if "Object" in mode:
            st.subheader("🔍 Hasil Deteksi Objek")
            if yolo:
                with st.spinner("Mendeteksi objek..."):
                    res = yolo(img)
                    result_img = Image.fromarray(res[0].plot())
                    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
                    st.image(result_img, caption="Hasil Deteksi", width=350)
                    st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.error("❌ Model YOLO tidak ditemukan!")
        else:
            st.subheader("🧠 Hasil Klasifikasi Gambar")
            if classifier:
                img_resized = img.resize((224, 224))
                arr = keras_image.img_to_array(img_resized)
                arr = np.expand_dims(arr, axis=0) / 255.0
                pred = classifier.predict(arr)
                class_idx = np.argmax(pred)
                conf = np.max(pred)
                classes = ["Half Water 💧", "Full Water 💦", "Overflowing 🚰"]
                result_class = classes[class_idx] if class_idx < len(classes) else "Tidak Dikenal"
                st.success(f"✅ Kelas terdeteksi: **{result_class}** (probabilitas {conf:.2%})")
            else:
                st.error("❌ Model Keras tidak ditemukan!")

        progress.progress(100)
        st.markdown("<div class='done-box'>✅ Selesai!</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; margin-top:10px;'><a href='?page=about' style='color:#007acc; font-weight:600;'>ℹ Lihat Tentang Web Ini</a></div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='info-box'>📘 Silakan unggah gambar terlebih dahulu.</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<footer>© 2025 WaterVision — Created by Vanissa Aulya Putri 💧</footer>", unsafe_allow_html=True)


# ==============================
# ABOUT PAGE
# ==============================
def about_page():
    st.set_page_config(page_title="About — WaterVision", layout="centered")
    add_custom_css()
    st.markdown("<h1>🌐 Tentang WaterVision</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='info-box' style='max-width:700px; margin:auto;'>
        <p><b>WaterVision</b> adalah aplikasi berbasis AI yang dikembangkan untuk mendeteksi dan mengklasifikasikan tingkat air menggunakan teknologi <b>Deep Learning</b> dan <b>Computer Vision</b>.</p>
        <p>Aplikasi ini mengenali tiga kondisi utama air pada wadah:</p>
        <ul style='text-align:left;'>
            <li>💧 <b>Half Water</b> — air setengah penuh</li>
            <li>💦 <b>Full Water</b> — air penuh</li>
            <li>🚰 <b>Overflowing</b> — air meluap</li>
        </ul>
        <p>Website ini dibuat untuk membantu pengguna dalam melakukan pengawasan otomatis berbasis citra visual, terutama di bidang pengelolaan air dan sistem sensor cerdas.</p>
        <br>
        <p><b>Developer:</b> Vanissa Aulya Putri<br>
        <b>Email:</b> watervision@gmail.com</p>
        <a href="/" style="display:inline-block;margin-top:15px;background:#0099ff;color:white;padding:8px 16px;border-radius:8px;text-decoration:none;">⬅ Kembali ke Halaman Utama</a>
    </div>
    """, unsafe_allow_html=True)


# ==============================
# ROUTER
# ==============================
if __name__ == "__main__":
    query_params = st.query_params
    if "page" in query_params and query_params["page"] == "about":
        about_page()
    else:
        main()
