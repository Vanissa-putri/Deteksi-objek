import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
import base64, os, time, io

# ==============================
# CONFIGURATIONS
# ==============================
APP_TITLE = "💧 WaterVision — Smart Water Classification"
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
        h2, h3, h4 {{
            color: #052c48;
        }}
        .stButton>button {{
            background: linear-gradient(90deg, #0099ff, #33bbff);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6em 1.2em;
            font-weight: bold;
            box-shadow: 0 4px 10px rgba(0,0,0,0.25);
        }}
        .stButton>button:hover {{
            background: linear-gradient(90deg, #007acc, #1ebfff);
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
        .highlight-box {{
            text-align:center;
            background-color: rgba(255,255,255,0.9);
            padding: 10px 18px;
            border-radius: 10px;
            display:inline-block;
            box-shadow:0 3px 8px rgba(0,0,0,0.2);
            margin-top:10px;
            font-weight:600;
        }}
        footer {{
            text-align: center;
            color: white;
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
            <img src='data:image/jpeg;base64,{encoded}' width='200' style='border-radius:12px; box-shadow:0 4px 10px rgba(0,0,0,0.3);'>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        🌊 Website ini digunakan untuk mendeteksi dan mengklasifikasikan tingkat air ke dalam tiga kelas utama:  
        <b>Full Water</b> 💦, <b>Half Water</b> 💧, dan <b>Overflowing</b> 🚰.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center; margin-top:25px;">
        <a class="contact-btn whatsapp" href="https://wa.me/{WHATSAPP_NUMBER}" target="_blank">💬 WhatsApp</a>
        <a class="contact-btn gmail" href="mailto:{EMAIL_WATERVISION}" target="_blank">✉ Gmail</a>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("📤 Upload Image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        # 🔹 Gambar upload jadi sedang (bukan full-width)
        st.image(img, caption="📷 Uploaded Image", width=450)

        progress = st.progress(0)
        for i in range(0, 101, 25):
            time.sleep(0.05)
            progress.progress(i)

        yolo, classifier = load_models()

        result_img = None

        if "Object" in mode:
            st.subheader("🔍 Hasil Deteksi Objek" if lang == "Indonesia" else "🔍 Object Detection Result")
            if yolo:
                with st.spinner("Mendeteksi objek..." if lang == "Indonesia" else "Detecting objects..."):
                    res = yolo(img)
                    result_img = Image.fromarray(res[0].plot())
                    st.image(result_img, use_container_width=False, width=450)
            else:
                st.error("❌ Model YOLO tidak ditemukan!")
        else:
            st.subheader("🧠 Hasil Klasifikasi Gambar" if lang == "Indonesia" else "🧠 Image Classification Result")
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
                result_img = img.copy().resize((400, 400))
            else:
                st.error("❌ Model Keras tidak ditemukan!")

        progress.progress(100)

        st.markdown("<div style='text-align:center; margin-top:15px;'><div class='highlight-box'>✅ Selesai!</div></div>", unsafe_allow_html=True)

        if result_img:
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            st.download_button("📥 Download Hasil Gambar", data=byte_im, file_name="hasil_watervision.png", mime="image/png")

        st.markdown("""
        <div style='text-align:center; margin-top:25px;'>
            <div class='highlight-box'>
                <a href='?page=about' style='color:#007acc; text-decoration:none; font-weight:600;'>
                🔗 Pelajari Lebih Lanjut Tentang Web Ini
                </a>
            </div>
        </div>
        """, unsafe_allow_html=True)
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
        <p>Website ini dibuat untuk membantu pengguna dalam melakukan pengawasan otomatis berbasis citra visual.</p>
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
