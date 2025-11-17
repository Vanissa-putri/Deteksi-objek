import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
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
        .contact-btn {{
            padding: 8px 15px;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 600;
            font-size: 14px;
            margin-right: 6px;
        }}
        .whatsapp {{ background-color: #25d366; color: white; }}
        .gmail {{ background-color: #d93025; color: white; }}
        .info-box {{
            text-align: center;
            padding: 15px;
            background-color: rgba(255,255,255,0.85);
            border-radius: 10px;
            margin-top: 25px;
        }}
        .highlight-box {{
            text-align:center;
            background-color: white;
            padding: 10px 18px;
            border-radius: 10px;
            display:inline-block;
            box-shadow:0 3px 8px rgba(0,0,0,0.2);
            margin-top:10px;
            font-weight:600;
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

    st.sidebar.title("⚙️ Pengaturan")
    lang = st.sidebar.selectbox("🌐 Bahasa", ["Indonesia", "English"])

    # ==========================
    # TEXT DICTIONARY
    # ==========================
    if lang == "English":
        title_text = "💧 WaterVision — Smart Water Classification"
        subtitle_text = "Application for intelligent water level detection and classification 💧"
        info_box_text = "🌊 This website detects and classifies water levels into: <b>Full</b>, <b>Half</b>, and <b>Overflowing</b>."
        upload_text = "📤 Upload Image (JPG/PNG)"
        mode_object = "Object Detection (YOLO)"
        mode_classify = "Image Classification"
        spinner_detecting = "Detecting objects..."
        subheader_classification = "🧠 Image Classification Result"
        detected_class_text = "Detected Class"
        done_text = "Done!"
        placeholder_no_image = "Please upload an image first."
        yolo_error_text = "YOLO model not found!"
        keras_error_text = "Keras model not found!"
        about_link_text = "Learn More About This Web"
        download_button_text = "Download Result Image"
    else:
        title_text = "💧 WaterVision — Smart Water Classification"
        subtitle_text = "Aplikasi untuk mendeteksi tingkat air secara cerdas 💧"
        info_box_text = "🌊 Website ini mengklasifikasikan air menjadi: <b>Full</b>, <b>Half</b>, dan <b>Overflowing</b>."
        upload_text = "📤 Unggah gambar (JPG/PNG)"
        mode_object = "Deteksi Objek (YOLO)"
        mode_classify = "Klasifikasi Gambar"
        spinner_detecting = "Mendeteksi objek..."
        subheader_classification = "🧠 Hasil Klasifikasi Gambar"
        detected_class_text = "Kelas terdeteksi"
        done_text = "Selesai!"
        placeholder_no_image = "Unggah gambar terlebih dahulu."
        yolo_error_text = "Model YOLO tidak ditemukan!"
        keras_error_text = "Model Keras tidak ditemukan!"
        about_link_text = "Pelajari Tentang Web Ini"
        download_button_text = "Download Hasil Gambar"

    # ==========================
    # MODE SELECTION
    # ==========================
    st.sidebar.markdown("---")
    mode = st.sidebar.radio("", [mode_object, mode_classify])

    # ==========================
    # MAIN CONTENT
    # ==========================
    st.markdown(f"<h1>{title_text}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center; font-size:18px;'>{subtitle_text}</p>", unsafe_allow_html=True)

    uploaded = st.file_uploader(upload_text, type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="📷 Uploaded Image", width=450)

        yolo, classifier = load_models()
        result_img = None

        if mode == mode_object:
            st.subheader("🔍 Hasil Deteksi Objek")
            if yolo:
                with st.spinner(spinner_detecting):
                    res = yolo(img)
                    result_img = Image.fromarray(res[0].plot())
                    st.image(result_img, width=450)
            else:
                st.error(yolo_error_text)

        else:
            st.subheader(subheader_classification)
            if classifier:
                target_size = classifier.input_shape[1:3]
                img_resized = img.resize(target_size)
                arr = tf.keras.utils.img_to_array(img_resized)
                arr = np.expand_dims(arr, axis=0) / 255.0
                arr = arr.astype("float32")

                pred = classifier.predict(arr)
                class_idx = np.argmax(pred)
                conf = np.max(pred)

                classes = ["Half Water 💧", "Full Water 💦", "Overflowing 🚰"]
                result_class = classes[class_idx] if class_idx < len(classes) else "Unknown"

                st.markdown(
                    f"<div class='highlight-box'>{detected_class_text}: <b>{result_class}</b> ({conf:.2%})</div>",
                    unsafe_allow_html=True
                )

                result_img = img.copy().resize((400, 400))
            else:
                st.error(keras_error_text)

        if result_img:
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            st.download_button(download_button_text, buf.getvalue(), "hasil_watervision.png", "image/png")

    else:
        st.markdown(f"<div class='info-box'>{placeholder_no_image}</div>", unsafe_allow_html=True)


# ==============================
# ROUTING
# ==============================
if __name__ == "__main__":
    main()
