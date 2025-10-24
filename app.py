import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import os

# ===================== CONFIGURASI DASAR =====================
st.set_page_config(page_title="Water Level Classifier", page_icon="💧", layout="wide")

# CSS custom (background, style, tombol, dll)
st.markdown("""
    <style>
        body {
            background-color: #f5f9ff;
        }
        [data-testid="stAppViewContainer"] {
            background-image: url("assets/background.jpg");
            background-size: cover;
            background-position: center;
        }
        .main-title {
            font-size: 38px;
            font-weight: bold;
            color: #003366;
            text-align: center;
            margin-bottom: -10px;
        }
        .subtitle {
            text-align: center;
            color: #555;
            font-size: 18px;
            margin-bottom: 30px;
        }
        .upload-box {
            background: rgba(255, 255, 255, 0.85);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        }
        .stButton>button {
            background-color: #007bff;
            color: white;
            border-radius: 10px;
            height: 3em;
            width: 10em;
            border: none;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #0056b3;
            color: white;
        }
        .link-top {
            position: absolute;
            top: 25px;
            right: 50px;
            font-size: 16px;
            color: #0047ab;
            text-decoration: none;
            font-weight: 600;
        }
        .link-top:hover {
            color: #002b80;
        }
    </style>
""", unsafe_allow_html=True)

# ===================== MENU NAVIGASI =====================
menu = st.sidebar.radio("📍 Navigasi", ["Dashboard", "Tentang Web"])

# ===================== HALAMAN DASHBOARD =====================
if menu == "Dashboard":
    st.markdown('<div class="main-title">💧 Water Level Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Aplikasi ini digunakan untuk mendeteksi tingkat air dalam botol: Full, Half, atau Overflowing.</div>', unsafe_allow_html=True)

    st.markdown('<a href="?menu=Tentang Web" class="link-top">Tentang Web →</a>', unsafe_allow_html=True)

    st.write("")
    st.write("### 📤 Upload Gambar untuk Deteksi")

    uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Diupload", use_container_width=False, width=400)

        model_path = "best.pt"
        if not os.path.exists(model_path):
            st.error("Model YOLO tidak ditemukan. Pastikan file 'best.pt' tersedia.")
        else:
            model = YOLO(model_path)
            results = model.predict(np.array(image))

            for r in results:
                res_plotted = r.plot()
                st.image(res_plotted, caption="Hasil Deteksi", use_container_width=False, width=500)

            label_counts = {}
            for r in results:
                for c in r.boxes.cls:
                    label = model.names[int(c)]
                    label_counts[label] = label_counts.get(label, 0) + 1

            st.write("### 📊 Hasil Klasifikasi:")
            for label, count in label_counts.items():
                st.write(f"- **{label}**: {count} objek terdeteksi")

# ===================== HALAMAN TENTANG WEB =====================
elif menu == "Tentang Web":
    st.title("ℹ️ Tentang Web Klasifikasi Air 💧")
    st.write("""
        Website ini dirancang untuk mengidentifikasi **tingkat air dalam botol** menggunakan model deep learning **YOLO (You Only Look Once)**.  
        Model ini mampu membedakan antara tiga kondisi air:
        """)

    st.markdown("""
        - 🟦 **Full Water** — botol terisi penuh air  
        - 🟨 **Half Water** — botol setengah terisi  
        - 🟥 **Overflowing** — air meluap dari botol  
    """)

    st.write("""
        Tujuan dari web ini adalah untuk membantu proses **deteksi otomatis tingkat air** dengan cara yang cepat dan efisien, 
        serta menampilkan hasil deteksi langsung dari gambar yang diunggah.
    """)

    st.markdown("---")
    st.subheader("👩‍💻 Pengembang")
    st.write("""
        **Nama:** [Nama Kamu]  
        **Email:** [emailkamu@example.com]  
        **Model:** YOLOv8 - Deep Learning for Object Detection
    """)
